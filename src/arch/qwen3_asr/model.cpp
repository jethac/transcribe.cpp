#include "asr_resource_snapshot.h"
// arch/qwen3_asr/model.cpp - Qwen3-ASR family handler.

#include "causal_lm/causal_lm.h"
#include "decoder.h"
#include "encoder.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "qwen3_asr.h"
#include "transcribe-arch.h"
#include "transcribe-backend.h"
#include "transcribe-batch-util.h"
#include "transcribe-debug.h"
#include "transcribe-decode-budget.h"
#include "transcribe-env.h"
#include "transcribe-flash-policy.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-log.h"
#include "transcribe-mel.h"
#include "transcribe-meta.h"
#include "transcribe-prompting.h"
#include "transcribe-repetition-guard.h"
#include "weights.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace transcribe::qwen3_asr {

extern const Arch arch;

struct WarmupState {
    transcribe_session * session;
    bool streaming;
    int completed_steps = 0;
};

static thread_local WarmupState * active_warmup = nullptr;

static bool khr_gpu_model_policy(const QwenAsrModel * cm) {
    const char * policy = std::getenv("TRANSCRIBE_VULKAN_KHR_PROFILE");
    return policy && !std::strcmp(policy, "1") && cm->plan.requested == TRANSCRIBE_BACKEND_VULKAN;
}

static bool khr_gpu_graph_policy(const QwenAsrSession * cc) {
    return khr_gpu_model_policy(static_cast<const QwenAsrModel *>(cc->model));
}

static int initial_khr_kv_capacity(int64_t needed, int limit) {
    int64_t capacity = 16;
    while (capacity < needed && capacity < limit) capacity *= 2;
    return static_cast<int>(std::min<int64_t>(capacity, limit));
}

static transcribe_status grow_khr_kv(QwenAsrSession * cc, int capacity) {
    auto * cm = static_cast<QwenAsrModel *>(cc->model);
    if (cc->sched) {
        ggml_backend_sched_synchronize(cc->sched);
        ggml_backend_sched_reset(cc->sched);
    }
    const int old_capacity = cc->kv_cache.n_ctx;
    if (!transcribe::causal_lm::kv_grow(cc->kv_cache, cm->plan.primary, capacity,
                                     cm->hparams.dec_n_kv_heads, cm->hparams.dec_head_dim,
                                     cm->hparams.dec_n_layers)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr: KV growth failed (%d -> %d)", old_capacity, capacity);
        return TRANSCRIBE_ERR_OOM;
    }
    log_msg(TRANSCRIBE_LOG_LEVEL_INFO,
            "qwen3_asr KHR KV grow: old_capacity=%d capacity=%d rows=%d head=%d",
            old_capacity, cc->kv_cache.n_ctx, cc->kv_cache.n, cc->kv_cache.head);
    return TRANSCRIBE_OK;
}

static transcribe_status release_subsample_scratch(QwenAsrSession * cc) {
    if (!khr_gpu_graph_policy(cc) || !cc->sched) return TRANSCRIBE_OK;
    auto * cm = static_cast<QwenAsrModel *>(cc->model);
    ggml_backend_sched_synchronize(cc->sched);
    for (int i = 0; i < ggml_backend_sched_get_n_backends(cc->sched); ++i) {
        const auto backend = ggml_backend_sched_get_backend(cc->sched, i);
        log_msg(TRANSCRIBE_LOG_LEVEL_INFO,
                "qwen3_asr KHR phase release: phase=subsample backend=%s scratch_bytes=%zu",
                ggml_backend_name(backend), ggml_backend_sched_get_buffer_size(cc->sched, backend));
    }
    transcribe::safe_sched_free(cc->sched);
    cc->sched = nullptr;
    cc->sched = ggml_backend_sched_new(cm->plan.scheduler_list.data(), nullptr,
                                      static_cast<int>(cm->plan.scheduler_list.size()), 16384, false, true);
    if (!cc->sched) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr: encoder scheduler recreation failed");
        return TRANSCRIBE_ERR_BACKEND;
    }
    return TRANSCRIBE_OK;
}

static size_t compute_metadata_bytes(QwenAsrSession * cc, size_t default_bytes, int n_conv_chunks,
                                     bool encoder, size_t decoder_graph_size) {
    if (!khr_gpu_graph_policy(cc)) return default_bytes;
    const auto * cm = static_cast<const QwenAsrModel *>(cc->model);
    if (n_conv_chunks < 0 || cm->hparams.dec_n_layers < 0 ||
        cc->encoder_use_flash || cc->decoder_use_flash) return 0;
    size_t bytes = 0;
    size_t tensor_slots = 256;
    const auto add = [](size_t & dst, size_t value) {
        if (value > std::numeric_limits<size_t>::max() - dst) return false;
        dst += value;
        return true;
    };
    const auto add_tensors = [&](size_t count, size_t per_item) {
        return count <= std::numeric_limits<size_t>::max() / per_item &&
               add(tensor_slots, count * per_item);
    };
    if (encoder && (!add_tensors(cm->weights.enc_blocks.size(), 256) ||
                    !add(bytes, ggml_graph_overhead_custom(8192, false)))) return 0;
    if (decoder_graph_size &&
        (!add_tensors(static_cast<size_t>(cm->hparams.dec_n_layers), 256) ||
         !add(bytes, ggml_graph_overhead_custom(decoder_graph_size, false)))) return 0;
    const int passes[] = { n_conv_chunks >= kConvPassChunks ? kConvPassChunks : 0,
                           n_conv_chunks % kConvPassChunks };
    for (int chunks : passes) {
        if (!chunks) continue;
        const size_t groups = (static_cast<size_t>(chunks) + kConvChunkGroup - 1) / kConvChunkGroup;
        if (!add_tensors(groups, 128) || !add(bytes, ggml_graph_overhead_custom(64 + 48 * groups, false))) return 0;
    }
    const size_t tensor_bytes = ggml_tensor_overhead();
    if (tensor_slots > std::numeric_limits<size_t>::max() / tensor_bytes ||
        !add(bytes, tensor_slots * tensor_bytes)) return 0;
    log_msg(TRANSCRIBE_LOG_LEVEL_INFO,
            "qwen3_asr KHR metadata: bytes=%zu tensor_slots=%zu encoder=%d decoder_graph=%zu conv_chunks=%d",
            bytes, tensor_slots, encoder ? 1 : 0, decoder_graph_size, n_conv_chunks);
    return bytes;
}

static bool check_qwen_graph_placement(QwenAsrSession * cc, ggml_cgraph * graph) {
    if (khr_gpu_graph_policy(cc)) {
        const auto * cm = static_cast<const QwenAsrModel *>(cc->model);
        const char * primary = cm->plan.primary ? ggml_backend_name(cm->plan.primary) : "none";
        if (!cm->plan.primary || cm->plan.primary_kind != BackendKind::Vulkan) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr KHR profile requires Vulkan, got %s", primary);
            return false;
        }
        size_t gpu = 0, metadata = 0, rejected = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
                node->op == GGML_OP_VIEW || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_PERMUTE) {
                ++metadata;
                continue;
            }
            ggml_backend_t actual = ggml_backend_sched_get_tensor_backend(cc->sched, node);
            if (actual != cm->plan.primary) {
                ++rejected;
                log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                        "qwen3_asr KHR placement rejected before graph compute: node=%d name=%s op=%s backend=%s",
                        i, node->name, ggml_op_name(node->op), actual ? ggml_backend_name(actual) : "unassigned");
            } else {
                ++gpu;
            }
        }
        log_msg(rejected ? TRANSCRIBE_LOG_LEVEL_ERROR : TRANSCRIBE_LOG_LEVEL_INFO,
                "qwen3_asr KHR placement: compute_gpu=%zu compute_rejected=%zu metadata=%zu policy=no_cpu_fallback",
                gpu, rejected, metadata);
        if (rejected) return false;
    }
    return true;
}


static ggml_status compute_qwen_graph(QwenAsrSession * cc, ggml_cgraph * graph) {
    if (!check_qwen_graph_placement(cc, graph)) return GGML_STATUS_FAILED;
    static std::atomic<unsigned> diagnostic_graphs{0};
    const bool diagnostics = khr_gpu_graph_policy(cc) && transcribe::env::flag("TRANSCRIBE_RESOURCE_DIAGNOSTICS") &&
        diagnostic_graphs.fetch_add(1, std::memory_order_relaxed) < 64;
    char label[160] = {};
    if (diagnostics) {
        const char *mode = active_warmup && active_warmup->session == cc ?
            (active_warmup->streaming ? "warm_streaming" : "warm_offline") : "request";
        std::snprintf(label, sizeof(label), "mode=%s nodes=%d kv_n=%d kv_head=%d kv_capacity=%d",
                      mode, ggml_graph_n_nodes(graph), cc->kv_cache.n, cc->kv_cache.head, cc->kv_cache.n_ctx);
        asr_resource_snapshot("graph_begin", label);
    }
    const ggml_status status = ggml_backend_sched_graph_compute(cc->sched, graph);
    if (diagnostics) asr_resource_snapshot("graph_end", label);
    return status;
}

static_assert(std::is_base_of_v<transcribe_model, QwenAsrModel>);
static_assert(std::is_base_of_v<transcribe_session, QwenAsrSession>);

QwenAsrSession::~QwenAsrSession() {
    kv_cache.free();
    kv_cache_batch.free();
}

QwenAsrModel::~QwenAsrModel() {
    if (ctx_meta != nullptr) {
        ggml_free(ctx_meta);
        ctx_meta = nullptr;
    }
    weight_storage.reset();
    packed_gate_up.free();
    for (auto it = plan.scheduler_list.rbegin(); it != plan.scheduler_list.rend(); ++it) {
        safe_backend_free(*it);
    }
    plan.scheduler_list.clear();
    plan.primary      = nullptr;
    plan.primary_kind = transcribe::BackendKind::Unknown;
}

namespace {

constexpr const char k_default_variant[] = "qwen3-asr";

// Input-length contract (see docs/input-limits.md). Qwen3-ASR is a
// hard-context-cap family: audio tokens + chat prompt + generation share the
// Qwen3 decoder's context window (dec_max_position_embeddings), clamped to that
// ceiling. Over-length input is rejected with TRANSCRIBE_ERR_INPUT_TOO_LONG; a
// transcript that fills the generation budget before end-of-stream is flagged
// via transcribe_was_truncated().

// Generation reserve: what the input gate keeps free, and the decode-budget floor.
constexpr int k_gen_reserve = 256;

// Effective decoder context ceiling, in tokens: the model's trained maximum,
// optionally lowered — never raised — by the caller's session n_ctx knob.
int qwen3_context_ceiling(int32_t n_ctx_knob, const QwenAsrHParams & hp) {
    int ceiling = hp.dec_max_position_embeddings;
    if (n_ctx_knob > 0 && n_ctx_knob < ceiling) {
        ceiling = n_ctx_knob;
    }
    return ceiling;
}

// Advisory transcribe_capabilities::max_audio_ms: the longest audio whose
// audio tokens plus a representative prompt and the generation reserve fit
// the context ceiling. The audio encoder downsamples mel frames 8x (three
// stride-2 convs, see aftercnn_len); inverting that gives ms. Returns 0
// ("unknown / unbounded") if the rate constants are missing. Note: even
// within this bound a long transcript may truncate at the generation budget
// (transcribe_was_truncated) — max_audio_ms is the input bound.
int64_t qwen3_max_audio_ms(const QwenAsrHParams & hp) {
    if (hp.dec_max_position_embeddings <= 0 || hp.fe_hop_length <= 0 || hp.fe_sample_rate <= 0) {
        return 0;
    }
    constexpr int k_prompt_overhead = 48;  // chat affixes; advisory
    const int     max_audio_tokens  = hp.dec_max_position_embeddings - k_prompt_overhead - k_gen_reserve;
    if (max_audio_tokens <= 0) {
        return 0;
    }
    // audio_tokens ≈ mel_frames / 8 ; mel_frames = ms * sr / (hop * 1000)
    //   => ms ≈ audio_tokens * 8 * hop * 1000 / sr
    const int64_t mel_frames = static_cast<int64_t>(max_audio_tokens) * 8;
    return mel_frames * hp.fe_hop_length * 1000 / hp.fe_sample_rate;
}

// Forward declarations for helpers defined further down in this file.
transcribe_status resolve_chat_tokens(const transcribe::Tokenizer & tok, ChatTokens & out);

transcribe_status load(Loader & loader, const transcribe_model_load_params * params, transcribe_model ** out_model) {
    const int64_t t_load_start = INT64_C(0);

    auto m       = std::make_unique<QwenAsrModel>();
    m->arch      = &arch;
    m->t_load_us = 0;
    m->variant   = loader.variant().empty() ? k_default_variant : loader.variant();
    m->backend.clear();

    apply_family_invariants(*m);
    m->caps.n_languages = 0;
    m->caps.languages   = nullptr;

    if (const transcribe_status st = read_capability_kv(loader.gguf(), m->caps); st != TRANSCRIBE_OK) {
        return st;
    }
    if (const transcribe_status st = read_languages_kv(loader.gguf(), *m); st != TRANSCRIBE_OK) {
        return st;
    }

    // Tokenizer (byte-level BPE; loader handles the "gpt2" model tag).
    if (const transcribe_status st = m->tok.load(loader.gguf()); st != TRANSCRIBE_OK) {
        return st;
    }

    // Chat template (read here so its absence surfaces at load, not mid-decode).
    (void) read_optional_string_kv(loader.gguf(), "tokenizer.chat_template", "qwen3_asr", "", m->chat_template);

    // Resolve chat-template special-token ids at load so a vocab drift surfaces
    // here instead of silently producing a wrong prompt at decode time.
    if (const transcribe_status st = resolve_chat_tokens(m->tok, m->chat_tokens); st != TRANSCRIBE_OK) {
        return st;
    }

    if (const transcribe_status st = read_qwen3_asr_hparams(loader.gguf(), m->hparams); st != TRANSCRIBE_OK) {
        return st;
    }

    // Publish the input-length ceiling now that the decoder context window
    // and frontend rate are known.
    m->caps.max_audio_ms = qwen3_max_audio_ms(m->hparams);

    // Basis for transcribe_session_get_limits: the same constants
    // qwen3_max_audio_ms uses, so the limit recomputes at a lowered n_ctx.
    if (m->hparams.dec_max_position_embeddings > 0 && m->hparams.fe_hop_length > 0 && m->hparams.fe_sample_rate > 0) {
        m->limits.has_context_cap    = true;
        m->limits.model_max_ctx      = m->hparams.dec_max_position_embeddings;
        m->limits.prompt_overhead    = 48;
        m->limits.gen_reserve        = k_gen_reserve;
        // audio_tokens ≈ mel_frames / 8 ; mel_frames = ms*sr/(hop*1000)
        m->limits.ms_per_audio_token = 8.0 * m->hparams.fe_hop_length * 1000.0 / m->hparams.fe_sample_rate;
        m->limits.kv_elems_per_ctx_token =
            (int64_t) m->hparams.dec_n_kv_heads * m->hparams.dec_head_dim * m->hparams.dec_n_layers * 2;
    }

    m->hparams.vocab_size   = m->tok.n_tokens();
    m->hparams.bos_token_id = m->tok.bos_id();
    m->hparams.eos_token_id = m->tok.eos_id();

    if (m->hparams.vocab_size != m->hparams.dec_vocab_size) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr: tokenizer vocab (%d) != decoder vocab_size (%d)",
                m->hparams.vocab_size, m->hparams.dec_vocab_size);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (m->hparams.eos_token_id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr: GGUF tokenizer has no eos_token_id");
        return TRANSCRIBE_ERR_GGUF;
    }

    // Mel frontend (Whisper-style 128-bin log-mel at 16 kHz, 30 s max).
    {
        transcribe::MelConfig cfg{};
        cfg.sample_rate  = m->hparams.fe_sample_rate;
        cfg.num_mels     = m->hparams.fe_num_mels;
        cfg.n_fft        = m->hparams.fe_n_fft;
        cfg.win_length   = m->hparams.fe_win_length;
        cfg.hop_length   = m->hparams.fe_hop_length;
        cfg.pre_emphasis = m->hparams.fe_pre_emphasis;
        cfg.f_min        = m->hparams.fe_f_min;
        cfg.f_max        = m->hparams.fe_f_max;
        cfg.pad_mode     = m->hparams.fe_pad_mode;
        cfg.window_type  = m->hparams.fe_window;     // "hann_periodic"
        cfg.normalize    = m->hparams.fe_normalize;  // "per_utterance" (Whisper)

        // Optional filterbank + window buffers baked by the converter; if
        // present MelFrontend uses them instead of reconstructing from hparams.
        {
            using R               = transcribe::load_common::ReadF32Result;
            const size_t fb_elems = static_cast<size_t>(cfg.num_mels) * static_cast<size_t>(cfg.n_fft / 2 + 1);
            const auto   fb_rc    = transcribe::load_common::read_f32_tensor_checked(
                loader.gguf(), loader.path(), "frontend.mel_filterbank", fb_elems, "qwen3_asr", cfg.filterbank);
            if (fb_rc != R::Ok && fb_rc != R::Absent) {
                return TRANSCRIBE_ERR_GGUF;
            }
            const size_t win_elems = static_cast<size_t>(cfg.win_length);
            const auto   win_rc    = transcribe::load_common::read_f32_tensor_checked(
                loader.gguf(), loader.path(), "frontend.window", win_elems, "qwen3_asr", cfg.window);
            if (win_rc != R::Ok && win_rc != R::Absent) {
                return TRANSCRIBE_ERR_GGUF;
            }
        }

        m->mel.emplace(cfg);
        if (cfg.normalize == "per_utterance" && cfg.pad_mode == "reflect") {
            transcribe::MelConfig raw = cfg;
            raw.normalize             = "log10";
            m->mel_log10.emplace(raw);
        }
        // Streaming runs the conv stem on each 1 s chunk while the audio
        // arrives and the rest at finalize, with the offline transcript (see
        // stream_feed). It needs the raw frontend above.
        m->caps.supports_streaming = m->mel_log10.has_value();
    }

    // Reopen with no_alloc to build the tensor catalog.
    gguf_init_params init_params{};
    init_params.no_alloc = true;
    init_params.ctx      = &m->ctx_meta;

    gguf_context * gguf_data = gguf_init_from_file(loader.path().c_str(), init_params);
    if (gguf_data == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    if (const transcribe_status st = build_qwen3_asr_weights(m->ctx_meta, m->hparams, m->weights);
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }

    // Backend plan.
    const transcribe_backend_request backend_req = (params != nullptr) ? params->backend : TRANSCRIBE_BACKEND_AUTO;
    if (const transcribe_status st = transcribe::load_common::init_backends(
            backend_req, (params != nullptr) ? params->device : nullptr, "qwen3_asr", m->plan);
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }
    if (khr_gpu_model_policy(m.get()) && (!m->plan.primary || m->plan.primary_kind != BackendKind::Vulkan)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "qwen3_asr KHR profile: explicitly requested Vulkan resolved to %s before weight allocation",
                m->plan.primary ? ggml_backend_name(m->plan.primary) : "none");
        gguf_free(gguf_data);
        return TRANSCRIBE_ERR_BACKEND;
    }
    m->backend         = ggml_backend_name(m->plan.primary);
    m->primary_backend = m->plan.primary;

    if (const transcribe_status st = transcribe::load_common::load_weights(
            loader.path(), gguf_data, m->ctx_meta, m->plan, params, "qwen3_asr", m->weight_storage);
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }
    gguf_free(gguf_data);

    // Pack gate+up into a separate session + backend buffer so the FFN
    // can run a single mul_mat instead of two. ctx_meta is sized
    // exactly for GGUF file tensors with no headroom, so packed
    // tensors live in their own context owned by `causal_lm::pack_gate_up`.
    {
        std::vector<transcribe::causal_lm::GateUpEntry> entries;
        entries.reserve(m->weights.dec_blocks.size());
        for (auto & b : m->weights.dec_blocks) {
            entries.push_back({ b.ffn_gate_w, b.ffn_up_w, &b.ffn_gate_up_w });
        }
        if (const transcribe_status st =
                transcribe::causal_lm::pack_gate_up(m->plan.primary, m->hparams.dec_hidden, m->hparams.dec_intermediate,
                                                    entries, m->packed_gate_up, "qwen3_asr");
            st != TRANSCRIBE_OK) {
            m->packed_gate_up.free();
            return st;
        }
    }

    m->t_load_us = INT64_C(0) - t_load_start;
    *out_model   = m.release();
    return TRANSCRIBE_OK;
}

transcribe_status init_context(transcribe_model *                model,
                               const transcribe_session_params * params,
                               transcribe_session **             out_ctx) {
    if (model->arch != &arch) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto cc       = std::make_unique<QwenAsrSession>();
    cc->model     = model;
    cc->n_threads = params->n_threads;
    cc->kv_type   = params->kv_type;
    cc->n_ctx     = transcribe_session_params_n_ctx(params);

    // Encoder flash is what keeps long audio bounded: without it every block
    // materialises [T_enc, T_enc, n_heads] f32 scores (~0.9 GB at 5 minutes,
    // ~13.6 GB at 20). Measured on M4 Metal at 306 s it is also faster
    // (encode 2843 -> 2120 ms) with an identical transcript; CPU is neutral.
    // TRANSCRIBE_NO_FLASH still turns it off.
    cc->encoder_use_flash = true;
    cc->decoder_use_flash = true;
    transcribe::flash::apply_env_overrides(cc->encoder_use_flash, cc->decoder_use_flash);
    if (khr_gpu_graph_policy(cc.get())) {
        const char * forced = std::getenv("TRANSCRIBE_FORCE_FLASH");
        if (forced && forced[0] && forced[0] != '0') {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr KHR profile conflicts with TRANSCRIBE_FORCE_FLASH");
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
        cc->encoder_use_flash = false;
        cc->decoder_use_flash = false;
        log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "qwen3_asr KHR profile: non-flash GPU graph, no CPU compute fallback");
    }


    // Pre-allocate KV cache at context creation so the first run
    // doesn't pay the allocation cost inside the decode phase.
    auto * cm = static_cast<QwenAsrModel *>(model);
    if (khr_gpu_graph_policy(cc.get())) {
        log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "qwen3_asr KHR profile: KV allocation deferred to request capacity");
    } else {
        ggml_type kv_type = GGML_TYPE_F16;
        if (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) {
            kv_type = GGML_TYPE_F32;
        }
        if (!transcribe::causal_lm::kv_init(cc->kv_cache, cm->plan.primary,
                                            /*n_ctx=*/2048, cm->hparams.dec_n_kv_heads, cm->hparams.dec_head_dim,
                                            cm->hparams.dec_n_layers, kv_type)) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "qwen3_asr init_context: KV cache allocation failed "
                                "(n_ctx=2048, %d kv-heads x %d head-dim x %d layers) — "
                                "out of memory.",
                                cm->hparams.dec_n_kv_heads, cm->hparams.dec_head_dim, cm->hparams.dec_n_layers);
            return TRANSCRIBE_ERR_OOM;
        }
    }

    *out_ctx = cc.release();
    return TRANSCRIBE_OK;
}

// Resolve the chat-template piece strings against the loaded tokenizer.
// Hard-fails with TRANSCRIBE_ERR_GGUF on a missing piece (a vocab reorder
// surfaces here at load). The newline is stored in its GPT-2 byte-level form
// (\n → U+010A "Ċ", \xC4\x8A); roles and <|im_*|> tokens are verbatim.
transcribe_status resolve_chat_tokens(const transcribe::Tokenizer & tok, ChatTokens & out) {
    struct PieceSlot {
        const char * piece;
        int32_t *    slot;
    };

    const PieceSlot pieces[] = {
        { "<|im_start|>", &out.im_start       },
        { "<|im_end|>",   &out.im_end         },
        { "\xC4\x8A",     &out.newline        }, // "Ċ" = byte-level \n
        { "system",       &out.role_system    },
        { "user",         &out.role_user      },
        { "assistant",    &out.role_assistant },
    };
    for (const auto & p : pieces) {
        const int id = tok.find(p.piece);
        if (id < 0) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr: chat-template piece \"%s\" not in tokenizer", p.piece);
            return TRANSCRIBE_ERR_GGUF;
        }
        *p.slot = id;
    }
    return TRANSCRIBE_OK;
}

// Build the prompt token sequence + audio-position list, mirroring the
// Qwen3-ASR chat template at the token level:
//
//   <|im_start|>system\n<|im_end|>\n
//   <|im_start|>user\n<|audio_start|><|audio_pad|>*T_enc<|audio_end|><|im_end|>\n
//   <|im_start|>assistant\n[language {Name}<asr_text>]?
//
// The system message carries the generic prompting context (`system_ids`,
// empty by default). A non-null `lang_prefix_ids` (resolved via
// encode_language_prefix) is appended after the trailing newline to force an
// output language; kept out of here so this stays a pure token-id assembler.
void build_prompt_tokens(const QwenAsrHParams &       hp,
                         const ChatTokens &           ct,
                         int                          T_enc,
                         const std::vector<int32_t> & system_ids,
                         const std::vector<int32_t> * lang_prefix_ids,
                         std::vector<int32_t> &       out_ids,
                         std::vector<int64_t> &       out_audio_positions) {
    out_ids.clear();
    out_audio_positions.clear();

    out_ids.push_back(ct.im_start);
    out_ids.push_back(ct.role_system);
    out_ids.push_back(ct.newline);
    out_ids.insert(out_ids.end(), system_ids.begin(), system_ids.end());
    out_ids.push_back(ct.im_end);
    out_ids.push_back(ct.newline);

    out_ids.push_back(ct.im_start);
    out_ids.push_back(ct.role_user);
    out_ids.push_back(ct.newline);

    out_ids.push_back(hp.audio_start_token_id);
    const int64_t audio_start_pos = static_cast<int64_t>(out_ids.size());
    for (int i = 0; i < T_enc; ++i) {
        out_ids.push_back(hp.audio_token_id);
        out_audio_positions.push_back(audio_start_pos + i);
    }
    out_ids.push_back(hp.audio_end_token_id);

    out_ids.push_back(ct.im_end);
    out_ids.push_back(ct.newline);

    out_ids.push_back(ct.im_start);
    out_ids.push_back(ct.role_assistant);
    out_ids.push_back(ct.newline);

    if (lang_prefix_ids != nullptr && !lang_prefix_ids->empty()) {
        out_ids.insert(out_ids.end(), lang_prefix_ids->begin(), lang_prefix_ids->end());
    }
}

// Ids build_prompt_tokens emits besides the system ids, the audio pads and
// the language prefix (the role / newline / audio-boundary tokens above).
constexpr int k_chat_frame_tokens = 15;

// Length build_prompt_tokens produces with an empty system message, so the
// system context can be budgeted before the prompt is built.
int prompt_tokens_without_system(int T_enc, const std::vector<int32_t> * lang_prefix_ids) {
    return k_chat_frame_tokens + T_enc + (lang_prefix_ids != nullptr ? static_cast<int>(lang_prefix_ids->size()) : 0);
}

// Generic prompting -> system-message ids: vocabulary joined " " (measured
// better than ", ": fewer whole-dictionary dumps into the output), then
// " " + prompt verbatim. `budget` is the room the context window leaves
// after the rest of the prompt and the generation reserve; overflow trims
// the context first, then terms (see fit_terms_and_context).
transcribe_status encode_system_context(const transcribe::Tokenizer & tok,
                                        const transcribe_run_params * params,
                                        int                           budget,
                                        std::vector<int32_t> &        out) {
    out.clear();
    if (params == nullptr) {
        return TRANSCRIBE_OK;
    }
    const std::vector<std::string> terms = transcribe::prompting::terms(params);
    std::string                    ctx   = params->prompt != nullptr ? params->prompt : "";
    if (!terms.empty() && !ctx.empty()) {
        ctx = " " + ctx;
    }
    transcribe::prompting::FittedPrompt fit;
    if (const transcribe_status st = transcribe::prompting::fit_terms_and_context(
            tok, terms, { "", " ", "" }, ctx, std::max(budget, 0), "qwen3_asr run", fit);
        st != TRANSCRIBE_OK) {
        return st;
    }
    out = std::move(fit.term_ids);
    out.insert(out.end(), fit.ctx_ids.begin(), fit.ctx_ids.end());
    return TRANSCRIBE_OK;
}

// The prompt-prefix cache is on unless TRANSCRIBE_QWEN3_NO_PREFIX_CACHE is
// set (for A/B runs); its results are bit-identical to the uncached path on
// the CPU backend. The header draft is opt-in (TRANSCRIBE_QWEN3_HEADER_DRAFT):
// its verification compares logits from the prefill graph with what the
// step graph would have produced, which round differently, so it can change
// a near-tie token.
bool prefix_cache_enabled() {
    return !transcribe::env::flag("TRANSCRIBE_QWEN3_NO_PREFIX_CACHE");
}
bool header_draft_enabled() {
    return prefix_cache_enabled() && transcribe::env::flag("TRANSCRIBE_QWEN3_HEADER_DRAFT");
}

// Rows of the prompt-prefix cache usable for a prompt starting with the n ids
// at b whose single-graph prefill takes the `tiled` attention path: the
// common prefix of the ids, among rows computed on that same path.
int common_prefix(const QwenAsrSession * cc, const int32_t * b, int n, bool tiled) {
    const int m = std::min(static_cast<int>(cc->kv_ids.size()), n);
    int       i = 0;
    while (i < m && cc->kv_ids[i] == b[i] && (cc->kv_tiled[i] != 0) == tiled) {
        ++i;
    }
    return i;
}

void clear_prefix_cache(QwenAsrSession * cc) {
    cc->kv_ids.clear();
    cc->kv_tiled.clear();
}

// fa_q_rows for a chunk of T_chunk query rows that must take the `tiled`
// path (see build_prefill_chunk_graph).
int chunk_fa_q_rows(int T_chunk, bool tiled) {
    if (tiled) {
        return T_chunk >= k_fa_tile_rows ? 0 : -k_fa_tile_rows;
    }
    return T_chunk >= k_fa_tile_rows ? k_fa_tile_rows / 2 : 0;
}

// Upload a chunk mask: fill_prefill_chunk_mask for the T_chunk real query
// rows, all-visible rows for the padding (their outputs are dropped).
void set_chunk_mask(const PrefillChunkBuild & pb, int max_n_kv, int T_chunk, int n_past) {
    std::vector<ggml_fp16_t> mask(static_cast<size_t>(max_n_kv) * pb.mask_rows, ggml_fp32_to_fp16(0.0f));
    transcribe::causal_lm::fill_prefill_chunk_mask(mask.data(), max_n_kv, T_chunk, n_past);
    ggml_backend_tensor_set(pb.mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
}

}  // namespace

// below; encode_language_prefix matches the qwen3_asr.h declaration.)

// BCP-47 → publisher canonical name ("English", "Chinese", ...), which the
// prompt renders instead of the code. Frozen per release
// (qwen_asr.inference.utils.SUPPORTED_LANGUAGES); this table is the update
// point for a future variant.
struct LangNameEntry {
    const char * bcp47;
    const char * pub_name;
};

constexpr LangNameEntry k_qwen3_asr_language_names[] = {
    { "zh",  "Chinese"    },
    { "en",  "English"    },
    { "yue", "Cantonese"  },
    { "ar",  "Arabic"     },
    { "de",  "German"     },
    { "fr",  "French"     },
    { "es",  "Spanish"    },
    { "pt",  "Portuguese" },
    { "id",  "Indonesian" },
    { "it",  "Italian"    },
    { "ko",  "Korean"     },
    { "ru",  "Russian"    },
    { "th",  "Thai"       },
    { "vi",  "Vietnamese" },
    { "ja",  "Japanese"   },
    { "tr",  "Turkish"    },
    { "hi",  "Hindi"      },
    { "ms",  "Malay"      },
    { "nl",  "Dutch"      },
    { "sv",  "Swedish"    },
    { "da",  "Danish"     },
    { "fi",  "Finnish"    },
    { "pl",  "Polish"     },
    { "cs",  "Czech"      },
    { "fil", "Filipino"   },
    { "fa",  "Persian"    },
    { "el",  "Greek"      },
    { "ro",  "Romanian"   },
    { "hu",  "Hungarian"  },
    { "mk",  "Macedonian" },
};

// Resolve a caller-supplied BCP-47 code to the token-id sequence the chat
// template expects: BPE("language {Name}") + the `<asr_text>` special id.
// encode() doesn't split special tokens out, so we look up the <asr_text> id
// directly from the vocab (never hardcoded) and append it by hand. The
// dispatcher validates `bcp47` against caps.languages first, so an unknown
// code here means converter/map drift — surface as UNSUPPORTED_LANGUAGE.
transcribe_status encode_language_prefix(const transcribe::Tokenizer & tok,
                                         const char *                  bcp47,
                                         std::vector<int32_t> &        out_ids) {
    out_ids.clear();
    if (bcp47 == nullptr || bcp47[0] == '\0') {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const char * pub_name = nullptr;
    for (const auto & e : k_qwen3_asr_language_names) {
        if (std::strcmp(e.bcp47, bcp47) == 0) {
            pub_name = e.pub_name;
            break;
        }
    }
    if (pub_name == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "qwen3_asr: no canonical publisher name for "
                "language=\"%s\"",
                bcp47);
        return TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE;
    }
    if (!tok.has_encoder()) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "qwen3_asr: tokenizer missing encoder (merges "
                "unavailable); cannot render language hint");
        return TRANSCRIBE_ERR_GGUF;
    }
    const int asr_text_id = tok.find("<asr_text>");
    if (asr_text_id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "qwen3_asr: tokenizer vocab missing <asr_text> "
                "special token");
        return TRANSCRIBE_ERR_GGUF;
    }
    std::string text = "language ";
    text += pub_name;
    if (const transcribe_status st = tok.encode(text, out_ids); st != TRANSCRIBE_OK) {
        return st;
    }
    out_ids.push_back(asr_text_id);
    return TRANSCRIBE_OK;
}

namespace {  // reopen anon for the rest of the file's helpers.

// Host-side pack [n_mels, T_mel] mel into batched chunks
// [mel_per_chunk, n_mels, 1, n_chunks]. Chunks shorter than
// mel_per_chunk are zero-padded.
void pack_mel_chunks(const float *         mel,  // [n_mels, T_mel]
                     int                   n_mels,
                     int                   n_mel_frames,
                     const EncoderTiming & t,
                     std::vector<float> &  out) {
    const size_t per_chunk_elems = static_cast<size_t>(t.mel_per_chunk) * n_mels;
    out.assign(per_chunk_elems * t.n_chunks, 0.0f);

    for (int c = 0; c < t.n_chunks; ++c) {
        const int tail = (c == t.n_chunks - 1) ? t.last_chunk_real_mel : t.mel_per_chunk;
        for (int m = 0; m < n_mels; ++m) {
            const float * src = mel + static_cast<size_t>(m) * n_mel_frames + static_cast<size_t>(c) * t.mel_per_chunk;
            float *       dst =
                out.data() + static_cast<size_t>(c) * per_chunk_elems + static_cast<size_t>(m) * t.mel_per_chunk;
            std::memcpy(dst, src, tail * sizeof(float));
            // trailing frames in the tail chunk stay 0.0 (zero pad).
        }
    }
}

// Conv subsampler over `n_chunks` packed mel chunks (pack_mel_chunks layout)
// into out = [d_model, per_chunk_aftercnn, n_chunks], in passes of
// kConvPassChunks chunks. Full passes share one graph, allocated once and
// re-run; the remainder gets one graph of its own size. Neither node count
// nor scratch grows with the audio, and chunks are independent, so the
// result matches one graph over every chunk. Built in cc->compute_ctx;
// resets cc->sched, and the caller resets it again before its next graph.
transcribe_status run_subsample(QwenAsrSession *     cc,
                                QwenAsrModel *       cm,
                                const float *        mel,
                                int                  n_chunks,
                                std::vector<float> & out) {
    const size_t in_chunk  = static_cast<size_t>(cm->hparams.enc_n_window) * 2 * cm->hparams.enc_num_mel_bins;
    const size_t out_chunk = static_cast<size_t>(cm->hparams.enc_d_model) * aftercnn_len(cm->hparams.enc_n_window * 2);
    out.resize(out_chunk * n_chunks);

    // Runs chunks [c_begin, c_end) through one graph of `pass` chunks.
    auto run_passes = [&](int pass, int c_begin, int c_end) -> transcribe_status {
        SubsampleBuild sb = build_subsample_graph(cc->compute_ctx, cm->weights, cm->hparams, pass);
        if (sb.graph == nullptr || sb.out == nullptr) {
            return TRANSCRIBE_ERR_GGUF;
        }
        ggml_backend_sched_reset(cc->sched);
        if (!TRANSCRIBE_ALLOC_SCHED_GRAPH(cc->sched, sb.graph, "qwen3_asr")) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "qwen3_asr run: subsample graph allocation failed — out of memory.");
            return TRANSCRIBE_ERR_OOM;
        }
        transcribe::configure_sched_n_threads(cc->sched, cc->n_threads);
        for (int c0 = c_begin; c0 < c_end; c0 += pass) {
            ggml_backend_tensor_set(sb.mel_in, mel + static_cast<size_t>(c0) * in_chunk, 0, ggml_nbytes(sb.mel_in));
            if (const ggml_status gs = compute_qwen_graph(cc, sb.graph);
                gs != GGML_STATUS_SUCCESS) {
                log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr run: subsample graph compute failed (%d)",
                        static_cast<int>(gs));
                return TRANSCRIBE_ERR_BACKEND;
            }
            ggml_backend_tensor_get(sb.out, out.data() + static_cast<size_t>(c0) * out_chunk, 0, ggml_nbytes(sb.out));
        }
        return TRANSCRIBE_OK;
    };

    // Chunks have independent convolution inputs and padding. Bound the live
    // im2col/activation arena on the constrained profile while preserving every
    // chunk and the original output order; one graph is reused across passes.
    const int pass_chunks = khr_gpu_graph_policy(cc) ? 1 : kConvPassChunks;
    if (khr_gpu_graph_policy(cc)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "qwen3_asr KHR subsample: chunks=%d pass_chunks=%d",
                n_chunks, pass_chunks);
    }
    const int n_full = n_chunks - n_chunks % pass_chunks;
    if (n_full > 0) {
        if (const transcribe_status st = run_passes(pass_chunks, 0, n_full); st != TRANSCRIBE_OK) {
            return st;
        }
    }
    if (n_full < n_chunks) {
        return run_passes(n_chunks - n_full, n_full, n_chunks);
    }
    return TRANSCRIBE_OK;
}

// Mel and conv stem already computed by the streaming path (stream_finalize):
// run_impl skips both and starts at the encoder transformer.
struct PrecomputedStem {
    EncoderTiming      timing{};
    std::vector<float> subsample;  // [d_model, per_chunk_aftercnn, n_chunks]
    int64_t            t_subsample_us = 0;
};

transcribe_status run_impl(transcribe_session *          session,
                           const float *                 pcm,
                           int                           n_samples,
                           const transcribe_run_params * params,
                           PrecomputedStem *             pre) {
    if (session == nullptr || pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto * cc = static_cast<QwenAsrSession *>(session);
    auto * cm = static_cast<QwenAsrModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Pre-run abort check (Qwen3-ASR is single-shot, so this is the only
    // observation point).
    if (cc->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }

    // Language hint. Null/empty == auto-detect (the LM emits its own
    // "language X<asr_text>" prefix, stripped by the output parser below). A
    // non-null code is resolved to "language {Name}<asr_text>" tokens that seed
    // the assistant turn; a resolve failure surfaces as UNSUPPORTED_LANGUAGE.
    std::vector<int32_t>         lang_prefix_ids;
    const std::vector<int32_t> * lang_prefix_ptr = nullptr;
    if (params != nullptr && params->language != nullptr && params->language[0] != '\0') {
        if (const transcribe_status st = encode_language_prefix(cm->tok, params->language, lang_prefix_ids);
            st != TRANSCRIBE_OK) {
            return st;
        }
        lang_prefix_ptr = &lang_prefix_ids;
    }
    // Experiment (TRANSCRIBE_QWEN3_FORCE_LAST_HEADER, off by default): with
    // no hint but a context prompt (a later phrase of an utterance), force
    // the header the last auto-detect run generated instead of detecting.
    // Unlike the header draft, nothing checks it against the audio.
    bool forced_last = false;
    if (lang_prefix_ptr == nullptr && params != nullptr && params->prompt != nullptr && params->prompt[0] != '\0' &&
        !cc->header_draft.empty() && prefix_cache_enabled() &&
        transcribe::env::flag("TRANSCRIBE_QWEN3_FORCE_LAST_HEADER")) {
        lang_prefix_ids = cc->header_draft;
        lang_prefix_ptr = &lang_prefix_ids;
        forced_last     = true;
    }

    transcribe::debug::init();

    // Mel front-end.
    if (!cm->mel.has_value()) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr run: model has no MelFrontend");
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const int64_t t_mel_start  = INT64_C(0);
    int           mel_n_mels   = cm->mel->num_mels();
    int           mel_n_frames = (pre != nullptr) ? pre->timing.n_mel_frames : 0;
    if (pre == nullptr) {
        if (const transcribe_status mst = cm->mel->compute(pcm, static_cast<size_t>(n_samples), cc->mel_buf,
                                                           mel_n_mels, mel_n_frames, cc->n_threads);
            mst != TRANSCRIBE_OK) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr run: MelFrontend::compute failed (%s)",
                    transcribe_status_string(mst));
            return mst;
        }
        cc->t_mel_us = INT64_C(0) - t_mel_start;
    }

    // Dump the post-frontend mel in the reference's contract shape
    // [n_mels, T_mel]. The batched graph input is a reshaped view of
    // the same data, so the comparison point lives on the host. (The
    // streaming path fills mel_buf for this only when dumping.)
    if (transcribe::debug::enabled() &&
        cc->mel_buf.size() == static_cast<size_t>(mel_n_mels) * static_cast<size_t>(mel_n_frames)) {
        const long long shape[2] = { mel_n_mels, mel_n_frames };
        transcribe::debug::dump_host_f32("enc.mel.in", cc->mel_buf.data(), static_cast<long long>(cc->mel_buf.size()),
                                         shape, 2, "frontend.mel.norm");
    }

    // Compute encoder timing + reject unsupported shapes.
    EncoderTiming timing = (pre != nullptr) ? pre->timing : compute_encoder_timing(mel_n_frames, cm->hparams);
    if (timing.n_chunks <= 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "qwen3_asr run: encoder timing is degenerate "
                "(n_mel_frames=%d)",
                mel_n_frames);
        return TRANSCRIBE_ERR_GGUF;
    }

    // Input-length gate, before encoding: audio + the prompt without system
    // context + generation must fit the decoder context window. The system
    // context is fitted to what is left (encode_system_context), so this is
    // the same bound as the prompt gate below, checked before an over-length
    // clip pays for the encoder.
    const int ceiling = qwen3_context_ceiling(cc->n_ctx, cm->hparams);
    if (const int T_min = prompt_tokens_without_system(timing.T_enc, lang_prefix_ptr);
        T_min + k_gen_reserve > ceiling) {
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                            "qwen3_asr run: input too long — %d audio + %d prompt tokens "
                            "leave no room for output within the %d-token context (need %d). "
                            "Shorten the audio (see transcribe_capabilities.max_audio_ms) or "
                            "split it into segments.",
                            timing.T_enc, T_min - timing.T_enc, ceiling, T_min + k_gen_reserve);
        return TRANSCRIBE_ERR_INPUT_TOO_LONG;
    }

    // Reset per-call compute state. The phase timers below break out
    // per-run cost (graph build, sched alloc, uploads, prefill compute)
    // that the public transcribe_timings (mel/encode/decode) doesn't
    // expose, for the TRANSCRIBE_PERF_DEBUG breakdown.
    const int64_t t_enc_build_start    = INT64_C(0);
    int64_t       t_enc_build_us       = 0;
    int64_t       t_enc_d2h_us         = 0;
    int64_t       t_prefill_build_us   = 0;
    int64_t       t_prefill_compute_us = 0;
    int64_t       t_prefill_logits_us  = 0;
    int64_t       t_step_loop_us       = 0;
    int           n_steps              = 0;

    if (cc->compute_ctx != nullptr) {
        ggml_free(cc->compute_ctx);
        cc->compute_ctx = nullptr;
    }

    {
        ggml_init_params ip{};
        ip.mem_size     = compute_metadata_bytes(cc, 16 * 1024 * 1024, pre ? 0 : timing.n_chunks, true, 16384);
        if (!ip.mem_size) return TRANSCRIBE_ERR_OOM;
        ip.mem_buffer   = nullptr;
        ip.no_alloc     = true;
        cc->compute_ctx = ggml_init(ip);
        if (cc->compute_ctx == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr run: ggml_init for compute_ctx failed");
            return TRANSCRIBE_ERR_OOM;
        }
    }

    if (cc->sched == nullptr) {
        cc->sched = ggml_backend_sched_new(cm->plan.scheduler_list.data(), nullptr,
                                           static_cast<int>(cm->plan.scheduler_list.size()), 16384, false, true);
        if (cc->sched == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr run: ggml_backend_sched_new failed");
            return TRANSCRIBE_ERR_BACKEND;
        }
    }

    // Conv subsampler, in passes of up to kConvPassChunks chunks.
    std::vector<float> subsample_host;
    int64_t            t_subsample_us = 0;
    if (pre != nullptr) {
        subsample_host = std::move(pre->subsample);
        t_subsample_us = pre->t_subsample_us;
    } else {
        std::vector<float> mel_batched;
        pack_mel_chunks(cc->mel_buf.data(), mel_n_mels, mel_n_frames, timing, mel_batched);
        const int64_t t_sub0 = INT64_C(0);
        if (const transcribe_status st = run_subsample(cc, cm, mel_batched.data(), timing.n_chunks, subsample_host);
            st != TRANSCRIBE_OK) {
            return st;
        }
        t_subsample_us = INT64_C(0) - t_sub0;
    }

    // Subsample outputs are complete in host storage; no later graph uses its arena.
    if (const auto st = release_subsample_scratch(cc); st != TRANSCRIBE_OK) return st;

    // Build encoder graph.
    EncoderBuild eb = build_encoder_graph(cc->compute_ctx, cm->weights, cm->hparams, timing, cc->encoder_use_flash);
    if (eb.graph == nullptr || eb.out == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    std::unique_ptr<ggml_backend_buffer, decltype(&transcribe::safe_buffer_free)> encoder_handoff(
        nullptr, &transcribe::safe_buffer_free);
    if (khr_gpu_graph_policy(cc)) {
        if (eb.out->view_src || !ggml_is_contiguous(eb.out) || eb.out->type != GGML_TYPE_F32)
            return TRANSCRIBE_ERR_BACKEND;
        const auto buft = ggml_backend_get_default_buffer_type(cm->plan.primary);
        const size_t bytes = ggml_backend_buft_get_alloc_size(buft, eb.out);
        encoder_handoff.reset(ggml_backend_alloc_buffer(cm->plan.primary, bytes));
        if (!encoder_handoff) return TRANSCRIBE_ERR_OOM;
        if (ggml_backend_tensor_alloc(encoder_handoff.get(), eb.out,
                                     ggml_backend_buffer_get_base(encoder_handoff.get())) != GGML_STATUS_SUCCESS)
            return TRANSCRIBE_ERR_BACKEND;
    }
    const auto bind_encoder_input = [&](ggml_tensor * input) {
        if (!encoder_handoff) return true;
        if (!input || input->view_src || !ggml_are_same_shape(input, eb.out) ||
            input->type != eb.out->type || !ggml_is_contiguous(input)) return false;
        return ggml_backend_tensor_alloc(encoder_handoff.get(), input,
                   ggml_backend_buffer_get_base(encoder_handoff.get())) == GGML_STATUS_SUCCESS;
    };

    // Allocate + compute encoder graph.
    ggml_backend_sched_reset(cc->sched);
    if (!TRANSCRIBE_ALLOC_SCHED_GRAPH(cc->sched, eb.graph, "qwen3_asr")) {
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                            "qwen3_asr run: encoder graph allocation failed — out of memory.");
        return TRANSCRIBE_ERR_OOM;
    }

    ggml_backend_tensor_set(eb.subsample_in, subsample_host.data(), 0, subsample_host.size() * sizeof(float));
    if (khr_gpu_graph_policy(cc)) std::vector<float>().swap(subsample_host);

    // Positional embedding.
    {
        std::vector<float> pe = build_sinusoid_pe(cm->hparams.enc_d_model, timing.per_chunk_aftercnn);
        ggml_backend_tensor_set(eb.pos_emb_in, pe.data(), 0, pe.size() * sizeof(float));
    }

    transcribe::configure_sched_n_threads(cc->sched, cc->n_threads);

    const int64_t t_enc_start = INT64_C(0);
    t_enc_build_us            = t_enc_start - t_enc_build_start - t_subsample_us;
    if (const ggml_status gs = compute_qwen_graph(cc, eb.graph); gs != GGML_STATUS_SUCCESS) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr run: encoder graph compute failed (%d)", static_cast<int>(gs));
        return TRANSCRIBE_ERR_BACKEND;
    }
    cc->t_encode_us = INT64_C(0) - t_enc_start + t_subsample_us;

    // Dump encoder intermediates.
    auto try_dump = [](const char * name, ggml_tensor * t, const char * stage) {
        if (t != nullptr) {
            transcribe::debug::dump_tensor(name, t, stage);
        }
    };
    try_dump("enc.subsample.out", eb.dumps.subsample_out, "enc.subsample");
    try_dump("enc.pos_add.out", eb.dumps.pos_add_out, "enc.pos_add");
    try_dump("enc.block.0.out", eb.dumps.block_0_out, "enc.block.0");
    {
        char bname[64];
        std::snprintf(bname, sizeof(bname), "enc.block.%d.out", cm->hparams.enc_n_layers - 1);
        try_dump(bname, eb.dumps.block_last_out, "enc.block.last");
    }
    try_dump("enc.ln_post.out", eb.dumps.ln_post_out, "enc.ln_post");
    try_dump("enc.proj.out", eb.dumps.proj_out, "enc.proj");

    // Preserve encoder output across the encoder/prefill scheduler reset. The graph already
    // dropped the aftercnn pad rows (see encoder.cpp), so eb.out is exactly
    // [d_enc, T_enc] — the reference's `padded_embed[padded_mask_after_cnn]`
    // shape.
    const int d_enc = static_cast<int>(eb.out->ne[0]);
    const int T_enc = static_cast<int>(eb.out->ne[1]);
    if (encoder_handoff) {
        std::vector<float>().swap(cc->enc_host);
    } else {
        cc->enc_host.resize(static_cast<size_t>(d_enc) * static_cast<size_t>(T_enc));
        const int64_t t_d2h_start = INT64_C(0);
        ggml_backend_tensor_get(eb.out, cc->enc_host.data(), 0, cc->enc_host.size() * sizeof(float));
        t_enc_d2h_us = INT64_C(0) - t_d2h_start;
    }

    // Decode phase begins. t_dec_start covers prompt + KV init + prefill
    // build/compute + step loop (prefill is part of "decode" to users).
    const int64_t t_dec_start           = INT64_C(0);
    const int64_t t_prefill_build_start = t_dec_start;

    // Prompt construction.
    std::vector<int32_t> prompt_ids;
    std::vector<int64_t> audio_positions;
    std::vector<int32_t> system_ids;
    if (const transcribe_status st = encode_system_context(
            cm->tok, params, ceiling - k_gen_reserve - prompt_tokens_without_system(T_enc, lang_prefix_ptr),
            system_ids);
        st != TRANSCRIBE_OK) {
        return st;
    }
    build_prompt_tokens(cm->hparams, cm->chat_tokens, T_enc, system_ids, lang_prefix_ptr, prompt_ids, audio_positions);
    const int T_prompt   = static_cast<int>(prompt_ids.size());
    const int prefix_len = audio_positions.empty() ? 0 : static_cast<int>(audio_positions.front());
    const int suffix_len = T_prompt - prefix_len - T_enc;
    (void) audio_positions;

    // Input-length gate: audio + prompt + generation must fit the decoder
    // context window. Reject an over-length clip here, before prefill/decode.
    if (T_prompt + k_gen_reserve > ceiling) {
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                            "qwen3_asr run: input too long — %d audio + %d prompt tokens "
                            "leave no room for output within the %d-token context (need %d). "
                            "Shorten the audio (see transcribe_capabilities.max_audio_ms) or "
                            "split it into segments.",
                            T_enc, prefix_len + suffix_len, ceiling, T_prompt + k_gen_reserve);
        return TRANSCRIBE_ERR_INPUT_TOO_LONG;
    }

    const int max_new = transcribe::pick_decode_budget(
        transcribe::predict_transcript_tokens(T_enc, cm->limits.ms_per_audio_token), k_gen_reserve, T_prompt, ceiling);

    // KV cache init (grow-to-fit, clamped to the context ceiling). Size to
    // hold prompt + decode budget, rounded up to a power of two (the step
    // graph's flash-attn path wants pow2 attention width). A pre-allocated
    // smaller cache is freed and re-allocated.
    int want_n_ctx = 1024;
    if (khr_gpu_graph_policy(cc)) {
        want_n_ctx = static_cast<int>(std::min<int64_t>(ceiling,
            ((static_cast<int64_t>(T_prompt) + max_new + 15) / 16) * 16));
    } else {
        while (want_n_ctx < T_prompt + max_new) {
            want_n_ctx *= 2;
        }
    }
    if (want_n_ctx > ceiling) {
        want_n_ctx = ceiling;
    }
    const int kv_capacity_limit = want_n_ctx;
    if (khr_gpu_graph_policy(cc)) {
        const int64_t draft_room = std::min<int64_t>(max_new, std::max<size_t>(1, cc->header_draft.size()));
        want_n_ctx = initial_khr_kv_capacity(static_cast<int64_t>(T_prompt) + draft_room, kv_capacity_limit);
        log_msg(TRANSCRIBE_LOG_LEVEL_INFO,
                "qwen3_asr KHR KV request: prompt=%d decode_budget=%d capacity=%d limit=%d retained=%d",
                T_prompt, max_new, want_n_ctx, kv_capacity_limit, cc->kv_cache.n_ctx);
    }
    // Prompt-prefix cache: rows [0, n_past) of the KV cache already hold this
    // prompt's first n_past tokens from an earlier run (the chat frame and
    // the system context, both before the audio), so only the rest is
    // prefilled. The audio and everything after it are always new.
    const bool reuse = prefix_cache_enabled();
    if (cc->kv_cache.ctx != nullptr && cc->kv_cache.n_ctx < want_n_ctx) {
        if (khr_gpu_graph_policy(cc)) {
            if (const auto st = grow_khr_kv(cc, want_n_ctx); st != TRANSCRIBE_OK) return st;
        } else {
            cc->kv_cache.free();
        }
    }
    if (cc->kv_cache.ctx == nullptr) {
        clear_prefix_cache(cc);
        ggml_type kv_type = GGML_TYPE_F16;
        if (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) {
            kv_type = GGML_TYPE_F32;
        }
        if (!transcribe::causal_lm::kv_init(cc->kv_cache, cm->plan.primary, want_n_ctx, cm->hparams.dec_n_kv_heads,
                                            cm->hparams.dec_head_dim, cm->hparams.dec_n_layers, kv_type)) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "qwen3_asr run: KV cache allocation failed (n_ctx=%d, "
                                "%d kv-heads x %d head-dim x %d layers) — out of memory. "
                                "Lower transcribe_session_params.n_ctx or shorten the audio.",
                                want_n_ctx, cm->hparams.dec_n_kv_heads, cm->hparams.dec_head_dim,
                                cm->hparams.dec_n_layers);
            return TRANSCRIBE_ERR_OOM;
        }
    } else if (!reuse) {
        // Clear stale positions for a fresh prefill.
        clear_prefix_cache(cc);
        if (cc->kv_cache.buffer != nullptr) {
            ggml_backend_buffer_clear(cc->kv_cache.buffer, 0);
        }
        cc->kv_cache.n    = 0;
        cc->kv_cache.head = 0;
    }
    // Stale rows past n_past need no clearing: every attention mask here
    // excludes them (-inf), and the step loop writes each row before reading it.
    // Every row is computed on the attention path the single prefill graph
    // over this prompt would take, so the result does not depend on which
    // rows came from the cache.
    const bool ref_tiled = T_prompt >= k_fa_tile_rows;
    const int  n_past    = reuse ? common_prefix(cc, prompt_ids.data(), prefix_len, ref_tiled) : 0;
    if (khr_gpu_graph_policy(cc) && transcribe::env::flag("TRANSCRIBE_RESOURCE_DIAGNOSTICS")) {
        std::fprintf(stderr, "ASR_PREFIX_REUSE requested=%d reused=%d valid=%d head=%d capacity=%d\n",
                     prefix_len, n_past, cc->kv_cache.n, cc->kv_cache.head, cc->kv_cache.n_ctx);
    }
    // Header draft: the previous auto-detect run's "language X<asr_text>" ids,
    // appended to the prompt; the prefill's logits at those positions show
    // how many of them the model predicts itself (greedy), and those are
    // taken as generated, saving a decode step each.
    std::vector<int32_t> draft;
    if (lang_prefix_ptr == nullptr && header_draft_enabled() &&
        static_cast<int>(cc->header_draft.size()) + 1 < max_new &&
        T_prompt + static_cast<int>(cc->header_draft.size()) <= cc->kv_cache.n_ctx) {
        draft = cc->header_draft;
    }
    const int n_draft = static_cast<int>(draft.size());
    cc->kv_reused    = n_past;
    cc->kv_prefilled = T_prompt - n_past;
    cc->draft_n      = n_draft;
    cc->draft_ok     = 0;

    const int vocab  = cm->hparams.dec_vocab_size;
    auto      argmax = [](const float * v, int n) -> int32_t {
        int32_t best   = 0;
        float   best_v = v[0];
        for (int32_t i = 1; i < n; ++i) {
            if (v[i] > best_v) {
                best_v = v[i];
                best   = i;
            }
        }
        return best;
    };
    std::vector<int32_t> generated_ids;
    float                logit0_max = 0.0f;  // perf log: max logit after the prompt (bitwise checks)
    int32_t              next_tok = 0;
    int                  n_accepted = 0;  // header draft tokens taken as generated

    if (n_past == 0 && n_draft == 0) {
        // Prefill graph. slice_last false: last block's FFN + final norm run on
        // every position (needed for dump parity). true: slice to just the final
        // position before the last FFN (llama.cpp's inp_out_ids trick, ~25 ms).
        const bool   dumps_on   = transcribe::debug::enabled();
        const bool   slice_last = !dumps_on;
        PrefillBuild pb = build_prefill_graph(cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache, T_prompt, T_enc,
                                              prefix_len, suffix_len,
                                              /*use_flash=*/cc->decoder_use_flash, slice_last);
        if (pb.graph == nullptr || pb.out == nullptr) {
            return TRANSCRIBE_ERR_GGUF;
        }

        // Allocate + compute prefill on the same scheduler.
        if (!bind_encoder_input(pb.enc_out_in)) return TRANSCRIBE_ERR_BACKEND;
        ggml_backend_sched_reset(cc->sched);
        if (!TRANSCRIBE_ALLOC_SCHED_GRAPH(cc->sched, pb.graph, "qwen3_asr")) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "qwen3_asr run: prefill graph allocation failed (T_prompt=%d) — "
                                "out of memory. Lower transcribe_session_params.n_ctx or shorten "
                                "the audio.",
                                T_prompt);
            return TRANSCRIBE_ERR_OOM;
        }

        // Upload prefill inputs.
        ggml_backend_tensor_set(pb.input_ids_in, prompt_ids.data(), 0, prompt_ids.size() * sizeof(int32_t));
        if (!encoder_handoff)
            ggml_backend_tensor_set(pb.enc_out_in, cc->enc_host.data(), 0, cc->enc_host.size() * sizeof(float));

        {
            std::vector<int32_t> positions(T_prompt);
            for (int i = 0; i < T_prompt; ++i) {
                positions[i] = i;
            }
            ggml_backend_tensor_set(pb.positions_in, positions.data(), 0, positions.size() * sizeof(int32_t));
        }

        {
            // Causal mask in F16 (matches pb.mask_in): row r col c is 0 if c <= r,
            // else -inf, row-major. F16 upload avoids a per-layer ggml_cast.
            const ggml_fp16_t        mask_zero    = ggml_fp32_to_fp16(0.0f);
            const ggml_fp16_t        mask_neg_inf = ggml_fp32_to_fp16(-INFINITY);
            std::vector<ggml_fp16_t> mask(static_cast<size_t>(T_prompt) * T_prompt, mask_neg_inf);
            for (int r = 0; r < T_prompt; ++r) {
                for (int c = 0; c <= r; ++c) {
                    mask[static_cast<size_t>(r) * T_prompt + c] = mask_zero;
                }
            }
            ggml_backend_tensor_set(pb.mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        }

        const int64_t t_prefill_compute_start = INT64_C(0);
        t_prefill_build_us                    = t_prefill_compute_start - t_prefill_build_start;
        if (const ggml_status gs = compute_qwen_graph(cc, pb.graph); gs != GGML_STATUS_SUCCESS) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr run: prefill graph compute failed (%d)", static_cast<int>(gs));
            return TRANSCRIBE_ERR_BACKEND;
        }
        t_prefill_compute_us = INT64_C(0) - t_prefill_compute_start;

        cc->kv_cache.n    = T_prompt;
        cc->kv_cache.head = T_prompt;

        // Dump dec.* intermediates.
        try_dump("dec.token_emb", pb.dumps.token_emb, "dec.token_emb");
        try_dump("dec.audio_injected", pb.dumps.audio_injected, "dec.audio_injected");
        try_dump("dec.block.0.out", pb.dumps.block_0_out, "dec.block.0");
        {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "dec.block.%d.out", cm->hparams.dec_n_layers - 1);
            try_dump(nm, pb.dumps.block_last_out, "dec.block.last");
        }
        try_dump("dec.out_before_head", pb.dumps.out_before_head, "dec.out_before_head");
        try_dump("dec.logits_raw", pb.dumps.logits_raw, "dec.logits_raw");

        // Read prefill logits + first argmax.
        const int64_t      t_prefill_logits_start = INT64_C(0);
        std::vector<float> logits(vocab);
        ggml_backend_tensor_get(pb.out, logits.data(), 0, logits.size() * sizeof(float));


        next_tok = argmax(logits.data(), vocab);
        logit0_max = logits[next_tok];
        generated_ids.push_back(next_tok);
        t_prefill_logits_us = INT64_C(0) - t_prefill_logits_start;
    } else {
        // Prefill [n_past, T_prompt + n_draft) on top of the cached rows.
        // (Debug dumps of decoder intermediates are on the path above only.)
        const int T_full   = T_prompt + n_draft;
        const int T_chunk  = T_full - n_past;
        const int n_logits = 1 + n_draft;
        cc->kv_ids.resize(static_cast<size_t>(n_past));  // rows past n_past are rewritten below
        cc->kv_tiled.resize(static_cast<size_t>(n_past));

        std::vector<int32_t> full_ids(prompt_ids);
        full_ids.insert(full_ids.end(), draft.begin(), draft.end());

        PrefillChunkBuild pb =
            build_prefill_chunk_graph(cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache, T_chunk, T_full,
                                      prefix_len - n_past, T_enc, suffix_len + n_draft, n_logits,
                                      /*use_flash=*/cc->decoder_use_flash, chunk_fa_q_rows(T_chunk, ref_tiled));
        if (pb.graph == nullptr || pb.out == nullptr) {
            return TRANSCRIBE_ERR_GGUF;
        }
        if (!bind_encoder_input(pb.enc_out_in)) return TRANSCRIBE_ERR_BACKEND;
        ggml_backend_sched_reset(cc->sched);
        if (!TRANSCRIBE_ALLOC_SCHED_GRAPH(cc->sched, pb.graph, "qwen3_asr")) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "qwen3_asr run: prefill graph allocation failed (T_prompt=%d) — "
                                "out of memory. Lower transcribe_session_params.n_ctx or shorten "
                                "the audio.",
                                T_prompt);
            return TRANSCRIBE_ERR_OOM;
        }
        if (pb.input_ids_in != nullptr) {
            ggml_backend_tensor_set(pb.input_ids_in, full_ids.data() + n_past, 0,
                                    static_cast<size_t>(T_chunk) * sizeof(int32_t));
        }
        if (!encoder_handoff)
            ggml_backend_tensor_set(pb.enc_out_in, cc->enc_host.data(), 0, cc->enc_host.size() * sizeof(float));
        {
            std::vector<int32_t> positions(T_chunk);
            std::vector<int64_t> kv_idx(T_chunk);
            for (int i = 0; i < T_chunk; ++i) {
                positions[i] = n_past + i;
                kv_idx[i]    = n_past + i;
            }
            ggml_backend_tensor_set(pb.positions_in, positions.data(), 0, positions.size() * sizeof(int32_t));
            ggml_backend_tensor_set(pb.kv_idx_in, kv_idx.data(), 0, kv_idx.size() * sizeof(int64_t));
        }
        set_chunk_mask(pb, T_full, T_chunk, n_past);

        const int64_t t_prefill_compute_start = INT64_C(0);
        t_prefill_build_us                    = t_prefill_compute_start - t_prefill_build_start;
        if (const ggml_status gs = compute_qwen_graph(cc, pb.graph); gs != GGML_STATUS_SUCCESS) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr run: prefill graph compute failed (%d)",
                    static_cast<int>(gs));
            return TRANSCRIBE_ERR_BACKEND;
        }
        t_prefill_compute_us = INT64_C(0) - t_prefill_compute_start;

        // Logits row j predicts position T_prompt + j: keep the draft while
        // the model agrees with it, then its own token.
        const int64_t      t_prefill_logits_start = INT64_C(0);
        std::vector<float> logits(static_cast<size_t>(vocab) * n_logits);
        ggml_backend_tensor_get(pb.out, logits.data(), 0, logits.size() * sizeof(float));
        logit0_max = logits[argmax(logits.data(), vocab)];
        while (n_accepted < n_draft &&
               argmax(logits.data() + static_cast<size_t>(vocab) * n_accepted, vocab) == draft[n_accepted]) {
            generated_ids.push_back(draft[n_accepted]);
            ++n_accepted;
        }
        next_tok = argmax(logits.data() + static_cast<size_t>(vocab) * n_accepted, vocab);
        generated_ids.push_back(next_tok);
        t_prefill_logits_us = INT64_C(0) - t_prefill_logits_start;

        cc->kv_cache.n    = T_prompt + n_accepted;
        cc->kv_cache.head = T_prompt + n_accepted;
        cc->draft_ok      = n_accepted;
    }
    encoder_handoff.reset();
    if (reuse) {
        cc->kv_ids.assign(prompt_ids.begin(), prompt_ids.begin() + prefix_len);
        cc->kv_tiled.assign(static_cast<size_t>(prefix_len), ref_tiled ? 1 : 0);
    }
    cc->t_prefill_us = t_prefill_build_us + t_prefill_compute_us + t_prefill_logits_us;

    // Step loop.
    const int32_t eos_id   = cm->hparams.eos_token_id;
    int           cur_past = T_prompt + n_accepted;

    // Build the step graph ONCE and reuse every step, sized for the actual
    // workload (T_prompt written + up to max_new generated). Metal's flash-attn
    // kernels dispatch ~30% faster (M4 Max) when K/V ne[1] is a power of 2, so
    // round up; floor of 1024 (smaller just hits the slow-misaligned branch).
    int max_n_kv = 1024;
    while (max_n_kv < T_prompt + max_new) {
        max_n_kv *= 2;
    }
    if (max_n_kv > cc->kv_cache.n_ctx) {
        max_n_kv = cc->kv_cache.n_ctx;
    }
    if (khr_gpu_graph_policy(cc)) max_n_kv = std::min(cc->kv_cache.n_ctx, kv_capacity_limit);
    const int step_capacity_limit = khr_gpu_graph_policy(cc) ? kv_capacity_limit : max_n_kv;
    const int64_t t_step_build_start = INT64_C(0);
    const ggml_fp16_t mask_zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t mask_neg_inf = ggml_fp32_to_fp16(-INFINITY);
    StepBuild sb{};
    std::vector<ggml_fp16_t> step_mask;
    bool mask_needs_prefix = true;
    auto rebuild_step_graph = [&]() -> transcribe_status {
        if (cc->compute_ctx != nullptr) {
            ggml_free(cc->compute_ctx);
            cc->compute_ctx = nullptr;
        }
        ggml_init_params ip{};
        ip.mem_size = compute_metadata_bytes(cc, 8 * 1024 * 1024, 0, false, 8192);
        if (!ip.mem_size) return TRANSCRIBE_ERR_OOM;
        ip.mem_buffer = nullptr;
        ip.no_alloc = true;
        cc->compute_ctx = ggml_init(ip);
        if (!cc->compute_ctx) return TRANSCRIBE_ERR_OOM;
        sb = build_step_graph(cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache, max_n_kv,
                              /*use_flash=*/cc->decoder_use_flash);
        if (!sb.graph || !sb.out) return TRANSCRIBE_ERR_GGUF;
        ggml_backend_sched_reset(cc->sched);
        if (!TRANSCRIBE_ALLOC_SCHED_GRAPH(cc->sched, sb.graph, "qwen3_asr")) return TRANSCRIBE_ERR_OOM;
        step_mask.assign(max_n_kv, mask_neg_inf);
        mask_needs_prefix = true;
        return TRANSCRIBE_OK;
    };
    if (const auto st = rebuild_step_graph(); st != TRANSCRIBE_OK) return st;
    const int64_t t_step_build_once_us = INT64_C(0) - t_step_build_start;

    // Per-step timers.
    int64_t       t_step_set_us     = 0;
    int64_t       t_step_comp_us    = 0;
    int64_t       t_step_get_us     = 0;
    const int64_t t_step_loop_start = INT64_C(0);
    bool          repeating         = false;
    const bool warming = active_warmup && active_warmup->session == session;
    while ((next_tok != eos_id || warming) && static_cast<int32_t>(generated_ids.size()) < max_new &&
           cur_past + 1 <= step_capacity_limit) {
        if (cur_past >= max_n_kv) {
            ggml_backend_sched_synchronize(cc->sched);
            ggml_backend_sched_reset(cc->sched);
            sb = {};
            ggml_free(cc->compute_ctx);
            cc->compute_ctx = nullptr;
            const int capacity = static_cast<int>(std::min<int64_t>(step_capacity_limit,
                std::max<int64_t>(static_cast<int64_t>(cur_past) + 1,
                                  static_cast<int64_t>(cc->kv_cache.n_ctx) * 2)));
            if (const auto st = grow_khr_kv(cc, capacity); st != TRANSCRIBE_OK) return st;
            max_n_kv = cc->kv_cache.n_ctx;
            if (const auto st = rebuild_step_graph(); st != TRANSCRIBE_OK) return st;
        }
        const int64_t t_set0 = INT64_C(0);

        ggml_backend_tensor_set(sb.input_id_in, &next_tok, 0, sizeof(int32_t));
        const int32_t pos_val = cur_past;
        ggml_backend_tensor_set(sb.position_in, &pos_val, 0, sizeof(int32_t));
        const int64_t kv_idx_val = cur_past;
        ggml_backend_tensor_set(sb.kv_idx_in, &kv_idx_val, 0, sizeof(int64_t));

        // Mask: mark the newly-added position as attendable. Positions
        // [0, cur_past) were zeroed in prior iterations; just set the
        // new one. (On iter 0, zero everything in [0, cur_past].)
        if (mask_needs_prefix) {
            std::fill(step_mask.begin(), step_mask.begin() + cur_past + 1, mask_zero);
            mask_needs_prefix = false;
        } else {
            step_mask[cur_past] = mask_zero;
        }
        ggml_backend_tensor_set(sb.mask_in, step_mask.data(), 0, static_cast<size_t>(max_n_kv) * sizeof(ggml_fp16_t));

        const int64_t t_set1 = INT64_C(0);
        t_step_set_us += t_set1 - t_set0;

        if (const ggml_status gs = compute_qwen_graph(cc, sb.graph); gs != GGML_STATUS_SUCCESS) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr step: graph compute failed (%d)", static_cast<int>(gs));
            return TRANSCRIBE_ERR_BACKEND;
        }
        const int64_t t_comp1 = INT64_C(0);
        t_step_comp_us += t_comp1 - t_set1;

        int32_t argmax_tok = 0;
        ggml_backend_tensor_get(sb.out, &argmax_tok, 0, sizeof(int32_t));
        next_tok = argmax_tok;
        generated_ids.push_back(next_tok);
        cur_past += 1;
        cc->kv_cache.n    = cur_past + 1;
        cc->kv_cache.head = cur_past + 1;
        t_step_get_us += INT64_C(0) - t_comp1;
        if (warming) {
            ++active_warmup->completed_steps;
            if (transcribe::env::flag("TRANSCRIBE_RESOURCE_DIAGNOSTICS")) {
                std::fprintf(stderr, "ASR_WARMUP_STEP mode=%s completed=1\n",
                             active_warmup->streaming ? "streaming" : "offline");
            }
            break;
        }
        if (next_tok != eos_id && transcribe::stop_on_repetition(generated_ids, "qwen3_asr run")) {
            cc->mark_repetition_stop();
            repeating = true;
            break;
        }
    }
    if (warming) {
        return active_warmup->completed_steps == 1 ? TRANSCRIBE_OK : TRANSCRIBE_ERR_BACKEND;
    }
    t_step_loop_us = INT64_C(0) - t_step_loop_start;
    n_steps        = static_cast<int>(generated_ids.size()) - 1 - n_accepted;

    // The header this run generated, as the next run's draft.
    if (lang_prefix_ptr == nullptr && prefix_cache_enabled()) {
        cc->header_draft.clear();
        const int asr_text_id = cm->tok.find("<asr_text>");
        for (size_t i = 0; asr_text_id >= 0 && i < generated_ids.size() && i < 8; ++i) {
            if (generated_ids[i] == asr_text_id) {
                cc->header_draft.assign(generated_ids.begin(), generated_ids.begin() + static_cast<long>(i) + 1);
                break;
            }
        }
    }

    // Decode stopped at EOS (complete) or the generation budget / context width
    // (truncated). Surface the latter via transcribe_was_truncated() + WARN; a
    // repetition stop has already done both.
    if (!repeating && next_tok != eos_id) {
        cc->was_truncated = true;
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_WARN,
                            "qwen3_asr run: output truncated at %d tokens — decode reached the "
                            "generation budget before end-of-stream; the transcript may be "
                            "incomplete.",
                            static_cast<int>(generated_ids.size()));
        transcribe::trim_repetition_at_budget_stop(generated_ids, "qwen3_asr run");
    }

    // Map granular counters to the debug-print shape. With graph reuse all
    // per-step overhead collapses to tensor_set; build/alloc/ctx_reset are
    // amortized in t_step_build_once_us.
    int64_t t_step_ctx_us   = 0;
    int64_t t_step_build_us = t_step_build_once_us;
    int64_t t_step_alloc_us = 0;

    // Strip trailing EOS if present (match the reference transcript).
    if (!generated_ids.empty() && generated_ids.back() == eos_id) {
        generated_ids.pop_back();
    }

    // Decode generated ids to text (Tokenizer::decode handles the "gpt2"
    // byte-level inversion natively).
    std::string raw_text = cm->tok.decode(generated_ids.data(), static_cast<int>(generated_ids.size()));
    if (forced_last) {  // report the forced header like a detected one
        raw_text = cm->tok.decode(lang_prefix_ids.data(), static_cast<int>(lang_prefix_ids.size())) + raw_text;
    }
    cc->raw_text         = raw_text;  // pre-envelope text, via transcribe_raw_text

    // Parse Qwen3-ASR output: auto-detect emits "language X<asr_text>text"
    // (strip prefix); forced emits "text" (we already seeded the prefix). The
    // split is unconditional — absent prefix leaves transcript_text = raw_text.
    std::string transcript_text = raw_text;
    if (auto sep = raw_text.find("<asr_text>"); sep != std::string::npos) {
        // Auto-detect path: surface the model-picked language name as
        // detected_language (reverse-mapped to BCP-47), but only when the
        // caller did NOT supply a hint (the field reports what the model told
        // us, not what we told it).
        if (lang_prefix_ptr == nullptr || forced_last) {
            constexpr const char k_prefix[] = "language ";
            const size_t         name_start = raw_text.find(k_prefix);
            if (name_start != std::string::npos && name_start < sep) {
                const size_t ns   = name_start + (sizeof(k_prefix) - 1);
                std::string  name = raw_text.substr(ns, sep - ns);
                while (!name.empty() && (name.back() == ' ' || name.back() == '\t' || name.back() == '\n')) {
                    name.pop_back();
                }
                for (const auto & e : k_qwen3_asr_language_names) {
                    if (name == e.pub_name) {
                        cc->detected_language = e.bcp47;
                        break;
                    }
                }
            }
        }
        transcript_text = raw_text.substr(sep + std::strlen("<asr_text>"));
    }

    // Write full_text + a single segment (no timestamps; TIMESTAMPS_NONE).
    cc->t_decode_us = INT64_C(0) - t_dec_start;

    // Optional perf breakdown (finer split than the public mel/encode/decode
    // timings), gated on env var.
    if (transcribe::env::flag("TRANSCRIBE_PERF_DEBUG")) {
        log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
                "qwen3_asr prefill: T_prompt=%d audio=%d prefix=%d reused=%d prefilled=%d draft=%d/%d "
                "prefill_ms=%.2f steps=%d step_ms=%.2f decode_ms=%.2f l0=%a",
                T_prompt, T_enc, prefix_len, cc->kv_reused, cc->kv_prefilled + n_draft, cc->draft_ok, n_draft,
                cc->t_prefill_us / 1000.0, n_steps, t_step_loop_us / 1000.0,
                (INT64_C(0) - t_dec_start) / 1000.0, static_cast<double>(logit0_max));
        const double ms     = 1.0 / 1000.0;
        const double sum_ms = (cc->t_mel_us + t_enc_build_us + cc->t_encode_us + t_enc_d2h_us + t_prefill_build_us +
                               t_prefill_compute_us + t_prefill_logits_us + t_step_loop_us) *
                              ms;
        const double per_step_ms = (n_steps > 0) ? (t_step_loop_us * ms / n_steps) : 0.0;
        log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
                "qwen3_asr perf breakdown:\n"
                "  mel              %8.2f ms\n"
                "  enc_build        %8.2f ms  (graph + sched + uploads)\n"
                "  enc_compute      %8.2f ms\n"
                "    conv_stem    %8.2f ms  (%d chunks)\n"
                "  enc_d2h          %8.2f ms  (%d floats)\n"
                "  prefill_build    %8.2f ms  (kv_init + prompt + graph + sched + uploads)\n"
                "  prefill_compute  %8.2f ms  (T_prompt=%d)\n"
                "  prefill_logits   %8.2f ms  (readback + argmax, vocab=%d)\n"
                "  step_loop        %8.2f ms  (%d steps, %.2f ms/step)\n"
                "    ctx_reset    %8.2f ms  (%.3f ms/step)\n"
                "    graph_build  %8.2f ms  (%.3f ms/step)\n"
                "    sched_alloc  %8.2f ms  (%.3f ms/step)\n"
                "    tensor_set   %8.2f ms  (%.3f ms/step)\n"
                "    compute      %8.2f ms  (%.3f ms/step)\n"
                "    tensor_get   %8.2f ms  (%.3f ms/step)\n"
                "  ---\n"
                "  sum              %8.2f ms",
                cc->t_mel_us * ms, t_enc_build_us * ms, cc->t_encode_us * ms, t_subsample_us * ms, timing.n_chunks,
                t_enc_d2h_us * ms,
                static_cast<int>(cc->enc_host.size()), t_prefill_build_us * ms, t_prefill_compute_us * ms, T_prompt,
                t_prefill_logits_us * ms, vocab, t_step_loop_us * ms, n_steps, per_step_ms, t_step_ctx_us * ms,
                (n_steps > 0) ? (t_step_ctx_us * ms / n_steps) : 0.0, t_step_build_us * ms,
                (n_steps > 0) ? (t_step_build_us * ms / n_steps) : 0.0, t_step_alloc_us * ms,
                (n_steps > 0) ? (t_step_alloc_us * ms / n_steps) : 0.0, t_step_set_us * ms,
                (n_steps > 0) ? (t_step_set_us * ms / n_steps) : 0.0, t_step_comp_us * ms,
                (n_steps > 0) ? (t_step_comp_us * ms / n_steps) : 0.0, t_step_get_us * ms,
                (n_steps > 0) ? (t_step_get_us * ms / n_steps) : 0.0, sum_ms);
    }

    cc->full_text   = transcript_text;
    cc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    cc->has_result  = true;
    transcribe_session::SegmentEntry seg{};
    seg.text  = transcript_text;
    seg.t0_ms = 0;
    seg.t1_ms = static_cast<int64_t>(n_samples) * 1000 / static_cast<int64_t>(cm->hparams.fe_sample_rate);
    cc->segments.push_back(std::move(seg));

    // A truncated decode returns OUTPUT_TRUNCATED; the partial transcript above
    // stays readable (like an aborted run).
    return cc->truncation_status();
}

transcribe_status run(transcribe_session *          session,
                      const float *                 pcm,
                      int                           n_samples,
                      const transcribe_run_params * params) {
    return run_impl(session, pcm, n_samples, params, /*pre=*/nullptr);
}

// ===========================================================================
// Offline batched decode (transcribe_run_batch)
// ===========================================================================
//
// Prefill each utterance serially into its OWN slab of a batched KV cache
// (byte-identical to single-shot prefill, so it inherits correctness), then
// batch only the autoregressive step loop. Encoder + prefill stay
// per-utterance; the batch-axis math lives in the batched step graph.

namespace {

// Apply the session thread count to every backend behind the scheduler. The
// setting persists across sched_reset, so callers only need this once.
void apply_sched_threads(QwenAsrSession * cc) {
    transcribe::configure_sched_n_threads(cc->sched, cc->n_threads);
}

// Fresh per-utterance graph arena. Frees any prior compute_ctx and inits a
// no_alloc metadata context of `mb` MiB.
transcribe_status reset_compute_ctx(QwenAsrSession * cc, int mb, size_t metadata_bytes = SIZE_MAX) {
    if (cc->compute_ctx != nullptr) {
        ggml_free(cc->compute_ctx);
        cc->compute_ctx = nullptr;
    }
    ggml_init_params ip{};
    ip.mem_size     = metadata_bytes == SIZE_MAX ? static_cast<size_t>(mb) * 1024 * 1024 : metadata_bytes;
    if (!ip.mem_size) return TRANSCRIBE_ERR_OOM;
    ip.mem_buffer   = nullptr;
    ip.no_alloc     = true;
    cc->compute_ctx = ggml_init(ip);
    return cc->compute_ctx != nullptr ? TRANSCRIBE_OK : TRANSCRIBE_ERR_OOM;
}

// Batched encoder: mel (parallel) + one encoder graph over all B utterances
// on the batch axis. Fills enc_hosts[b] = [d_enc, T_enc[b]], T_enc[b], valid[b].
// Real-row outputs are bit-identical to encode_one per utterance. A row with
// more than T_enc_cap audio tokens is rejected (fail_status INPUT_TOO_LONG)
// before it can size the batch.
transcribe_status encode_all_batched(QwenAsrSession *                  cc,
                                     QwenAsrModel *                    cm,
                                     const float * const *             pcm,
                                     const int *                       n_samples,
                                     int                               n,
                                     int                               T_enc_cap,
                                     std::vector<char> &               valid,
                                     std::vector<transcribe_status> &  fail_status,
                                     std::vector<std::vector<float>> & enc_hosts,
                                     std::vector<int> &                T_enc_out,
                                     int64_t &                         mel_us,
                                     int64_t &                         enc_us) {
    const int n_mels = cm->hparams.enc_num_mel_bins;

    // Mel (parallel across utterances).
    std::vector<std::vector<float>> mels(n);
    std::vector<int>                mel_nf(n, 0);
    int                             n_threads = cc->n_threads;
    if (n_threads <= 0) {
        n_threads = transcribe::default_n_threads();
    }
    const int64_t t_mel0 = INT64_C(0);
    transcribe::parallel_for_all(n, n_threads, [&](int b) {
        if (pcm[b] == nullptr || n_samples[b] <= 0) {
            return true;  // valid stays 0
        }
        int nm = 0, nf = 0;
        if (cm->mel->compute(pcm[b], static_cast<size_t>(n_samples[b]), mels[b], nm, nf, /*n_threads=*/1) !=
            TRANSCRIBE_OK) {
            return true;  // leave invalid
        }
        mel_nf[b] = nf;
        valid[b]  = 1;
        return true;
    });
    mel_us += INT64_C(0) - t_mel0;

    // Per-utterance timing + packing geometry.
    std::vector<EncoderTiming> timings(n);
    int                        n_chunks_max = 1;
    bool                       any          = false;
    for (int b = 0; b < n; ++b) {
        if (!valid[b]) {
            continue;
        }
        timings[b] = compute_encoder_timing(mel_nf[b], cm->hparams);
        if (timings[b].n_chunks <= 0) {
            valid[b] = 0;
            continue;
        }
        // Same input-length gate as run(), before the row pads every other
        // row's encoder input to its length.
        if (timings[b].T_enc > T_enc_cap) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "qwen3_asr run_batch: utterance %d input too long — %d audio tokens, "
                                "at most %d fit the context with the prompt and output. See "
                                "transcribe_capabilities.max_audio_ms.",
                                b, timings[b].T_enc, T_enc_cap);
            valid[b]       = 0;
            fail_status[b] = TRANSCRIBE_ERR_INPUT_TOO_LONG;
            continue;
        }
        n_chunks_max = std::max(n_chunks_max, timings[b].n_chunks);
        T_enc_out[b] = timings[b].T_enc;
        any          = true;
    }
    if (!any) {
        return TRANSCRIBE_OK;  // caller emits per-row errors
    }

    if (const transcribe_status st = reset_compute_ctx(cc, 16); st != TRANSCRIBE_OK) {
        return st;
    }
    if (cc->sched == nullptr) {
        cc->sched = ggml_backend_sched_new(cm->plan.scheduler_list.data(), nullptr,
                                           static_cast<int>(cm->plan.scheduler_list.size()), 16384, false, true);
        if (cc->sched == nullptr) {
            return TRANSCRIBE_ERR_BACKEND;
        }
    }

    // Conv subsampler over every packed chunk: [mel_per_chunk, n_mels, 1,
    // n*n_chunks_max]; utterance b's chunk c at N = b*n_chunks_max + c,
    // zero-padded beyond n_chunks[b].
    const int          mel_per_chunk = cm->hparams.enc_n_window * 2;
    std::vector<float> subsample_host;
    {
        const size_t       per_chunk_elems = static_cast<size_t>(mel_per_chunk) * n_mels;
        std::vector<float> packed(per_chunk_elems * static_cast<size_t>(n) * n_chunks_max, 0.0f);
        for (int b = 0; b < n; ++b) {
            if (!valid[b]) {
                continue;
            }
            const EncoderTiming & t = timings[b];
            for (int c = 0; c < t.n_chunks; ++c) {
                const int    tail = (c == t.n_chunks - 1) ? t.last_chunk_real_mel : mel_per_chunk;
                const size_t nidx = static_cast<size_t>(b) * n_chunks_max + c;
                for (int m = 0; m < n_mels; ++m) {
                    const float * src =
                        mels[b].data() + static_cast<size_t>(m) * mel_nf[b] + static_cast<size_t>(c) * mel_per_chunk;
                    float * dst = packed.data() + nidx * per_chunk_elems + static_cast<size_t>(m) * mel_per_chunk;
                    std::memcpy(dst, src, static_cast<size_t>(tail) * sizeof(float));
                }
            }
        }
        const int64_t t_sub0 = INT64_C(0);
        if (const transcribe_status st = run_subsample(cc, cm, packed.data(), n * n_chunks_max, subsample_host);
            st != TRANSCRIBE_OK) {
            return st;
        }
        enc_us += INT64_C(0) - t_sub0;
    }

    // Subsample outputs are complete in host storage; no later graph uses its arena.
    if (const auto st = release_subsample_scratch(cc); st != TRANSCRIBE_OK) return st;

    // Build batched encoder graph.
    EncoderBuildBatched eb =
        build_encoder_graph_batched(cc->compute_ctx, cm->weights, cm->hparams, n_chunks_max, n, cc->encoder_use_flash);
    if (eb.graph == nullptr || eb.out == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_sched_reset(cc->sched);
    if (!TRANSCRIBE_ALLOC_SCHED_GRAPH(cc->sched, eb.graph, "qwen3_asr")) {
        return TRANSCRIBE_ERR_OOM;
    }

    const int T_per_chunk = eb.T_per_chunk;
    const int T_pad_max   = eb.T_pad_max;

    ggml_backend_tensor_set(eb.subsample_in, subsample_host.data(), 0, subsample_host.size() * sizeof(float));

    // Positional embedding (shared across chunks/utterances).
    {
        std::vector<float> pe = build_sinusoid_pe(cm->hparams.enc_d_model, T_per_chunk);
        ggml_backend_tensor_set(eb.pos_emb_in, pe.data(), 0, pe.size() * sizeof(float));
    }

    // Key-pad mask [T_pad_max, T_pad_max, 1, n]: row b attends keys k < T_enc[b].
    {
        const size_t plane = static_cast<size_t>(T_pad_max) * T_pad_max;
        if (cc->encoder_use_flash) {
            const ggml_fp16_t        mz = ggml_fp32_to_fp16(0.0f);
            const ggml_fp16_t        mn = ggml_fp32_to_fp16(-INFINITY);
            std::vector<ggml_fp16_t> mask(plane * n, mn);
            for (int b = 0; b < n; ++b) {
                const int     real = valid[b] ? std::max(1, T_enc_out[b]) : 1;
                ggml_fp16_t * base = mask.data() + plane * b;
                for (int q = 0; q < T_pad_max; ++q) {
                    std::fill(base + static_cast<size_t>(q) * T_pad_max,
                              base + static_cast<size_t>(q) * T_pad_max + real, mz);
                }
            }
            ggml_backend_tensor_set(eb.mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        } else {
            const float        mn = -INFINITY;
            std::vector<float> mask(plane * n, mn);
            for (int b = 0; b < n; ++b) {
                const int real = valid[b] ? std::max(1, T_enc_out[b]) : 1;
                float *   base = mask.data() + plane * b;
                for (int q = 0; q < T_pad_max; ++q) {
                    std::fill(base + static_cast<size_t>(q) * T_pad_max,
                              base + static_cast<size_t>(q) * T_pad_max + real, 0.0f);
                }
            }
            ggml_backend_tensor_set(eb.mask_in, mask.data(), 0, mask.size() * sizeof(float));
        }
    }

    apply_sched_threads(cc);

    const int64_t t_enc0 = INT64_C(0);
    if (compute_qwen_graph(cc, eb.graph) != GGML_STATUS_SUCCESS) {
        return TRANSCRIBE_ERR_BACKEND;
    }
    enc_us += INT64_C(0) - t_enc0;

    // Readback [d_enc, T_pad_max, n]; slice each utterance's first T_enc[b] rows.
    const int          d_enc = static_cast<int>(eb.out->ne[0]);
    std::vector<float> out_all(static_cast<size_t>(d_enc) * T_pad_max * n);
    ggml_backend_tensor_get(eb.out, out_all.data(), 0, out_all.size() * sizeof(float));
    for (int b = 0; b < n; ++b) {
        if (!valid[b]) {
            continue;
        }
        const int te = T_enc_out[b];
        enc_hosts[b].resize(static_cast<size_t>(d_enc) * te);
        const float * src = out_all.data() + static_cast<size_t>(b) * T_pad_max * d_enc;
        std::memcpy(enc_hosts[b].data(), src, static_cast<size_t>(d_enc) * te * sizeof(float));
    }
    return TRANSCRIBE_OK;
}

// Batched prefill: one graph processes all B prompts, writing each utterance's
// KV into its slab and returning the first generated token per utterance. The
// caller must have allocated cc->kv_cache_batch with n_ctx >= max T_prompt and
// computed prompt_ids[b] / T_prompt[b] / prefix_len. Collapses B per-utterance
// prefill graph-builds + sched-allocs into one.
transcribe_status prefill_all_batched(QwenAsrSession *                          cc,
                                      QwenAsrModel *                            cm,
                                      const std::vector<char> &                 valid,
                                      const std::vector<std::vector<int32_t>> & prompt_ids,
                                      const std::vector<int> &                  T_prompt,
                                      int                                       prefix_len,
                                      const std::vector<std::vector<float>> &   enc_hosts,
                                      const std::vector<int> &                  T_enc,
                                      int                                       n,
                                      std::vector<int32_t> &                    first_tok_out) {
    int T_prompt_max = 0, T_enc_max = 0;
    for (int b = 0; b < n; ++b) {
        if (!valid[b]) {
            continue;
        }
        T_prompt_max = std::max(T_prompt_max, T_prompt[b]);
        T_enc_max    = std::max(T_enc_max, T_enc[b]);
    }
    if (T_prompt_max == 0) {
        return TRANSCRIBE_OK;
    }
    T_enc_max = std::max(1, T_enc_max);

    if (const transcribe_status st = reset_compute_ctx(cc, 32); st != TRANSCRIBE_OK) {
        return st;
    }
    PrefillBuildBatched pb = build_prefill_graph_batched(cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache_batch,
                                                         T_prompt_max, T_enc_max, n, cc->decoder_use_flash);
    if (pb.graph == nullptr || pb.out == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    ggml_backend_sched_reset(cc->sched);
    if (!TRANSCRIBE_ALLOC_SCHED_GRAPH(cc->sched, pb.graph, "qwen3_asr")) {
        return TRANSCRIBE_ERR_OOM;
    }

    const int d_enc = cm->hparams.enc_output_dim;

    // input_ids [T_prompt_max, n] (pad 0).
    {
        std::vector<int32_t> ids(static_cast<size_t>(T_prompt_max) * n, 0);
        for (int b = 0; b < n; ++b) {
            if (!valid[b]) {
                continue;
            }
            std::memcpy(ids.data() + static_cast<size_t>(b) * T_prompt_max, prompt_ids[b].data(),
                        static_cast<size_t>(T_prompt[b]) * sizeof(int32_t));
        }
        ggml_backend_tensor_set(pb.input_ids_in, ids.data(), 0, ids.size() * sizeof(int32_t));
    }
    // Audio injection by elementwise blend (see decoder.h): audio_dense holds
    // each utterance's enc_out embeds scattered into their prompt positions,
    // keep is 0 there and 1 elsewhere. (d_enc == dec_hidden, enforced.)
    {
        std::vector<float> audio_dense(static_cast<size_t>(d_enc) * T_prompt_max * n, 0.0f);
        std::vector<float> keep(static_cast<size_t>(T_prompt_max) * n, 1.0f);
        for (int b = 0; b < n; ++b) {
            const int te = valid[b] ? T_enc[b] : 0;
            // enc_hosts[b] is [d_enc, te] column-major; audio token j lands at
            // prompt position prefix_len+j, flat column b*T_prompt_max+pos.
            for (int j = 0; j < te; ++j) {
                const size_t dst_col = static_cast<size_t>(b) * T_prompt_max + (prefix_len + j);
                std::memcpy(audio_dense.data() + dst_col * d_enc, enc_hosts[b].data() + static_cast<size_t>(j) * d_enc,
                            static_cast<size_t>(d_enc) * sizeof(float));
                keep[dst_col] = 0.0f;
            }
        }
        ggml_backend_tensor_set(pb.audio_dense_in, audio_dense.data(), 0, audio_dense.size() * sizeof(float));
        ggml_backend_tensor_set(pb.keep_mask_in, keep.data(), 0, keep.size() * sizeof(float));
    }
    // positions [T_prompt_max] 0..T-1 (shared).
    {
        std::vector<int32_t> pos(T_prompt_max);
        for (int t = 0; t < T_prompt_max; ++t) {
            pos[t] = t;
        }
        ggml_backend_tensor_set(pb.positions_in, pos.data(), 0, pos.size() * sizeof(int32_t));
    }
    // causal mask [T_prompt_max, T_prompt_max] f16 (shared): m[q*T+k]=0 if k<=q.
    {
        const ggml_fp16_t        mz = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t        mn = ggml_fp32_to_fp16(-INFINITY);
        std::vector<ggml_fp16_t> mask(static_cast<size_t>(T_prompt_max) * T_prompt_max, mn);
        for (int q = 0; q < T_prompt_max; ++q) {
            std::fill(mask.begin() + static_cast<size_t>(q) * T_prompt_max,
                      mask.begin() + static_cast<size_t>(q) * T_prompt_max + q + 1, mz);
        }
        ggml_backend_tensor_set(pb.mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }
    // kv_idx [T_prompt_max, n] i64: idx[t,b] = t.
    {
        std::vector<int64_t> kidx(static_cast<size_t>(T_prompt_max) * n);
        for (int b = 0; b < n; ++b) {
            for (int t = 0; t < T_prompt_max; ++t) {
                kidx[static_cast<size_t>(b) * T_prompt_max + t] = t;
            }
        }
        ggml_backend_tensor_set(pb.kv_idx_in, kidx.data(), 0, kidx.size() * sizeof(int64_t));
    }
    // last_idx [1, n] i32: each utterance's last real position.
    {
        std::vector<int32_t> lidx(n, 0);
        for (int b = 0; b < n; ++b) {
            lidx[b] = valid[b] ? (T_prompt[b] - 1) : 0;
        }
        ggml_backend_tensor_set(pb.last_idx_in, lidx.data(), 0, lidx.size() * sizeof(int32_t));
    }

    apply_sched_threads(cc);
    if (compute_qwen_graph(cc, pb.graph) != GGML_STATUS_SUCCESS) {
        return TRANSCRIBE_ERR_BACKEND;
    }

    std::vector<int32_t> amax(n, 0);
    ggml_backend_tensor_get(pb.out, amax.data(), 0, amax.size() * sizeof(int32_t));
    for (int b = 0; b < n; ++b) {
        first_tok_out[b] = amax[b];
    }
    return TRANSCRIBE_OK;
}

// Decode one utterance's generated token ids into a ResultSet (strip EOS,
// detokenize, parse the Qwen3-ASR "language X<asr_text>…" envelope). Mirrors
// run()'s output-parsing tail; `lang_prefix_ptr` non-null means the caller
// forced a language (so we don't surface a detected one).
transcribe_session::ResultSet finalize_utterance(QwenAsrModel *               cm,
                                                 std::vector<int32_t>         generated_ids,
                                                 const std::vector<int32_t> * lang_prefix_ptr,
                                                 int                          n_samples) {
    transcribe_session::ResultSet rs;
    const int32_t                 eos_id = cm->hparams.eos_token_id;
    if (!generated_ids.empty() && generated_ids.back() == eos_id) {
        generated_ids.pop_back();
    }

    std::string raw_text = cm->tok.decode(generated_ids.data(), static_cast<int>(generated_ids.size()));

    std::string transcript_text = raw_text;
    if (auto sep = raw_text.find("<asr_text>"); sep != std::string::npos) {
        if (lang_prefix_ptr == nullptr) {
            constexpr const char k_prefix[] = "language ";
            const size_t         name_start = raw_text.find(k_prefix);
            if (name_start != std::string::npos && name_start < sep) {
                const size_t ns   = name_start + (sizeof(k_prefix) - 1);
                std::string  name = raw_text.substr(ns, sep - ns);
                while (!name.empty() && (name.back() == ' ' || name.back() == '\t' || name.back() == '\n')) {
                    name.pop_back();
                }
                for (const auto & e : k_qwen3_asr_language_names) {
                    if (name == e.pub_name) {
                        rs.detected_language = e.bcp47;
                        break;
                    }
                }
            }
        }
        transcript_text = raw_text.substr(sep + std::strlen("<asr_text>"));
    }

    rs.full_text   = transcript_text;
    rs.raw_text    = raw_text;
    rs.result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    rs.has_result  = true;
    rs.status      = TRANSCRIBE_OK;
    transcribe_session::SegmentEntry seg{};
    seg.text  = transcript_text;
    seg.t0_ms = 0;
    seg.t1_ms = static_cast<int64_t>(n_samples) * 1000 / static_cast<int64_t>(cm->hparams.fe_sample_rate);
    rs.segments.push_back(std::move(seg));
    return rs;
}

// Serial fallback: run() per utterance, capturing each into batch_results.
// Used when batched decode is unavailable (flash off) or as the safe path.
transcribe_status run_batch_serial(QwenAsrSession *              cc,
                                   const float * const *         pcm,
                                   const int *                   n_samples,
                                   int                           n,
                                   const transcribe_run_params * params) {
    return transcribe::run_batch_serial(cc, pcm, n_samples, n,
                                        [&](const float * p, int ns) { return run(cc, p, ns, params); });
}

}  // namespace

transcribe_status run_batch(transcribe_session *          session,
                            const float * const *         pcm,
                            const int *                   n_samples,
                            int                           n,
                            const transcribe_run_params * params) {
    if (session == nullptr || pcm == nullptr || n_samples == nullptr || n <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto * cc = static_cast<QwenAsrSession *>(session);
    auto * cm = static_cast<QwenAsrModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (!cm->mel.has_value()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Batched decode requires the flash-attention step path and dump-free
    // operation. Fall back to the serial loop otherwise (same results).
    if (!cc->decoder_use_flash || transcribe::debug::enabled() || n == 1) {
        return run_batch_serial(cc, pcm, n_samples, n, params);
    }

    transcribe::debug::init();

    // Shared language hint (v1: one run_params across the batch).
    std::vector<int32_t>         lang_prefix_ids;
    const std::vector<int32_t> * lang_prefix_ptr = nullptr;
    if (params != nullptr && params->language != nullptr && params->language[0] != '\0') {
        if (encode_language_prefix(cm->tok, params->language, lang_prefix_ids) != TRANSCRIBE_OK) {
            return TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE;
        }
        lang_prefix_ptr = &lang_prefix_ids;
    }

    // Shared system context (vocabulary / prompt), one run_params per batch,
    // fitted as if there were no audio. A row whose own budget is smaller than
    // that fit would get a different system context from run(), so the batch
    // then goes serial (see fit_terms_and_context: otherwise the fits match).
    const int ceiling       = qwen3_context_ceiling(cc->n_ctx, cm->hparams);
    auto      system_budget = [&](int T_enc) {
        return std::max(ceiling - k_gen_reserve - prompt_tokens_without_system(T_enc, lang_prefix_ptr), 0);
    };
    std::vector<int32_t> system_ids;
    if (encode_system_context(cm->tok, params, system_budget(0), system_ids) != TRANSCRIBE_OK) {
        return run_batch_serial(cc, pcm, n_samples, n, params);
    }

    // Pass 1: per-utterance encoder + prefill into KV slabs.
    std::vector<std::vector<int32_t>> generated(n);
    std::vector<int>                  T_prompt(n, 0);
    std::vector<int32_t>              next_tok(n, 0);
    std::vector<int>                  n_past(n, 0);
    std::vector<char>                 valid(n, 0);  // utterance produced a usable prefill
    int64_t                           mel_us = 0, enc_us = 0;

    // Per-utterance terminal status for rejected rows. Defaults to INVALID_ARG;
    // over-length rows are upgraded to INPUT_TOO_LONG.
    std::vector<transcribe_status> fail_status(n, TRANSCRIBE_ERR_INVALID_ARG);

    // Encode every utterance first so we can size the batched cache to the
    // real max prompt length before allocating it.
    std::vector<std::vector<float>> enc_hosts(n);
    std::vector<int>                T_enc(n, 0);
    const int     T_enc_cap  = ceiling - k_gen_reserve - prompt_tokens_without_system(0, lang_prefix_ptr);
    const int64_t t_encpass0 = INT64_C(0);
    if (const transcribe_status st = encode_all_batched(cc, cm, pcm, n_samples, n, T_enc_cap, valid, fail_status,
                                                        enc_hosts, T_enc, mel_us, enc_us);
        st != TRANSCRIBE_OK) {
        return st;
    }
    const int64_t enc_pass_us = INT64_C(0) - t_encpass0;

    // Prompt length bound → max_n_kv and batched-cache n_ctx. Build and keep
    // each utterance's prompt token ids for the batched prefill.
    int                               max_T_prompt = 0;
    int                               max_T_enc    = 0;
    int                               prefix_len   = 0;
    std::vector<std::vector<int32_t>> prompt_ids(n);
    for (int b = 0; b < n; ++b) {
        if (!valid[b]) {
            continue;
        }
        if (static_cast<int>(system_ids.size()) > system_budget(T_enc[b])) {
            return run_batch_serial(cc, pcm, n_samples, n, params);
        }
        std::vector<int64_t> ap;
        build_prompt_tokens(cm->hparams, cm->chat_tokens, T_enc[b], system_ids, lang_prefix_ptr, prompt_ids[b], ap);
        T_prompt[b] = static_cast<int>(prompt_ids[b].size());
        prefix_len  = ap.empty() ? 0 : static_cast<int>(ap.front());
        // Same gate as single-shot run(); the rest of the batch still runs.
        if (T_prompt[b] + k_gen_reserve > ceiling) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "qwen3_asr run_batch: utterance %d input too long — %d audio + "
                                "%d prompt tokens exceed the %d-token context. See "
                                "transcribe_capabilities.max_audio_ms.",
                                b, T_enc[b], T_prompt[b] - T_enc[b], ceiling);
            valid[b]       = 0;
            fail_status[b] = TRANSCRIBE_ERR_INPUT_TOO_LONG;
            continue;
        }
        max_T_prompt = std::max(max_T_prompt, T_prompt[b]);
        max_T_enc    = std::max(max_T_enc, T_enc[b]);
    }
    if (max_T_prompt == 0) {
        // No usable utterance — emit per-row errors and return.
        for (int b = 0; b < n; ++b) {
            transcribe_session::ResultSet rs;
            rs.status = fail_status[b];
            cc->batch_results.push_back(std::move(rs));
        }
        return TRANSCRIBE_OK;
    }
    const int max_new =
        transcribe::pick_decode_budget(transcribe::predict_transcript_tokens(max_T_enc, cm->limits.ms_per_audio_token),
                                       k_gen_reserve, max_T_prompt, ceiling);
    int max_n_kv = 1024;
    while (max_n_kv < max_T_prompt + max_new) {
        max_n_kv *= 2;
    }
    // Clamp the pow2 round-up to the ceiling (the per-utterance gate guarantees
    // every valid row still fits).
    if (max_n_kv > ceiling) {
        max_n_kv = ceiling;
    }

    // Allocate / reuse the batched KV cache (n_ctx == max_n_kv, n slabs).
    ggml_type kv_type = (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) ? GGML_TYPE_F32 : GGML_TYPE_F16;
    if (cc->kv_cache_batch.self_k == nullptr || cc->kv_batch_cap != n || cc->kv_batch_n_ctx != max_n_kv) {
        cc->kv_cache_batch.free();
        if (!transcribe::causal_lm::kv_init_batched(cc->kv_cache_batch, cm->plan.primary, max_n_kv,
                                                    cm->hparams.dec_n_kv_heads, cm->hparams.dec_head_dim,
                                                    cm->hparams.dec_n_layers, n, kv_type)) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "qwen3_asr run_batch: batched KV cache allocation failed "
                                "(n_ctx=%d x %d utterances) — out of memory. Lower "
                                "transcribe_session_params.n_ctx or the batch size.",
                                max_n_kv, n);
            return TRANSCRIBE_ERR_OOM;
        }
        cc->kv_batch_cap   = n;
        cc->kv_batch_n_ctx = max_n_kv;
    } else if (cc->kv_cache_batch.buffer != nullptr) {
        ggml_backend_buffer_clear(cc->kv_cache_batch.buffer, 0);
    }

    const int64_t t_prefpass0 = INT64_C(0);
    {
        std::vector<int32_t> first_tok(n, 0);
        if (const transcribe_status st =
                prefill_all_batched(cc, cm, valid, prompt_ids, T_prompt, prefix_len, enc_hosts, T_enc, n, first_tok);
            st != TRANSCRIBE_OK) {
            return st;
        }
        for (int b = 0; b < n; ++b) {
            if (!valid[b]) {
                continue;
            }
            n_past[b]   = T_prompt[b];
            next_tok[b] = first_tok[b];
            generated[b].push_back(first_tok[b]);
        }
    }
    const int64_t prefill_pass_us = INT64_C(0) - t_prefpass0;

    // Pass 2: batched step loop (shared causal_lm driver).
    const int32_t eos_id = cm->hparams.eos_token_id;

    if (const transcribe_status st = reset_compute_ctx(cc, 16); st != TRANSCRIBE_OK) {
        return st;
    }
    StepBuildBatched sb = build_step_graph_batched(cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache_batch,
                                                   max_n_kv, n, cc->decoder_use_flash);
    if (sb.graph == nullptr || sb.out == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_sched_reset(cc->sched);
    if (!TRANSCRIBE_ALLOC_SCHED_GRAPH(cc->sched, sb.graph, "qwen3_asr")) {
        return TRANSCRIBE_ERR_OOM;
    }

    transcribe::causal_lm::StepBatchedIO io{};
    io.input_ids = sb.input_ids_in;
    io.positions = sb.position_in;
    io.kv_idx    = sb.kv_idx_in;
    io.mask      = sb.mask_in;
    io.argmax    = sb.out;
    io.graph     = sb.graph;

    transcribe::causal_lm::StepBatchedState step_state;
    step_state.valid    = valid;
    step_state.next_tok = next_tok;
    step_state.n_past   = n_past;

    transcribe::causal_lm::StepLoopStats step_stats;
    std::vector<char>                    truncated;
    if (!check_qwen_graph_placement(cc, sb.graph)) return TRANSCRIBE_ERR_BACKEND;
    if (const transcribe_status st = transcribe::causal_lm::run_batched_step_loop(
            cc, cc->sched, io, n, max_n_kv, eos_id, max_new, step_state, generated, &step_stats, &truncated);
        st != TRANSCRIBE_OK) {
        return st;
    }
    const int64_t step_us = step_stats.step_us;
    const int     n_steps = step_stats.n_steps;

    // Capture per-utterance results.
    const int valid_count = std::max(1, static_cast<int>(std::count(valid.begin(), valid.end(), char(1))));
    for (int b = 0; b < n; ++b) {
        if (!valid[b]) {
            transcribe_session::ResultSet rs;
            rs.status = fail_status[b];
            cc->batch_results.push_back(std::move(rs));
            continue;
        }
        transcribe_session::ResultSet rs = finalize_utterance(cm, generated[b], lang_prefix_ptr, n_samples[b]);
        // Per-utterance truncation parity with the single-shot path.
        if (b < static_cast<int>(truncated.size()) && truncated[b]) {
            cc->was_truncated = true;
            rs.status         = transcribe::decode_stop_status(truncated[b]);
        }
        rs.t_mel_us    = mel_us / valid_count;
        rs.t_encode_us = enc_us / valid_count;
        rs.t_decode_us = step_us / valid_count;
        cc->batch_results.push_back(std::move(rs));
    }

    if (transcribe::env::flag("TRANSCRIBE_PERF_DEBUG")) {
        log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
                "qwen3_asr run_batch: n=%d valid=%d max_n_kv=%d steps=%d (all phases batched x%d)\n"
                "  enc_pass=%.1fms (mel=%.1f parallel + enc_compute=%.1f, 1 graph)\n"
                "  prefill_pass=%.1fms (1 batched graph: build/sched/compute/readback)\n"
                "  step_loop=%.1fms (%.2fms/step)",
                n, valid_count, max_n_kv, n_steps, valid_count, enc_pass_us / 1000.0, mel_us / 1000.0, enc_us / 1000.0,
                prefill_pass_us / 1000.0, step_us / 1000.0, n_steps > 0 ? step_us / 1000.0 / n_steps : 0.0);
    }

    return TRANSCRIBE_OK;
}

// ===========================================================================
// Streaming: the conv stem runs while the audio arrives
// ===========================================================================
//
// The conv stem (3x Conv2d + conv_out) works on independent n_window*2-frame
// (1 s) mel chunks, so each chunk goes through it as soon as its audio has
// arrived. Finalize then runs only the last chunk, the encoder transformer
// and the decoder. The result is the offline one, bit for bit:
//
// * Mel frames. Frame t is the STFT of input samples [t*hop - n_fft/2,
//   t*hop + n_fft/2) of the reflect-padded audio, so it is final once sample
//   t*hop + n_fft/2 has arrived: the end-of-input reflection can no longer
//   reach it. mel_log10 computes frames over a slice of the received audio
//   that starts n_fft/2/hop + 1 hops before the first wanted frame, so the
//   slice's own start padding never reaches a wanted frame, and each frame's
//   STFT is computed from the same samples with the same code as offline.
// * Normalization. per_utterance clamps every value to (max over the whole
//   utterance) - 8 before scaling, so a chunk normalized with the running max
//   can differ from offline once a louder frame arrives later. Each time a
//   chunk completes, every earlier chunk is renormalized with the current max
//   and the ones whose normalized mel changed (bitwise compare) are re-run
//   with it; finalize does the same with the final max. So every chunk's conv
//   input ends up exactly the offline one, and only a max that rises in the
//   last partial chunk leaves re-runs for finalize.
// * Conv stem. A chunk's output depends only on that chunk's mel (offline
//   runs the chunks as an independent batch, and each output element is one
//   dot product over that chunk's im2col), so it does not matter when or
//   alongside which other chunks it was computed.
//
// The decoder sees the same encoder output, so the transcript is the
// offline transcript; only the work after the end of the audio shrinks.

transcribe_status ensure_sched(QwenAsrSession * cc, QwenAsrModel * cm) {
    if (cc->sched == nullptr) {
        cc->sched = ggml_backend_sched_new(cm->plan.scheduler_list.data(), nullptr,
                                           static_cast<int>(cm->plan.scheduler_list.size()), 16384, false, true);
        if (cc->sched == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr stream: ggml_backend_sched_new failed");
            return TRANSCRIBE_ERR_BACKEND;
        }
    }
    return TRANSCRIBE_OK;
}

// Append raw log10 mel frames [stream_n_frames, t_end) computed from the first
// n_avail received samples. Every frame in the range must be final for
// n_avail samples, or n_avail must be the whole utterance.
transcribe_status stream_add_frames(QwenAsrSession * cc, QwenAsrModel * cm, size_t n_avail, int t_end) {
    const int t0 = cc->stream_n_frames;
    if (t_end <= t0) {
        return TRANSCRIBE_OK;
    }
    const transcribe::MelConfig & cfg    = cm->mel_log10->config();
    const int                     hop    = cfg.hop_length;
    const int                     n_mels = cfg.num_mels;
    const int                     ctx    = (cfg.n_fft / 2) / hop + 1;
    const int                     s0     = std::max(0, t0 - ctx);
    const size_t                  off    = static_cast<size_t>(s0) * hop;

    std::vector<float> raw;
    int                nm = 0, nf = 0;
    if (const transcribe_status st =
            cm->mel_log10->compute(cc->stream_pcm.data() + off, n_avail - off, raw, nm, nf, cc->n_threads);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (nm != n_mels || t_end - s0 > nf) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    cc->stream_log_mel.resize(static_cast<size_t>(t_end) * n_mels);
    for (int t = t0; t < t_end; ++t) {
        float * dst = cc->stream_log_mel.data() + static_cast<size_t>(t) * n_mels;
        for (int m = 0; m < n_mels; ++m) {
            const float v = raw[static_cast<size_t>(m) * nf + (t - s0)];
            dst[m]        = v;
            if (v > cc->stream_max) {
                cc->stream_max = v;
            }
        }
    }
    cc->stream_n_frames = t_end;
    return TRANSCRIBE_OK;
}

// MelFrontend's per_utterance clamp and scaling for chunk c, with `maxv` as
// the utterance max, into dst = [n_mels, mel_per_chunk] (pack_mel_chunks
// layout; frames past the end of the audio stay zero).
void stream_normalize_chunk(const QwenAsrSession * cc, int n_mels, int mel_per_chunk, int c, float maxv, float * dst) {
    const double floor_val = static_cast<double>(maxv) - 8.0;
    const int    f0        = c * mel_per_chunk;
    const int    n_real    = std::min(mel_per_chunk, cc->stream_n_frames - f0);
    for (int m = 0; m < n_mels; ++m) {
        float * row = dst + static_cast<size_t>(m) * mel_per_chunk;
        for (int t = 0; t < n_real; ++t) {
            double v = static_cast<double>(cc->stream_log_mel[static_cast<size_t>(f0 + t) * n_mels + m]);
            if (v < floor_val) {
                v = floor_val;
            }
            row[t] = static_cast<float>((v + 4.0) / 4.0);
        }
        std::fill(row + std::max(0, n_real), row + mel_per_chunk, 0.0f);
    }
}

// Bring the conv stem output up to date for chunks [0, n_chunks) at the
// current max: chunks not run yet, and earlier chunks whose normalized mel
// changed since they were run (the max rose). All of them go through one
// subsample graph. Returns the number of chunks run in *n_run, of which
// *n_rerun were re-runs.
transcribe_status stream_run_chunks(QwenAsrSession * cc, QwenAsrModel * cm, int n_chunks, int * n_run, int * n_rerun) {
    const int    n_mels        = cm->mel_log10->num_mels();
    const int    mel_per_chunk = cm->hparams.enc_n_window * 2;
    const size_t per_chunk     = static_cast<size_t>(mel_per_chunk) * n_mels;
    const size_t out_chunk     = static_cast<size_t>(cm->hparams.enc_d_model) * aftercnn_len(mel_per_chunk);
    const int    n_done        = cc->stream_n_chunks;

    std::vector<int>   todo;
    std::vector<float> packed;
    std::vector<float> chunk(per_chunk);
    *n_rerun = 0;
    for (int c = 0; c < n_chunks; ++c) {
        stream_normalize_chunk(cc, n_mels, mel_per_chunk, c, cc->stream_max, chunk.data());
        if (c < n_done) {
            float * prev = cc->stream_chunk_mel.data() + per_chunk * c;
            if (std::memcmp(chunk.data(), prev, per_chunk * sizeof(float)) == 0) {
                continue;
            }
            ++*n_rerun;
        }
        todo.push_back(c);
        packed.insert(packed.end(), chunk.begin(), chunk.end());
    }
    *n_run = static_cast<int>(todo.size());
    if (todo.empty()) {
        return TRANSCRIBE_OK;
    }
    if (const transcribe_status st = ensure_sched(cc, cm); st != TRANSCRIBE_OK) {
        return st;
    }
    if (reset_compute_ctx(cc, 4, compute_metadata_bytes(cc, 4 * 1024 * 1024,
                                                       static_cast<int>(todo.size()), false, 0)) != TRANSCRIBE_OK) {
        return TRANSCRIBE_ERR_OOM;
    }
    std::vector<float> out;
    if (const transcribe_status st = run_subsample(cc, cm, packed.data(), static_cast<int>(todo.size()), out);
        st != TRANSCRIBE_OK) {
        return st;
    }
    cc->stream_chunk_mel.resize(per_chunk * std::max(n_chunks, n_done));
    cc->stream_subsample.resize(out_chunk * std::max(n_chunks, n_done));
    for (size_t i = 0; i < todo.size(); ++i) {
        std::memcpy(cc->stream_chunk_mel.data() + per_chunk * todo[i], packed.data() + per_chunk * i,
                    per_chunk * sizeof(float));
        std::memcpy(cc->stream_subsample.data() + out_chunk * todo[i], out.data() + out_chunk * i,
                    out_chunk * sizeof(float));
    }
    cc->stream_n_chunks = std::max(n_chunks, n_done);
    return TRANSCRIBE_OK;
}

// Fill the prompt-prefix cache for a run with these params before its audio
// is known: the chat frame and the system context, up to and including
// <|audio_start|>, which is everything the prompt has before the audio. A
// stream begins while the phrase is still being spoken, so this work is off
// the path from the end of the audio to the text; finalize then prefills
// only the audio and what follows it.
//
// The rows must be computed on the attention path of the final prompt
// (tiled iff it has 64+ rows, decoder.h), which depends on the audio length.
// T_enc_min is a lower bound on it (the audio so far): the prefix is
// computed for the path that bound implies, and stream_feed calls this again
// once the audio has grown enough to change it. The system context is fitted
// as for a run with no audio; if the real run fits it differently, its
// prompt simply shares less with the cache.
transcribe_status prefill_prompt_prefix(QwenAsrSession * cc, QwenAsrModel * cm, const transcribe_run_params * params,
                                        int T_enc_min) {
    if (!prefix_cache_enabled()) {
        return TRANSCRIBE_OK;
    }
    const int64_t        t0      = INT64_C(0);
    const int            ceiling = qwen3_context_ceiling(cc->n_ctx, cm->hparams);
    std::vector<int32_t> system_ids;
    if (const transcribe_status st = encode_system_context(
            cm->tok, params, ceiling - k_gen_reserve - prompt_tokens_without_system(0, nullptr), system_ids);
        st != TRANSCRIBE_OK) {
        return st;
    }
    std::vector<int32_t> lang_ids;
    if (params->language != nullptr && params->language[0] != '\0') {
        if (const transcribe_status st = encode_language_prefix(cm->tok, params->language, lang_ids);
            st != TRANSCRIBE_OK) {
            return st;
        }
    }
    std::vector<int32_t> ids;
    std::vector<int64_t> audio_positions;
    build_prompt_tokens(cm->hparams, cm->chat_tokens, /*T_enc=*/0, system_ids, &lang_ids, ids, audio_positions);
    // <|im_start|>system\n + system + <|im_end|>\n<|im_start|>user\n<|audio_start|>
    const int prefix_len = static_cast<int>(system_ids.size()) + 9;
    if (prefix_len > static_cast<int>(ids.size()) || ids[prefix_len - 1] != cm->hparams.audio_start_token_id) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const bool tiled        = static_cast<int>(ids.size()) + T_enc_min >= k_fa_tile_rows;
    cc->stream_prompt_rows  = static_cast<int>(ids.size());
    cc->stream_prefix_tiled = tiled ? 1 : 0;
    const int n_past        = common_prefix(cc, ids.data(), prefix_len, tiled);
    if (n_past == prefix_len) {
        return TRANSCRIBE_OK;
    }
    if (cc->kv_cache.ctx == nullptr) {
        int n_ctx = std::min(1024, ceiling);
        if (khr_gpu_graph_policy(cc)) {
            n_ctx = initial_khr_kv_capacity(prefix_len, ceiling);
            log_msg(TRANSCRIBE_LOG_LEVEL_INFO,
                    "qwen3_asr KHR KV prefix: prompt=%zu audio_min=%d generation_budget=%d capacity=%d",
                    ids.size(), T_enc_min, k_gen_reserve, n_ctx);
        }
        if (prefix_len > n_ctx) {
            return TRANSCRIBE_OK;  // left to the run, which sizes the cache itself
        }
        const ggml_type kv_type = cc->kv_type == TRANSCRIBE_KV_TYPE_F32 ? GGML_TYPE_F32 : GGML_TYPE_F16;
        if (!transcribe::causal_lm::kv_init(cc->kv_cache, cm->plan.primary, n_ctx, cm->hparams.dec_n_kv_heads,
                                            cm->hparams.dec_head_dim, cm->hparams.dec_n_layers, kv_type)) {
            return TRANSCRIBE_ERR_OOM;
        }
        clear_prefix_cache(cc);
    } else if (prefix_len > cc->kv_cache.n_ctx) {
        return TRANSCRIBE_OK;
    }
    if (const transcribe_status st = ensure_sched(cc, cm); st != TRANSCRIBE_OK) {
        return st;
    }
    if (reset_compute_ctx(cc, 16, compute_metadata_bytes(cc, 16 * 1024 * 1024, 0, false, 16384)) != TRANSCRIBE_OK) {
        return TRANSCRIBE_ERR_OOM;
    }
    const int         T_chunk = prefix_len - n_past;
    PrefillChunkBuild pb      = build_prefill_chunk_graph(cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache,
                                                          T_chunk, prefix_len, T_chunk, 0, 0, /*n_logits=*/0,
                                                          cc->decoder_use_flash, chunk_fa_q_rows(T_chunk, tiled));
    if (pb.graph == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_sched_reset(cc->sched);
    if (!TRANSCRIBE_ALLOC_SCHED_GRAPH(cc->sched, pb.graph, "qwen3_asr")) {
        return TRANSCRIBE_ERR_OOM;
    }
    apply_sched_threads(cc);
    cc->kv_ids.resize(static_cast<size_t>(n_past));
    cc->kv_tiled.resize(static_cast<size_t>(n_past));
    ggml_backend_tensor_set(pb.input_ids_in, ids.data() + n_past, 0, static_cast<size_t>(T_chunk) * sizeof(int32_t));
    {
        std::vector<int32_t> positions(T_chunk);
        std::vector<int64_t> kv_idx(T_chunk);
        for (int i = 0; i < T_chunk; ++i) {
            positions[i] = n_past + i;
            kv_idx[i]    = n_past + i;
        }
        ggml_backend_tensor_set(pb.positions_in, positions.data(), 0, positions.size() * sizeof(int32_t));
        ggml_backend_tensor_set(pb.kv_idx_in, kv_idx.data(), 0, kv_idx.size() * sizeof(int64_t));
    }
    set_chunk_mask(pb, prefix_len, T_chunk, n_past);
    if (compute_qwen_graph(cc, pb.graph) != GGML_STATUS_SUCCESS) {
        return TRANSCRIBE_ERR_BACKEND;
    }
    cc->kv_cache.n = prefix_len;
    cc->kv_cache.head = prefix_len;
    cc->kv_ids.assign(ids.begin(), ids.begin() + prefix_len);
    cc->kv_tiled.assign(static_cast<size_t>(prefix_len), tiled ? 1 : 0);
    if (khr_gpu_graph_policy(cc) && transcribe::env::flag("TRANSCRIBE_RESOURCE_DIAGNOSTICS")) {
        std::fprintf(stderr, "ASR_PREFIX_COMMIT rows=%d n=%d head=%d capacity=%d\n",
                     prefix_len, cc->kv_cache.n, cc->kv_cache.head, cc->kv_cache.n_ctx);
    }
    const int64_t dt = INT64_C(0) - t0;
    cc->begin_prefilled += T_chunk;
    cc->t_begin_prefill_us += dt;
    if (transcribe::env::flag("TRANSCRIBE_PERF_DEBUG")) {
        log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
                "qwen3_asr stream begin: prompt prefix %d tokens (%s), %d cached, %d prefilled in %.2f ms",
                prefix_len, tiled ? "tiled" : "rows", n_past, T_chunk, dt / 1000.0);
    }
    return TRANSCRIBE_OK;
}

transcribe_status stream_validate(const transcribe_session * session,
                                  const transcribe_run_params * /*run_params*/,
                                  const transcribe_stream_params * stream_params) {
    const auto * cm = static_cast<const QwenAsrModel *>(session->model);
    if (cm == nullptr || !cm->mel_log10.has_value()) {
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }
    if (stream_params != nullptr && stream_params->family != nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;  // no family stream extension
    }
    return TRANSCRIBE_OK;
}

void stream_clear(QwenAsrSession * cc) {
    cc->stream_pcm.clear();
    cc->stream_log_mel.clear();
    cc->stream_n_frames = 0;
    cc->stream_max      = -std::numeric_limits<float>::infinity();
    cc->stream_n_chunks = 0;
    cc->stream_chunk_mel.clear();
    cc->stream_subsample.clear();
    cc->stream_feed_us    = 0;
    cc->stream_conv_us    = 0;
    cc->stream_recomputed = 0;
    cc->stream_rerun_feed = 0;
}

transcribe_status stream_begin(transcribe_session *          session,
                               const transcribe_run_params * run_params,
                               const transcribe_stream_params * /*stream_params*/) {
    auto * cc = static_cast<QwenAsrSession *>(session);
    auto * cm = static_cast<QwenAsrModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty() || !cm->mel_log10.has_value()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    // Reject a bad language hint now rather than at finalize.
    if (run_params->language != nullptr && run_params->language[0] != '\0') {
        std::vector<int32_t> ids;
        if (const transcribe_status st = encode_language_prefix(cm->tok, run_params->language, ids);
            st != TRANSCRIBE_OK) {
            return st;
        }
    }
    // The dispatcher owns the strings behind run_params for the stream's life.
    cc->stream_run_params = *run_params;
    stream_clear(cc);
    transcribe::debug::init();
    // The prompt before the audio is known now: prefill it while the audio
    // arrives. A failure here only costs the cache.
    cc->stream_prefix_tiled = -1;
    cc->begin_prefilled     = 0;
    cc->t_begin_prefill_us  = 0;
    if (prefill_prompt_prefix(cc, cm, run_params, /*T_enc_min=*/0) != TRANSCRIBE_OK) {
        clear_prefix_cache(cc);
        cc->stream_prefix_tiled = -1;
    }
    return TRANSCRIBE_OK;
}

transcribe_status stream_feed(transcribe_session *       session,
                              const float *              pcm,
                              int                        n_samples,
                              transcribe_stream_update * update) {
    auto * cc = static_cast<QwenAsrSession *>(session);
    auto * cm = static_cast<QwenAsrModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty() || !cm->mel_log10.has_value()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (cc->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }
    const int64_t t_start = INT64_C(0);
    cc->stream_pcm.insert(cc->stream_pcm.end(), pcm, pcm + n_samples);

    const transcribe::MelConfig & cfg           = cm->mel_log10->config();
    const int                     hop           = cfg.hop_length;
    const int                     pad           = cfg.n_fft / 2;
    const int                     mel_per_chunk = cm->hparams.enc_n_window * 2;
    const size_t                  n_avail       = cc->stream_pcm.size();

    // Frames t with t*hop + pad < n_avail are final (see above); computing
    // them on every feed keeps the running max current. The conv stem runs
    // when a chunk completes, together with any earlier chunk the max has
    // changed since, so those re-runs also happen while the audio arrives.
    const int n_final = (n_avail > static_cast<size_t>(pad)) ?
                            static_cast<int>((n_avail - static_cast<size_t>(pad) - 1) / hop) + 1 :
                            0;
    if (const transcribe_status st = stream_add_frames(cc, cm, n_avail, n_final); st != TRANSCRIBE_OK) {
        return st;
    }
    const int n_ready = cc->stream_n_frames / mel_per_chunk;
    if (n_ready > cc->stream_n_chunks) {
        const int64_t t_conv0 = INT64_C(0);
        int           n_run = 0, n_rerun = 0;
        if (const transcribe_status st = stream_run_chunks(cc, cm, n_ready, &n_run, &n_rerun); st != TRANSCRIBE_OK) {
            return st;
        }
        cc->stream_rerun_feed += n_rerun;
        cc->stream_conv_us += INT64_C(0) - t_conv0;
    }
    // The audio so far may now be long enough that the final prompt takes the
    // tiled attention path: recompute the cached prefix for it, still while
    // the audio arrives.
    if (cc->stream_prefix_tiled == 0 && cc->stream_n_frames > 0) {
        const int T_enc_min = compute_encoder_timing(cc->stream_n_frames, cm->hparams).T_enc;
        if (cc->stream_prompt_rows + T_enc_min >= k_fa_tile_rows &&
            prefill_prompt_prefix(cc, cm, &cc->stream_run_params, T_enc_min) != TRANSCRIBE_OK) {
            clear_prefix_cache(cc);
            cc->stream_prefix_tiled = -1;
        }
    }
    cc->stream_feed_us += INT64_C(0) - t_start;

    if (update != nullptr) {
        const int     sr      = std::max(1, cm->hparams.fe_sample_rate);
        const int64_t done    = static_cast<int64_t>(cc->stream_n_chunks) * mel_per_chunk * hop;
        update->revision          = cc->stream_revision;
        update->input_received_ms = static_cast<int64_t>(n_avail) * 1000 / sr;
        update->buffered_ms       = (static_cast<int64_t>(n_avail) - done) * 1000 / sr;
    }
    return TRANSCRIBE_OK;
}

transcribe_status stream_finalize(transcribe_session * session, transcribe_stream_update * update) {
    auto * cc = static_cast<QwenAsrSession *>(session);
    auto * cm = static_cast<QwenAsrModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty() || !cm->mel_log10.has_value()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (cc->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }
    const float * pcm       = cc->stream_pcm.data();
    const int     n_samples = static_cast<int>(cc->stream_pcm.size());

    transcribe_status st = TRANSCRIBE_OK;
    if (cc->stream_n_chunks == 0) {
        // Less than one chunk arrived: nothing was precomputed.
        st = run_impl(session, pcm, n_samples, &cc->stream_run_params, /*pre=*/nullptr);
    } else {
        const int    n_mels        = cm->mel_log10->num_mels();
        const int    mel_per_chunk = cm->hparams.enc_n_window * 2;
        const size_t per_chunk     = static_cast<size_t>(mel_per_chunk) * n_mels;

        const int n_feed_chunks = cc->stream_n_chunks;
        // The rest of the mel: per_utterance emits n_samples / hop frames.
        const int64_t t_mel0 = INT64_C(0);
        const int     n_out  = cm->mel->n_frames_for(static_cast<size_t>(n_samples)) - 1;
        if (const transcribe_status mst = stream_add_frames(cc, cm, static_cast<size_t>(n_samples), n_out);
            mst != TRANSCRIBE_OK) {
            return mst;
        }
        cc->t_mel_us = INT64_C(0) - t_mel0;

        // Conv stem for the chunks that were not run yet, and for any early
        // chunk whose normalized mel changed because the max rose after it.
        const int64_t   t_conv0 = INT64_C(0);
        PrecomputedStem pre{};
        pre.timing        = compute_encoder_timing(n_out, cm->hparams);
        const int n_chunk = pre.timing.n_chunks;
        int       n_run = 0, n_rerun = 0;
        if (const transcribe_status sst = stream_run_chunks(cc, cm, n_chunk, &n_run, &n_rerun); sst != TRANSCRIBE_OK) {
            return sst;
        }
        cc->stream_recomputed = n_rerun;
        pre.subsample      = cc->stream_subsample;
        pre.t_subsample_us = INT64_C(0) - t_conv0;

        // The assembled mel, for the debug dump only.
        cc->mel_buf.clear();
        if (transcribe::debug::enabled()) {
            std::vector<float> chunk(per_chunk);
            cc->mel_buf.assign(static_cast<size_t>(n_mels) * n_out, 0.0f);
            for (int c = 0; c < n_chunk; ++c) {
                stream_normalize_chunk(cc, n_mels, mel_per_chunk, c, cc->stream_max, chunk.data());
                const int n_real = std::min(mel_per_chunk, n_out - c * mel_per_chunk);
                for (int m = 0; m < n_mels; ++m) {
                    std::memcpy(cc->mel_buf.data() + static_cast<size_t>(m) * n_out + c * mel_per_chunk,
                                chunk.data() + static_cast<size_t>(m) * mel_per_chunk, n_real * sizeof(float));
                }
            }
        }

        if (transcribe::env::flag("TRANSCRIBE_PERF_DEBUG")) {
            log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
                    "qwen3_asr stream: %d of %d chunks through the conv stem during feed (%d re-runs; "
                    "feed %.2f ms, conv stem %.2f ms); at finalize: %d chunks (%d re-run after the max "
                    "rose), mel %.2f ms, conv stem %.2f ms",
                    n_feed_chunks, n_chunk, cc->stream_rerun_feed, cc->stream_feed_us / 1000.0,
                    cc->stream_conv_us / 1000.0, n_run, n_rerun, cc->t_mel_us / 1000.0,
                    pre.t_subsample_us / 1000.0);
        }
        st = run_impl(session, pcm, n_samples, &cc->stream_run_params, &pre);
    }

    // A stream reports truncation through transcribe_was_truncated() only.
    if (st == TRANSCRIBE_ERR_OUTPUT_TRUNCATED || st == TRANSCRIBE_ERR_OUTPUT_REPETITION) {
        st = TRANSCRIBE_OK;
    }
    if (st != TRANSCRIBE_OK) {
        return st;
    }
    if (update != nullptr) {
        const int     sr       = std::max(1, cm->hparams.fe_sample_rate);
        const int64_t audio_ms = static_cast<int64_t>(n_samples) * 1000 / sr;
        update->result_changed     = true;
        update->revision           = cc->stream_revision;
        update->input_received_ms  = audio_ms;
        update->audio_committed_ms = audio_ms;
        update->buffered_ms        = 0;
    }
    return TRANSCRIBE_OK;
}

void stream_reset(transcribe_session * session) {
    stream_clear(static_cast<QwenAsrSession *>(session));
}

}  // namespace

extern const Arch arch = {
    /* .name             = */ "qwen3_asr",
    /* .load             = */ load,
    /* .init_context     = */ init_context,
    /* .run              = */ run,
    /* .run_batch        = */ run_batch,
    /* .stream_validate  = */ stream_validate,
    /* .stream_begin     = */ stream_begin,
    /* .stream_feed      = */ stream_feed,
    /* .stream_finalize  = */ stream_finalize,
    /* .stream_reset     = */ stream_reset,
    /* .accepts_ext_kind = */ nullptr,
};


extern "C" transcribe_status transcribe_qwen3_asr_warmup(
    transcribe_model * model, const transcribe_session_params * session_params,
    const float * pcm, int n_samples, const transcribe_run_params * params, bool streaming) {
    if (!model || model->arch != &arch || !khr_gpu_model_policy(static_cast<QwenAsrModel *>(model)) ||
        active_warmup || !pcm || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    transcribe_session * temporary = nullptr;
    transcribe_status status = transcribe_session_init(model, session_params, &temporary);
    if (status != TRANSCRIBE_OK) return status;
    WarmupState state{temporary, streaming};
    {
        struct Scope {
            WarmupState * previous;
            transcribe_session * session;
            ~Scope() {
                if (session->sched) ggml_backend_sched_synchronize(session->sched);
                const char *mode = active_warmup->streaming ? "streaming" : "offline";
                asr_resource_snapshot("warm_cleanup_before", mode);
                active_warmup = previous;
                transcribe_session_free(session);
                asr_resource_snapshot("warm_cleanup_after", mode);
                asr_release_unused_heap("warm_cleanup", mode,
                    transcribe::env::flag("TRANSCRIBE_RESOURCE_DIAGNOSTICS"));
            }
        } scope{active_warmup, temporary};
        active_warmup = &state;
        if (streaming) {
            status = transcribe_stream_begin(temporary, params, nullptr);
            if (status == TRANSCRIBE_OK) status = transcribe_stream_feed(temporary, pcm, n_samples, nullptr);
            if (status == TRANSCRIBE_OK) status = transcribe_stream_finalize(temporary, nullptr);
        } else {
            status = transcribe_run(temporary, pcm, n_samples, params);
        }
        if (status == TRANSCRIBE_OK && state.completed_steps != 1) status = TRANSCRIBE_ERR_BACKEND;
    }
    if (transcribe::env::flag("TRANSCRIBE_RESOURCE_DIAGNOSTICS")) {
        std::fprintf(stderr,
                     "ASR_WARMUP_DONE mode=%s steps=%d temporary_session=1 synchronized=1 status=%d\n",
                     streaming ? "streaming" : "offline", state.completed_steps, static_cast<int>(status));
    }
    return status;
}

}  // namespace transcribe::qwen3_asr
