// arch/qwen3_asr/qwen3_asr.h - Qwen3-ASR model and context types.
//
// INTERNAL to src/arch/qwen3_asr/. Concrete transcribe_model /
// transcribe_session subclasses for the Qwen3-ASR family (audio-LLM: audio
// encoder + Qwen3 causal LM with audio-token injection).

#pragma once

#include "causal_lm/causal_lm.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "transcribe-backend.h"
#include "transcribe-load-common.h"
#include "transcribe-mel.h"
#include "transcribe-model.h"
#include "transcribe-session.h"
#include "transcribe-tokenizer.h"
#include "weights.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_backend_sched;
typedef struct ggml_backend *        ggml_backend_t;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct ggml_backend_sched *  ggml_backend_sched_t;

namespace transcribe::qwen3_asr {

void apply_family_invariants(transcribe_model & model);

// Encode "language {Name}<asr_text>" for the given BCP-47 code: the token-id
// sequence the chat template seeds the assistant turn with on a forced hint.
// Declared here (not in model.cpp's anon namespace) so the BPE parity test can
// verify the full prefix against the HF reference.
//
// Returns:
//   TRANSCRIBE_OK                     on success, out_ids populated.
//   TRANSCRIBE_ERR_INVALID_ARG        bcp47 null or empty.
//   TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE  bcp47 not in the publisher's
//                                     support list (the static map
//                                     in model.cpp is the source of
//                                     truth; drift vs. caps.languages
//                                     surfaces here).
//   TRANSCRIBE_ERR_GGUF               tokenizer has no encoder
//                                     (missing merges) or vocab
//                                     missing <asr_text> special.
transcribe_status encode_language_prefix(const transcribe::Tokenizer & tok,
                                         const char *                  bcp47,
                                         std::vector<int32_t> &        out_ids);

// ---------------------------------------------------------------------------
// Model / Context
// ---------------------------------------------------------------------------

// Qwen3 chat-template special-token ids, resolved through the loaded tokenizer
// at load time so a future vocab reorder fails loudly. Reference ids on the
// 0.6B/1.7B vocab are stable but never hardcoded at runtime.
struct ChatTokens {
    int32_t im_start       = -1;
    int32_t im_end         = -1;
    int32_t newline        = -1;
    int32_t role_system    = -1;
    int32_t role_user      = -1;
    int32_t role_assistant = -1;
};

struct QwenAsrModel final : public transcribe_model {
    Tokenizer      tok;
    QwenAsrHParams hparams;
    QwenAsrWeights weights;
    ggml_context * ctx_meta = nullptr;

    transcribe::BackendPlan                    plan;
    transcribe::load_common::WeightStorage     weight_storage;
    transcribe::causal_lm::PackedGateUpHandles packed_gate_up;

    std::optional<transcribe::MelFrontend> mel;

    // Same frontend with normalize = "log10": the raw per-frame log-mel the
    // streaming path normalizes itself (see stream_feed in model.cpp). Set
    // only when the offline frontend is the Whisper per_utterance one with
    // centered reflect padding, the case the streaming path reproduces.
    std::optional<transcribe::MelFrontend> mel_log10;

    // Jinja chat template string from the GGUF KV (empty == none).
    std::string chat_template;

    // Resolved chat-template token ids (see resolve_chat_tokens in model.cpp).
    ChatTokens chat_tokens;

    QwenAsrModel() = default;
    ~QwenAsrModel() override;

    const transcribe::Tokenizer * tokenizer() const override { return &tok; }
};

struct QwenAsrSession final : public transcribe_session {
    transcribe::causal_lm::KvCache kv_cache;

    // Batched KV cache for offline transcribe_run_batch (n_batch slabs).
    // Allocated/resized lazily by run_batch; freed in the destructor.
    transcribe::causal_lm::KvCache kv_cache_batch;
    int                            kv_batch_cap   = 0;  // allocated n_batch
    int                            kv_batch_n_ctx = 0;  // allocated n_ctx

    std::vector<float> mel_buf;
    std::vector<float> enc_host;  // audio encoder output, pre-injection

    bool encoder_use_flash = true;
    bool decoder_use_flash = true;

    // Streaming (transcribe_stream_*): the conv stem runs on each 1 s mel
    // chunk as soon as its audio has arrived; finalize runs the last chunk,
    // the encoder transformer and the decoder. See stream_feed in model.cpp.
    transcribe_run_params stream_run_params{};    // strings owned by the session
    std::vector<float>    stream_pcm;             // all audio received
    std::vector<float>    stream_log_mel;         // raw log10 mel, frame-major [t][n_mels]
    int                   stream_n_frames = 0;    // frames in stream_log_mel (all final)
    float                 stream_max      = 0.0f; // max over stream_log_mel
    int                   stream_n_chunks = 0;    // chunks through the conv stem
    std::vector<float>    stream_chunk_mel;       // normalized mel each chunk was run with
    std::vector<float>    stream_subsample;       // conv stem output per chunk
    int64_t               stream_feed_us   = 0;   // time spent inside stream_feed
    int64_t               stream_conv_us   = 0;   // of which conv stem
    int                   stream_rerun_feed = 0;  // chunks re-run during feed (max rose)
    int                   stream_recomputed = 0;  // chunks re-run at finalize

    // Prompt-prefix cache (see run_impl). kv_ids are the prompt tokens whose
    // K/V sit in kv_cache rows [0, kv_ids.size()): the chat frame and the
    // system context up to and including <|audio_start|> of the last prompt.
    // A run whose prompt starts with the same tokens prefills only the rest.
    std::vector<int32_t> kv_ids;
    // Per cached row: computed by ggml's tiled flash-attention path (a graph
    // of 64+ query rows) or row by row; a row is reused only by a prompt
    // whose single-graph prefill would take the same path (decoder.h).
    std::vector<uint8_t> kv_tiled;
    // The "language X<asr_text>" ids the last auto-detect run generated. The
    // next auto-detect run appends them to its prompt as a draft and keeps
    // the ones the model itself predicts (greedy verification).
    std::vector<int32_t> header_draft;
    // Last run / stream_begin, for the perf log.
    int     kv_reused        = 0;  // prompt rows taken from the cache
    int     kv_prefilled     = 0;  // prompt rows computed
    int     draft_n          = 0;  // header tokens drafted
    int     draft_ok         = 0;  // of which accepted
    int64_t t_prefill_us     = 0;  // prefill graph build + compute
    int     begin_prefilled  = 0;  // rows prefilled by stream_begin / stream_feed
    int64_t t_begin_prefill_us = 0;
    int     stream_prefix_tiled = -1;  // path of the stream's cached prefix (-1: none)
    int     stream_prompt_rows  = 0;   // the stream's prompt rows besides the audio

    QwenAsrSession() = default;
    ~QwenAsrSession() override;
};

}  // namespace transcribe::qwen3_asr
