#ifndef ASR_RESOURCE_SNAPSHOT_H
#define ASR_RESOURCE_SNAPSHOT_H
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(__linux__) && defined(__GLIBC__)
#include <malloc.h>
#include <fcntl.h>
#include <unistd.h>
static bool asr_resource_read(const char *path, char *data, size_t size) {
    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) { data[0] = 0; return false; }
    size_t used = 0;
    while (used + 1 < size) {
        const ssize_t n = read(fd, data + used, size - used - 1);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        used += static_cast<size_t>(n);
    }
    close(fd); data[used] = 0; return used != 0;
}
static long long asr_resource_value(const char *data, const char *key) {
    const size_t n = std::strlen(key);
    for (const char *line = data; line && *line; ) {
        if (!std::strncmp(line, key, n) && (line[n] == ':' || line[n] == ' ' || line[n] == '\t')) {
            const char *value = line + n + 1;
            while (*value == ' ' || *value == '\t') ++value;
            char *end = nullptr; const long long result = std::strtoll(value, &end, 10);
            return end != value ? result : -1;
        }
        line = std::strchr(line, '\n'); if (line) ++line;
    }
    return -1;
}
static void asr_resource_snapshot(const char *phase, const char *label) {
    const int saved_errno = errno;
    struct RestoreErrno { int value; ~RestoreErrno() { errno = value; } } restore{saved_errno};
    const char *enabled = std::getenv("TRANSCRIBE_RESOURCE_DIAGNOSTICS");
    static std::atomic<unsigned> samples{0};
    if (!enabled || std::strcmp(enabled, "1") || samples.fetch_add(1, std::memory_order_relaxed) >= 256) return;
    const struct mallinfo2 heap = mallinfo2();
    char status[4096], rollup[4096], membership[2048], stat[4096] = {}, current[64] = {}, swap[64] = {}, path[2304];
    asr_resource_read("/proc/self/status", status, sizeof(status));
    asr_resource_read("/proc/self/smaps_rollup", rollup, sizeof(rollup));
    bool cgroup = false;
    if (asr_resource_read("/proc/self/cgroup", membership, sizeof(membership))) {
        for (char *line = membership; line && *line; ) {
            char *end = std::strchr(line, '\n'); if (end) *end = 0;
            if (!std::strncmp(line, "0::/", 4)) {
                const char *base = line + 3;
                const int n = std::snprintf(path, sizeof(path), "/sys/fs/cgroup%s/memory.stat", base);
                if (n > 0 && static_cast<size_t>(n) < sizeof(path)) cgroup = asr_resource_read(path, stat, sizeof(stat));
                std::snprintf(path, sizeof(path), "/sys/fs/cgroup%s/memory.current", base);
                asr_resource_read(path, current, sizeof(current));
                std::snprintf(path, sizeof(path), "/sys/fs/cgroup%s/memory.swap.current", base);
                asr_resource_read(path, swap, sizeof(swap));
                break;
            }
            line = end ? end + 1 : nullptr;
        }
    }
    flockfile(stderr);
    std::fprintf(stderr, "ASR_OWNERSHIP phase=%.64s label=%.160s heap_arena_bytes=%zu heap_used_bytes=%zu heap_free_bytes=%zu heap_mmap_bytes=%zu heap_mmap_regions=%zu heap_top_free_bytes=%zu\n", phase, label, heap.arena, heap.uordblks, heap.fordblks, heap.hblkhd, heap.hblks, heap.keepcost);
    std::fprintf(stderr, "ASR_OWNERSHIP_PROC phase=%.64s rss_kib=%lld anon_kib=%lld file_kib=%lld shmem_kib=%lld swap_kib=%lld pss_kib=%lld private_dirty_kib=%lld anonymous_kib=%lld\n", phase, asr_resource_value(status,"VmRSS"),asr_resource_value(status,"RssAnon"),asr_resource_value(status,"RssFile"),asr_resource_value(status,"RssShmem"),asr_resource_value(status,"VmSwap"),asr_resource_value(rollup,"Pss"),asr_resource_value(rollup,"Private_Dirty"),asr_resource_value(rollup,"Anonymous"));
    std::fprintf(stderr, "ASR_OWNERSHIP_CGROUP phase=%.64s available=%u current_bytes=%lld swap_bytes=%lld anon_bytes=%lld file_bytes=%lld shmem_bytes=%lld kernel_bytes=%lld pagetables_bytes=%lld unevictable_bytes=%lld\n", phase,cgroup?1u:0u,current[0]?std::strtoll(current,nullptr,10):-1,swap[0]?std::strtoll(swap,nullptr,10):-1,asr_resource_value(stat,"anon"),asr_resource_value(stat,"file"),asr_resource_value(stat,"shmem"),asr_resource_value(stat,"kernel"),asr_resource_value(stat,"pagetables"),asr_resource_value(stat,"unevictable"));
    std::fflush(stderr); funlockfile(stderr);
}
static void asr_release_unused_heap(const char *phase, const char *label, bool diagnostics) {
    const int saved_errno = errno;
    struct RestoreErrno { int value; ~RestoreErrno() { errno = value; } } restore{saved_errno};
    const char *policy = std::getenv("TRANSCRIBE_VULKAN_KHR_PROFILE");
    if (!policy || std::strcmp(policy, "1")) return;
    const int result = malloc_trim(0);
    if (diagnostics) {
        std::fprintf(stderr, "ASR_HEAP_RELEASE phase=%.64s label=%.160s result=%d policy=khr_linux_glibc\n", phase, label, result);
        std::fflush(stderr);
        asr_resource_snapshot("heap_release_after", label);
    }
}
#else
static void asr_resource_snapshot(const char *, const char *) {}
static void asr_release_unused_heap(const char *, const char *, bool) {}
#endif
#endif
