#include "whisper.h"
#include "ggml-backend.h"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

#if defined(_WIN32)
#define VOICE_API extern "C" __declspec(dllexport)
#else
#define VOICE_API extern "C" __attribute__((visibility("default")))
#endif

struct VoiceContext {
    whisper_context *model;
    std::atomic<bool> cancelled{false};
    std::string text;
    int threads;
};

static std::once_flag backends;

static bool cancelled(void *data) {
    return static_cast<VoiceContext *>(data)->cancelled.load();
}

VOICE_API int voice_abi_version() { return 1; }
VOICE_API const char *voice_engine_version() { return whisper_version(); }

VOICE_API VoiceContext *voice_open(const char *model, const char *directory,
                                  int threads, char *error, size_t capacity) {
    if (!model || !directory || threads < 1 || threads > 32) {
        std::snprintf(error, capacity, "Invalid model path, backend directory or thread count");
        return nullptr;
    }
    if (std::strcmp(whisper_version(), "1.9.2") != 0) {
        std::snprintf(error, capacity, "Expected whisper.cpp 1.9.2");
        return nullptr;
    }
    std::call_once(backends, [directory]() { ggml_backend_load_all_from_path(directory); });
    if (!ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
        std::snprintf(error, capacity, "CPU backend was not found in the engine directory");
        return nullptr;
    }
    auto params = whisper_context_default_params();
    params.use_gpu = false;
    auto *context = whisper_init_from_file_with_params(model, params);
    if (!context) {
        std::snprintf(error, capacity, "Whisper model initialization failed");
        return nullptr;
    }
    auto *voice = new VoiceContext;
    voice->model = context;
    voice->threads = threads;
    return voice;
}

VOICE_API int voice_transcribe(VoiceContext *voice, const float *samples, int count,
                               char *error, size_t capacity) {
    if (!voice || !samples || count < 1 || count > 320000) {
        std::snprintf(error, capacity, "Expected one nonempty audio segment of at most 20 seconds");
        return 1;
    }
    if (voice->cancelled.load()) {
        std::snprintf(error, capacity, "Transcription cancelled");
        return 2;
    }
    voice->text.clear();
    auto params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads = voice->threads;
    params.language = "auto";
    params.no_context = true;
    params.translate = false;
    params.temperature = 0;
    params.temperature_inc = 0;
    params.greedy.best_of = 1;
    params.suppress_nst = true;
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.abort_callback = cancelled;
    params.abort_callback_user_data = voice;
    int status = whisper_full(voice->model, params, samples, count);
    if (voice->cancelled.load()) {
        std::snprintf(error, capacity, "Transcription cancelled");
        return 2;
    }
    if (status != 0) {
        std::snprintf(error, capacity, "Whisper inference failed with status %d", status);
        return 3;
    }
    for (int i = 0; i < whisper_full_n_segments(voice->model); ++i) {
        voice->text += whisper_full_get_segment_text(voice->model, i);
    }
    return 0;
}

VOICE_API const char *voice_text(VoiceContext *voice) { return voice->text.c_str(); }
VOICE_API void voice_cancel(VoiceContext *voice) { voice->cancelled.store(true); }
VOICE_API void voice_close(VoiceContext *voice) {
    whisper_free(voice->model);
    delete voice;
}
