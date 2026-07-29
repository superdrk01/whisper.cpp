#include <jni.h>
#include <android/log.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "whisper.h"

#define LOG_TAG "WhisperAndroid"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

struct Engine {
    whisper_context * ctx = nullptr;
    std::mutex mutex;
    std::atomic<bool> abort_requested{false};
    ~Engine() { if (ctx) whisper_free(ctx); }
};

static std::string jstring_to_utf8(JNIEnv * env, jstring value) {
    if (!value) return {};
    const char * chars = env->GetStringUTFChars(value, nullptr);
    if (!chars) return {};
    std::string out(chars);
    env->ReleaseStringUTFChars(value, chars);
    return out;
}

static std::string json_escape(const char * input) {
    if (!input) return "";
    std::ostringstream out;
    for (const unsigned char c : std::string(input)) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"':  out << "\\\""; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out << buf;
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

static bool abort_callback(void * user_data) {
    return static_cast<Engine *>(user_data)->abort_requested.load(std::memory_order_relaxed);
}

static Engine * from_handle(jlong handle) {
    return reinterpret_cast<Engine *>(static_cast<intptr_t>(handle));
}

static jlong native_create(JNIEnv * env, jclass, jstring model_path, jboolean use_gpu,
                           jboolean flash_attn, jint gpu_device, jboolean enable_dtw,
                           jint dtw_preset, jint dtw_n_top, jlong dtw_mem_size) {
    const std::string path = jstring_to_utf8(env, model_path);
    if (path.empty()) {
        return 0;
    }

    whisper_context_params p = whisper_context_default_params();
    p.use_gpu = use_gpu == JNI_TRUE;
    p.flash_attn = flash_attn == JNI_TRUE;
    p.gpu_device = gpu_device;
    p.dtw_token_timestamps = enable_dtw == JNI_TRUE;
    p.dtw_aheads_preset = static_cast<whisper_alignment_heads_preset>(dtw_preset);
    if (dtw_n_top >= 0) p.dtw_n_top = dtw_n_top;
    if (dtw_mem_size > 0) p.dtw_mem_size = static_cast<size_t>(dtw_mem_size);

    std::unique_ptr<Engine> engine(new Engine());
    engine->ctx = whisper_init_from_file_with_params(path.c_str(), p);
    if (!engine->ctx) {
        LOGE("Failed to initialize model: %s", path.c_str());
        return 0;
    }
    return static_cast<jlong>(reinterpret_cast<intptr_t>(engine.release()));
}

static void native_destroy(JNIEnv *, jclass, jlong handle) {
    delete from_handle(handle);
}

static void native_cancel(JNIEnv *, jclass, jlong handle) {
    if (Engine * e = from_handle(handle)) e->abort_requested.store(true);
}

static jstring native_system_info(JNIEnv * env, jclass) {
    return env->NewStringUTF(whisper_print_system_info());
}

static jstring native_transcribe(
        JNIEnv * env, jclass, jlong handle, jfloatArray audio,
        jstring language, jboolean translate, jint strategy, jint threads,
        jint offset_ms, jint duration_ms, jboolean no_context, jboolean single_segment,
        jboolean token_timestamps, jboolean split_on_word, jint max_len, jint max_tokens,
        jboolean suppress_blank, jboolean suppress_nst, jfloat temperature,
        jfloat temperature_inc, jfloat entropy_thold, jfloat logprob_thold,
        jfloat no_speech_thold, jfloat max_initial_ts, jfloat length_penalty,
        jint best_of, jint beam_size, jboolean vad_enabled, jstring vad_model_path,
        jfloat vad_threshold, jint vad_min_speech_ms, jint vad_min_silence_ms,
        jfloat vad_max_speech_s, jint vad_speech_pad_ms, jfloat vad_samples_overlap) {
    Engine * engine = from_handle(handle);
    if (!engine || !engine->ctx || !audio) return nullptr;
    if (vad_enabled == JNI_TRUE && jstring_to_utf8(env, vad_model_path).empty()) {
        return env->NewStringUTF("{\"status\":-3,\"error\":\"VAD model path is required\",\"segments\":[]}");
    }

    const jsize count = env->GetArrayLength(audio);
    if (count <= 0) return env->NewStringUTF("{\"status\":-2,\"error\":\"Audio array is empty\",\"segments\":[]}");
    std::vector<float> samples(static_cast<size_t>(count));
    env->GetFloatArrayRegion(audio, 0, count, samples.data());
    if (env->ExceptionCheck()) return nullptr;

    const std::string lang = jstring_to_utf8(env, language);
    const std::string vad_path = jstring_to_utf8(env, vad_model_path);
    const auto sampling = strategy == 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;
    whisper_full_params p = whisper_full_default_params(sampling);
    p.language = lang.empty() ? nullptr : lang.c_str();
    p.translate = translate == JNI_TRUE;
    p.n_threads = threads;
    p.offset_ms = offset_ms;
    p.duration_ms = duration_ms;
    p.no_context = no_context == JNI_TRUE;
    p.single_segment = single_segment == JNI_TRUE;
    p.print_progress = false;
    p.print_realtime = false;
    p.print_timestamps = false;
    p.token_timestamps = token_timestamps == JNI_TRUE;
    p.split_on_word = split_on_word == JNI_TRUE;
    p.max_len = max_len;
    p.max_tokens = max_tokens;
    p.suppress_blank = suppress_blank == JNI_TRUE;
    p.suppress_nst = suppress_nst == JNI_TRUE;
    p.temperature = temperature;
    p.temperature_inc = temperature_inc;
    p.entropy_thold = entropy_thold;
    p.logprob_thold = logprob_thold;
    p.no_speech_thold = no_speech_thold;
    p.max_initial_ts = max_initial_ts;
    p.length_penalty = length_penalty;
    p.greedy.best_of = best_of;
    p.beam_search.beam_size = beam_size;
    p.vad = vad_enabled == JNI_TRUE;
    p.vad_model_path = p.vad ? vad_path.c_str() : nullptr;
    p.vad_params.threshold = vad_threshold;
    p.vad_params.min_speech_duration_ms = vad_min_speech_ms;
    p.vad_params.min_silence_duration_ms = vad_min_silence_ms;
    p.vad_params.max_speech_duration_s = vad_max_speech_s;
    p.vad_params.speech_pad_ms = vad_speech_pad_ms;
    p.vad_params.samples_overlap = vad_samples_overlap;
    p.abort_callback = abort_callback;
    p.abort_callback_user_data = engine;

    std::lock_guard<std::mutex> lock(engine->mutex);
    engine->abort_requested.store(false);
    const int status = whisper_full(engine->ctx, p, samples.data(), static_cast<int>(samples.size()));

    std::ostringstream out;
    out << "{\"status\":" << status
        << ",\"cancelled\":" << (engine->abort_requested.load() ? "true" : "false")
        << ",\"detectedLanguageId\":" << whisper_full_lang_id(engine->ctx)
        << ",\"vadEnabled\":" << (p.vad ? "true" : "false")
        << ",\"segments\":[";

    if (status == 0) {
        const int n_segments = whisper_full_n_segments(engine->ctx);
        for (int s = 0; s < n_segments; ++s) {
            if (s) out << ',';
            out << "{\"startMs\":" << whisper_full_get_segment_t0(engine->ctx, s) * 10
                << ",\"endMs\":" << whisper_full_get_segment_t1(engine->ctx, s) * 10
                << ",\"text\":\"" << json_escape(whisper_full_get_segment_text(engine->ctx, s)) << "\""
                << ",\"tokens\":[";
            const int n_tokens = whisper_full_n_tokens(engine->ctx, s);
            for (int t = 0; t < n_tokens; ++t) {
                if (t) out << ',';
                const whisper_token_data d = whisper_full_get_token_data(engine->ctx, s, t);
                out << "{\"id\":" << d.id
                    << ",\"text\":\"" << json_escape(whisper_full_get_token_text(engine->ctx, s, t)) << "\""
                    << ",\"startMs\":" << whisper_full_get_token_t0(engine->ctx, s, t) * 10
                    << ",\"endMs\":" << whisper_full_get_token_t1(engine->ctx, s, t) * 10
                    << ",\"dtwMs\":" << d.t_dtw * 10
                    << ",\"probability\":" << d.p << '}';
            }
            out << "]}";
        }
    }
    out << "],\"vadSegments\":[";
    if (status == 0 && p.vad) {
        const int n_vad = whisper_full_n_vad_segments(engine->ctx);
        for (int i = 0; i < n_vad; ++i) {
            if (i) out << ',';
            out << "{\"startMs\":" << whisper_full_get_vad_segment_t0(engine->ctx, i) * 10
                << ",\"endMs\":" << whisper_full_get_vad_segment_t1(engine->ctx, i) * 10 << '}';
        }
    }
    out << "]}";
    return env->NewStringUTF(out.str().c_str());
}

static const JNINativeMethod METHODS[] = {
    {"nativeCreate", "(Ljava/lang/String;ZZIZIIJ)J", reinterpret_cast<void *>(native_create)},
    {"nativeDestroy", "(J)V", reinterpret_cast<void *>(native_destroy)},
    {"nativeCancel", "(J)V", reinterpret_cast<void *>(native_cancel)},
    {"nativeSystemInfo", "()Ljava/lang/String;", reinterpret_cast<void *>(native_system_info)},
    {"nativeTranscribe", "(J[FLjava/lang/String;ZIIIIZZZZIIZZFFFFFFFIIZLjava/lang/String;FIIFIF)Ljava/lang/String;", reinterpret_cast<void *>(native_transcribe)},
};

} // namespace

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM * vm, void *) {
    JNIEnv * env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) return JNI_ERR;
    jclass cls = env->FindClass("com/example/whisper/WhisperEngine");
    if (!cls) return JNI_ERR;
    if (env->RegisterNatives(cls, METHODS, sizeof(METHODS) / sizeof(METHODS[0])) != JNI_OK) return JNI_ERR;
    return JNI_VERSION_1_6;
}
