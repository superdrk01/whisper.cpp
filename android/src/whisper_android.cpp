#include "whisper_android.h"

#include "ggml-backend.h"

namespace {

bool valid_dtw_preset(enum whisper_alignment_heads_preset preset) {
    switch (preset) {
        case WHISPER_AHEADS_NONE:
        case WHISPER_AHEADS_N_TOP_MOST:
        case WHISPER_AHEADS_CUSTOM:
        case WHISPER_AHEADS_TINY_EN:
        case WHISPER_AHEADS_TINY:
        case WHISPER_AHEADS_BASE_EN:
        case WHISPER_AHEADS_BASE:
        case WHISPER_AHEADS_SMALL_EN:
        case WHISPER_AHEADS_SMALL:
        case WHISPER_AHEADS_MEDIUM_EN:
        case WHISPER_AHEADS_MEDIUM:
        case WHISPER_AHEADS_LARGE_V1:
        case WHISPER_AHEADS_LARGE_V2:
        case WHISPER_AHEADS_LARGE_V3:
        case WHISPER_AHEADS_LARGE_V3_TURBO:
            return true;
    }
    return false;
}

} // namespace

extern "C" WA_EXPORT struct whisper_context * wa_init_from_file(
        const char * model_path,
        bool use_gpu) {
    return wa_init_from_file_ex(
            model_path,
            use_gpu,
            false,
            WHISPER_AHEADS_NONE);
}

extern "C" WA_EXPORT struct whisper_context * wa_init_from_file_ex(
        const char * model_path,
        bool use_gpu,
        bool enable_dtw,
        enum whisper_alignment_heads_preset dtw_preset) {
    if (model_path == nullptr || model_path[0] == '\0') {
        return nullptr;
    }
    if (enable_dtw && !valid_dtw_preset(dtw_preset)) {
        return nullptr;
    }

    whisper_context_params params = whisper_context_default_params();
    params.use_gpu = use_gpu;
    params.dtw_token_timestamps = enable_dtw;
    params.dtw_aheads_preset = enable_dtw ? dtw_preset : WHISPER_AHEADS_NONE;

    return whisper_init_from_file_with_params(model_path, params);
}

extern "C" WA_EXPORT void wa_free(struct whisper_context * context) {
    if (context != nullptr) {
        whisper_free(context);
    }
}

extern "C" WA_EXPORT const char * wa_system_info(void) {
    return whisper_print_system_info();
}

extern "C" WA_EXPORT int wa_backend_count(void) {
    return static_cast<int>(ggml_backend_reg_count());
}
