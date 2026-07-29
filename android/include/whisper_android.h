#pragma once

#include <stdbool.h>
#include "whisper.h"

#if defined(__GNUC__)
#define WA_EXPORT __attribute__((visibility("default")))
#else
#define WA_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Compatibility initializer: GPU can be requested, DTW remains disabled.
WA_EXPORT struct whisper_context * wa_init_from_file(
        const char * model_path,
        bool use_gpu);

// Extended initializer for subtitle/alignment applications.
// When enable_dtw is true, dtw_preset must match the loaded Whisper model.
WA_EXPORT struct whisper_context * wa_init_from_file_ex(
        const char * model_path,
        bool use_gpu,
        bool enable_dtw,
        enum whisper_alignment_heads_preset dtw_preset);

WA_EXPORT void wa_free(struct whisper_context * context);

// The returned pointer is owned by whisper.cpp and must not be freed.
WA_EXPORT const char * wa_system_info(void);

// Number of registered ggml backends after backend initialization.
WA_EXPORT int wa_backend_count(void);

#ifdef __cplusplus
}
#endif
