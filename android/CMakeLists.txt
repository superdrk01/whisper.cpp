cmake_minimum_required(VERSION 3.22)
project(whisper_android LANGUAGES C CXX)

if(NOT ANDROID)
    message(FATAL_ERROR "This project must be configured with the Android NDK toolchain")
endif()
if(NOT DEFINED WHISPER_CPP_DIR)
    message(FATAL_ERROR "Pass -DWHISPER_CPP_DIR=/absolute/path/to/whisper.cpp")
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Produce Android 15/16 KB-page compatible DSOs even with older NDKs. NDK r28+
# already does this by default, but keeping the flags explicit makes the intent
# and CI verification unambiguous.
add_link_options("LINKER:-z,max-page-size=16384" "LINKER:-z,common-page-size=16384")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(GGML_VULKAN ON CACHE BOOL "" FORCE)
set(GGML_CPU ON CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory("${WHISPER_CPP_DIR}" whisper.cpp-build EXCLUDE_FROM_ALL)

add_library(whisper_android SHARED
    src/whisper_android.cpp
    src/whisper_jni.cpp
)

target_include_directories(whisper_android PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${WHISPER_CPP_DIR}/include"
    "${WHISPER_CPP_DIR}/ggml/include"
)

target_compile_definitions(whisper_android PRIVATE WHISPER_ANDROID_BUILD=1)
target_compile_options(whisper_android PRIVATE -O3 -ffunction-sections -fdata-sections -fvisibility=hidden)
target_link_options(whisper_android PRIVATE -Wl,--gc-sections -Wl,--exclude-libs,ALL)

target_link_libraries(whisper_android PRIVATE whisper android log vulkan)
