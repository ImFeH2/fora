cmake_minimum_required(VERSION 3.25)
project(huddol_voice LANGUAGES C CXX)
include(FetchContent)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
if(APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path")
else()
  set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

FetchContent_Declare(whisper_source
  URL https://codeload.github.com/ggml-org/whisper.cpp/tar.gz/306c88f4d1286aec1bf96e544632897886af5501
  URL_HASH SHA256=f3585ebf64df3e41b26c45d93bdb38b423ca1de0bf40cc4b6d320254867a75df
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

set(voice_destination huddol/adapters/voice/lib)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  FetchContent_Declare(whisper_runtime
    URL https://github.com/ggml-org/whisper.cpp/releases/download/v1.9.2/whisper-bin-ubuntu-x64.tar.gz
    URL_HASH SHA256=46811a3ecf584307480a220b9ef5ff81b7b22dc41577cbc274ce3afc61f753b1
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_GetProperties(whisper_source)
  if(NOT whisper_source_POPULATED)
    FetchContent_Populate(whisper_source)
  endif()
  FetchContent_GetProperties(whisper_runtime)
  if(NOT whisper_runtime_POPULATED)
    FetchContent_Populate(whisper_runtime)
  endif()
  find_library(WHISPER_LIBRARY NAMES whisper PATHS "${whisper_runtime_SOURCE_DIR}" "${whisper_runtime_SOURCE_DIR}/whisper-bin-ubuntu-x64" NO_DEFAULT_PATH REQUIRED)
  get_filename_component(runtime_directory "${WHISPER_LIBRARY}" DIRECTORY)
  find_library(GGML_LIBRARY NAMES ggml PATHS "${runtime_directory}" NO_DEFAULT_PATH REQUIRED)
  find_library(GGML_BASE_LIBRARY NAMES ggml-base PATHS "${runtime_directory}" NO_DEFAULT_PATH REQUIRED)
  add_library(voice_shim SHARED native/voice_shim.cpp)
  target_include_directories(voice_shim PRIVATE "${whisper_source_SOURCE_DIR}/include" "${whisper_source_SOURCE_DIR}/ggml/include")
  target_link_libraries(voice_shim PRIVATE "${WHISPER_LIBRARY}" "${GGML_LIBRARY}" "${GGML_BASE_LIBRARY}")
  install(DIRECTORY "${runtime_directory}/" DESTINATION "${voice_destination}" COMPONENT voice
    FILES_MATCHING PATTERN "libwhisper.so*" PATTERN "libggml*.so*")
else()
  FetchContent_MakeAvailable(whisper_source)
  add_library(voice_shim SHARED native/voice_shim.cpp)
  target_link_libraries(voice_shim PRIVATE whisper ggml ggml-base)
  install(TARGETS whisper ggml ggml-base ggml-cpu
    RUNTIME DESTINATION "${voice_destination}" COMPONENT voice
    LIBRARY DESTINATION "${voice_destination}" COMPONENT voice)
endif()

set_target_properties(voice_shim PROPERTIES CXX_VISIBILITY_PRESET hidden)
install(TARGETS voice_shim
  RUNTIME DESTINATION "${voice_destination}" COMPONENT voice
  LIBRARY DESTINATION "${voice_destination}" COMPONENT voice)
install(FILES "${whisper_source_SOURCE_DIR}/LICENSE" DESTINATION "${voice_destination}" RENAME LICENSE-whisper COMPONENT voice)
