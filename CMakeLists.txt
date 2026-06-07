cmake_minimum_required(VERSION 3.20)
project(pose-matching VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_PROFILER "Link gperftools CPU profiler" OFF)

find_package(GTSAM REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

include(FetchContent)
FetchContent_Declare(rerun_sdk URL
    https://github.com/rerun-io/rerun/releases/download/0.33.0/rerun_cpp_sdk.zip
)
FetchContent_MakeAvailable(rerun_sdk)

add_subdirectory(mask-generation)

set(POSE_MATCHING_SOURCES
    src/main.cpp
    src/pose_estimator.cpp
    src/rerun_visualizer.cpp
)

add_executable(pose_matching ${POSE_MATCHING_SOURCES})

target_include_directories(pose_matching PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(pose_matching
    PRIVATE
        maskgen
        ${OpenCV_LIBS}
        nlohmann_json::nlohmann_json
        rerun_sdk
)

add_executable(verify_pose src/verify_pose.cpp)

target_include_directories(verify_pose PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(verify_pose
    PRIVATE
        maskgen
        ${OpenCV_LIBS}
        nlohmann_json::nlohmann_json
        rerun_sdk
)

set(CACHE_LIB_SOURCES
    src/cache_format.cpp
)

add_library(cache_lib STATIC ${CACHE_LIB_SOURCES})

target_include_directories(cache_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(cache_lib
    PUBLIC
        ${OpenCV_LIBS}
        ZLIB::ZLIB
        OpenSSL::Crypto
)

add_executable(pose_cache_builder src/cache_builder_main.cpp)

target_include_directories(pose_cache_builder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(pose_cache_builder
    PRIVATE
        maskgen
        cache_lib
        ${OpenCV_LIBS}
        nlohmann_json::nlohmann_json
)

set(CACHED_SOURCES
    src/cached_main.cpp
    src/cached_pose_estimator.cpp
    src/contour_sampler.cpp
    src/refine_lm.cpp
    src/rerun_visualizer.cpp
)

add_executable(pose_matching_cached ${CACHED_SOURCES})

target_include_directories(pose_matching_cached PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(pose_matching_cached
    PRIVATE
        maskgen
        cache_lib
        ${OpenCV_LIBS}
        nlohmann_json::nlohmann_json
        rerun_sdk
        OpenMP::OpenMP_CXX
)

if(GTSAM_FOUND)
    find_library(GTSAM_LIBRARY gtsam REQUIRED)
    target_link_libraries(pose_matching_cached PRIVATE ${GTSAM_LIBRARY} tbb)
    target_include_directories(pose_matching_cached PRIVATE ${GTSAM_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR})
endif()

if(ENABLE_PROFILER)
    find_library(PROFILER_LIB profiler REQUIRED)
    target_link_libraries(pose_matching_cached PRIVATE ${PROFILER_LIB})
    target_compile_definitions(pose_matching_cached PRIVATE ENABLE_PROFILER)
endif()

add_executable(visualize_poses src/visualize_poses_main.cpp)

target_include_directories(visualize_poses PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(visualize_poses
    PRIVATE
        maskgen
        ${OpenCV_LIBS}
        nlohmann_json::nlohmann_json
        rerun_sdk
)
