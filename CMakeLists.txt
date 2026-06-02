cmake_minimum_required(VERSION 3.20)
project(pose-matching VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)

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
