cmake_minimum_required(VERSION 3.11)

# Exacutable project

project(
  LANE_DETECTION # Name of the project
  LANGUAGES CXX # Language used
  VERSION 0.1.0 # Version of the project
  DESCRIPTION "Lane Detection in Avi project" # Description of the project
)

set(${PROJECT_NAME}_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(${PROJECT_NAME}_SOURCE_PATH ${${PROJECT_NAME}_PATH}/src)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17) # 11, 14, 17, 20, 23
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_VERBOSE_MAKEFILE ON)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(${PROJECT_NAME}
  ${${PROJECT_NAME}_SOURCE_PATH}/main.cpp
)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_compile_options(${PROJECT_NAME} PRIVATE
  -Wall -Wextra -Wpedantic -Werror
)
