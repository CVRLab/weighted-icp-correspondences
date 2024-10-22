cmake_minimum_required(VERSION 3.22.1)
project(alignment_experiments)

message(STATUS "BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# Configure CXX standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")

# Find OpenCV libraries
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

# Find MRPT libraries
find_package(MRPT 2.0
  REQUIRED
  COMPONENTS core containers img io math maps poses system
)
message(STATUS "MRPT_VERSION: ${MRPT_VERSION}")
message(STATUS "MRPT_LIBRARIES: ${MRPT_LIBRARIES}")

# Find PCL
find_package(PCL REQUIRED)
message(STATUS "PCL_VERSION: ${PCL_VERSION}")
message(STATUS "PCL_INCLUDE_DIRS: ${PCL_INCLUDE_DIRS}")
message(STATUS "PCL_LIBRARIES: ${PCL_LIBRARIES}")
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Configure include directories
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${MRPT_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME} alignment_experiments.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${MRPT_LIBRARIES} ${PCL_LIBRARIES})
