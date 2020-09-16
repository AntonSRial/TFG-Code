cmake_minimum_required(VERSION 3.8)
project(cubemos-samples)

# Find cubemos Skeleton Tracking package
find_package(CUBEMOS_SKELETON_TRACKING
			REQUIRED
			CONFIG
			PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../lib/cmake/CUBEMOS_SKELETON_TRACKING"
			)

# Find OpenCV package
if(WIN32)
    set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dep/opencv")
else()
    set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dep/opencv/lib/cmake/opencv4")
endif()

find_package(OpenCV REQUIRED CONFIG)


# Find realsense package
find_package(realsense2 
			REQUIRED 
			CONFIG 
			PATHS "${CMAKE_CURRENT_SOURCE_DIR}/dep/realsense/lib/cmake/realsense2"
			)


# set compiler flags
if (WIN32)
	include (cmake/windows_compiler_flags.cmake)
elseif(UNIX)
	include (cmake/linux_compiler_flags.cmake)
elseif(ANDROID)
	message(FATAL_ERROR "Android is not yet supported.")
else()
	message(FATAL_ERROR "Unsupported operating system")
endif()

# set C++ standard
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

include(GNUInstallDirs)

add_subdirectory(cpp/webcam)
add_subdirectory(cpp/realsense)
add_subdirectory(cpp/image)

if (WIN32)
    set(DOTNET_VERSION_LIBRARY "4.0" CACHE STRING ".Net Version, defaulting to '4.0'")
    set(DOTNET_VERSION_EXAMPLES "4.5" CACHE STRING ".Net Version, defaulting to '4.5'")
    include(CSharpUtilities)
    add_subdirectory(csharp/realsense)
    add_subdirectory(csharp/interactive)
    add_subdirectory(csharp/image)
    add_subdirectory(csharp/videoseries)
endif()



