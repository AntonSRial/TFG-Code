# Copyright cubemos gmbh
# Since the realsense does not get deployed with cmake config in windows
# we use a small workaround to find the package in windows 
cmake_minimum_required (VERSION 3.8)

# Save the command line compile commands in the build output
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Simple non robust way to find the librealsense library
project(cpp-realsense LANGUAGES CXX)
add_executable(${PROJECT_NAME} cpp_realsense.cpp ../samples.h)
target_link_libraries(${PROJECT_NAME} PRIVATE cubemos_skeleton_tracking)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${realsense2_LIBRARY})

set_target_properties (${PROJECT_NAME} PROPERTIES
	FOLDER samples
)

install(TARGETS ${PROJECT_NAME}
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)