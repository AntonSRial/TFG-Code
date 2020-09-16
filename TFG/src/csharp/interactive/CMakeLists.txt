cmake_minimum_required( VERSION 3.8 )
project(interactive LANGUAGES CSharp)

add_executable(${PROJECT_NAME} 
	Program.cs
	LogOutput.cs
	Window.xaml
	Window.xaml.cs
	WindowCameraControlOptions.cs
	WindowRenderingControlOptions.cs
    PoseEstimationRendering.cs
	DepthMapHelpers.cs
	Properties/AssemblyInfo.cs
	InfoDialog.xaml
	InfoDialog.xaml.cs
	../Common.cs
	dep/res/cubemoslogo.png
	dep/res/usageGuide.png
)
target_link_libraries(${PROJECT_NAME} Cubemos.Core)
target_link_libraries(${PROJECT_NAME} Cubemos.SkeletonTracking)

set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DOTNET_TARGET_FRAMEWORK_VERSION "v${DOTNET_VERSION_EXAMPLES}")
set_property(TARGET ${PROJECT_NAME} PROPERTY VS_GLOBAL_ApplicationIcon "${CMAKE_SOURCE_DIR}/res/icon/logo.ico")

set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DOTNET_REFERENCES
	"PresentationCore"
	"PresentationFramework"
	"System"
	"System.Data"
	"System.Drawing"
	"System.Windows.Forms"
	"System.Xaml"
	"WindowsBase" )

set(resources "dep/res/cubemoslogo.png" "dep/res/usageGuide.png")

set_property(SOURCE ${resources} PROPERTY VS_TOOL_OVERRIDE "Resource")

set_target_properties(${PROJECT_NAME} PROPERTIES
	VS_DOTNET_REFERENCE_Intel.Realsense "${CMAKE_CURRENT_SOURCE_DIR}/../realsense/dep/librealsense/Intel.Realsense.dll")


set_target_properties (${PROJECT_NAME} PROPERTIES
	FOLDER wrappers/csharp/samples
	RUNTIME_OUTPUT_NAME csharp-${PROJECT_NAME})

target_compile_options(${PROJECT_NAME} PUBLIC "/unsafe")

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY
	"${CMAKE_CURRENT_SOURCE_DIR}/../realsense/dep/librealsense/"
    DESTINATION ${CMAKE_INSTALL_BINDIR}
)
