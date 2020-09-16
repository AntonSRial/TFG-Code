cmake_minimum_required( VERSION 3.12 )

project(image LANGUAGES CSharp)

add_executable(${PROJECT_NAME} 
	Program.cs	
	Properties/AssemblyInfo.cs
	../Common.cs
)
target_link_libraries(${PROJECT_NAME} Cubemos.Core)
target_link_libraries(${PROJECT_NAME} Cubemos.SkeletonTracking)

set_property(TARGET ${PROJECT_NAME} PROPERTY VS_GLOBAL_ApplicationIcon "${CMAKE_SOURCE_DIR}/res/icon/logo.ico")
set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DOTNET_TARGET_FRAMEWORK_VERSION "v${DOTNET_VERSION_EXAMPLES}")

set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DOTNET_REFERENCES	
	"System"
	"System.Data"
	"System.Drawing"
	"System.Xaml"	
)


set_target_properties (${PROJECT_NAME} PROPERTIES
	FOLDER wrappers/csharp/samples
	RUNTIME_OUTPUT_NAME csharp-${PROJECT_NAME})


install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})


