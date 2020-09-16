#include <chrono>
#include <iostream>
#include <string>

#include <windows.h>
#include <WinUser.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>

#include <cubemos/engine.h>
#include <cubemos/skeleton_tracking.h>

#include "../samples.h"
#include <fstream>

using CUBEMOS_SKEL_Buffer_Ptr = std::unique_ptr<CM_SKEL_Buffer, void (*)(CM_SKEL_Buffer*)>;

static const cv::Scalar skeletonColor = cv::Scalar(100, 254, 213);
static const cv::Scalar jointColor = cv::Scalar(0, 255, 0);
static const cv::Scalar screenColor = cv::Scalar(0, 0, 255);
int frameCounter = 0;
bool isClicked = false;
std::ofstream fileAxisX;
std::ofstream fileAxisY;


struct cmPoint {
	float color_pixel[2];
	float point3d[3];
	std::string to_string() const
	{
		char buffer[100];
		int cx = snprintf(buffer, 100, "(%.2f, %.2f, %.2f)", point3d[0], point3d[1], point3d[2]);
		return std::string(buffer);
	}
};

cmPoint
get_skeleton_point_3d(rs2::depth_frame const& depthFrame, int x, int y)
{
	// Get the distance at the given pixel
	auto distance = depthFrame.get_distance(x, y);

	cmPoint point;
	point.color_pixel[0] = static_cast<float>(x);
	point.color_pixel[1] = static_cast<float>(y);

	// Deproject from pixel to point in 3D
	rs2_intrinsics intr = depthFrame.get_profile().as<rs2::video_stream_profile>().get_intrinsics(); // Calibration data
	rs2_deproject_pixel_to_point(point.point3d, &intr, point.color_pixel, distance);

	return point;
}

CUBEMOS_SKEL_Buffer_Ptr
create_skel_buffer()
{
	return CUBEMOS_SKEL_Buffer_Ptr(new CM_SKEL_Buffer(), [](CM_SKEL_Buffer* pb) {
		cm_skel_release_buffer(pb);
		delete pb;
		});
}


float
calibrateScreen(rs2::depth_frame const& depth_frame, cv::Mat& image) {
	//Si queremos poner todas las coordenadas:	
	//cmPoint point3d = get_skeleton_point_3d(depth_frame, depth_frame.get_width()/10, depth_frame.get_height()/7);
	float top_left = depth_frame.get_distance(depth_frame.get_width() / 3.5, depth_frame.get_height() / 3.5);
	float top_right = depth_frame.get_distance(1280 - (depth_frame.get_width() / 3.5), depth_frame.get_height() / 3.5);
	float bot_left = depth_frame.get_distance(depth_frame.get_width() / 3.5, 720 - depth_frame.get_height() / 3.5);
	float bot_right = depth_frame.get_distance(1280 - (depth_frame.get_width() / 3.5), 720 - depth_frame.get_height() / 3.5);
	//cv::putText(image, "O "+ std::to_string(top_left), cv::Point( depth_frame.get_width() / 3.5, depth_frame.get_height() / 3.5), cv::FONT_HERSHEY_COMPLEX, 1, screenColor);
	cv::putText(image, "O ", cv::Point( depth_frame.get_width() / 3.5, depth_frame.get_height() / 3.5), cv::FONT_HERSHEY_COMPLEX, 1, screenColor);
	cv::putText(image, "O " , cv::Point(1280-(depth_frame.get_width() / 3.5), depth_frame.get_height() / 3.5), cv::FONT_HERSHEY_COMPLEX, 1, screenColor);
	cv::putText(image, "O " , cv::Point(depth_frame.get_width() / 3.5, 720 - depth_frame.get_height() / 3.5), cv::FONT_HERSHEY_COMPLEX, 1, screenColor);
	cv::putText(image, "O " , cv::Point(1280 - (depth_frame.get_width() / 3.5),720 - depth_frame.get_height() / 3.5), cv::FONT_HERSHEY_COMPLEX, 1, screenColor);
	//cv::putText(image, "X", cv::Point(1280 - (depth_frame.get_width() /2), 720 - depth_frame.get_height() / 2), cv::FONT_HERSHEY_COMPLEX, 1, screenColor);
	
	//Check mouse position
	/*POINT point;
	if (GetCursorPos(&point)) {
		cv::putText(image, std::to_string(point.x)+", "+ std::to_string(point.y), cv::Point(point.x, point.y), cv::FONT_HERSHEY_COMPLEX, 1, screenColor);
		std::cout << "Las coordenadas son " << point.x <<","<<point.y << " meters away \r";
	}
	cv::putText(image, "+", cv::Point(1280, 720), cv::FONT_HERSHEY_COMPLEX, 1, screenColor); */
	return std::max(top_left, std::max(bot_left, std::max(top_right, bot_right)));
}

/*
Render skeletons and tracking ids on top of the color image
*/
float
renderSkeletons(const CM_SKEL_Buffer* skeletons_buffer, rs2::depth_frame const& depth_frame, cv::Mat& image, float distance)
{
	CV_Assert(image.type() == CV_8UC3);
	const cv::Point2f absentKeypoint(-1.0f, -1.0f);

	//Cada uno de los "huesos" del esqueleto
	const std::vector<std::pair<int, int>> limbKeypointsIds = { { 1, 2 },   { 1, 5 },   { 2, 3 }, { 3, 4 },  { 5, 6 },
																{ 6, 7 },   { 1, 8 },   { 8, 9 }, { 9, 10 }, { 1, 11 },
																{ 11, 12 }, { 12, 13 }, { 1, 0 }, { 0, 14 }, { 14, 16 },
																{ 0, 15 },  { 15, 17 } };

	for (int i = 0; i < skeletons_buffer->numSkeletons; i++) {
		CV_Assert(skeletons_buffer->skeletons[i].numKeyPoints == 18);

		int id = skeletons_buffer->skeletons[i].id;
		cv::Point2f keyPointHead(skeletons_buffer->skeletons[i].keypoints_coord_x[0],
			skeletons_buffer->skeletons[i].keypoints_coord_y[0]);

		for (size_t keypointIdx = 0; keypointIdx < skeletons_buffer->skeletons[i].numKeyPoints; keypointIdx++) {
			const cv::Point2f keyPoint(skeletons_buffer->skeletons[i].keypoints_coord_x[keypointIdx],
				skeletons_buffer->skeletons[i].keypoints_coord_y[keypointIdx]);
			if (keyPoint != absentKeypoint && keypointIdx == 4) {
				cv::circle(image, keyPoint, 4, jointColor, -1);

				// get the 3d point and render it on the joints
				//Aquí sacamos el texto que se pone en cada uno de los puntos de articulación.
				//DERECHA
				cmPoint point3d =
					get_skeleton_point_3d(depth_frame, static_cast<int>(keyPoint.x), static_cast<int>(keyPoint.y));
				cv::putText(image, "("+std::to_string((int)keyPoint.x)+", "+ std::to_string((int)keyPoint.y) + "," + std::to_string(depth_frame.get_distance(static_cast<int>(keyPoint.x), static_cast<int>(keyPoint.y))) +")", keyPoint, cv::FONT_HERSHEY_COMPLEX, 1, jointColor);

				//Si la mano derecha toca la pared y está dentro de los margenes, se mueve el cursor
				if ((distance + distance * 0.05) > depth_frame.get_distance(static_cast<int>(keyPoint.x), static_cast<int>(keyPoint.y))
					&& (distance - distance * 0.05) < depth_frame.get_distance(static_cast<int>(keyPoint.x), static_cast<int>(keyPoint.y))
					&& static_cast<int>(keyPoint.x) > depth_frame.get_width() / 3.5
					&& static_cast<int>(keyPoint.x) < 1330 - (depth_frame.get_width() / 3.5)
					&& static_cast<int>(keyPoint.y) > depth_frame.get_height() / 3.5
					&& static_cast<int>(keyPoint.y) < 720 - (depth_frame.get_height() / 3.5)){
					//Tenemos que normalizar las coordenadas del ratón: 1535x863 vs 1280x720
					SetCursorPos(-150+(keyPoint.x-365)* 1535 /550,(keyPoint.y-190)* 863 /310);
					//SetCursorPos(100 + (keyPoint.x - 390) * 1280 / 550, 80 + (keyPoint.y - 200) * 720 / 310);
					mouse_event(MOUSEEVENTF_LEFTDOWN , 0, 0, 0, 0);
					//Para escribir en ficheros
					//fileAxisX << std::to_string(-225 + ((int)keyPoint.x - 365) * 1535 / 550) + "\n";
					//fileAxisY << std::to_string(((int)keyPoint.y - 205) * 863 / 310) + "\n";
					
				}
				else {
					mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
					
					isClicked = false;
				}
			}

			if (keyPoint != absentKeypoint && keypointIdx == 7) {
				cv::circle(image, keyPoint, 4, jointColor, -1);

				//IZQUIERDA
				cmPoint point3d =
					get_skeleton_point_3d(depth_frame, static_cast<int>(keyPoint.x), static_cast<int>(keyPoint.y));
				cv::putText(image, "(" + std::to_string((int)keyPoint.x) + ", " + std::to_string((int)keyPoint.y) + "," + std::to_string(depth_frame.get_distance(static_cast<int>(keyPoint.x), static_cast<int>(keyPoint.y))) + ")", keyPoint, cv::FONT_HERSHEY_COMPLEX, 1, jointColor);

				//Si la mano derecha toca la pared y está dentro de los margenes, se mueve el cursor
				if ((distance + distance * 0.05) > depth_frame.get_distance(static_cast<int>(keyPoint.x), static_cast<int>(keyPoint.y))
					&& (distance - distance * 0.05) < depth_frame.get_distance(static_cast<int>(keyPoint.x), static_cast<int>(keyPoint.y))) {
					//&& static_cast<int>(keyPoint.x) > depth_frame.get_width() / 3.5
					//&& static_cast<int>(keyPoint.x) < 1330 - (depth_frame.get_width() / 3.5)
					//&& static_cast<int>(keyPoint.y) > depth_frame.get_height() / 3.5
					//&& static_cast<int>(keyPoint.y) < 720 - (depth_frame.get_height() / 3.5)) 
					//Tenemos que normalizar las coordenadas del ratón: 1535x863 vs 1280x720
					SetCursorPos(-150 + (keyPoint.x) * 1535 / 550, (keyPoint.y - 190) * 863 / 310);
					//SetCursorPos(100 + (keyPoint.x - 390) * 1280 / 550, 80 + (keyPoint.y - 200) * 720 / 310);
					mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
				}
				else {
					mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
					isClicked = false;
				}
			}

		}

		for (const auto& limbKeypointsId : limbKeypointsIds) {
			const cv::Point2f keyPointFirst(skeletons_buffer->skeletons[i].keypoints_coord_x[limbKeypointsId.first],
				skeletons_buffer->skeletons[i].keypoints_coord_y[limbKeypointsId.first]);

			const cv::Point2f keyPointSecond(skeletons_buffer->skeletons[i].keypoints_coord_x[limbKeypointsId.second],
				skeletons_buffer->skeletons[i].keypoints_coord_y[limbKeypointsId.second]);

			if (keyPointFirst == absentKeypoint || keyPointSecond == absentKeypoint) {
				continue;
			}

			cv::line(image, keyPointFirst, keyPointSecond, skeletonColor, 2, cv::LINE_AA);
		}
		for (size_t keypointIdx = 0; keypointIdx < skeletons_buffer->skeletons[i].numKeyPoints; keypointIdx++) {
			const cv::Point2f keyPoint(skeletons_buffer->skeletons[i].keypoints_coord_x[keypointIdx],
				skeletons_buffer->skeletons[i].keypoints_coord_y[keypointIdx]);
			if (keyPoint != absentKeypoint) {
				// found a valid keypoint and displaying the skeleton tracking id next to it
				cv::putText(image,
					(std::to_string(id)),
					cv::Point2f(keyPoint.x, keyPoint.y - 20),
					cv::FONT_HERSHEY_COMPLEX,
					1,
					skeletonColor);
				break;
			}
		}
	}
}

int
main(int argc, char* argv[])
{
	// set up the intel realsense pipeline
	rs2::pipeline pipe;
	rs2::config cfg;
	rs2::context ctx;
	fileAxisX.open("C:/Users/Valiant/Desktop/ejex.txt");
	fileAxisY.open("C:/Users/Valiant/Desktop/ejeY.txt");

	if (ctx.query_devices().size() == 0) {
		EXIT_PROGRAM("No realsense device connected.");
	}

	cfg.enable_stream(RS2_STREAM_COLOR, -1, 1280, 720, RS2_FORMAT_BGR8, 30);
	cfg.enable_stream(RS2_STREAM_DEPTH, -1, 1280, 720, RS2_FORMAT_ANY, 30);
	rs2::align align_to_color(RS2_STREAM_COLOR);

	rs2::pipeline_profile profile;
	try {
		profile = pipe.start(cfg);
	}
	catch (std::exception ex)
	{
		EXIT_PROGRAM(std::string("Exception encountered starting the RealSense pipeline: ") + ex.what());
	}
	auto sensor = profile.get_device().first<rs2::depth_sensor>();

	auto range = sensor.get_option_range(RS2_OPTION_VISUAL_PRESET);
	for (auto i = range.min; i < range.max; i += range.step)
		if (std::string(sensor.get_option_value_description(RS2_OPTION_VISUAL_PRESET, i)) == "High Density")
			sensor.set_option(RS2_OPTION_VISUAL_PRESET, i);

	CM_TargetComputeDevice enInferenceMode = CM_TargetComputeDevice::CM_CPU;

	if (argc > 1) {
		if (strcmp(argv[1], "CPU") == 0 || strcmp(argv[1], "MYRIAD") == 0 || strcmp(argv[1], "GPU") == 0) {
			if (strcmp(argv[1], "MYRIAD") == 0)
				enInferenceMode = CM_TargetComputeDevice::CM_MYRIAD;
			else if (strcmp(argv[1], "CPU") == 0)
				enInferenceMode = CM_TargetComputeDevice::CM_CPU;
			else if (strcmp(argv[1], "GPU") == 0)
				enInferenceMode = CM_TargetComputeDevice::CM_GPU;
		}
	}

	// set up the cubemos skeleton tracking api pipeline
	CM_SKEL_Handle* handle = nullptr;
	// Output all messages with severity level INFO or higher to the console and to a file
	cm_initialise_logging(CM_LogLevel::CM_LL_INFO, true, default_log_dir().c_str());

	CM_ReturnCode retCode = cm_skel_create_handle(&handle, default_license_dir().c_str());
	CHECK_HANDLE_CREATION(retCode);

	std::string modelName = default_model_dir();
	if (enInferenceMode == CM_TargetComputeDevice::CM_CPU) {
		modelName += std::string("/fp32/skeleton-tracking.cubemos");
	}
	else {
		modelName += std::string("/fp16/skeleton-tracking.cubemos");
	}
	retCode = cm_skel_load_model(handle, enInferenceMode, modelName.c_str());
	if (retCode != CM_SUCCESS) {
		EXIT_PROGRAM("Model loading failed.");
	}

	std::string cvWindowName = "Skeleton tracking with intel realsense camera C/C++";
	cv::namedWindow(cvWindowName, cv::WINDOW_NORMAL);
	cv::setWindowProperty(cvWindowName, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN); 

	cv::Mat capturedFrame;

	const int nHeight = 192; // height of the image with which the DNN model will run inference

	// cache the first inference to get started with tracking
	// let some empty frames to run
	// el tamaño es de 1280 de ancho x 720 de alto (en pixeles)
	for (int k = 0; k < 30; k++) {
		rs2::frameset data = pipe.wait_for_frames();
		rs2::frame colorFrame = data.get_color_frame();
		rs2::frame depthFrame = data.get_depth_frame();
		capturedFrame = cv::Mat(
			cv::Size(colorFrame.as<rs2::video_frame>().get_width(), colorFrame.as<rs2::video_frame>().get_height()),
			CV_8UC3,
			(void*)colorFrame.get_data(),
			cv::Mat::AUTO_STEP);
	}

	CM_Image imageLast = {
		capturedFrame.data,         CM_UINT8, capturedFrame.cols, capturedFrame.rows, capturedFrame.channels(),
		(int)capturedFrame.step[0], CM_HWC
	};

	CUBEMOS_SKEL_Buffer_Ptr skeletonsPresent = create_skel_buffer();
	CUBEMOS_SKEL_Buffer_Ptr skeletonsLast = create_skel_buffer();
	int nTimeoutMs = 1000;

	// Get the skeleton keypoints for the first frame
	CM_ReturnCode retCodeFirstFrame = cm_skel_estimate_keypoints(handle, &imageLast, nHeight, skeletonsLast.get());

	// continue to loop through acquisition and display until the escape key is hit
	int frameCount = 0;
	std::string fpsTest = "Frame rate: ";

	// start measuring the time taken for execution
	std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::system_clock::now();

	while (cv::waitKey(1) != 27) {
		// capture image
		rs2::frameset data = pipe.wait_for_frames();
		data = align_to_color.process(data);

		rs2::frame colorFrame = data.get_color_frame();
		rs2::depth_frame depthFrame = data.get_depth_frame();

		capturedFrame = cv::Mat(
			cv::Size(colorFrame.as<rs2::video_frame>().get_width(), colorFrame.as<rs2::video_frame>().get_height()),
			CV_8UC3,
			(void*)colorFrame.get_data(),
			cv::Mat::AUTO_STEP);

		// exit the loop if the captured frame is empty
		if (capturedFrame.empty()) {
			std::cerr << "No new frame could be captured using the input source. Exiting the loop." << std::endl;
			break;
		}

		CM_Image imagePresent = {
			capturedFrame.data,         CM_UINT8, capturedFrame.cols, capturedFrame.rows, capturedFrame.channels(),
			(int)capturedFrame.step[0], CM_HWC
		};

		//Calibramos y medimos la distancia a la pantalla
		float dist = calibrateScreen(depthFrame, capturedFrame);

		// Run Skeleton Tracking and display the results
		retCode = cm_skel_estimate_keypoints(handle, &imagePresent, nHeight, skeletonsPresent.get());
		//std::cout << "The camera is facing an object " << dist << " meters away \r";

		// track the skeletons in case of successful skeleton estimation
		if (retCode == CM_SUCCESS) {
			if (skeletonsPresent->numSkeletons > 0) {
				// Assign tracking ids to the skeletons in the present frame
				cm_skel_update_tracking_id(handle, skeletonsLast.get(), skeletonsPresent.get());
				// Render skeleton overlays with tracking ids
				renderSkeletons(skeletonsPresent.get(), depthFrame, capturedFrame, dist);
				// Set the present frame as last one to track the next frame
				skeletonsLast.swap(skeletonsPresent);
				// Free memory of the latest frame
				cm_skel_release_buffer(skeletonsPresent.get());
				

			}
		}

		frameCount++;
		if (frameCount % 25 == 0) {
			auto timePassed =
				std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startTime)
				.count();
			auto fps = 25000.0 / timePassed;

			fpsTest = "Frame rate: " + std::to_string(fps) + " FPS";
			startTime = std::chrono::system_clock::now();
		}
		cv::putText(capturedFrame, fpsTest, cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, 1, skeletonColor);

	
		cv::imshow(cvWindowName, capturedFrame);

	}

	// release the memory which is managed by the cubemos framework
	cm_skel_destroy_handle(&handle);
	return 0;
}
