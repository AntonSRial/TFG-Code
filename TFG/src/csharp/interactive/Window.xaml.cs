using System;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Controls;
using System.Drawing;
using System.Linq;
using System.Windows.Media.Imaging;
using System.IO;
using System.Collections.Generic;
using static Cubemos.SkeletonTracking.Api;
using Intel.RealSense;
using Cubemos.SkeletonTracking;
using System.Diagnostics;

///
/// \brief This demo code shows how to use the cubemos Skeleton Tracking C# API using a RealSense Camera for image input
///
namespace Cubemos.Samples
{
    /// Device to use for computation: Intel CPU or Intel GPU
    public enum ComputationMode { CPU, GPU, MYRIAD }

    /// Interaction logic for the Window.xaml
    public partial class CaptureWindow : Window {
        // Declare the necessary control variables
        private bool bHideRenderImage = false;

        // Logger
        LogOutput logger;

        // Declare the necessary realsense variables, pipeline and exit events
        Device selectedDevice;
        String selectedDeviceSerial = "";
        private DepthFrame depthFrame;
        private Intrinsics intrinsicsDepthImagerMaster;

        private Context context;
        private Pipeline pipeline = new Pipeline();
        private Config cfg = new Config();
        private PipelineProfile pp;
        private bool hasDepthSensor = false;
        private CancellationTokenSource tokenSource = new CancellationTokenSource();
        // Convert the realsense frame to C# bitmap
        private Bitmap FrameToBitmap(
          VideoFrame frame,
          System.Drawing.Imaging.PixelFormat format = System.Drawing.Imaging.PixelFormat.Format24bppRgb)
        {
            if (frame.Width == 0)
                return null;
            return new Bitmap(frame.Width, frame.Height, frame.Stride, format, frame.Data);
        }

        public CaptureWindow()
        {
            InitializeComponent();

            logger = new LogOutput(logTextBox);
            Console.SetOut(logger);
            settingsPanel.Visibility = Visibility.Hidden;

            context = new Intel.RealSense.Context();
            availableDevices = context.QueryDevices(include_platform_camera : false);

            foreach (Device device in availableDevices) {
                camera_source.Items.Add(
                  String.Format("{0}: {1}", device.Info[CameraInfo.Name], device.Info[CameraInfo.SerialNumber]));
            }

            foreach (Resolution resolution in sensorResolutions) {
                resolutionOptionBox.Items.Add(resolution.getResolutionAsString());
            }
            resolutionOptionBox.SelectedIndex = 2;

            bool bFoundRealSense = QueryRealSenseDevices();
            if (bFoundRealSense) {
                camera_source.SelectedIndex = 0;
                selectedDevice = availableDevices[camera_source.SelectedIndex];
                selectedDeviceSerial = selectedDevice.Info[CameraInfo.SerialNumber];
            }
            else
                toggleStartStop.IsEnabled = false;

            presetOptionBox.ItemsSource = Enum.GetValues(typeof(Intel.RealSense.Rs400VisualPreset));
            presetOptionBox.SelectedIndex = 0;

            string acq_msg = String.Format("Acquisition Status:\t OFFLINE");
            acquisition_status.Dispatcher.BeginInvoke((Action) delegate { acquisition_status.Text = acq_msg; });
            SetPostProcessingRanges();
     

            Cubemos.Api.InitialiseLogging(Cubemos.LogLevel.CM_LL_INFO, true, Common.DefaultLogDir());
        }
        private void StartCapture(int networkHeight, ComputationMode computationMode)
        {
            try {
                bool bDevicesFound = QueryRealSenseDevices();
                if (bDevicesFound == false) {
                    Console.WriteLine("Cannot start acquisition as no RealSense is connected.");
                    toggleStartStop.IsChecked = false;
                    this.toggleStartStop.Content = "\uF5B0";
                    // Enable all controls
                    this.computationBackend.IsEnabled = true;
                    this.networkSlider.IsEnabled = true;
                    // Stop demo

                    string acq_msg = string.Format("Acquisition Status:\t OFFLINE");
                    acquisition_status.Dispatcher.BeginInvoke((Action) delegate { acquisition_status.Text = acq_msg; });
                    return;
                }

                // get the selected image width and height
                int nImageWidth = sensorResolutions[resolutionOptionBox.SelectedIndex].Width;
                int nImageHeight = sensorResolutions[resolutionOptionBox.SelectedIndex].Height;

                Console.WriteLine(
                  string.Format("Enabling the {0} S.No: {1}",
                                availableDevices[camera_source.SelectedIndex].Info[CameraInfo.Name],
                                availableDevices[camera_source.SelectedIndex].Info[CameraInfo.SerialNumber]));
                Console.WriteLine(
                  string.Format("Selected resolution for the image acquisition is {0}x{1}", nImageWidth, nImageHeight));
                Console.WriteLine(string.Format("Selected network size: {0} along with {1} as the computation device",
                                                networkHeight,
                                                computationMode));
                selectedDeviceSerial = availableDevices[camera_source.SelectedIndex].Info[CameraInfo.SerialNumber];
                // Create and config the pipeline to stream color and depth frames.
                cfg.EnableDevice(availableDevices[camera_source.SelectedIndex].Info[CameraInfo.SerialNumber]);
                cfg.EnableStream(Intel.RealSense.Stream.Color, nImageWidth, nImageHeight, Format.Bgr8, framerate : 30);
                cfg.EnableStream(Intel.RealSense.Stream.Depth, nImageWidth, nImageHeight, framerate : 30);

                Task.Factory.StartNew(() => {
                    try {
                        // Create and config the pipeline to stream color and depth frames.
                        pp = pipeline.Start(cfg);
                        intrinsicsDepthImagerMaster =
                          (pp.GetStream(Intel.RealSense.Stream.Depth).As<VideoStreamProfile>()).GetIntrinsics();

                        // Initialise cubemos DNN framework with the required deep learning model and the target compute
                        // device. Currently CPU and GPU are supported target devices. FP32 model is necessary for the
                        // CPU and FP16 model is required by the Myriad device and GPU

                        Cubemos.SkeletonTracking.Api skeletontrackingApi;

                        String cubemosModelDir = Common.DefaultModelDir();                       

                        var computeDevice = Cubemos.TargetComputeDevice.CM_CPU;
                        var modelFile = cubemosModelDir + "\\fp32\\skeleton-tracking.cubemos";

                        if (computationMode == ComputationMode.GPU) {
                            computeDevice = Cubemos.TargetComputeDevice.CM_GPU;
                            modelFile = cubemosModelDir + "\\fp16\\skeleton-tracking.cubemos";
                        }
                        else if (computationMode == ComputationMode.MYRIAD)
                        {
                            computeDevice = Cubemos.TargetComputeDevice.CM_MYRIAD;
                            modelFile = cubemosModelDir + "\\fp16\\skeleton-tracking.cubemos";
                        }

                        var licenseFolder = Common.DefaultLicenseDir();
                        try {
                            skeletontrackingApi = new SkeletonTracking.Api(licenseFolder);
                        }
                        catch (Exception ex) {
                            throw new Cubemos.Exception(
                              String.Format("Activation key or license key not found in {0}.\n " +
                              "If you haven't activated the SDK yet, please run post_installation script as described in the Getting Started Guide to activate your license.", 
                              licenseFolder));
                        }

                        try {
                            skeletontrackingApi.LoadModel(computeDevice, modelFile);
                        }
                        catch (Exception ex) {
                            if (File.Exists(modelFile)) {
                                throw new Cubemos.Exception(
                                  "Internal error occured during model initialization. Please make sure your compute device satisfies the hardware system requirements.");
                            }
                            else {
                                throw new Cubemos.Exception(
                                  string.Format("Model file \"{0}\" not found. Details: \"{1}\"", modelFile, ex));
                            }
                        }

                        Console.WriteLine("Finished initialization");

                        Stopwatch fpsStopwatch = new Stopwatch();
                        double fps = 0.0;
                        int nFrameCnt = 0;

                        bool firstRun = true;

                        Console.WriteLine("Starting image acquisition and skeleton keypoints");
                        while (!tokenSource.Token.IsCancellationRequested) {

                            int pipelineID = 1;
                            if (bEnableTracking == false) {
                                pipelineID = 0;
                            }

                            fpsStopwatch.Restart();

                            // We wait for the next available FrameSet and using it as a releaser object that would
                            // track all newly allocated .NET frames, and ensure deterministic finalization at the end
                            // of scope.
                            using(var releaser = new FramesReleaser())
                            {
                                using(var frames = pipeline.WaitForFrames())
                                {
                                    if (frames.Count != 2)
                                        Console.WriteLine("Not all frames are available...");

                                    var f = frames.ApplyFilter(align).DisposeWith(releaser).AsFrameSet().DisposeWith(
                                      releaser);

                                    var colorFrame = f.ColorFrame.DisposeWith(releaser);
                                    depthFrame = f.DepthFrame.DisposeWith(releaser);

                                    var alignedDepthFrame = align.Process<DepthFrame>(depthFrame).DisposeWith(f);

                                    if (temporalFilterEnabled) {
                                        alignedDepthFrame = temp.Process<DepthFrame>(alignedDepthFrame).DisposeWith(f);
                                    }

                                    // We colorize the depth frame for visualization purposes
                                    var colorizedDepth =
                                      colorizer.Process<VideoFrame>(alignedDepthFrame).DisposeWith(f);

                                    // Preprocess the input image
                                    Bitmap inputImage = FrameToBitmap(colorFrame);
                                    Bitmap inputDepthMap = FrameToBitmap((VideoFrame) colorizedDepth);

                                    // Run the inference on the preprocessed image
                                    List<SkeletonKeypoints> skeletonKeypoints;
                                    skeletontrackingApi.RunSkeletonTracking(
                                      ref inputImage, networkHeight, out skeletonKeypoints, pipelineID);

                                    if (firstRun) {
                                        Cnv2.Dispatcher.BeginInvoke((Action) delegate { Panel.SetZIndex(Cnv2, -1); },
                                                                    System.Windows.Threading.DispatcherPriority.Render);

                                        toggleStartStop.Dispatcher.BeginInvoke(
                                          (Action) delegate { toggleStartStop.IsEnabled = true; });

                                        firstRun = false;
                                    }

                                    Bitmap displayImage;
                                    if (bShowOnlySkeletons) {
                                        displayImage = new Bitmap(inputImage.Width, inputImage.Height);
                                        using(Graphics g = Graphics.FromImage(displayImage))
                                        {
                                            g.Clear(System.Drawing.Color.Black);
                                        }
                                    }
                                    else {
                                        displayImage = new Bitmap(inputImage);
                                    }

                                    Graphics graphics = Graphics.FromImage(displayImage);

                                    // Render the correct skeletons detected from the inference
                                    if (true == bRenderSkeletons) {
                                        renderSkeletons(
                                          skeletonKeypoints, nImageWidth, nImageHeight, bEnableTracking, graphics);
                                    }

                                    if (true == bRenderCoordinates) {
                                        renderCoordinates(skeletonKeypoints, nImageWidth, graphics);
                                    }

                                    if (false == bHideRenderImage) { // Render the final frame onto the display window
                                        imgColor.Dispatcher.BeginInvoke(renderDelegate, imgColor, displayImage);
                                    }
                                    if (true == bRenderDepthMap) { // Overlay the depth map onto the display window
                                        imgColor.Dispatcher.BeginInvoke(renderDelegate, imgDepth, inputDepthMap);
                                    }

                                    nFrameCnt++;
                                    fps += (double)(1000.0 / (double) fpsStopwatch.ElapsedMilliseconds);

                                    if (nFrameCnt % 25 == 0) {
                                        string msg = String.Format("FPS:\t\t\t{0:F2}", fps / nFrameCnt);
                                        fps_output.Dispatcher.BeginInvoke((Action) delegate { fps_output.Text = msg; });
                                        fps = 0;
                                        nFrameCnt = 0;
                                    }

                                    string msg_person_count =
                                      string.Format("Person Count:\t\t{0}", skeletonKeypoints.Count);
                                    person_count.Dispatcher.BeginInvoke(
                                      (Action) delegate { person_count.Text = msg_person_count; });
                                }
                            }
                        }
                    }
                    catch (System.Exception exT) {
                        string errorMsg = string.Format(
                          "Internal Error Occured. Application will now close.\nError Details:\n\n\"{0}\"",
                          exT.Message);
                        Cnv2.Dispatcher.BeginInvoke(
                          new InfoDialog.ShowInfoDialogDelegate(InfoDialog.ShowInfoDialog), "Error", errorMsg);
                    }
                }, tokenSource.Token);
            }
            catch (System.Exception ex) {
                string errorMsg = string.Format(
                  "Internal Error Occured. Application will now close.\nError Details:\n\n\"{0}\"", ex.Message);
                Cnv2.Dispatcher.BeginInvoke(
                  new InfoDialog.ShowInfoDialogDelegate(InfoDialog.ShowInfoDialog), "Error", errorMsg);
            }
        }
        private void StopCapture()
        {
            try {
                tokenSource.Cancel();
            }
            catch (ObjectDisposedException ex) {
                Console.WriteLine("An exception occured during capture stop. Message: " + ex.Message);
            }
        }
        private void control_Closing(object sender, System.ComponentModel.CancelEventArgs e) { StopCapture(); }
        private void NetworkSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            int val = Convert.ToInt32(e.NewValue);
            string msg = String.Format("Network input size: {0}", val);
            this.textBlock1.Text = msg;
        }
        private void ToggleButtonHelp_Click(object sender, RoutedEventArgs e)
        {
            if ((bool) this.toggleHelp.IsChecked) {
                Panel.SetZIndex(usageGuideImage, 12);
                bHideRenderImage = true;
            }
            else {
                bHideRenderImage = false;
                Panel.SetZIndex(usageGuideImage, -12);
            }
        }
        private void ButtonScreenShot_Click(object sender, RoutedEventArgs e)
        {
            if (imgColor.Source == null) {
                Console.WriteLine(
                  "No image is available to take a screenshot of. Make sure that the stream has been started and an image is rendered onto the display.");
                return;
            }

            BitmapSource img = imgColor.Source as BitmapSource;
            JpegBitmapEncoder encoder = new JpegBitmapEncoder();
            BitmapFrame outputFrame = BitmapFrame.Create(img);
            encoder.Frames.Add(outputFrame);

            String filePath = Environment.GetFolderPath(Environment.SpecialFolder.MyPictures) +
                              "\\CUBEMOS_skeletontracking_" + DateTime.Now.ToString("yyyyMMdd");
            Directory.CreateDirectory(filePath);

            filePath = filePath + "\\CUBEMOS_" + DateTime.Now.ToString("yyyyMMdd_HHmmss_fff") + ".jpeg";
            using(FileStream file = File.OpenWrite(filePath)) { encoder.Save(file); }
            Console.WriteLine("Screenshot saved at: " + filePath);
        }
        private void ToggleButtonSettings_Click(object sender, RoutedEventArgs e)
        {
            settingsPanel.Visibility =
              (Visibility.Visible == settingsPanel.Visibility) ? Visibility.Hidden : Visibility.Visible;
        }
    }
}
