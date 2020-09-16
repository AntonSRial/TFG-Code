using System;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Controls;
using System.Drawing;
using System.Linq;
using System.IO;
using System.Collections.Generic;
using Intel.RealSense;
using System.Diagnostics;
using System.Windows.Shapes;
using System.Collections.ObjectModel;

/// 
/// \brief This demo code shows how to use the cubemos Skeleton Tracking C# API
/// 
namespace Cubemos.Samples
{
    // Interaction logic for the Window.xaml
    public partial class CaptureWindow : Window
    {
        struct Resolution
        {
            public int Width;
            public int Height;
            public Resolution(int w, int h)
            {
                this.Width = w;
                this.Height = h;
            }
            public string getResolutionAsString()
            {
                string retVal = (Width + " x " + Height);
                return retVal;
            }
        }

        // Depth map post-processing
        private Colorizer colorizer = new Colorizer();
        private Align align = new Align(Intel.RealSense.Stream.Color);
        private DecimationFilter decimate = new DecimationFilter();
        private SpatialFilter spatial = new SpatialFilter();
        private TemporalFilter temp = new TemporalFilter();

        private List<Resolution> sensorResolutions = new List<Resolution>( new[] {
        new Resolution(640, 360 ),
        new Resolution(848, 480 ),
        new Resolution(1280, 720 )} );

        private DeviceList availableDevices;
        private ReadOnlyCollection<Sensor> activeDeviceSensors;

        private bool temporalFilterEnabled = false;
        private bool spatialFilterEnabled = false;
       
        DateTime LastBrightnessChange = DateTime.Now;
        TimeSpan BrightnessChangeDelay = TimeSpan.FromMilliseconds(100);



        private void SetPostProcessingRanges()
        {            
            slSpatialMagnitude.Maximum = spatial.Options[Option.FilterMagnitude].Max;
            slSpatialMagnitude.Minimum = spatial.Options[Option.FilterMagnitude].Min;
            slSpatialMagnitude.Value = spatial.Options[Option.FilterMagnitude].Default;

            slSpatialSmoothAlpha.Maximum = spatial.Options[Option.FilterSmoothAlpha].Max;
            slSpatialSmoothAlpha.Minimum = spatial.Options[Option.FilterSmoothAlpha].Min;
            slSpatialSmoothAlpha.Value = spatial.Options[Option.FilterSmoothAlpha].Default;

            slSpatialSmoothDelta.Maximum = spatial.Options[Option.FilterSmoothDelta].Max;
            slSpatialSmoothDelta.Minimum = spatial.Options[Option.FilterSmoothDelta].Min;
            slSpatialSmoothDelta.Value = spatial.Options[Option.FilterSmoothDelta].Default;

            slTemporalSmoothAlpha.Maximum = temp.Options[Option.FilterSmoothAlpha].Max;
            slTemporalSmoothAlpha.Minimum = temp.Options[Option.FilterSmoothAlpha].Min;
            slTemporalSmoothAlpha.Value = temp.Options[Option.FilterSmoothAlpha].Default;

            slTemporalSmoothDelta.Maximum = temp.Options[Option.FilterSmoothDelta].Max;
            slTemporalSmoothDelta.Minimum = temp.Options[Option.FilterSmoothDelta].Min;
            slTemporalSmoothDelta.Value = temp.Options[Option.FilterSmoothDelta].Default;
        }

        private void ToggleStartStop_Click(object sender, RoutedEventArgs e)
        {
            if ((bool)this.toggleStartStop.IsChecked)
            {
                this.toggleStartStop.IsEnabled = false;
                Panel.SetZIndex(Cnv2, 1);
                loadingText.Visibility = Visibility.Visible;
                Console.WriteLine("Starting the application");
                // Disable all controls
                this.resolutionOptionBox.IsEnabled = false;
                this.toggleStartStop.Content = "\uE73C";
                this.computationBackend.IsEnabled = false;
                this.networkSlider.IsEnabled = false;
                this.camera_source.IsEnabled = false;

                int networksize = (int)networkSlider.Value;
                ComputationMode computationMode = ComputationMode.CPU;
                System.Windows.Controls.ComboBoxItem item = computationBackend.SelectedItem as System.Windows.Controls.ComboBoxItem;
                string itemstring = (string)item.Content;
                if (itemstring == "Target Device: Intel GPU")
                {
                    computationMode = ComputationMode.GPU;
                }

                if (itemstring == "Target Device: Intel Myriad")
                {
                    computationMode = ComputationMode.MYRIAD;
                }

                try
                {
                    if(tokenSource != null)
                        tokenSource.Dispose();
                }
                catch (ObjectDisposedException)
                {

                }
                tokenSource = new CancellationTokenSource();
                string acq_msg = String.Format("Acquisition Status:\t ONLINE");

                acquisition_status.Dispatcher.BeginInvoke((Action)delegate
                {
                    acquisition_status.Text = acq_msg;
                });

                StartCapture(networksize, computationMode);
            }
            else
            {
                this.toggleStartStop.IsEnabled = false;
                Console.WriteLine("Stopping the application");


                // Enable all controls
                this.toggleStartStop.Content = "\uF5B0";
                this.resolutionOptionBox.IsEnabled = true;
                this.computationBackend.IsEnabled = true;
                this.networkSlider.IsEnabled = true;
                this.camera_source.IsEnabled = true;

                // Stop demo
                string acq_msg = String.Format("Acquisition Status:\t OFFLINE");
                acquisition_status.Dispatcher.BeginInvoke((Action)delegate
                {
                    acquisition_status.Text = acq_msg;
                });

                StopCapture();
                try
                {
                    pipeline.Stop();
                }
                catch
                { }

                person_count.Dispatcher.BeginInvoke((Action)delegate
                {
                    person_count.Text = "Person Count:";
                });

                this.toggleStartStop.IsEnabled = true;
            }
        }
        private void Camera_source_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            int index = camera_source.SelectedIndex;
            if (index >= availableDevices.Count || index < 0)
                return;
            var device = availableDevices[index];

            activeDeviceSensors = device.QuerySensors<Sensor>();
            hasDepthSensor = false;
            foreach (Sensor sensor in activeDeviceSensors)
            {
                IOptionsContainer options = sensor.Options;
                bool isDepthSensor = sensor.Options.Supports(Option.VisualPreset);
                if (isDepthSensor)
                {
                    hasDepthSensor = true;

                    if (sensor.Options.Supports(Option.LaserPower))
                    {
                        IOption emitterPowerOption = options[Option.LaserPower];
                        laserPower.Visibility = Visibility.Visible;
                        laserPower.Minimum = emitterPowerOption.Min;
                        laserPower.Maximum = emitterPowerOption.Max;
                        laserPower.Value = 0.5 * (laserPower.Minimum + laserPower.Maximum);
                    }
                    else
                    {
                        laserPower.Visibility = Visibility.Hidden;
                    }
                }
                else
                {

                    if (sensor.Options.Supports(Option.Brightness))
                    {
                        IOption brightnes = options[Option.Brightness];
                        rgbBrightness.Visibility = Visibility.Visible;
                        rgbBrightness.Minimum = brightnes.Min;
                        rgbBrightness.Maximum = brightnes.Max;
                    }
                    else
                    {
                        rgbBrightness.Visibility = Visibility.Hidden;
                    }
                    rgbBacklightCompensation.Visibility = sensor.Options.Supports(Option.BacklightCompensation) ? Visibility.Visible : Visibility.Hidden;
                }
            }
            if (hasDepthSensor)
            {
                depth_sensor_options_expander.Visibility = Visibility.Visible;
            }
            else
            {
                depth_sensor_options_expander.Visibility = Visibility.Hidden;
            }
        }

        private void RgbBrightness_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {            
            if (DateTime.Now - LastBrightnessChange <= BrightnessChangeDelay)
                return;
                             
            LastBrightnessChange = DateTime.Now;

            float val = Convert.ToSingle(e.NewValue);
            int sensorIndex = hasDepthSensor ? 1 : 0;
               
            IOption option = activeDeviceSensors[sensorIndex].Options[Option.Brightness];
            option.Value = val;                          
        }

        private void RgbBacklightCompensation_Checked(object sender, RoutedEventArgs e)
        {
            bool isChecked = rgbBacklightCompensation.IsChecked ?? false;
            int sensorIndex = hasDepthSensor ? 1 : 0;
            IOption option = activeDeviceSensors[sensorIndex].Options[Option.BacklightCompensation];
            option.Value = isChecked ? 1 : 0;
        }
        private void DepthEmitterEnabled_Checked(object sender, RoutedEventArgs e)
        {
            bool isChecked = depthEmitterEnabled.IsChecked ?? false;
            if (!hasDepthSensor)
                return;

            int sensorIndex = 0;
            //set laser power in case it has changed
            IOption optionLaser = activeDeviceSensors[0].Options[Option.LaserPower];
            optionLaser.Value = Convert.ToSingle(laserPower.Value);
            IOption option = activeDeviceSensors[sensorIndex].Options[Option.EmitterEnabled];
            option.Value = isChecked ? 1 : 0;
        }
        private void LaserPower_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (!hasDepthSensor)
                return;
            if (!(depthEmitterEnabled.IsChecked ?? false))
                return;

            IOption option = activeDeviceSensors[0].Options[Option.LaserPower];
            float val = Convert.ToSingle(e.NewValue);
            option.Value = val;
        }
        private void PresetOptionBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (!hasDepthSensor || pp == null)
                return;
            var val = (Intel.RealSense.Rs400VisualPreset)presetOptionBox.SelectedIndex;
            IOption option = pp.Device.Sensors[0].Options[Option.VisualPreset];
            option.Value = (float)val;
            Console.WriteLine("Changing preset to {0}", val.ToString());
        }
        private void temporalFilter_Checked(object sender, RoutedEventArgs e)
        {
            temporalFilterEnabled = temporalFilter.IsChecked ?? false;
        }
                
        private void temporalFilterSmoothAlpha_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            temp.Options[Option.FilterSmoothAlpha].Value = Convert.ToSingle(e.NewValue);
        }
        private void temporalFilterSmoothDelta_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            temp.Options[Option.FilterSmoothDelta].Value = Convert.ToSingle(e.NewValue);
        }
        private void spatialFilter_Checked(object sender, RoutedEventArgs e)
        {
            spatialFilterEnabled = spatialFilter.IsChecked ?? false;
        }
        private void spatialFilterMagnitude_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            spatial.Options[Option.FilterMagnitude].Value = Convert.ToSingle(e.NewValue);
        }
        private void spatialFilterSmoothAlpha_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            spatial.Options[Option.FilterSmoothAlpha].Value = Convert.ToSingle(e.NewValue);
        }
        private void spatialFilterSmoothDelta_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            spatial.Options[Option.FilterSmoothDelta].Value = Convert.ToSingle(e.NewValue);
        }     
        private void ButtonRefreshDevices_Click(object sender, RoutedEventArgs e)
        {
            Console.WriteLine("Refreshing devices..");
            if(QueryRealSenseDevices())
            {
                availableDevices = context.QueryDevices(false);
                Console.WriteLine("Connected RealSense Cameras found");
               
                camera_source.Items.Clear();

                foreach (Device device in availableDevices)
                {                    
                    camera_source.Items.Add(String.Format("{0}: {1}", device.Info[CameraInfo.Name], device.Info[CameraInfo.SerialNumber]));   
                    if(device.Info[CameraInfo.SerialNumber] == selectedDeviceSerial)
                    {
                        camera_source.SelectedIndex = camera_source.Items.Count - 1;
                        selectedDevice = availableDevices[camera_source.SelectedIndex];
                    }
                }
                if(selectedDeviceSerial == "")
                {
                    camera_source.SelectedIndex = 0;
                    selectedDevice = availableDevices[camera_source.SelectedIndex];
                    selectedDeviceSerial = selectedDevice.Info[CameraInfo.SerialNumber];
                }
                toggleStartStop.IsEnabled = true;
            }
            else
            {
                toggleStartStop.IsEnabled = false;
                camera_source.Items.Clear();
                Console.WriteLine("No RealSense Cameras found");
            }
        }
        private bool QueryRealSenseDevices()
        {
            int nRealSenseCnt = context.QueryDevices(false).Count;
            if (nRealSenseCnt == 0)
            {
                toggleStartStop.Foreground.Opacity = 0.1;
            }
            else
            {
                toggleStartStop.Foreground.Opacity = 1.0;
            }
            return nRealSenseCnt > 0;
        }
    }
}