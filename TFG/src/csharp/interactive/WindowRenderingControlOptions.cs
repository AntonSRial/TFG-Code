using Intel.RealSense;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;


/// 
/// \brief This demo code shows how to use the cubemos Skeleton Tracking C# API
/// 
namespace Cubemos.Samples
{
    // Interaction logic for the Window.xaml
    public partial class CaptureWindow : Window
    {
        // Declare the necessary control variables
        private bool bRenderSkeletons = false;
        private bool bShowOnlySkeletons = false;
        private bool bRenderDepthMap = false;
        private bool bRenderCoordinates = false;
        private bool bEnableTracking = false;

        private Action<System.Windows.Controls.Image, Bitmap> renderDelegate = new Action<System.Windows.Controls.Image, Bitmap>(RenderImageFromBitmap);

        private void cbEnable_Tracking(object sender, RoutedEventArgs e)
        {
            bEnableTracking = !bEnableTracking;
        }

        private void cbRendering_SkeletonOverlay(object sender, RoutedEventArgs e)
        {
            bRenderSkeletons = !bRenderSkeletons;
        }
        private void cbRendering_OnlySkeletons(object sender, RoutedEventArgs e)
        {
            bShowOnlySkeletons = !bShowOnlySkeletons;
        }
        private void cbRendering_DepthMapOverlay(object sender, RoutedEventArgs e)
        {
            bRenderDepthMap = (!bRenderDepthMap);
            Panel.SetZIndex(imgDepth, (true == bRenderDepthMap) ? 1 : -1);
        }
        private void ImgColor_MouseLeftButtonDown(object sender, System.Windows.Input.MouseButtonEventArgs e)
        {
        }
        private void cbRendering_ShowCoordinates(object sender, RoutedEventArgs e)
        {
            bRenderCoordinates = !bRenderCoordinates;
        }      

        /// \brief Bitmap rendering
        /// \param imgWindow [in] display control to render the image on
        /// \param img [in] image to render
        static void RenderImageFromBitmap(System.Windows.Controls.Image imgWindow, Bitmap img)
        {
            BitmapImage bitmapImage = new BitmapImage();
            using (MemoryStream memory = new MemoryStream())
            {
                img.Save(memory, ImageFormat.Bmp);
                memory.Position = 0;
                bitmapImage.BeginInit();
                bitmapImage.StreamSource = memory;
                bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapImage.EndInit();
            }

            imgWindow.Source = bitmapImage;
        }                         
    }
   
}