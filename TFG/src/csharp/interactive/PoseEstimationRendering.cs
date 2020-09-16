using System;
using System.Windows;
using System.Drawing;
using System.Collections.Generic;
using System.Drawing.Imaging;
using static Cubemos.SkeletonTracking.Api;

///
/// \brief This demo code shows how to use the cubemos Skeleton Tracking C# API
///
namespace Cubemos.Samples
{
    // Skeletons rendering for Skeleton Tracking for the Window.xaml
    public partial class CaptureWindow : Window {
        // The list of colors to use for skeleton tracking
        static List<System.Drawing.Pen> skeletonRenderColors = new List<System.Drawing.Pen>(
          new[]{ new System.Drawing.Pen(System.Drawing.ColorTranslator.FromHtml("#d5fe64")),
                 new System.Drawing.Pen(System.Drawing.ColorTranslator.FromHtml("#fe6262")),
                 new System.Drawing.Pen(System.Drawing.ColorTranslator.FromHtml("#62fefe")),
                 new System.Drawing.Pen(System.Drawing.ColorTranslator.FromHtml("#6262fe")),
                 new System.Drawing.Pen(System.Drawing.ColorTranslator.FromHtml("#fe62fe")),
                 new System.Drawing.Pen(System.Drawing.ColorTranslator.FromHtml("#40ff00")),
                 new System.Drawing.Pen(System.Drawing.ColorTranslator.FromHtml("#ffbf00")) });

        // The list containing the ordering in which the skeleton key points are paired for rendering
        static List<Coordinate> renderSkeletonPointPairs = new List<Coordinate>(new[]{ new Coordinate(1, 2),
                                                                                       new Coordinate(1, 5),
                                                                                       new Coordinate(2, 3),
                                                                                       new Coordinate(3, 4),
                                                                                       new Coordinate(5, 6),
                                                                                       new Coordinate(6, 7),
                                                                                       new Coordinate(1, 8),
                                                                                       new Coordinate(8, 9),
                                                                                       new Coordinate(9, 10),
                                                                                       new Coordinate(1, 11),
                                                                                       new Coordinate(11, 12),
                                                                                       new Coordinate(12, 13),
                                                                                       new Coordinate(1, 0),
                                                                                       new Coordinate(0, 14),
                                                                                       new Coordinate(14, 16),
                                                                                       new Coordinate(0, 15),
                                                                                       new Coordinate(15, 17) });

        ///
        /// \brief Render skeletons: the joints and connections between them
        /// \param skeletonKeypoints [in] skeletons to render
        /// \param nImageWidth [in] image width in pixel
        /// \param nImageHeight [in] image height in pixel
        /// \param graphics [in] graphics object to display the skeletons
        public static void renderSkeletons(List<SkeletonKeypoints> skeletonKeypoints,
                                           int nImageWidth,
                                           int nImageHeight,
                                           bool bEnableTracking,
                                           Graphics graphics)
        {
            int nEllipseSize = (int)(nImageWidth / 64); // 20 at 1280 and 10 at 640 width

            for (int skeleton_index = 0; skeleton_index < skeletonKeypoints.Count; skeleton_index++) {
                var skeleton = skeletonKeypoints[skeleton_index];

                System.Drawing.Pen pen;
                if (bEnableTracking) {
                    pen = skeletonRenderColors[skeletonKeypoints[skeleton_index].id % skeletonRenderColors.Count];
                }
                else {
                    pen = new System.Drawing.Pen(System.Drawing.ColorTranslator.FromHtml("#d5fe64"));
                }

                // 4 at 1280 and 2 at 640 width
                pen.Width = nImageWidth / 320;

                for (int joint_index = 0; joint_index < skeleton.listJoints.Count; joint_index++) {
                    Coordinate coordinate = skeleton.listJoints[joint_index];
                    if (coordinate.x > 0 && coordinate.y > 0) {
                        graphics.FillEllipse(new SolidBrush(System.Drawing.ColorTranslator.FromHtml("#16379B")),
                                             (int) coordinate.x - (int)(nImageWidth / 128),
                                             (int) coordinate.y -
                                               (int)(nImageWidth / 128), // 10 at 1280 and 5 at 640 width
                                             nEllipseSize,
                                             nEllipseSize);
                    }
                }

                Coordinate coordinateValid = new Coordinate(-1.0, -1.0);
                coordinateValid =
                  skeleton.listJoints.Find(delegate(Coordinate crd) { return crd.x >= 0 && crd.y >= 0; });

                // Render the connecting lines between the detected keypoints
                for (int pair_index = 0; pair_index < renderSkeletonPointPairs.Count; pair_index++) {
                    var pointPair = renderSkeletonPointPairs[pair_index];
                    Coordinate startPoint = new Coordinate((int) skeleton.listJoints[(int) pointPair.x].x,
                                                           (int) skeleton.listJoints[(int) pointPair.x].y);
                    Coordinate endPoint = new Coordinate((int) skeleton.listJoints[(int) pointPair.y].x,
                                                         (int) skeleton.listJoints[(int) pointPair.y].y);

                    if (startPoint.x > 0 && startPoint.y > 0 && endPoint.y > 0 && endPoint.y > 0) {
                        graphics.DrawLine(pen,
                                          new System.Drawing.Point((int) startPoint.x, (int) startPoint.y),
                                          new System.Drawing.Point((int) endPoint.x, (int) endPoint.y));
                    }
                }
            }
        }
        ///
        /// \brief Render 3D coordinates of the joints relative to the camera coordinates. The order is X, Y, Z, where Z is the distance to the camera.
        /// \param skeletons [in] skeletons to render
        /// \param nImageWidth [in] image width in pixel
        /// \param graphics [in] graphics object to display the skeletons
        public void renderCoordinates(List<SkeletonKeypoints> skeletons, int nImageWidth, Graphics graphics)
        {
            foreach (var skeleton in skeletons) {
                foreach (Coordinate coordinate in skeleton.listJoints) {
                    if (coordinate.x <= 0 || coordinate.y <= 0)
                        continue;

                    // manage the drawing here
                    float[, ] depthValues = DepthMapHelpers.getDepthInKernel(
                      depthFrame, (int) coordinate.x, (int) coordinate.y, nKernelSize : 5);

                    float averageDepth = DepthMapHelpers.averageValidDepthFromNeighbourhood(depthValues);
                    if (averageDepth > 0) {
                        System.Windows.Media.Media3D.Point3D worldCoordinates =
                          DepthMapHelpers.WorldCoordinate(averageDepth,
                                                          (int) coordinate.x,
                                                          (int) coordinate.y,
                                                          intrinsicsDepthImagerMaster.fx,
                                                          intrinsicsDepthImagerMaster.fy,
                                                          intrinsicsDepthImagerMaster.ppx,
                                                          intrinsicsDepthImagerMaster.ppy);

                        string coordinateAsString = String.Format(
                          "({0:F2}, {1:F2}, {2:F2})", worldCoordinates.X, worldCoordinates.Y, worldCoordinates.Z);
                        graphics.DrawString(coordinateAsString,
                                            new Font("Avenir", (int)(nImageWidth / 106)),
                                            Brushes.GreenYellow,
                                            new PointF((float) coordinate.x, (float) coordinate.y));
                    }
                }
            }
        }
    }
}
