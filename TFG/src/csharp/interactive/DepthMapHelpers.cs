using System;
using System.Collections.Generic;

namespace Cubemos.Samples
{
    /// \brief Helper functions for working with depth maps and 3D-2D transformations
    public partial class DepthMapHelpers
    {
        /// \brief Returns average depth value of a square region in the depth map only taking into account valid depth values
        /// \param depthKernel [in] square region of depth values
        public static float averageValidDepthFromNeighbourhood(float[,] depthKernel)
        {
            float average = 0;
            int nValidDepths = 0;
            for(int row = 0; row < depthKernel.GetLength(0); row++)
            {
                for (int col = 0; col < depthKernel.GetLength(1); col++)
                {
                    float depth = depthKernel[row, col];
                    if (depth <= 0.0001)
                        continue;

                    average += depth;
                    nValidDepths++;
                }
            }

            if (nValidDepths == 0)
                return 0.0f;
            else
                return average / nValidDepths;
        }

        /// \brief Returns depth values for a square region with the side of the nKernelSize centered on the specified coordinate
        /// \param depthFrame [in] depth map containing float values of distances in mm for every image pixel
        /// \param nCol [in] x coordinate of the region center
        /// \param nRow [in] y coordinate of the region center
        /// \param nKernelSize [in] side length of the region, e.g. nKernelSize = 3 gives a square region 3x3 
        public static float[,] getDepthInKernel(Intel.RealSense.DepthFrame depthFrame, int nCol, int nRow, int nKernelSize)
        {
            if (nCol >= depthFrame.Width || nRow >= depthFrame.Height || nCol < 0 || nRow < 0)
                throw new IndexOutOfRangeException(String.Format("Requested coordinages x: {0}, y: {1} out of CM_Image Range: {2}*{3}", nCol, nRow, depthFrame.Width, depthFrame.Height));
            
            uint kernelSizeHalf = (uint)(nKernelSize / 2);
            
            uint unStartCol = Math.Max(0, (uint)(nCol - kernelSizeHalf));
            uint unEndCol = Math.Min((uint)depthFrame.Width, (uint)(nCol + kernelSizeHalf));

            uint unStartRow = Math.Max(0, (uint)(nRow - kernelSizeHalf));
            uint unEndRow = Math.Min((uint)depthFrame.Height, (uint)(nRow + kernelSizeHalf));

            
            float[,] depthNeigbourhood = new float[unEndRow - unStartRow + 1, unEndCol - unStartCol + 1];
            for (uint i = unStartCol; i <= unEndCol; i++)
            {
                for (uint j = unStartRow; j <= unEndRow; j++)
                {
                    float depth = depthFrame.GetDistance((int)i, (int)j);

                    depthNeigbourhood[j - unStartRow, i - unStartCol] = depth;
                }
            }

            return depthNeigbourhood;
        }

        /// \brief Calculates 3D world coordinates relative to the camera for a point in image coordinates (2D) and camera intrinsics
        /// \param depth [in] distance from the camera center for this image point
        /// \param pixelX [in] x image coordinate (column)
        /// \param pixelY [in] y image coordinate (row)
        /// \param focalLengthX [in] focal length
        /// \param focalLengthY [in] focal length
        /// \param centerX [in] x coordinate of the camera center
        /// \param centerY [in] y coordinate of the camera center
        public static System.Windows.Media.Media3D.Point3D WorldCoordinate(float depth, int pixelX, int pixelY, float focalLengthX, float focalLengthY, float centerX, float centerY)
        {
            float Z = depth;

            // calculate x and y
            // x = (u - ppx) * z / f;

            float X = ((float)pixelX - centerX) * depth / focalLengthX;
            float Y = ((float)pixelY - centerY) * depth / focalLengthY;

            return new System.Windows.Media.Media3D.Point3D(X, Y, Z);
        }

        /// \brief Calculates 2D image coordinates in pixels from 3D world coordinates of a single point and camera intrinsics
        /// \param worldCoordinates [in] 3D coordinates in the camera coordinates system
        /// \param focalLengthX [in] focal length
        /// \param focalLengthY [in] focal length
        /// \param centerX [in] x coordinate of the camera center
        /// \param centerY [in] y coordinate of the camera center
        public static System.Windows.Point getImageCoordinates(System.Windows.Media.Media3D.Point3D worldCoordinates, float focalLengthX, float focalLengthY, float centerX, float centerY)
        {
            
            double x = worldCoordinates.X * focalLengthX / worldCoordinates.Z + centerX;
            double y = worldCoordinates.Y * focalLengthY / worldCoordinates.Z + centerY;

            return new System.Windows.Point(x,y);
        }

        /// \brief Calculates 2D image coordinates in pixel for a list of 3D world coordinates and camera intrinsics 
        /// \param worldCoordinates [in] 3D coordinates in the camera coordinates system
        /// \param focalLengthX [in] focal length
        /// \param focalLengthY [in] focal length
        /// \param centerX [in] x coordinate of the camera center
        /// \param centerY [in] y coordinate of the camera center
        public static List<System.Windows.Point> getImageCoordinates(List<System.Windows.Media.Media3D.Point3D> worldCoordinates, float focalLengthX, float focalLengthY, float centerX, float centerY)
        {
            List<System.Windows.Point> points = new List<System.Windows.Point>();
            foreach(var worldCoord in worldCoordinates)
            {
                points.Add(getImageCoordinates(worldCoord, focalLengthX, focalLengthY, centerX, centerY));
            }

            return points;
        }
    }
}
