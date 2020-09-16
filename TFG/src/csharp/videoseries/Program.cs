using System;

// This sample shows how to use the cubemos Skeleton Tracking C# API in a console application
namespace Cubemos.Samples
{
    class Program {
        static void Main(string[] args)
        {
            Console.WriteLine("Skeleton tracking will be run on a pair of two images in the samples folder. ");

            // Initialize logging to output all messages with severity level INFO or higher to the console
            Cubemos.Api.InitialiseLogging(Cubemos.LogLevel.CM_LL_INFO, bWriteToConsole : true);

            // Create cubemos Skeleton tracking Api handle and specify the valid cubemos activation key file
            Cubemos.SkeletonTracking.Api skeletontrackingApi =
              new Cubemos.SkeletonTracking.Api(Common.DefaultLicenseDir());

            // Initialise cubemos SkeletonTracking framework with the required model
            skeletontrackingApi.LoadModel(Cubemos.TargetComputeDevice.CM_CPU,
                                          Common.DefaultModelDir() + "\\fp32\\skeleton-tracking.cubemos");

            // Read an RGB image of any size
            String cubemosSampleImagesDir = Common.DefaultResDir() + "\\images";
            System.Drawing.Bitmap image1 =
              new System.Drawing.Bitmap(cubemosSampleImagesDir + ".\\skeleton_tracking_1.jpg");
            System.Drawing.Bitmap image2 =
              new System.Drawing.Bitmap(cubemosSampleImagesDir + ".\\skeleton_tracking_2.jpg");

            // Set the network input size to 256 for better accuracy
            int networkHeight = 256;
            // Send inference request and get the poses
            // Results container
            System.Collections.Generic.List<Cubemos.SkeletonTracking.Api.SkeletonKeypoints> lastResult;
            // ID of the image pipeline required to group images for tracking
            int pipelineID = 1;

            skeletontrackingApi.RunSkeletonTracking(ref image1, networkHeight, out lastResult, pipelineID);
            skeletontrackingApi.RunSkeletonTracking(ref image2, networkHeight, out lastResult, pipelineID);

            // Output detected skeletons
            Console.WriteLine("# Persons detected: " + lastResult.Count);
            for (int skeleton_index = 0; skeleton_index < lastResult.Count; skeleton_index++) {
                var skeleton = lastResult[skeleton_index];
                Console.WriteLine("Skeleton #" + skeleton_index + " Tracking ID " + skeleton.id);
                for (int joint_index = 0; joint_index < skeleton.listJoints.Count; joint_index++) {
                    Cubemos.SkeletonTracking.Api.Coordinate coordinate = skeleton.listJoints[joint_index];
                    Console.WriteLine("\tJoint coordinate #" + joint_index + ": " + coordinate.x + "; " + coordinate.y);
                }
            }

            Console.Write("Press <Enter> to exit... ");
            Console.ReadLine();
        }
    }
}
