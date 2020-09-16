using System;

///
/// \brief This sample shows how to use the cubemos Skeleton Tracking C# API in a console application
///
namespace Cubemos.Samples
{
    class Program {
        static void Main(string[] args)
        {
            Console.WriteLine(
              "Skeleton tracking will be run on an image saved at %LOCALAPPDATA%\\res\\images\\skeleton_estimation.jpg... ");

            // Initialize logging to output all messages with severity level INFO or higher to the console
            Cubemos.Api.InitialiseLogging(Cubemos.LogLevel.CM_LL_INFO, bWriteToConsole : true);

            // Create cubemos Skeleton tracking Api handle and specify the directory containing a valid
            // cubemos_license.json or activation_key.json file
            Cubemos.SkeletonTracking.Api skeletontrackingApi;
            try {
                skeletontrackingApi = new Cubemos.SkeletonTracking.Api(Common.DefaultLicenseDir());
            }
            catch (Cubemos.Exception ex) {
                Console.WriteLine("If you haven't activated the SDK yet, please run post_installation script as described in the Getting Started Guide to activate your license.");
                Console.ReadLine();
                return;
            }

            // Initialise cubemos SkeletonTracking framework with the required model
            try
            {
                skeletontrackingApi.LoadModel(Cubemos.TargetComputeDevice.CM_CPU,
                                              Common.DefaultModelDir() + "\\fp32\\skeleton-tracking.cubemos");
            }
            catch (Cubemos.Exception ex)
            {
                Console.WriteLine(String.Format("Error during model loading. " +
                      "Please verify the model exists at the path {0}. Details: {1}", Common.DefaultModelDir() + "\\fp32\\skeleton-tracking.cubemos", ex));
                Console.ReadLine();
                return;
            }

            // Read an RGB image of any size
            System.Drawing.Bitmap inputImage =
              new System.Drawing.Bitmap(Common.DefaultResDir() + ".\\images\\skeleton_estimation.jpg");

            // Set the network input size to 128 for faster inference
            int networkHeight = 128;
            System.Collections.Generic.List<Cubemos.SkeletonTracking.Api.SkeletonKeypoints> skeletonKeypoints;
            // Send inference request and get the poses
            skeletontrackingApi.RunSkeletonTracking(ref inputImage, networkHeight, out skeletonKeypoints);

            // Output detected skeletons
            Console.WriteLine("# Persons detected: " + skeletonKeypoints.Count);
            for (int skeleton_index = 0; skeleton_index < skeletonKeypoints.Count; skeleton_index++) {
                var skeleton = skeletonKeypoints[skeleton_index];
                Console.WriteLine("Skeleton #" + skeleton_index);
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
