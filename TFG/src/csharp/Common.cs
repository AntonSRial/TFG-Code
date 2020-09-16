using System;

namespace Cubemos.Samples
{
    public static class Common
    {
        public static string DefaultArtifactsDir()
        {
            return Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData) + "\\Cubemos\\SkeletonTracking";
        }

        public static string DefaultLogDir()
        {
            return DefaultArtifactsDir() + "\\logs";
        }

        public static string DefaultLicenseDir()
        {
            return DefaultArtifactsDir() + "\\license";
        }

        public static string DefaultModelDir()
        {
            return DefaultArtifactsDir() + "\\models";
        }

        public static string DefaultResDir()
        {
            return DefaultArtifactsDir() + "\\res";
        }
    }
}