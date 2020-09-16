using System;
using System.Runtime.InteropServices;
using System.Linq;

namespace Cubemos.Samples
{
    class Program
    {
        [DllImport("kernel32.dll", SetLastError = true, ExactSpelling = true)]
        static extern bool FreeConsole();

        [STAThread]
        static void Main(string[] args)
        {
            FreeConsole();

            var w = new CaptureWindow();

            w.ShowDialog();
        }
    }
}
