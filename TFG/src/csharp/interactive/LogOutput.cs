
using System;
using System.IO;
using System.Text;
using System.Windows.Controls;


namespace Cubemos.Samples
{
    public class LogOutput : TextWriter
    {
        TextBox textBox = null;

        /// 
        /// \brief Sets the output for sample logs
        /// \param output [in] text box for logs
        /// 
        public LogOutput(TextBox output)
        {
            textBox = output;
        }

        /// 
        /// \brief Write a single character to the output
        /// \param value [in] character to write
        public override void Write(char value)
        {
            base.Write(value);
            textBox.Dispatcher.BeginInvoke(new Action(() =>
            {
                textBox.AppendText(value.ToString());
                textBox.ScrollToEnd();
            }));
        }

        public override Encoding Encoding
        {
            get { return System.Text.Encoding.UTF8; }
        }
    }
}