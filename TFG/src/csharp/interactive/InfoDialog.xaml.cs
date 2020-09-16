using System;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Controls;
using System.Drawing;
using System.Linq;
using Cubemos;
using System.Windows.Media.Imaging;
using System.IO;
using System.Collections.Generic;
using System.Diagnostics;
using System.Windows.Shapes;


namespace Cubemos.Samples
{
    /// 
    /// \brief Message box class
    public partial class InfoDialog : Window
    {

        /// 
        /// \brief InfoDialog Constructor
        /// \param question [in] message text inside the box
        /// \param title [in] title of the message box
        public InfoDialog(string question, string title)
        {
            InitializeComponent();
            this.Loaded += new RoutedEventHandler(PromptDialog_Loaded);
            txtQuestion.FontSize = 13;
            txtQuestion.Text = question;
            Title = title;
            
        }

        void PromptDialog_Loaded(object sender, RoutedEventArgs e)
        {
            
        }

        /// 
        /// \brief Displays a message box
        /// \param question [in] message text inside the box
        /// \param title [in] title of the message box
        public static string Prompt(string question, string title)
        {
            InfoDialog inst = new InfoDialog(question, title);
            inst.ShowDialog();
            inst.Focus();

            return null;
        }


        private void btnOk_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = true;
            Close();
        }

        private void btnCancel_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }

        public delegate void ShowInfoDialogDelegate(string title, string msg);


        /// \brief Info Dialog
        /// \param title [in] dialog title
        /// \param msg [in] message to display
        public static void ShowInfoDialog(string title, string msg)
        {
            var ob = InfoDialog.Prompt(msg, title);
            Environment.Exit(0);
        }        

    }
}
