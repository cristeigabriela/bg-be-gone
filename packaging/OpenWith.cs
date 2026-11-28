// Open a file with a given ProgID's "open" verb, the way Explorer's Open with
// menu does: ShellExecuteEx with an explicit class. test-windows-launch.ps1
// uses it to start bg-be-gone through the registration Setup wrote.
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

namespace BgBeGone
{
    public static class OpenWith
    {
        /// <summary>SHELLEXECUTEINFOW, <see href="https://learn.microsoft.com/windows/win32/api/shellapi/ns-shellapi-shellexecuteinfow"/>.</summary>
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        private struct SHELLEXECUTEINFOW
        {
            public int cbSize;
            public uint fMask;
            public IntPtr hwnd;
            public string lpVerb;
            public string lpFile;
            public string lpParameters;
            public string lpDirectory;
            public int nShow;
            public IntPtr hInstApp;
            public IntPtr lpIDList;
            public string lpClass;
            public IntPtr hkeyClass;
            public uint dwHotKey;
            public IntPtr hIcon;
            public IntPtr hProcess;
        }

        /// <summary>Use lpClass as the file's class (shellapi.h), <see href="https://learn.microsoft.com/windows/win32/api/shellapi/ns-shellapi-shellexecuteinfow"/>.</summary>
        private const uint SEE_MASK_CLASSNAME = 0x00000001;

        /// <summary>Wait for any DDE conversation to finish before returning (shellapi.h), same page.</summary>
        private const uint SEE_MASK_NOASYNC = 0x00000100;

        /// <summary>SW_SHOWNORMAL (winuser.h), <see href="https://learn.microsoft.com/windows/win32/api/winuser/nf-winuser-showwindow"/>.</summary>
        private const int SW_SHOWNORMAL = 1;

        [DllImport("shell32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern bool ShellExecuteExW(ref SHELLEXECUTEINFOW info);

        /// <summary>Run <paramref name="progId"/>'s open command on <paramref name="file"/>.</summary>
        public static void Open(string file, string progId)
        {
            var info = new SHELLEXECUTEINFOW
            {
                fMask = SEE_MASK_CLASSNAME | SEE_MASK_NOASYNC,
                lpVerb = "open",
                lpFile = file,
                lpClass = progId,
                nShow = SW_SHOWNORMAL,
            };
            info.cbSize = Marshal.SizeOf(info);
            if (!ShellExecuteExW(ref info))
            {
                throw new Win32Exception(Marshal.GetLastWin32Error(), "ShellExecuteEx failed for " + file);
            }
        }
    }
}
