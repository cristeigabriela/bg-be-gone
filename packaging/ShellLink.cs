// Shortcuts (.lnk) through the shell's own IShellLinkW, for BgBeGone.Windows.psm1.
//
// Both WScript.Shell's shortcut object and IShellLinkW::SetPath expand %VAR%
// in a target path, so a folder literally named "%TEMP%" turns into a bad
// path and is rejected. Save() names the target by an item ID list parsed
// from the literal path instead, and clears the "expand this" flags the shell
// sets on the icon and target, so every path is used exactly as written.
using System;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;
using System.Text;

namespace BgBeGone
{
    // CLSID_ShellLink, IID_IShellLinkW and IID_IShellLinkDataList from ShObjIdl_core.h
    // and ShlObj_core.h; .NET ships no binding for them.
    [ComImport, Guid("00021401-0000-0000-C000-000000000046")]
    class CShellLink { }

    [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("000214F9-0000-0000-C000-000000000046")]
    interface IShellLinkW
    {
        void GetPath([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder pszFile, int cch, IntPtr pfd, uint fFlags);
        void GetIDList(out IntPtr ppidl);
        void SetIDList(IntPtr pidl);
        void GetDescription([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder pszName, int cch);
        void SetDescription([MarshalAs(UnmanagedType.LPWStr)] string pszName);
        void GetWorkingDirectory([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder pszDir, int cch);
        void SetWorkingDirectory([MarshalAs(UnmanagedType.LPWStr)] string pszDir);
        void GetArguments([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder pszArgs, int cch);
        void SetArguments([MarshalAs(UnmanagedType.LPWStr)] string pszArgs);
        void GetHotkey(out short pwHotkey);
        void SetHotkey(short wHotkey);
        void GetShowCmd(out int piShowCmd);
        void SetShowCmd(int iShowCmd);
        void GetIconLocation([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder pszIconPath, int cch, out int piIcon);
        void SetIconLocation([MarshalAs(UnmanagedType.LPWStr)] string pszIconPath, int iIcon);
        void SetRelativePath([MarshalAs(UnmanagedType.LPWStr)] string pszPathRel, uint dwReserved);
        void Resolve(IntPtr hwnd, uint fFlags);
        void SetPath([MarshalAs(UnmanagedType.LPWStr)] string pszFile);
    }

    [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("45E2B4AE-B1C3-11D0-B92F-00A0C90312E1")]
    interface IShellLinkDataList
    {
        void AddDataBlock(IntPtr pDataBlock);
        void CopyDataBlock(uint dwSig, out IntPtr ppDataBlock);
        void RemoveDataBlock(uint dwSig);
        void GetFlags(out uint pdwFlags);
        void SetFlags(uint dwFlags);
    }

    public static class ShellLink
    {
        // ShlObj_core.h: SHELL_LINK_DATA_FLAGS and the data block signatures
        // that hold the to-be-expanded copies of the target and the icon.
        const uint SLDF_HAS_EXP_SZ = 0x00000200;
        const uint SLDF_HAS_EXP_ICON_SZ = 0x00004000;
        const uint EXP_SZ_LINK_SIG = 0xA0000001;
        const uint EXP_SZ_ICON_SIG = 0xA0000007;
        // ShObjIdl_core.h: SLGP_FLAGS.SLGP_RAWPATH, the path as stored, unexpanded.
        const uint SLGP_RAWPATH = 0x4;
        const int MaxPath = 32768;

        public static void Save(string lnk, string target, string arguments, string workingDirectory,
                                string icon, string description)
        {
            var link = (IShellLinkW)new CShellLink();
            IntPtr pidl = ILCreateFromPathW(target);
            if (pidl == IntPtr.Zero) throw new System.IO.FileNotFoundException("no such program", target);
            try { link.SetIDList(pidl); } finally { ILFree(pidl); }
            link.SetArguments(arguments);
            link.SetWorkingDirectory(workingDirectory);
            link.SetIconLocation(icon, 0);
            link.SetDescription(description);
            var data = (IShellLinkDataList)link;
            uint flags;
            data.GetFlags(out flags);
            if ((flags & SLDF_HAS_EXP_SZ) != 0) data.RemoveDataBlock(EXP_SZ_LINK_SIG);
            if ((flags & SLDF_HAS_EXP_ICON_SZ) != 0) data.RemoveDataBlock(EXP_SZ_ICON_SIG);
            data.SetFlags(flags & ~(SLDF_HAS_EXP_SZ | SLDF_HAS_EXP_ICON_SZ));
            ((IPersistFile)link).Save(lnk, true);
        }

        [DllImport("shell32.dll", CharSet = CharSet.Unicode)]
        static extern IntPtr ILCreateFromPathW(string pszPath);

        [DllImport("shell32.dll")]
        static extern void ILFree(IntPtr pidl);

        // [target, arguments] of a saved shortcut, both as stored.
        public static string[] Read(string lnk)
        {
            var link = (IShellLinkW)new CShellLink();
            ((IPersistFile)link).Load(lnk, 0);
            var target = new StringBuilder(MaxPath);
            link.GetPath(target, target.Capacity, IntPtr.Zero, SLGP_RAWPATH);
            var arguments = new StringBuilder(MaxPath);
            link.GetArguments(arguments, arguments.Capacity);
            return new[] { target.ToString(), arguments.ToString() };
        }
    }
}
