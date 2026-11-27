; Inno Setup script for bg-be-gone (Windows).
; Compiled by packaging/package-windows.ps1, which passes these defines:
;   AppVer, Variant, Bundle (dist\BgBeGone), Icon (.ico), OutDir, OutName
; Per-user install (no admin): the frontend runs the bundled mingw pythonw via
; BgBeGone.vbs; models download to %LOCALAPPDATA% at runtime.

#ifndef AppVer
  #define AppVer "0.0.0"
#endif
#ifndef Variant
  #define Variant "directml"
#endif

[Setup]
AppId={{4B1F6C2A-9E7D-4E3B-8C21-BG0BEG0NEWIN}
AppName=bg-be-gone
AppVersion={#AppVer}
AppPublisher=cristeigabriela
AppPublisherURL=https://github.com/cristeigabriela/bg-be-gone
DefaultDirName={autopf}\bg-be-gone
DefaultGroupName=bg-be-gone
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir={#OutDir}
OutputBaseFilename={#OutName}
SetupIconFile={#Icon}
UninstallDisplayIcon={app}\bg-be-gone.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "{#Bundle}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
; Shortcuts target the .vbs so no console window appears; wscript runs it.
Name: "{group}\bg-be-gone"; Filename: "{app}\BgBeGone.vbs"; WorkingDir: "{app}"; IconFilename: "{app}\bg-be-gone.ico"
Name: "{group}\Uninstall bg-be-gone"; Filename: "{uninstallexe}"
Name: "{userdesktop}\bg-be-gone"; Filename: "{app}\BgBeGone.vbs"; WorkingDir: "{app}"; IconFilename: "{app}\bg-be-gone.ico"; Tasks: desktopicon

[Registry]
; A ProgID + OpenWithProgids entries so bg-be-gone appears under "Open with"
; for common image types WITHOUT hijacking the user's default handler.
Root: HKCU; Subkey: "Software\Classes\bg-be-gone.Image"; ValueType: string; ValueName: ""; ValueData: "bg-be-gone image"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\bg-be-gone.Image\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\bg-be-gone.ico"
Root: HKCU; Subkey: "Software\Classes\bg-be-gone.Image\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{sys}\wscript.exe"" ""{app}\BgBeGone.vbs"" ""%1"""
Root: HKCU; Subkey: "Software\Classes\.png\OpenWithProgids";  ValueType: string; ValueName: "bg-be-gone.Image"; ValueData: ""; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.jpg\OpenWithProgids";  ValueType: string; ValueName: "bg-be-gone.Image"; ValueData: ""; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.jpeg\OpenWithProgids"; ValueType: string; ValueName: "bg-be-gone.Image"; ValueData: ""; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.webp\OpenWithProgids"; ValueType: string; ValueName: "bg-be-gone.Image"; ValueData: ""; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.bmp\OpenWithProgids";  ValueType: string; ValueName: "bg-be-gone.Image"; ValueData: ""; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.tif\OpenWithProgids";  ValueType: string; ValueName: "bg-be-gone.Image"; ValueData: ""; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.tiff\OpenWithProgids"; ValueType: string; ValueName: "bg-be-gone.Image"; ValueData: ""; Flags: uninsdeletevalue

[Run]
Filename: "{app}\BgBeGone.vbs"; Description: "Launch bg-be-gone"; Flags: postinstall nowait skipifsilent shellexec
