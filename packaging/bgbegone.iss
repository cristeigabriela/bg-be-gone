; Inno Setup script for bg-be-gone (Windows).
; Compiled by packaging/package-windows.ps1, which passes these defines:
;   AppVer, Variant, Bundle (dist\BgBeGone), Icon (.ico), OutDir, OutName
;
; Per-user install by default (no admin). The release lives in {app}\runtime:
; Setup extracts each one into a new folder beside it, and
; tools\setup-runtime.ps1 probes it and swaps it in, or removes it and keeps
; the old one. Uninstall removes the runtime by its inventory, so files
; bg-be-gone didn't write stay. Shortcuts and Open with run the bundled
; MSYS2 pythonw.exe on the bg_be_gone_launch module; models download to
; %LOCALAPPDATA%\bg-be-gone.

#ifndef AppVer
  #define AppVer "0.0.0"
#endif
#ifndef Variant
  #define Variant "directml"
#endif
; The longest path inside the bundle, measured by package-windows.ps1.
#ifndef LongestPath
  #define LongestPath "0"
#endif

; A test build (package-windows.ps1 -TestName X) installs as an app of its
; own: its AppId, Start Menu entry, Open with ProgID and default folder, so
; it can't change the real install's.
#ifdef TestName
  #define AppKey "bg-be-gone-" + TestName
  #define AppIdValue "io.github.cristeigabriela.BgBeGone." + TestName
#else
  #define AppKey "bg-be-gone"
  #define AppIdValue "io.github.cristeigabriela.BgBeGone"
#endif
#define ProgId AppKey + ".Image"

#define Launcher "{app}\runtime\bin\pythonw.exe"
#define LaunchArgs "-m bg_be_gone_launch"
#define AppIcon "{app}\runtime\bg-be-gone.ico"
#define PowerShell "{sys}\WindowsPowerShell\v1.0\powershell.exe"
; Bypass is scoped to the one process: Windows' default policy refuses every .ps1.
#define PsArgs "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File"

[Setup]
AppId={#AppIdValue}
AppName={#AppKey}
AppVersion={#AppVer}
AppVerName={#AppKey} {#AppVer} ({#Variant})
AppPublisher=cristeigabriela
AppPublisherURL=https://github.com/cristeigabriela/bg-be-gone
DefaultDirName={autopf}\{#AppKey}
DefaultGroupName={#AppKey}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir={#OutDir}
OutputBaseFilename={#OutName}
SetupIconFile={#Icon}
UninstallDisplayIcon={#AppIcon}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked
#if Variant != "seg"
Name: "model"; Description: "Download the default background-removal model now (BiRefNet General, about 1 GB). Otherwise the app downloads it on first use."; GroupDescription: "Models:"
#endif
#if Variant == "directml"
; Ticked by code below when an RX 6700/6750 XT (gfx1031) is present.
Name: "rocm"; Description: "Run BiRefNet on PyTorch with AMD ROCm (RX 6700/6750 XT only; downloads about 4 GB)"; GroupDescription: "AMD GPU:"; Flags: unchecked
#endif

[Files]
; Into a new folder of this run's own; CurStepChanged swaps it in. Setup's
; own uninstall leaves these alone: the files move, and setup-runtime.ps1
; removes the runtime by its inventory.
Source: "{#Bundle}\*"; DestDir: "{code:StagingDir}"; Flags: recursesubdirs createallsubdirs ignoreversion uninsneveruninstall

[Icons]
; No working folder: the shell would expand a %NAME% in it.
Name: "{group}\{#AppKey}"; Filename: "{#Launcher}"; Parameters: "{#LaunchArgs}"; IconFilename: "{#AppIcon}"
Name: "{group}\Uninstall {#AppKey}"; Filename: "{uninstallexe}"
Name: "{userdesktop}\{#AppKey}"; Filename: "{#Launcher}"; Parameters: "{#LaunchArgs}"; IconFilename: "{#AppIcon}"; Tasks: desktopicon

[Registry]
; A ProgID and OpenWithProgids entries, so bg-be-gone appears under Open with
; for the image types it reads, without taking over the default handler.
Root: HKCU; Subkey: "Software\Classes\{#ProgId}"; ValueType: string; ValueName: ""; ValueData: "bg-be-gone image"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\{#ProgId}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{#AppIcon}"
Root: HKCU; Subkey: "Software\Classes\{#ProgId}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{#Launcher}"" {#LaunchArgs} ""%1"""
Root: HKCU; Subkey: "Software\Classes\.png\OpenWithProgids";  ValueType: string; ValueName: "{#ProgId}"; ValueData: ""; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.jpg\OpenWithProgids";  ValueType: string; ValueName: "{#ProgId}"; ValueData: ""; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.jpeg\OpenWithProgids"; ValueType: string; ValueName: "{#ProgId}"; ValueData: ""; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.webp\OpenWithProgids"; ValueType: string; ValueName: "{#ProgId}"; ValueData: ""; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.gif\OpenWithProgids";  ValueType: string; ValueName: "{#ProgId}"; ValueData: ""; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.avif\OpenWithProgids"; ValueType: string; ValueName: "{#ProgId}"; ValueData: ""; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.heic\OpenWithProgids"; ValueType: string; ValueName: "{#ProgId}"; ValueData: ""; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.heif\OpenWithProgids"; ValueType: string; ValueName: "{#ProgId}"; ValueData: ""; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.bmp\OpenWithProgids";  ValueType: string; ValueName: "{#ProgId}"; ValueData: ""; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.tif\OpenWithProgids";  ValueType: string; ValueName: "{#ProgId}"; ValueData: ""; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.tiff\OpenWithProgids"; ValueType: string; ValueName: "{#ProgId}"; ValueData: ""; Flags: uninsdeletevalue

[Run]
#if Variant == "directml"
; In a visible console, so the 4 GB download shows its progress. On any
; failure the directml worker stays as it was.
Filename: "{#PowerShell}"; Parameters: "{#PsArgs} ""{app}\runtime\tools\Enable-AmdRocm.ps1"" -WaitOnExit"; WorkingDir: "{app}"; StatusMsg: "Enabling AMD ROCm (downloading PyTorch, about 4 GB)..."; Tasks: rocm; Check: RuntimeInstalled; Flags: waituntilterminated
; The PyTorch engine loads BiRefNet's original checkpoint, not rembg's ONNX file.
Filename: "{app}\runtime\worker\python.exe"; Parameters: "-I ""{app}\runtime\app\src\bgbg\plat\windows.py"" ""{app}\runtime\app\src\bgbg\prefetch.py"" torch birefnet-general ""{localappdata}\bg-be-gone\models\birefnet"""; WorkingDir: "{app}"; StatusMsg: "Downloading BiRefNet General (about 1 GB)..."; Tasks: rocm and model; Check: RuntimeInstalled; Flags: waituntilterminated runhidden
#endif
Filename: "{#Launcher}"; Parameters: "{#LaunchArgs}"; Description: "Launch bg-be-gone"; Check: RuntimeInstalled; Flags: postinstall nowait skipifsilent

[UninstallRun]
; From {app}, not from the runtime: a process's current folder can't be removed.
Filename: "{#PowerShell}"; Parameters: "{#PsArgs} ""{app}\runtime\tools\setup-runtime.ps1"" -Action Uninstall -App ""{app}"""; WorkingDir: "{app}"; Flags: runhidden waituntilterminated; RunOnceId: "RemoveRuntime"

[Code]
const
  // rembg 2.0.76's own URL and checksum for birefnet-general (the version the
  // worker locks pin). rembg re-checks the file when it loads it.
  ModelUrl = 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx';
  ModelMd5 = '7a35a0141cbbc80de11d9c9a28f52697';
  ModelFile = 'birefnet-general.onnx';
  InventoryName = 'bg-be-gone-files.txt';
  NotOurFolder = 'This folder is not empty and holds no bg-be-gone install. ' +
                 'Choose a new or empty folder, or the one bg-be-gone is installed in.';

type
  // Where the runtime swap got to: not yet run, in place, or refused.
  TRuntimeState = (rsPending, rsInstalled, rsFailed);

var
  DownloadPage: TDownloadWizardPage;
  ModelDownloaded: Boolean;
  RocmPreset: Boolean;
  Staging: String;
  RuntimeState: TRuntimeState;

function IsEmptyDir(const Dir: String): Boolean;
var
  FindRec: TFindRec;
begin
  Result := True;
  if FindFirst(AddBackslash(Dir) + '*', FindRec) then
  try
    repeat
      if (FindRec.Name <> '.') and (FindRec.Name <> '..') then begin
        Result := False;
        Break;
      end;
    until not FindNext(FindRec);
  finally
    FindClose(FindRec);
  end;
end;

// A new or empty folder, or one with bg-be-gone's runtime in it. Setup only
// ever removes files a runtime's inventory lists, but it won't move into a
// folder that is someone else's.
function IsOurFolder(const Dir: String): Boolean;
begin
  Result := (not DirExists(Dir)) or IsEmptyDir(Dir) or
            FileExists(AddBackslash(Dir) + 'runtime\' + InventoryName);
end;

// {app}\runtime.new-<8 hex>, a name no folder has yet; fixed for the run.
function StagingDir(Param: String): String;
begin
  if Staging = '' then
    repeat
      Staging := ExpandConstant('{app}\runtime.new-') + Format('%.8x', [Random($7FFFFFFF)]);
    until not DirExists(Staging);
  Result := Staging;
end;

function RuntimeInstalled: Boolean;
begin
  Result := RuntimeState = rsInstalled;
end;

// The RX 6700/6750 XT (Navi 22, gfx1031): the one card with a tested ROCm lock.
function HasGfx1031: Boolean;
var
  Locator, Service, Items, Item: Variant;
  I: Integer;
  Id: String;
begin
  Result := False;
  try
    Locator := CreateOleObject('WbemScripting.SWbemLocator');
    Service := Locator.ConnectServer('.', 'root\cimv2');
    Items := Service.ExecQuery('SELECT PNPDeviceID FROM Win32_VideoController');
    for I := 0 to Items.Count - 1 do begin
      Item := Items.ItemIndex(I);
      if VarIsNull(Item.PNPDeviceID) then
        Continue;
      Id := Item.PNPDeviceID;
      Id := Uppercase(Id);
      if (Pos('VEN_1002&DEV_73DF', Id) > 0) or (Pos('VEN_1002&DEV_73C3', Id) > 0) then
        Result := True;
    end;
  except
    Log('GPU query failed: ' + GetExceptionMessage);
  end;
end;

function ModelTarget: String;
begin
  Result := ExpandConstant('{localappdata}\bg-be-gone\u2net\') + ModelFile;
end;

function WantsOnnxModel: Boolean;
begin
  Result := False;
#if Variant != "seg"
  Result := WizardIsTaskSelected('model');
#endif
#if Variant == "directml"
  Result := Result and not WizardIsTaskSelected('rocm');
#endif
end;

procedure InitializeWizard;
begin
  DownloadPage := CreateDownloadPage('Downloading the default model',
    'BiRefNet General, about 1 GB. Abort skips it; bg-be-gone then downloads it on first use.', nil);
end;

procedure CurPageChanged(CurPageID: Integer);
begin
#if Variant == "directml"
  // Once, so a user who unticks it and goes back keeps their choice.
  if (CurPageID = wpSelectTasks) and not RocmPreset then begin
    RocmPreset := True;
    if HasGfx1031 then
      WizardSelectTasks('rocm');
  end;
#endif
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (CurPageID = wpSelectDir) and not IsOurFolder(WizardDirValue) then begin
    MsgBox(NotOurFolder, mbError, MB_OK);
    Result := False;
    Exit;
  end;
  // Before the files go in, into {tmp}; ssPostInstall moves it into place.
  // A failed or aborted download never stops the install.
  if (CurPageID = wpReady) and WantsOnnxModel and not FileExists(ModelTarget) then begin
    DownloadPage.Clear;
    DownloadPage.Add(ModelUrl, ModelFile, '');
    DownloadPage.Show;
    try
      try
        DownloadPage.Download;
        ModelDownloaded := GetMD5OfFile(ExpandConstant('{tmp}\') + ModelFile) = ModelMd5;
        if not ModelDownloaded then
          Log('The model download failed its MD5 check; the app will download it again.');
      except
        Log('Model download skipped: ' + GetExceptionMessage);
      end;
    finally
      DownloadPage.Hide;
    end;
  end;
end;

// The same guard for /SILENT and /DIR=, which skip the wizard pages and so
// NextButtonClick. It runs before anything is written.
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Room: Integer;
begin
  Result := '';
  if not IsOurFolder(ExpandConstant('{app}')) then begin
    Result := NotOurFolder;
    Exit;
  end;
  // Windows refuses paths of 260 characters and more (MAX_PATH), and the
  // runtime's longest file sits {#LongestPath} characters below the staging folder.
  Room := 259 - {#LongestPath} - Length(AddBackslash(StagingDir('')));
  if Room < 0 then
    Result := Format('This folder''s path is too long for bg-be-gone''s files. ' +
                     'Choose one of at most %d characters.', [Length(ExpandConstant('{app}')) + Room]);
end;

// Probe the extracted release and swap it in (tools\setup-runtime.ps1). Its
// output goes to Setup's log, and its last line to the user on failure.
procedure SwitchRuntime;
var
  Params, LogFile, Why: String;
  Lines: TArrayOfString;
  I, Code: Integer;
begin
  Why := 'setup-runtime.ps1 failed';
  LogFile := ExpandConstant('{tmp}\setup-runtime.log');
  Params := '{#PsArgs} "' + Staging + '\tools\setup-runtime.ps1" -Action Install' +
            ' -App "' + RemoveBackslash(ExpandConstant('{app}')) + '" -Staged "' + Staging + '"' +
            ' -LogFile "' + LogFile + '"';
  if not Exec(ExpandConstant('{#PowerShell}'), Params, ExpandConstant('{app}'), SW_HIDE,
              ewWaitUntilTerminated, Code) then
    Code := -1;
  if LoadStringsFromFile(LogFile, Lines) then
    for I := 0 to GetArrayLength(Lines) - 1 do begin
      Log('setup-runtime: ' + Lines[I]);
      // The script prints its error as "!! <message>".
      if Copy(Lines[I], 1, 3) = '!! ' then
        Why := Copy(Lines[I], 4, Length(Lines[I]));
    end;
  if Code = 0 then begin
    RuntimeState := rsInstalled;
    Exit;
  end;
  RuntimeState := rsFailed;
  SuppressibleMsgBox('bg-be-gone could not be installed (exit ' + IntToStr(Code) + '): ' + Why,
                     mbError, MB_OK, IDOK);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep <> ssPostInstall then
    Exit;
  SwitchRuntime;
  if (RuntimeState = rsInstalled) and ModelDownloaded then begin
    ForceDirectories(ExtractFileDir(ModelTarget));
    if not FileCopy(ExpandConstant('{tmp}\') + ModelFile, ModelTarget, False) then
      Log('Could not copy the model to ' + ModelTarget);
  end;
end;

// Nonzero when the runtime didn't go in, so a silent install can tell.
function GetCustomSetupExitCode: Integer;
begin
  Result := 0;
  if RuntimeState = rsFailed then
    Result := 1;
end;
