<#
.SYNOPSIS
  Windows install and packaging helpers: build, probe and swap the worker,
  remove only what an install recorded, write the launcher, make the icon.

.DESCRIPTION
  The worker is a standalone CPython (python-build-standalone, the build
  `uv python install` fetches) with the ML packages installed into its own
  Lib\site-packages. Unlike a venv it has no pyvenv.cfg pointing back at the
  machine that built it, so a release bundle starts on any PC.

  Deletion rules, the same for every script that uses this module:
  - A folder bg-be-gone builds carries an inventory, bg-be-gone-files.txt,
    listing every file in it. Removing the folder removes exactly those
    files, then the folders they leave empty. Anything else stays.
  - Nothing is removed through a reparse point (junction or symlink): each
    path is checked from the install root down before anything is deleted.
  - New folders get a name no one else has (worker.new-<random>), so a build
    never collides with a folder it didn't create.

  Used by install.ps1, uninstall.ps1, build-windows-worker.ps1 and the
  release packaging.
#>

# The CPython the worker runs on. 3.12 has wheels for onnxruntime, numba,
# llvmlite and TheRock's torch; the patch release is pinned so every build
# ships the same interpreter.
$script:PythonVersion = '3.12.14'

# The inventory a bg-be-gone folder carries at its top.
$script:InventoryName = 'bg-be-gone-files.txt'

$script:Utf8NoBom = New-Object Text.UTF8Encoding($false)

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
function Write-Utf8File {
    <# Write text as UTF-8 without a BOM, which Python's json.load accepts. #>
    param([string]$Path, [string]$Text)
    [IO.File]::WriteAllText($Path, $Text, $script:Utf8NoBom)
}

function Invoke-NativeShown {
    <#
    .SYNOPSIS
      Run a program whose output is for the user to read, with its stderr
      shown as plain text. The exit code is in $LASTEXITCODE, as for a plain
      call.

    .DESCRIPTION
      uv, pip and pacman write their progress to stderr. When the install's
      output is redirected (into a log, say), Windows PowerShell 5.1 turns
      each of those lines into an error record and prints a red
      NativeCommandError block, so a working install reads as a failed one.

      Not for a call whose output is parsed: everything goes to the host.
    #>
    param([Parameter(Mandatory)][string]$FilePath, [string[]]$ArgumentList = @())
    # Under 'Stop', 2>&1 on a program would throw at its first stderr line.
    $ErrorActionPreference = 'Continue'
    & $FilePath @ArgumentList 2>&1 | ForEach-Object { "$_" } | Out-Host
}

function New-UniqueName {
    <# A sibling name for $Path no other program uses: <Path><Suffix>-<8 hex>. #>
    param([Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)][string]$Suffix)
    "$Path$Suffix-" + [guid]::NewGuid().ToString('N').Substring(0, 8)
}

function New-PrivateTempDir {
    <# A new, empty folder under %TEMP% that this run alone uses. #>
    param([Parameter(Mandatory)][string]$Prefix)
    $dir = Join-Path ([IO.Path]::GetTempPath()) ("$Prefix-" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $dir -ErrorAction Stop | Out-Null
    $dir
}

# ---------------------------------------------------------------------------
# Reparse points and inventories
# ---------------------------------------------------------------------------
function Test-ReparsePoint {
    <# True when $Path is a junction, symlink or other reparse point. #>
    param([Parameter(Mandatory)][string]$Path)
    [bool]([IO.File]::GetAttributes($Path) -band [IO.FileAttributes]::ReparsePoint)
}

function Assert-NoReparseBelow {
    <#
    .SYNOPSIS
      Throw unless $Path is inside $Root and no folder from $Root down to
      $Path (both included) is a reparse point.

    .DESCRIPTION
      $Cache remembers folders already checked, so an inventory of thousands
      of files checks each folder once. Parts of the chain that don't exist
      are fine: there is nothing there to delete through.
    #>
    param(
        [Parameter(Mandatory)][string]$Root,
        [Parameter(Mandatory)][string]$Path,
        [hashtable]$Cache = @{}
    )
    $rootFull = [IO.Path]::GetFullPath($Root).TrimEnd('\')
    $full = [IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith($rootFull + '\', [StringComparison]::OrdinalIgnoreCase) -and $full -ne $rootFull) {
        throw "$Path is not inside $Root"
    }
    $step = $full
    while ($step.Length -ge $rootFull.Length) {
        if (-not $Cache.ContainsKey($step)) {
            $exists = [IO.Directory]::Exists($step) -or [IO.File]::Exists($step)
            $Cache[$step] = $exists -and (Test-ReparsePoint $step)
        }
        if ($Cache[$step]) { throw "$step is a junction or symbolic link; not removing anything through it" }
        if ($step -eq $rootFull) { break }
        $step = [IO.Path]::GetDirectoryName($step)
    }
}

function Get-TreeFiles {
    <#
    .SYNOPSIS
      Every file under $Root, relative to it, without following reparse points.

    .DESCRIPTION
      With -Include FilesAndFolders, folders are listed too, each with a
      trailing backslash. Throws on a reparse point anywhere in the tree:
      bg-be-gone never makes one, so one there means the tree is not what
      was built.
    #>
    param(
        [Parameter(Mandatory)][string]$Root,
        [ValidateSet('Files', 'FilesAndFolders')][string]$Include = 'Files'
    )
    $rootFull = [IO.Path]::GetFullPath($Root).TrimEnd('\')
    $pending = New-Object System.Collections.Generic.Stack[IO.DirectoryInfo]
    $pending.Push((New-Object IO.DirectoryInfo $rootFull))
    $entries = New-Object System.Collections.Generic.List[string]
    while ($pending.Count) {
        $dir = $pending.Pop()
        foreach ($entry in $dir.EnumerateFileSystemInfos()) {
            if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "$($entry.FullName) is a junction or symbolic link"
            }
            $rel = $entry.FullName.Substring($rootFull.Length + 1)
            if ($entry -is [IO.DirectoryInfo]) {
                $pending.Push($entry)
                if ($Include -eq 'FilesAndFolders') { $entries.Add($rel + '\') }
            } else {
                $entries.Add($rel)
            }
        }
    }
    $entries.Sort([StringComparer]::OrdinalIgnoreCase)
    $entries
}

function Write-Inventory {
    <#
    .SYNOPSIS
      Record every file and folder under $Root in $Root\bg-be-gone-files.txt,
      itself included. Folders end in a backslash.
    #>
    param([Parameter(Mandatory)][string]$Root)
    $list = @(Get-TreeFiles $Root -Include FilesAndFolders | Where-Object { $_ -ne $script:InventoryName }) +
        $script:InventoryName
    Write-Utf8File (Join-Path $Root $script:InventoryName) (($list -join "`r`n") + "`r`n")
}

function Test-Inventory {
    <# True when $Root carries an inventory. #>
    param([Parameter(Mandatory)][string]$Root)
    Test-Path -LiteralPath (Join-Path $Root $script:InventoryName) -PathType Leaf
}

function Read-Inventory {
    <# The relative paths $Root's inventory lists, or none. #>
    param([Parameter(Mandatory)][string]$Root)
    $path = Join-Path $Root $script:InventoryName
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return @() }
    @([IO.File]::ReadAllLines($path, $script:Utf8NoBom) | Where-Object { $_ })
}

function Update-Inventory {
    <#
    .SYNOPSIS
      Re-record $Root's inventory after a run that changed the tree.

    .DESCRIPTION
      The new inventory keeps what the old one listed and still exists, and
      adds what this run created: entries present now that weren't in
      $Before, the tree as Get-TreeFiles -Include FilesAndFolders saw it when
      the run started. A file someone else put there in between stays
      unlisted, so it is never removed.
    #>
    param([Parameter(Mandatory)][string]$Root, [string[]]$Before = @())
    $had = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    foreach ($rel in $Before) { [void]$had.Add($rel) }
    $listed = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    foreach ($rel in Read-Inventory $Root) { [void]$listed.Add($rel) }
    $keep = foreach ($rel in Get-TreeFiles $Root -Include FilesAndFolders) {
        if ($rel -eq $script:InventoryName) { continue }
        if ($listed.Contains($rel) -or -not $had.Contains($rel)) { $rel }
    }
    $list = @($keep) + $script:InventoryName
    Write-Utf8File (Join-Path $Root $script:InventoryName) (($list -join "`r`n") + "`r`n")
}

function Test-LegacyInstall {
    <#
    .SYNOPSIS
      True when $Prefix holds an install made by install.ps1 before it kept a
      manifest, judged from evidence only that version leaves.

    .DESCRIPTION
      That version wrote BgBeGone.cmd with the worker venv's own path in it,
      and a worker venv with onnxruntime. A folder that merely has a
      pyvenv.cfg, or a .cmd of the same name, is not enough.
    #>
    param([Parameter(Mandatory)][string]$Prefix)
    $cmd = Join-Path $Prefix 'BgBeGone.cmd'
    $worker = Join-Path $Prefix 'worker'
    if (-not (Test-Path -LiteralPath $cmd -PathType Leaf)) { return $false }
    $text = [IO.File]::ReadAllText($cmd)
    $venvLine = 'set "BGBG_VENV_PYTHON=' + (Join-Path $worker 'Scripts\python.exe') + '"'
    if (-not $text.StartsWith('@echo off') -or
        $text.IndexOf($venvLine, [StringComparison]::OrdinalIgnoreCase) -lt 0 -or
        $text -notmatch 'set "BGBG_WORKER=[^"]*\\src\\bgbg\\worker\.py"') { return $false }
    if (-not (Test-Path -LiteralPath (Join-Path $worker 'pyvenv.cfg') -PathType Leaf)) { return $false }
    [bool](Get-ChildItem -LiteralPath (Join-Path $worker 'Lib\site-packages') -Filter 'onnxruntime*.dist-info' -Directory -ErrorAction SilentlyContinue)
}

function Write-LegacyInventory {
    <#
    .SYNOPSIS
      Give a pre-manifest install's worker venv an inventory of only the
      files bg-be-gone provably put there.

    .DESCRIPTION
      Proof is a package's own record: the RECORD file in each
      site-packages\*.dist-info of a package bg-be-gone's locks name, plus
      the bytecode Python compiled next to a recorded .py, plus the files
      python -m venv writes. Anything else (a file the user added, a package
      they installed into the venv) stays.
    #>
    param([Parameter(Mandatory)][string]$WorkerDir, [Parameter(Mandatory)][string]$LockDir)
    function ConvertTo-PackageKey([string]$Name) { ($Name -replace '[-_.]+', '_').ToLowerInvariant() }
    $ours = @{}
    foreach ($lock in Get-ChildItem -LiteralPath $LockDir -Filter 'windows-*.txt' -File) {
        foreach ($line in [IO.File]::ReadAllLines($lock.FullName)) {
            if ($line -match '^([A-Za-z0-9_.-]+)==') { $ours[(ConvertTo-PackageKey $Matches[1])] = $true }
        }
    }
    $root = [IO.Path]::GetFullPath($WorkerDir).TrimEnd('\')
    $site = Join-Path $root 'Lib\site-packages'
    $files = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    # What python -m venv itself writes.
    foreach ($rel in 'pyvenv.cfg', 'Scripts\python.exe', 'Scripts\pythonw.exe', 'Scripts\activate',
                     'Scripts\activate.bat', 'Scripts\Activate.ps1', 'Scripts\deactivate.bat', '.gitignore') {
        if ([IO.File]::Exists((Join-Path $root $rel))) { [void]$files.Add($rel) }
    }
    foreach ($info in Get-ChildItem -LiteralPath $site -Directory -Filter '*.dist-info' -ErrorAction SilentlyContinue) {
        if ($info.Name -notmatch '^(.+?)-[^-]+\.dist-info$' -or -not $ours.ContainsKey((ConvertTo-PackageKey $Matches[1]))) { continue }
        $record = Join-Path $info.FullName 'RECORD'
        if (-not [IO.File]::Exists($record)) { continue }
        foreach ($line in [IO.File]::ReadAllLines($record)) {
            # path,hash,size: a path holding a comma is quoted.
            $rel = if ($line.StartsWith('"')) { $line.Substring(1, $line.IndexOf('"', 1) - 1) } else { ($line -split ',', 2)[0] }
            if (-not $rel) { continue }
            $full = [IO.Path]::GetFullPath((Join-Path $site $rel))
            if (-not $full.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) { continue }
            if (-not [IO.File]::Exists($full)) { continue }
            [void]$files.Add($full.Substring($root.Length + 1))
            if ($full.EndsWith('.py', [StringComparison]::OrdinalIgnoreCase)) {
                $cache = Join-Path ([IO.Path]::GetDirectoryName($full)) '__pycache__'
                $stem = [IO.Path]::GetFileNameWithoutExtension($full)
                if ([IO.Directory]::Exists($cache)) {
                    foreach ($pyc in [IO.Directory]::GetFiles($cache, "$stem.*.pyc")) { [void]$files.Add($pyc.Substring($root.Length + 1)) }
                }
            }
        }
    }
    $list = @($files | Sort-Object) + $script:InventoryName
    Write-Utf8File (Join-Path $root $script:InventoryName) (($list -join "`r`n") + "`r`n")
}

function Remove-Inventory {
    <#
    .SYNOPSIS
      Remove the files $Root's inventory lists, then the folders it lists or
      that are left empty.

    .DESCRIPTION
      Every path is checked first (inside $Root, no reparse point on the way);
      one bad path stops the removal before anything is deleted. Files the
      inventory doesn't list stay where they are, and so do the folders that
      hold them.

      A listed folder with an inventory of its own goes by that one first:
      the release's runtime lists its worker, and Enable-AmdRocm.ps1 adds
      files only the worker's inventory knows.

    .OUTPUTS
      The paths of what is left under $Root, relative; empty when $Root is gone.
    #>
    param([Parameter(Mandatory)][string]$Root)
    if (-not (Test-Path -LiteralPath $Root)) { return @() }
    $rootFull = [IO.Path]::GetFullPath($Root).TrimEnd('\')
    if (Test-ReparsePoint $rootFull) { throw "$rootFull is a junction or symbolic link; not removing it" }
    $inventory = Join-Path $rootFull $script:InventoryName
    if (-not (Test-Path -LiteralPath $inventory -PathType Leaf)) {
        throw "$rootFull has no $($script:InventoryName); bg-be-gone can't tell which of its files are its own"
    }

    $cache = @{}
    $targets = New-Object System.Collections.Generic.List[string]
    $nested = New-Object System.Collections.Generic.List[string]
    foreach ($rel in [IO.File]::ReadAllLines($inventory, $script:Utf8NoBom)) {
        if (-not $rel) { continue }
        if ([IO.Path]::IsPathRooted($rel) -or ($rel -split '[\\/]') -contains '..') {
            throw "$inventory lists '$rel', which is not a path inside the folder"
        }
        $full = [IO.Path]::Combine($rootFull, $rel)
        # Each folder once: an MSYS2 tree lists ~50,000 files in ~5,000 folders.
        $dir = [IO.Path]::GetDirectoryName($full)
        if (-not $cache.ContainsKey($dir)) { Assert-NoReparseBelow -Root $rootFull -Path $dir -Cache $cache }
        $targets.Add($full)
        if ($rel.EndsWith('\') -and [IO.File]::Exists([IO.Path]::Combine($full, $script:InventoryName))) {
            $nested.Add($full.TrimEnd('\'))
        }
    }
    foreach ($folder in $nested) { $null = Remove-Inventory $folder }

    # .NET calls rather than cmdlets in these loops: per file, Test-Path and
    # Remove-Item take minutes on a tree that size.
    $dirs = @{}
    foreach ($full in $targets) {
        if ([IO.File]::Exists($full)) {
            $attributes = [IO.File]::GetAttributes($full)
            if ($attributes -band [IO.FileAttributes]::ReadOnly) {
                [IO.File]::SetAttributes($full, $attributes -bxor [IO.FileAttributes]::ReadOnly)
            }
            [IO.File]::Delete($full)
        }
        for ($d = [IO.Path]::GetDirectoryName($full); $d.Length -gt $rootFull.Length -and -not $dirs.ContainsKey($d);
             $d = [IO.Path]::GetDirectoryName($d)) {
            $dirs[$d] = $true
        }
    }
    # Deepest first, so a parent is only looked at once its children are done.
    foreach ($d in ($dirs.Keys | Sort-Object Length -Descending)) {
        if ([IO.Directory]::Exists($d) -and -not [IO.Directory]::EnumerateFileSystemEntries($d).GetEnumerator().MoveNext()) {
            [IO.Directory]::Delete($d)
        }
    }
    if (-not [IO.Directory]::EnumerateFileSystemEntries($rootFull).GetEnumerator().MoveNext()) {
        [IO.Directory]::Delete($rootFull)
        return @()
    }
    @(Get-TreeFiles $rootFull)
}

function Get-DataUserRecord {
    <# The record file that says $Prefix uses $DataDir. #>
    param([Parameter(Mandatory)][string]$DataDir, [Parameter(Mandatory)][string]$Prefix)
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = $sha.ComputeHash($script:Utf8NoBom.GetBytes($Prefix.TrimEnd('\').ToLowerInvariant()))
    } finally {
        $sha.Dispose()
    }
    $id = -join ($bytes[0..7] | ForEach-Object { $_.ToString('x2') })
    Join-Path $DataDir "installs\$id.json"
}

function Register-DataUser {
    <#
    .SYNOPSIS
      Record that the install at $Prefix keeps its models and settings in
      $DataDir, which every install of this user shares.
    #>
    param([Parameter(Mandatory)][string]$DataDir, [Parameter(Mandatory)][string]$Prefix)
    $record = Get-DataUserRecord $DataDir $Prefix
    Assert-NoReparseBelow -Root $DataDir -Path $record
    New-Item -ItemType Directory -Force (Split-Path -Parent $record) -ErrorAction Stop | Out-Null
    Write-Utf8File $record (@{ root = $Prefix.TrimEnd('\') } | ConvertTo-Json)
}

function Unregister-DataUser {
    <# Drop $Prefix's record from $DataDir, and the records folder once empty. #>
    param([Parameter(Mandatory)][string]$DataDir, [Parameter(Mandatory)][string]$Prefix)
    $record = Get-DataUserRecord $DataDir $Prefix
    # From the data folder down to the record, so neither the record nor its
    # folder is deleted through a junction.
    Assert-NoReparseBelow -Root $DataDir -Path $record
    if (Test-Path -LiteralPath $record -PathType Leaf) { Remove-Item -LiteralPath $record -Force }
    $dir = Split-Path -Parent $record
    if ((Test-Path -LiteralPath $dir) -and -not (Get-ChildItem -LiteralPath $dir -Force | Select-Object -First 1)) {
        Remove-Item -LiteralPath $dir -Force
    }
}

function Test-InstallAt {
    <#
    True when $Path holds a bg-be-gone install: a manifest, or the files an
    install made before there were manifests. Generous on purpose: it decides
    whether shared data is kept.
    #>
    param([Parameter(Mandatory)][string]$Path)
    # The last is a Setup install's runtime (setup-runtime.ps1).
    foreach ($marker in 'bg-be-gone-install.json', 'BgBeGone.cmd', 'BgBeGone.vbs',
                        'worker\python.exe', 'worker\pyvenv.cfg', 'runtime\bg-be-gone-files.txt') {
        if (Test-Path -LiteralPath (Join-Path $Path $marker) -PathType Leaf) { return $true }
    }
    $false
}

function Get-OtherDataUser {
    <#
    .SYNOPSIS
      The other installs that still keep their models and settings in
      $DataDir: recorded ones whose folder still holds an install, and an
      install inside $DataDir itself (the default prefix), which may predate
      the records.
    #>
    param([Parameter(Mandatory)][string]$DataDir, [Parameter(Mandatory)][string]$Prefix)
    $me = $Prefix.TrimEnd('\')
    $roots = @()
    $records = Join-Path $DataDir 'installs'
    if (Test-Path -LiteralPath $records) {
        foreach ($file in Get-ChildItem -LiteralPath $records -Filter '*.json' -File) {
            try { $roots += (Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8 | ConvertFrom-Json).root } catch {
                Write-Warning "Unreadable install record $($file.FullName): $_"
            }
        }
    }
    $roots += $DataDir.TrimEnd('\')
    @($roots | Where-Object { $_ -and -not [string]::Equals($_.TrimEnd('\'), $me, [StringComparison]::OrdinalIgnoreCase) } |
        Select-Object -Unique | Where-Object { Test-InstallAt $_ })
}

function Get-OwnedDataFile {
    <#
    .SYNOPSIS
      The files under $DataDir that bg-be-gone downloads or generates, for
      uninstall -Purge. Anything else there (a model the user put in
      models\) is not listed.

    .DESCRIPTION
      Each pattern names the code that writes it.
    #>
    param([Parameter(Mandatory)][string]$DataDir)
    $patterns = @(
        # rembg's downloads: U2NET_HOME is this folder, its files <model>.onnx (plat/windows.py).
        @{ Dir = 'u2net'; Filter = '*.onnx' }
        # engines/torch_birefnet.py: BiRefNet checkpoints, their check records, partial downloads.
        @{ Dir = 'models\birefnet'; Filter = 'BiRefNet-*.pth' }
        @{ Dir = 'models\birefnet'; Filter = 'BiRefNet-*.pth.verified' }
        @{ Dir = 'models\birefnet'; Filter = '*.part' }
        # segmentation.py: MobileSAM's two files, SAM 2.1's zips and their extracted pairs.
        @{ Dir = 'models'; Filter = 'mobile_sam_image_encoder.onnx*' }
        @{ Dir = 'models'; Filter = 'sam_mask_decoder_single.onnx*' }
        @{ Dir = 'models'; Filter = 'sam2.1_hiera_*.zip*' }
        @{ Dir = 'models\sam2.1_hiera_*'; Filter = 'sam2.1_hiera_*.*coder.onnx' }
        # Settings the app writes (engines/selection.py, segmentation.py).
        @{ Dir = 'config'; Filter = 'settings.json*' }
        @{ Dir = 'config'; Filter = 'seg.json' }
        # The frontend's and worker's log (plat/windows.py).
        @{ Dir = '.'; Filter = 'bg-be-gone-worker.log' }
        # Install records (Register-DataUser).
        @{ Dir = 'installs'; Filter = '*.json' }
    )
    foreach ($p in $patterns) {
        foreach ($dir in @(Get-Item -Path (Join-Path $DataDir $p.Dir) -ErrorAction SilentlyContinue | Where-Object PSIsContainer)) {
            Get-ChildItem -LiteralPath $dir.FullName -Filter $p.Filter -File -Force | ForEach-Object FullName
        }
    }
    # MIOpen's kernel cache (plat.kernel_cache_dir()): everything in it is generated.
    $miopen = Join-Path $DataDir 'miopen'
    if (Test-Path -LiteralPath $miopen -PathType Container) {
        Get-ChildItem -LiteralPath $miopen -Recurse -File -Force | ForEach-Object FullName
    }
}

function Remove-OwnedData {
    <#
    .SYNOPSIS
      Delete what Get-OwnedDataFile lists, then the folders that leaves
      empty, down from $DataDir. Nothing is deleted through a reparse point.

    .OUTPUTS
      How many files were removed.
    #>
    param([Parameter(Mandatory)][string]$DataDir)
    $root = [IO.Path]::GetFullPath($DataDir).TrimEnd('\')
    $cache = @{}
    $files = @(Get-OwnedDataFile $root)
    foreach ($f in $files) { Assert-NoReparseBelow -Root $root -Path $f -Cache $cache }
    $dirs = @{}
    foreach ($f in $files) {
        [IO.File]::SetAttributes($f, [IO.FileAttributes]::Normal)
        [IO.File]::Delete($f)
        for ($d = [IO.Path]::GetDirectoryName($f); $d.Length -gt $root.Length; $d = [IO.Path]::GetDirectoryName($d)) { $dirs[$d] = $true }
    }
    foreach ($d in ($dirs.Keys | Sort-Object Length -Descending)) {
        if ([IO.Directory]::Exists($d) -and -not [IO.Directory]::EnumerateFileSystemEntries($d).GetEnumerator().MoveNext()) {
            [IO.Directory]::Delete($d)
        }
    }
    $files.Count
}

function Get-WorkerPython {
    <# The interpreter inside a worker folder. #>
    param([Parameter(Mandatory)][string]$WorkerDir)
    Join-Path $WorkerDir 'python.exe'
}

function New-WorkerPython {
    <#
    .SYNOPSIS
      Put a fresh standalone CPython at $Dest (which must not exist yet).
    #>
    param([Parameter(Mandatory)][string]$Dest)
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv not found on PATH. Install it (winget install astral-sh.uv) and retry."
    }
    if (Test-Path -LiteralPath $Dest) { throw "$Dest already exists" }

    # Unpacked in a folder of its own next to $Dest, so the move below stays
    # on one volume. No registry entry and no python.exe shim: this copy is
    # the worker's alone.
    $unpack = New-UniqueName $Dest '.python'
    Invoke-NativeShown uv @('python', 'install', $script:PythonVersion, '--install-dir', $unpack, '--no-registry', '--no-bin')
    if ($LASTEXITCODE -ne 0) { throw "uv python install $($script:PythonVersion) failed" }
    try {
        $tree = Get-ChildItem -LiteralPath $unpack -Directory |
            Where-Object { $_.Name -like "cpython-$($script:PythonVersion)-*" -and -not $_.LinkType } |
            Select-Object -First 1
        if (-not $tree) { throw "uv put no cpython-$($script:PythonVersion) folder in $unpack" }
        Move-Item -LiteralPath $tree.FullName $Dest -ErrorAction Stop
    } finally {
        # Left: uv's lock file and a cpython-3.12-... junction to the folder
        # just moved. Each entry goes on its own, so the junction is unlinked,
        # never followed.
        foreach ($entry in Get-ChildItem -LiteralPath $unpack -Force) {
            if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { $entry.Delete() }
            else { Remove-Item -LiteralPath $entry.FullName -Recurse -Force }
        }
        Remove-Item -LiteralPath $unpack -Force
    }

    # python-build-standalone marks itself as managed by the tool that
    # installed it. This copy belongs to the worker, which installs into it.
    Remove-Item -LiteralPath (Join-Path $Dest 'Lib\EXTERNALLY-MANAGED') -Force -ErrorAction SilentlyContinue
}

function Install-LockedPackage {
    <#
    .SYNOPSIS
      Install a lock file's exact files into a worker, nothing more.

    .DESCRIPTION
      --no-deps with --require-hashes: every package and its hash comes from
      the lock, so nothing resolves at install time. uv is used where it
      exists (builds, source installs); the worker's own pip on a user's PC.
    #>
    param(
        [Parameter(Mandatory)][string]$WorkerDir,
        [Parameter(Mandatory)][string]$Lock,
        [string]$IndexUrl
    )
    $py = Get-WorkerPython $WorkerDir
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        $pipArgs = @('pip', 'install', '--python', $py, '--no-deps', '--require-hashes', '-r', $Lock)
        if ($IndexUrl) { $pipArgs += @('--index-url', $IndexUrl) }
        Invoke-NativeShown uv $pipArgs
    } else {
        $pipArgs = @('-I', '-m', 'pip', 'install', '--no-deps', '--require-hashes',
                     '--disable-pip-version-check', '--no-warn-script-location', '-r', $Lock)
        if ($IndexUrl) { $pipArgs += @('--index-url', $IndexUrl) }
        Invoke-NativeShown $py $pipArgs
    }
    if ($LASTEXITCODE -ne 0) { throw "Installing $Lock into $WorkerDir failed" }
}

function New-WorkerStartInfo {
    <#
    .SYNOPSIS
      A ProcessStartInfo that runs $Arguments on the worker's interpreter with
      the environment the frontend gives the worker.

    .DESCRIPTION
      It goes through `python -I plat\windows.py`, which applies the one
      worker-environment policy (plat/windows.py worker_env) and runs the
      rest. $AppDir is the folder holding plat\ and worker.py.
    #>
    param(
        [Parameter(Mandatory)][string]$WorkerDir,
        [Parameter(Mandatory)][string]$AppDir,
        [string[]]$Arguments
    )
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = Get-WorkerPython $WorkerDir
    $all = @('-I', (Join-Path $AppDir 'plat\windows.py')) + $Arguments
    # Windows paths hold no double quote, so quoting each argument is enough,
    # once a trailing backslash is doubled (else it escapes the closing quote).
    $psi.Arguments = ($all | ForEach-Object { '"' + ($_ -replace '(\\+)$', '$1$1') + '"' }) -join ' '
    $psi.UseShellExecute = $false
    $psi
}

function Invoke-WorkerPython {
    <# Run a script on the worker's interpreter in this console; return its exit code. #>
    param(
        [Parameter(Mandatory)][string]$WorkerDir,
        [Parameter(Mandatory)][string]$AppDir,
        [string[]]$Arguments
    )
    $proc = [Diagnostics.Process]::Start((New-WorkerStartInfo -WorkerDir $WorkerDir -AppDir $AppDir -Arguments $Arguments))
    try {
        $proc.WaitForExit()
        $proc.ExitCode
    } finally {
        $proc.Dispose()
    }
}

function Stop-ProcessTree {
    <# The runner's child outlives a plain Kill, so end the whole tree. #>
    param([Parameter(Mandatory)][Diagnostics.Process]$Process)
    if (-not $Process.HasExited) { & taskkill.exe /F /T /PID $Process.Id 2>&1 | Out-Null }
}

function Test-Worker {
    <#
    .SYNOPSIS
      Start worker.py on a worker's interpreter and return its "ready" message.

    .DESCRIPTION
      Runs with the frontend's environment for the worker. By default with
      BGBG_NO_TORCH too, so the probe never initialises a GPU.

      -Engine Torch instead runs the worker's real PyTorch start-up (the GPU
      runtime, torchvision's compiled ops, BiRefNet's code) and asks for its
      engine_status. It returns @{ Ready; EngineStatus }, and the caller
      keeps a worker only when EngineStatus.state is "ready".

      Throws when the worker exits or stays silent instead of answering.
    #>
    param(
        [Parameter(Mandatory)][string]$WorkerDir,
        [Parameter(Mandatory)][string]$AppDir,
        [ValidateSet('None', 'Torch')][string]$Engine = 'None',
        [int]$TimeoutSeconds = 300
    )
    $psi = New-WorkerStartInfo -WorkerDir $WorkerDir -AppDir $AppDir -Arguments @((Join-Path $AppDir 'worker.py'))
    $psi.RedirectStandardInput = $true
    $psi.RedirectStandardOutput = $true
    $psi.StandardOutputEncoding = [Text.Encoding]::UTF8
    $psi.CreateNoWindow = $true
    if ($Engine -eq 'Torch') {
        $psi.EnvironmentVariables['BGBG_ENGINE'] = 'torch'
        foreach ($switch in 'BGBG_NO_TORCH', 'BGBG_NO_ROCM') { $psi.EnvironmentVariables.Remove($switch) }
    } else {
        $psi.EnvironmentVariables['BGBG_NO_TORCH'] = '1'
    }

    # Windows PowerShell's .NET encodes a child's stdin with the console's
    # input encoding, and its UTF-8 writes a byte order mark into the pipe at
    # start-up. The worker can't parse a first request behind it.
    $consoleEncoding = [Console]::InputEncoding
    [Console]::InputEncoding = New-Object Text.UTF8Encoding $false
    try {
        $proc = [Diagnostics.Process]::Start($psi)
    } finally {
        [Console]::InputEncoding = $consoleEncoding
    }
    try {
        $stdin = $proc.StandardInput
        $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
        function Wait-Reply {
            param([string]$Type)
            while ($true) {
                $left = [int]($deadline - (Get-Date)).TotalMilliseconds
                $read = $proc.StandardOutput.ReadLineAsync()
                if ($left -le 0 -or -not $read.Wait($left)) {
                    throw "The worker in $WorkerDir sent no $Type within $TimeoutSeconds s"
                }
                $line = $read.Result
                if ($null -eq $line) { throw "The worker in $WorkerDir exited before it sent $Type" }
                try { $msg = $line | ConvertFrom-Json } catch { continue }
                if ($msg.type -eq $Type) { return $msg }
            }
        }
        $ready = Wait-Reply 'ready'
        $status = $null
        if ($Engine -eq 'Torch') {
            $stdin.WriteLine('{"op":"engine_status","id":1}')
            $stdin.Flush()
            $status = Wait-Reply 'engine_status'
        }
        $stdin.WriteLine('{"op":"shutdown"}')
        $stdin.Close()
        if (-not $proc.WaitForExit(15000)) { Stop-ProcessTree $proc }
        if ($Engine -eq 'Torch') { return @{ Ready = $ready; EngineStatus = $status } }
        return $ready
    } finally {
        Stop-ProcessTree $proc
        $proc.Dispose()
    }
}

function Write-WorkerRecord {
    <#
    .SYNOPSIS
      Write profile.json and versions.json into a worker folder.

    .DESCRIPTION
      profile.json tells the worker which engine runs BiRefNet. It sits in the
      interpreter's sys.prefix: {"engine": "ort"|"torch", "variant": ...,
      "gfx": ...}, gfx only for torch on AMD. No file means ort.

      versions.json records what this worker was built from, so a bug report
      or a release asset says which onnxruntime, torch and Python it ran.
    #>
    param(
        [Parameter(Mandatory)][string]$WorkerDir,
        [Parameter(Mandatory)][ValidateSet('ort', 'torch')][string]$Engine,
        [Parameter(Mandatory)][string]$Variant,
        [string]$GfxArch
    )
    $engineProfile = [ordered]@{ engine = $Engine; variant = $Variant }
    if ($GfxArch) { $engineProfile.gfx = $GfxArch }
    Write-Utf8File (Join-Path $WorkerDir 'profile.json') ($engineProfile | ConvertTo-Json)

    # importlib.metadata reads the installed dist-infos, so this is what is
    # there, not what a lock asked for. Single quotes only: PowerShell 5.1
    # does not escape double quotes inside a native command's argument.
    $query = @'
import json, platform
from importlib import metadata
names = ('onnxruntime', 'onnxruntime-directml', 'onnxruntime-gpu', 'rembg',
         'numpy', 'pi-heif', 'torch', 'torchvision')
pkgs = {}
for n in names:
    try:
        pkgs[n] = metadata.version(n)
    except metadata.PackageNotFoundError:
        pass
print(json.dumps({'python': platform.python_version(), 'packages': pkgs}))
'@
    $versions = & (Get-WorkerPython $WorkerDir) -I -c $query
    if ($LASTEXITCODE -ne 0) { throw "Could not read the package versions in $WorkerDir" }
    Write-Utf8File (Join-Path $WorkerDir 'versions.json') $versions
}

function Complete-Worker {
    <#
    .SYNOPSIS
      Compile the worker's bytecode, then write its inventory.

    .DESCRIPTION
      Compiled now, the .pyc files are in the inventory, and the worker never
      writes files of its own beside its code at run time.
    #>
    param([Parameter(Mandatory)][string]$WorkerDir)
    # Lib\ only: the stdlib and site-packages. tcl\ ships a .py data file that
    # isn't valid Python 3 and is never imported.
    & (Get-WorkerPython $WorkerDir) -I -m compileall -q -j 0 (Join-Path $WorkerDir 'Lib') | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Compiling the worker's bytecode failed" }
    Write-Inventory $WorkerDir
}

function Switch-OwnedFolder {
    <#
    .SYNOPSIS
      Put the probed folder at $Staged in place of $Path: a worker, or the
      release's runtime.

    .DESCRIPTION
      The old folder is renamed aside first and removed by its inventory only
      once the new one is in place, so at every step one complete copy
      exists. Files someone added to the old folder stay, in the renamed
      folder, and are reported.
    #>
    param([Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)][string]$Staged)
    $old = $null
    if (Test-Path -LiteralPath $Path) {
        if (-not (Test-Inventory $Path)) {
            throw "$Path has no $($script:InventoryName), so it may not be bg-be-gone's; not replacing it"
        }
        $old = New-UniqueName $Path '.old'
        Rename-Item -LiteralPath $Path (Split-Path -Leaf $old) -ErrorAction Stop
    }
    try {
        Rename-Item -LiteralPath $Staged (Split-Path -Leaf $Path) -ErrorAction Stop
    } catch {
        if ($old) { Rename-Item -LiteralPath $old (Split-Path -Leaf $Path) }
        throw "Could not move $Staged into place (is bg-be-gone still running?): $_"
    }
    # The rename above is the commit point: the new copy is live. Removing
    # the old one is cleanup, so a failure there is reported, not thrown, and
    # its inventory stays for the next run to finish the job.
    if ($old) {
        try {
            $left = @(Remove-Inventory $old)
            if ($left.Count) {
                Write-Warning "Kept $($left.Count) file(s) in $old that its inventory doesn't list"
            }
        } catch {
            Write-Warning "$Path is in place; the old copy at $old is not fully removed yet: $_"
        }
    }
}

# ---------------------------------------------------------------------------
# Launcher and icon
# ---------------------------------------------------------------------------
function Get-LaunchCommand {
    <#
    .SYNOPSIS
      The program and arguments that start bg-be-gone: the one place that
      decides how it is launched.

    .DESCRIPTION
      MSYS2's windowless pythonw.exe runs src\bgbg\plat\winlaunch.py, which
      tells the app where its worker is and runs it in the same process.

      The shell expands %VAR% in a shortcut's arguments, so -Mode Module
      passes no path at all: `-m bg_be_gone_launch`, a module in the
      interpreter's own site-packages that holds the paths (see
      Install-LaunchModule). -Mode Paths names them on the command line, for
      an MSYS2 that isn't bg-be-gone's to write into; a path holding a '%'
      then can't be launched from a shortcut.

    .OUTPUTS
      @{ Program = <pythonw.exe>; Arguments = <string[]> }
    #>
    param(
        [Parameter(Mandatory)][string]$GtkRoot,
        [Parameter(Mandatory)][ValidateSet('Module', 'Paths')][string]$Mode,
        [string]$AppDir,
        [string]$Install
    )
    if ($Mode -eq 'Module') {
        $arguments = @('-m', 'bg_be_gone_launch')
    } else {
        $arguments = @(Join-Path $AppDir 'plat\winlaunch.py')
        if ($Install) { $arguments += "--install=$Install" }
    }
    @{ Program = (Join-Path $GtkRoot 'bin\pythonw.exe'); Arguments = $arguments }
}

function Install-LaunchModule {
    <#
    .SYNOPSIS
      Put bg_be_gone_launch.py in the frontend interpreter's site-packages.

    .DESCRIPTION
      Without -AppDir it is the release bundle's copy, which finds the app
      from its own location. A source install passes its checkout's src\bgbg
      and its install folder, written into the file as Python strings.

    .OUTPUTS
      The file written.
    #>
    param(
        [Parameter(Mandatory)][string]$GtkRoot,
        [string]$AppDir,
        [string]$Install
    )
    $site = Get-ChildItem -LiteralPath (Join-Path $GtkRoot 'lib') -Directory -Filter 'python3.*' |
        ForEach-Object { Join-Path $_.FullName 'site-packages' } |
        Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $site) { throw "No python3.x\site-packages under $GtkRoot\lib" }
    $text = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'bg_be_gone_launch.py') -Raw
    # r"..." holds any Windows path: none ends in a backslash or holds a quote.
    if ($AppDir) { $text = $text.Replace('APP_DIR = None   # @APP_DIR@', "APP_DIR = r`"$AppDir`"") }
    if ($Install) { $text = $text.Replace('INSTALL = None   # @INSTALL@', "INSTALL = r`"$Install`"") }
    $path = Join-Path $site 'bg_be_gone_launch.py'
    Write-Utf8File $path $text
    # Compiled now, so its .pyc is part of the tree's inventory instead of a
    # file the first start leaves behind.
    & (Join-Path $GtkRoot 'bin\python.exe') -I -m py_compile $path
    if ($LASTEXITCODE -ne 0) { throw "Compiling $path failed" }
    $path
}

function ConvertTo-CommandLine {
    <# Quote arguments for CreateProcess: each in quotes, a trailing backslash doubled. #>
    param([string[]]$Arguments)
    ($Arguments | ForEach-Object { '"' + ($_ -replace '(\\+)$', '$1$1') + '"' }) -join ' '
}

function New-AppShortcut {
    <#
    .SYNOPSIS
      A shortcut at $Path that runs the launch command, with the app's icon.

    .DESCRIPTION
      No working folder: the shell expands %VAR% in it too, and the app needs
      none. The icon is shown as stored, without expansion (ShellLink.cs).
    #>
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][hashtable]$Command,
        [Parameter(Mandatory)][string]$Icon
    )
    Import-ShellLink
    [BgBeGone.ShellLink]::Save($Path, $Command.Program, (ConvertTo-CommandLine $Command.Arguments),
        '', $Icon, 'Local image background remover and object segmenter')
}

function Read-AppShortcut {
    <# A shortcut's target and arguments as stored: @{ Target; Arguments }. #>
    param([Parameter(Mandatory)][string]$Path)
    Import-ShellLink
    $target, $arguments = [BgBeGone.ShellLink]::Read($Path)
    @{ Target = $target; Arguments = $arguments }
}

function Import-ShellLink {
    <# Compile ShellLink.cs once per session (see its header for why not WScript.Shell). #>
    if (-not ('BgBeGone.ShellLink' -as [type])) {
        Add-Type -Path (Join-Path $PSScriptRoot 'ShellLink.cs') -ErrorAction Stop
    }
}

function New-AppIcon {
    <#
    .SYNOPSIS
      Wrap the app's 512 px PNG in an .ico container for shortcuts and Setup.

    .DESCRIPTION
      Windows Vista and later read PNG-compressed icon entries, so the .ico is
      one ICONDIR header, one entry and the PNG bytes unchanged.
    #>
    param([Parameter(Mandatory)][string]$Png, [Parameter(Mandatory)][string]$Ico)
    $bytes = [IO.File]::ReadAllBytes($Png)
    $ms = New-Object IO.MemoryStream
    $bw = New-Object IO.BinaryWriter($ms)
    # ICONDIR: reserved 0, type 1 (icon), 1 image.
    foreach ($v in @([UInt16]0, [UInt16]1, [UInt16]1)) { $bw.Write($v) }
    # ICONDIRENTRY: width and height 0 mean 256 or more; no palette; reserved.
    $bw.Write([Byte]0); $bw.Write([Byte]0); $bw.Write([Byte]0); $bw.Write([Byte]0)
    $bw.Write([UInt16]1); $bw.Write([UInt16]32)                   # planes, bits per pixel
    $bw.Write([UInt32]$bytes.Length); $bw.Write([UInt32]22)       # image size, offset (6 + 16)
    $bw.Write($bytes); $bw.Flush()
    [IO.File]::WriteAllBytes($Ico, $ms.ToArray())
    $bw.Dispose(); $ms.Dispose()
}

Export-ModuleMember -Function Write-Utf8File, Invoke-NativeShown, New-UniqueName, New-PrivateTempDir,
    Test-ReparsePoint, Assert-NoReparseBelow, Get-TreeFiles, Write-Inventory, Test-Inventory,
    Read-Inventory, Update-Inventory, Test-LegacyInstall,
    Remove-Inventory, Write-LegacyInventory, Get-OwnedDataFile, Remove-OwnedData, Register-DataUser, Unregister-DataUser, Test-InstallAt, Get-OtherDataUser,
    Get-WorkerPython, New-WorkerPython,
    Install-LockedPackage, New-WorkerStartInfo, Invoke-WorkerPython, Test-Worker,
    Write-WorkerRecord, Complete-Worker, Switch-OwnedFolder, Get-LaunchCommand, Install-LaunchModule,
    ConvertTo-CommandLine,
    New-AppShortcut, Read-AppShortcut, New-AppIcon
