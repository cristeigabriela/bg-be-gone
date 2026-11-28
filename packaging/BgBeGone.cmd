@echo off
rem bg-be-gone with a console, for reading what it prints. The Start Menu entry
rem runs bin\pythonw.exe on the same script, with no window.
rem
rem %~dp0 is expanded once, so a folder name like "%TEMP%" stays as it is.
"%~dp0bin\python.exe" "%~dp0app\src\bgbg\plat\winlaunch.py" %*
