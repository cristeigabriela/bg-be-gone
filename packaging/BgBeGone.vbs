' bg-be-gone launcher wrapper. Runs BgBeGone.cmd fully hidden so no console
' window flashes on start. Any arguments (e.g. an "Open with" file path) are
' quoted and forwarded through to the app.
Dim shell, fso, here, cmd, args, i
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
here = fso.GetParentFolderName(WScript.ScriptFullName)
args = ""
For i = 0 To WScript.Arguments.Count - 1
    args = args & " " & """" & WScript.Arguments(i) & """"
Next
cmd = """" & here & "\BgBeGone.cmd""" & args
shell.Run cmd, 0, False
