"""GPU rendering for the GTK frontend on Windows, and a way back to Cairo.

HACK: GTK 4.24's Windows backend only uses its GPU renderers (GL, Vulkan)
through Direct Composition, which is opt-in behind GDK_DEBUG=dcomp (GTK MR
!9593), and then draws the transparent window corners black. Both go away
once GTK enables dcomp by default with alpha on its GL window (the call GTK
MR !10153 proposed); then delete this module.

- Without dcomp every frame is painted in software (Cairo): the Generate
  shimmer and the Segment outlines took 2-4 CPU cores and lagged.
- With dcomp, GTK's GL renderer draws into a cloaked popup window of class
  "GdkWin32GL" and shows it through the toplevel
  (gdk_win32_gl_context_surface_attach, GTK 4.24.1). GTK marks the toplevel
  as alpha-carrying but not that window, so DWM composes it opaque.
  enable_alpha() makes MR !10153's DwmEnableBlurBehindWindow call, with an
  empty blur region, so DWM uses the alpha and nothing is blurred.

Since dcomp is still a debug path, a start that never shows its window
(a driver that crashes under it) must not take every later start down too.
choose_renderer() marks each dcomp start with its process, and the first
painted frame (or a handoff to a bg-be-gone already running) clears the
mark. A start that finds a mark whose process has exited starts once with
Cairo, says why in the log, and the next one tries the GPU again.
"""
import ctypes
import ctypes.wintypes as wt
import enum
import logging
import os
from typing import NamedTuple

# The marker file in the data folder: present while a dcomp start hasn't
# shown its window yet. It holds the process id that wrote it.
START_MARKER = "dcomp-start"

# Python ships no binding for these. dwmapi.h, DWM_BLURBEHIND's dwFlags:
# https://learn.microsoft.com/windows/win32/api/dwmapi/ns-dwmapi-dwm_blurbehind
_DWM_BB_ENABLE = 0x1
_DWM_BB_BLURREGION = 0x2
# winuser.h, event constants and SetWinEventHook's flags:
# https://learn.microsoft.com/windows/win32/winauto/event-constants
# https://learn.microsoft.com/windows/win32/api/winuser/nf-winuser-setwineventhook
_EVENT_OBJECT_CREATE = 0x8000
_WINEVENT_OUTOFCONTEXT = 0x0000
_OBJID_WINDOW = 0

_GL_WINDOW_CLASS = "GdkWin32GL"


class Renderer(enum.Enum):
    DCOMP = "dcomp"   # GTK's GL renderer through Direct Composition
    CAIRO = "cairo"   # GTK's software renderer, its default on Windows


class Choice(NamedTuple):
    renderer: Renderer
    reason: str | None   # why Cairo was chosen, for the log


# winnt.h's access right for GetProcessTimes on another process:
# https://learn.microsoft.com/windows/win32/procthread/process-security-and-access-rights
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _started_at(pid):
    """When process `pid` started (a FILETIME count), or None when no such
    process runs. With the pid it names one process: pids are reused."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wt.HANDLE
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        created, exited, kernel, user = (wt.FILETIME() for _ in range(4))
        if not kernel32.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                        ctypes.byref(kernel), ctypes.byref(user)):
            return None
        return (created.dwHighDateTime << 32) | created.dwLowDateTime
    finally:
        kernel32.CloseHandle(handle)


def _this_start():
    """This process, as the marker names it: "<pid> <start time>"."""
    return f"{os.getpid()} {_started_at(os.getpid())}"


def _read(marker):
    try:
        with open(marker, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def _owner_running(text):
    """Whether the start that wrote `text` is still running."""
    try:
        pid, started = (int(part) for part in text.split())
    except ValueError:
        return False
    return _started_at(pid) == started


def start_confirmed(marker):
    """This start showed its window, or handed its files to a bg-be-gone
    already running: clear its mark. Another start's mark is left alone."""
    if _read(marker) == _this_start():
        try:
            os.remove(marker)
        except OSError:
            pass


def _use_cairo():
    flags = [f for f in os.environ.get("GDK_DEBUG", "").split(",") if f and f != "dcomp"]
    if flags:
        os.environ["GDK_DEBUG"] = ",".join(flags)
    else:
        os.environ.pop("GDK_DEBUG", None)
    os.environ["GSK_RENDERER"] = "cairo"


def choose_renderer(marker):
    """Pick the renderer for this start, before GTK initialises.

    A mark whose start still runs is another start in flight: this one uses
    dcomp too and leaves the mark alone. A mark whose start has exited never
    showed a window, so this start uses Cairo, and only this one.
    """
    found = _read(marker)
    if found is not None and _owner_running(found):
        _request_dcomp()
        return Choice(Renderer.DCOMP, None)
    if found is not None:
        try:
            os.remove(marker)
        except OSError:
            pass
        _use_cairo()
        return Choice(Renderer.CAIRO, "the last start with Direct Composition (GDK_DEBUG=dcomp) "
                                      "never showed its window, so this one uses Cairo")
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, "w", encoding="utf-8") as f:
        f.write(_this_start())
    _request_dcomp()
    return Choice(Renderer.DCOMP, None)


def _request_dcomp():
    flags = [f for f in os.environ.get("GDK_DEBUG", "").split(",") if f]
    if "dcomp" not in flags:
        os.environ["GDK_DEBUG"] = ",".join(flags + ["dcomp"])


def confirm_on_first_frame(window, marker):
    """Clear the mark once `window` has painted its first frame."""
    def on_map(widget):
        clock = widget.get_frame_clock()
        handler = None

        def on_paint(_clock):
            clock.disconnect(handler)
            start_confirmed(marker)

        handler = clock.connect("after-paint", on_paint)

    if window.get_mapped():
        on_map(window)
    else:
        window.connect("map", on_map)


class _DwmBlurBehind(ctypes.Structure):
    _fields_ = [("dwFlags", wt.DWORD), ("fEnable", wt.BOOL), ("hRgnBlur", wt.HANDLE),
                ("fTransitionOnMaximized", wt.BOOL)]


def _gl_windows():
    """This process's top-level windows of GTK's GL class."""
    user32 = ctypes.WinDLL("user32")
    me = os.getpid()
    found = []

    def visit(hwnd, _):
        pid = wt.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        cls = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(hwnd, cls, len(cls))
        if pid.value == me and cls.value == _GL_WINDOW_CLASS:
            found.append(hwnd)
        return True

    user32.EnumWindows(ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)(visit), 0)
    return found


def enable_alpha(hwnd):
    dwmapi = ctypes.WinDLL("dwmapi")
    gdi32 = ctypes.WinDLL("gdi32")
    gdi32.CreateRectRgn.restype = wt.HANDLE
    empty = gdi32.CreateRectRgn(0, 0, -1, -1)
    blur = _DwmBlurBehind(_DWM_BB_ENABLE | _DWM_BB_BLURREGION, True, empty, False)
    hr = dwmapi.DwmEnableBlurBehindWindow(wt.HWND(hwnd), ctypes.byref(blur))
    gdi32.DeleteObject(wt.HANDLE(empty))
    if hr != 0:   # an HRESULT; 0 is S_OK
        logging.warning("DwmEnableBlurBehindWindow failed (0x%08x): a GL window keeps "
                        "black corners", hr & 0xFFFFFFFF)


_WINEVENTPROC = ctypes.WINFUNCTYPE(None, wt.HANDLE, wt.DWORD, wt.HWND, wt.LONG, wt.LONG,
                                   wt.DWORD, wt.DWORD)
_hook = None        # (hook handle, callback): kept alive for the process's lifetime


def watch_gl_windows():
    """Fix every GL window GTK creates from now on, and the ones it already has.

    Each popup surface (menus, popovers, tooltips, dialogs) gets its own GL
    window when it attaches, so fixing only the main window left their
    corners black. An out-of-context WinEvent hook on window creation,
    filtered to this process, runs on the GTK main thread's message loop.
    """
    global _hook
    if _hook is not None:
        return
    user32 = ctypes.WinDLL("user32")
    user32.SetWinEventHook.restype = wt.HANDLE

    def on_create(_hook_handle, _event, hwnd, id_object, _id_child, _thread, _time):
        if id_object != _OBJID_WINDOW or not hwnd:
            return
        cls = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(hwnd, cls, len(cls))
        if cls.value == _GL_WINDOW_CLASS:
            enable_alpha(hwnd)

    callback = _WINEVENTPROC(on_create)
    handle = user32.SetWinEventHook(_EVENT_OBJECT_CREATE, _EVENT_OBJECT_CREATE, None,
                                    callback, os.getpid(), 0, _WINEVENT_OUTOFCONTEXT)
    if handle:
        _hook = (handle, callback)
    else:
        # Left unset, so the next call tries again.
        logging.warning("SetWinEventHook failed: menus and popovers opened later keep "
                        "black corners")
    for hwnd in _gl_windows():
        enable_alpha(hwnd)
