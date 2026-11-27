"""GPU rendering for the GTK frontend on Windows.

GTK 4.24's Windows backend only uses its GPU renderers (GL, Vulkan) through
Direct Composition, and that is opt-in behind GDK_DEBUG=dcomp (GTK MR !9593).
Without it every frame is painted in software (Cairo): the Generate shimmer and
the Segment outlines then took 2-4 CPU cores and lagged.

With Direct Composition on, GTK's GL renderer draws into a separate, cloaked
popup window of class "GdkWin32GL" and shows it through the toplevel with
IDCompositionDevice::CreateSurfaceFromHwnd (gdkglcontext-win32.c,
gdk_win32_gl_context_surface_attach, GTK 4.24.1). GTK marks the toplevel as
alpha-carrying (gdk_win32_surface_enable_transparency) but not that window, so
DWM composes it opaque: the transparent margin GTK draws the window shadow and
rounded corners into came out black. enable_gl_window_alpha() makes the same
DwmEnableBlurBehindWindow call GTK MR !10153 proposed (closed unmerged), with an
empty blur region, so DWM uses the window's alpha and nothing is blurred.

Every surface (each window, menu, popover and tooltip) gets its own GL window,
and GTK recreates it when the surface re-attaches, so watch_gl_windows() fixes
each one as it is created. The Vulkan renderer uses its own window class
(GdkWin32Vulkan) and is not covered: it isn't GTK's default under dcomp, and
the fix is untested there.
"""
import ctypes
import ctypes.wintypes as wt
import os

# dwmapi.h; Python ships no binding for these.
_DWM_BB_ENABLE = 0x1
_DWM_BB_BLURREGION = 0x2
_GL_WINDOW_CLASS = "GdkWin32GL"


class _DwmBlurBehind(ctypes.Structure):
    _fields_ = [("dwFlags", wt.DWORD), ("fEnable", wt.BOOL), ("hRgnBlur", wt.HANDLE),
                ("fTransitionOnMaximized", wt.BOOL)]


def request_gpu_rendering():
    """Opt into Direct Composition. Call before GTK initialises."""
    flags = [f for f in os.environ.get("GDK_DEBUG", "").split(",") if f]
    if "dcomp" not in flags:
        os.environ["GDK_DEBUG"] = ",".join(flags + ["dcomp"])


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


def _enable_alpha(hwnd):
    dwmapi = ctypes.WinDLL("dwmapi")
    gdi32 = ctypes.WinDLL("gdi32")
    gdi32.CreateRectRgn.restype = wt.HANDLE
    empty = gdi32.CreateRectRgn(0, 0, -1, -1)
    blur = _DwmBlurBehind(_DWM_BB_ENABLE | _DWM_BB_BLURREGION, True, empty, False)
    dwmapi.DwmEnableBlurBehindWindow(wt.HWND(hwnd), ctypes.byref(blur))
    gdi32.DeleteObject(wt.HANDLE(empty))


# winuser.h; Python ships no binding for these.
_EVENT_OBJECT_CREATE = 0x8000
_WINEVENT_OUTOFCONTEXT = 0x0000
_OBJID_WINDOW = 0
_WINEVENTPROC = ctypes.WINFUNCTYPE(None, wt.HANDLE, wt.DWORD, wt.HWND, wt.LONG, wt.LONG,
                                   wt.DWORD, wt.DWORD)
_hook = None        # (hook handle, callback): kept alive for the process's lifetime


def watch_gl_windows():
    """Fix every GL window GTK creates from now on, and the ones it already has.

    Each popup surface (menus, popovers, tooltips, dialogs) gets its own GL
    window when it attaches, so hooking only the main window left their corners
    black. An out-of-context WinEvent hook on window creation, filtered to this
    process, is delivered through the GTK main thread's message loop.
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
            _enable_alpha(hwnd)

    callback = _WINEVENTPROC(on_create)
    handle = user32.SetWinEventHook(_EVENT_OBJECT_CREATE, _EVENT_OBJECT_CREATE, None,
                                    callback, os.getpid(), 0, _WINEVENT_OUTOFCONTEXT)
    _hook = (handle, callback)
    for hwnd in _gl_windows():
        _enable_alpha(hwnd)
