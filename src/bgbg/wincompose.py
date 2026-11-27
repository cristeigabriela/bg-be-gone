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

GTK recreates the GL window when a surface re-attaches, so the call is made on
every map. The Vulkan renderer uses its own window class (GdkWin32Vulkan) and is
not covered: it isn't GTK's default under dcomp, and the fix is untested there.
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


def enable_gl_window_alpha():
    """Let DWM compose GTK's GL window with its alpha. Safe to call repeatedly;
    a no-op when GTK renders without that window (Cairo, Vulkan)."""
    dwmapi = ctypes.WinDLL("dwmapi")
    gdi32 = ctypes.WinDLL("gdi32")
    gdi32.CreateRectRgn.restype = wt.HANDLE
    for hwnd in _gl_windows():
        empty = gdi32.CreateRectRgn(0, 0, -1, -1)
        blur = _DwmBlurBehind(_DWM_BB_ENABLE | _DWM_BB_BLURREGION, True, empty, False)
        dwmapi.DwmEnableBlurBehindWindow(wt.HWND(hwnd), ctypes.byref(blur))
        gdi32.DeleteObject(wt.HANDLE(empty))
