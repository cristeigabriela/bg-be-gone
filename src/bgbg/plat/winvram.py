"""Total and free VRAM of the main GPU on Windows, with no extra dependencies.

DXGI names the adapters and their dedicated memory; the "GPU Adapter Memory"
performance counter gives the memory in use on each adapter across all
processes. DXGI's own per-process budget is not used for "free": on an RX
6750 XT it reported 11475 MiB of budget while the desktop already held about
2 GiB of the 12243 MiB, so it does not account for other processes.
"""
import ctypes
import ctypes.wintypes as wt
import uuid


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16),
                ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_ubyte * 8)]


def _guid(s):
    u = uuid.UUID(s)
    g = _GUID(u.time_low, u.time_mid, u.time_hi_version)
    g.Data4[:] = list(u.bytes[8:])
    return g


# Interface ids and constants from dxgi.h / pdh.h; Python ships no binding for them.
_IID_IDXGIFactory1 = _guid("770aae78-f26f-4dba-a829-253c83d1b387")
_DXGI_ADAPTER_FLAG_SOFTWARE = 2
_PDH_FMT_LARGE = 0x00000400

# COM vtable slots. IUnknown: 0-2. IDXGIObject: 3-6. IDXGIFactory: 7-11, so
# IDXGIFactory1.EnumAdapters1 is 12. IDXGIAdapter: 7-9, so
# IDXGIAdapter1.GetDesc1 is 10.
_SLOT_RELEASE = 2
_SLOT_ENUM_ADAPTERS1 = 12
_SLOT_GET_DESC1 = 10


class _AdapterDesc1(ctypes.Structure):
    _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", ctypes.c_uint),
                ("DeviceId", ctypes.c_uint), ("SubSysId", ctypes.c_uint),
                ("Revision", ctypes.c_uint), ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory", ctypes.c_size_t),
                ("LuidLow", wt.DWORD), ("LuidHigh", wt.LONG), ("Flags", ctypes.c_uint)]


class _PdhValue(ctypes.Structure):
    _fields_ = [("CStatus", wt.DWORD), ("largeValue", ctypes.c_longlong)]


def _vcall(obj, slot, *argtypes):
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    return ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, *argtypes)(vtbl[slot])


def _release(obj):
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtbl[_SLOT_RELEASE])(obj)


def _main_adapter():
    """(description, dedicated bytes, luid low, luid high) of the hardware
    adapter with the most dedicated memory, or None."""
    factory = ctypes.c_void_p()
    if ctypes.windll.dxgi.CreateDXGIFactory1(ctypes.byref(_IID_IDXGIFactory1),
                                             ctypes.byref(factory)) != 0:
        return None
    best = None
    try:
        enum = _vcall(factory, _SLOT_ENUM_ADAPTERS1, wt.UINT, ctypes.POINTER(ctypes.c_void_p))
        i = 0
        while True:
            adapter = ctypes.c_void_p()
            try:
                enum(factory, i, ctypes.byref(adapter))
            except OSError:          # DXGI_ERROR_NOT_FOUND: no more adapters
                break
            d = _AdapterDesc1()
            try:
                _vcall(adapter, _SLOT_GET_DESC1, ctypes.POINTER(_AdapterDesc1))(
                    adapter, ctypes.byref(d))
            finally:
                _release(adapter)
            if not d.Flags & _DXGI_ADAPTER_FLAG_SOFTWARE and (
                    best is None or d.DedicatedVideoMemory > best[1]):
                best = (d.Description, d.DedicatedVideoMemory, d.LuidLow, d.LuidHigh)
            i += 1
    finally:
        _release(factory)
    return best


def _used_bytes(luid_low, luid_high):
    """Dedicated memory in use on that adapter, all processes, or None."""
    pdh = ctypes.WinDLL("pdh")
    for fn in ("PdhOpenQueryW", "PdhAddEnglishCounterW", "PdhCollectQueryData",
               "PdhGetFormattedCounterValue", "PdhCloseQuery"):
        getattr(pdh, fn).restype = wt.DWORD    # PDH_STATUS is unsigned
    # The counter names its instance by the LUID as two unsigned hex halves;
    # the mask prints DXGI's signed high half as unsigned.
    path = (r"\GPU Adapter Memory(luid_0x%08x_0x%08x_phys_0)\Dedicated Usage"
            % (luid_high & 0xFFFFFFFF, luid_low))
    query, counter = wt.HANDLE(), wt.HANDLE()
    if pdh.PdhOpenQueryW(None, None, ctypes.byref(query)):
        return None
    try:
        if (pdh.PdhAddEnglishCounterW(query, path, None, ctypes.byref(counter))
                or pdh.PdhCollectQueryData(query)):
            return None
        v = _PdhValue()
        if pdh.PdhGetFormattedCounterValue(counter, _PDH_FMT_LARGE, None, ctypes.byref(v)):
            return None
        return v.largeValue
    finally:
        pdh.PdhCloseQuery(query)


def probe():
    """``{"total": MiB, "free": MiB, "gpu": True, "name": str}`` or None."""
    adapter = _main_adapter()
    if adapter is None or not adapter[1]:
        return None
    name, total, luid_low, luid_high = adapter
    used = _used_bytes(luid_low, luid_high)
    # Unknown usage is no answer, not "all free". On DirectML an oversized
    # model raises no out-of-memory error, it spills into system RAM, so the
    # caller's step-down never fires and this number is the only guard.
    if used is None:
        return None
    total_mb = total >> 20
    return {"total": total_mb, "free": max(0, total_mb - (used >> 20)), "gpu": True, "name": name}


if __name__ == "__main__":
    print(probe())
