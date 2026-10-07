import ctypes
from ctypes import wintypes


kernel32 = ctypes.WinDLL(
    "kernel32",
    use_last_error=True,
)

GetCurrentProcess = kernel32.GetCurrentProcess
GetCurrentProcess.argtypes = []
GetCurrentProcess.restype = wintypes.HANDLE

SetProcessAffinityMask = kernel32.SetProcessAffinityMask
SetProcessAffinityMask.argtypes = [
    wintypes.HANDLE,
    ctypes.c_size_t,
]
SetProcessAffinityMask.restype = wintypes.BOOL

GetProcessAffinityMask = kernel32.GetProcessAffinityMask
GetProcessAffinityMask.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
]
GetProcessAffinityMask.restype = wintypes.BOOL


def cpu_list_to_mask(cpus):
    mask = 0

    for cpu in cpus:
        mask |= 1 << cpu

    return mask


def set_affinity(cpus):
    mask = cpu_list_to_mask(cpus)

    ok = SetProcessAffinityMask(
        GetCurrentProcess(),
        mask,
    )

    if not ok:
        raise OSError(
            ctypes.get_last_error(),
            "SetProcessAffinityMask failed",
        )


def get_affinity():
    process_mask = ctypes.c_size_t()
    system_mask = ctypes.c_size_t()

    ok = GetProcessAffinityMask(
        GetCurrentProcess(),
        ctypes.byref(process_mask),
        ctypes.byref(system_mask),
    )

    if not ok:
        raise OSError(
            ctypes.get_last_error(),
            "GetProcessAffinityMask failed",
        )

    return {
        "process_mask": process_mask.value,
        "system_mask": system_mask.value,
    }


if __name__ == "__main__":
    print(get_affinity())
