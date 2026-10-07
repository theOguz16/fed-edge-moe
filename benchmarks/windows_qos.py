import ctypes
from ctypes import wintypes


PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1

# PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
ProcessPowerThrottling = 4


class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
    _fields_ = [
        ("Version", wintypes.ULONG),
        ("ControlMask", wintypes.ULONG),
        ("StateMask", wintypes.ULONG),
    ]


kernel32 = ctypes.WinDLL(
    "kernel32",
    use_last_error=True,
)


GetCurrentProcess = kernel32.GetCurrentProcess
GetCurrentProcess.argtypes = []
GetCurrentProcess.restype = wintypes.HANDLE


SetProcessInformation = kernel32.SetProcessInformation
SetProcessInformation.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
]
SetProcessInformation.restype = wintypes.BOOL


GetProcessInformation = kernel32.GetProcessInformation
GetProcessInformation.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
]
GetProcessInformation.restype = wintypes.BOOL


def _current_process():
    handle = GetCurrentProcess()

    if not handle:
        raise OSError(
            ctypes.get_last_error(),
            "GetCurrentProcess failed",
        )

    return handle


def _set_state(control_mask, state_mask):
    state = PROCESS_POWER_THROTTLING_STATE()

    state.Version = (
        PROCESS_POWER_THROTTLING_CURRENT_VERSION
    )

    state.ControlMask = control_mask
    state.StateMask = state_mask

    ok = SetProcessInformation(
        _current_process(),
        ProcessPowerThrottling,
        ctypes.byref(state),
        ctypes.sizeof(state),
    )

    if not ok:
        err = ctypes.get_last_error()

        raise OSError(
            err,
            f"SetProcessInformation failed "
            f"(control={control_mask:#x}, "
            f"state={state_mask:#x})",
        )


def set_process_qos(mode):
    if mode == "system":
        # Windows decides.
        _set_state(
            0,
            0,
        )

    elif mode == "eco":
        # EcoQoS:
        # take control and enable execution-speed throttling.
        _set_state(
            PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
            PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
        )

    elif mode == "high":
        # HighQoS:
        # take control but disable execution-speed throttling.
        _set_state(
            PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
            0,
        )

    else:
        raise ValueError(
            f"Unknown QoS mode: {mode}"
        )


def get_process_qos_state():
    state = PROCESS_POWER_THROTTLING_STATE()

    state.Version = (
        PROCESS_POWER_THROTTLING_CURRENT_VERSION
    )

    ok = GetProcessInformation(
        _current_process(),
        ProcessPowerThrottling,
        ctypes.byref(state),
        ctypes.sizeof(state),
    )

    if not ok:
        err = ctypes.get_last_error()

        raise OSError(
            err,
            "GetProcessInformation failed",
        )

    controlled = bool(
        state.ControlMask
        & PROCESS_POWER_THROTTLING_EXECUTION_SPEED
    )

    throttled = bool(
        state.StateMask
        & PROCESS_POWER_THROTTLING_EXECUTION_SPEED
    )

    return {
        "version": int(state.Version),
        "control_mask": int(state.ControlMask),
        "state_mask": int(state.StateMask),

        "execution_speed_controlled":
            controlled,

        "execution_speed_throttled":
            throttled,
    }


if __name__ == "__main__":
    print(
        "process handle:",
        _current_process(),
    )

    for mode in [
        "system",
        "eco",
        "high",
    ]:
        set_process_qos(mode)

        print(
            mode,
            "=>",
            get_process_qos_state(),
        )

    # Restore normal Windows behavior.
    set_process_qos("system")
