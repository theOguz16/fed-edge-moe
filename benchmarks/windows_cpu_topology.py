import ctypes
import struct
from collections import defaultdict

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

GetSystemCpuSetInformation = kernel32.GetSystemCpuSetInformation
GetSystemCpuSetInformation.argtypes = [
    ctypes.c_void_p,
    ctypes.c_ulong,
    ctypes.POINTER(ctypes.c_ulong),
    ctypes.c_void_p,
    ctypes.c_ulong,
]
GetSystemCpuSetInformation.restype = ctypes.c_bool

needed = ctypes.c_ulong(0)

GetSystemCpuSetInformation(
    None,
    0,
    ctypes.byref(needed),
    None,
    0,
)

if needed.value == 0:
    raise RuntimeError(
        f"GetSystemCpuSetInformation size query failed: "
        f"{ctypes.get_last_error()}"
    )

buf = ctypes.create_string_buffer(needed.value)

ok = GetSystemCpuSetInformation(
    buf,
    needed.value,
    ctypes.byref(needed),
    None,
    0,
)

if not ok:
    raise RuntimeError(
        f"GetSystemCpuSetInformation failed: "
        f"{ctypes.get_last_error()}"
    )

raw = buf.raw[:needed.value]

rows = []
offset = 0

while offset < len(raw):
    size, info_type = struct.unpack_from("<II", raw, offset)

    if size <= 0:
        raise RuntimeError("Invalid CPU Set record size")

    # CpuSetInformation == 0
    if info_type == 0 and size >= 20:
        cpu_set_id = struct.unpack_from("<I", raw, offset + 8)[0]
        group = struct.unpack_from("<H", raw, offset + 12)[0]

        logical = raw[offset + 14]
        core = raw[offset + 15]
        llc = raw[offset + 16]
        numa = raw[offset + 17]
        efficiency = raw[offset + 18]
        flags = raw[offset + 19]

        rows.append({
            "id": cpu_set_id,
            "group": group,
            "logical": logical,
            "core": core,
            "llc": llc,
            "numa": numa,
            "efficiency": efficiency,
            "parked": flags & 0x1,
        })

    offset += size


print(
    f"{'SetID':>5} "
    f"{'Group':>5} "
    f"{'Logical':>7} "
    f"{'Core':>5} "
    f"{'Eff':>5} "
    f"{'Parked':>7}"
)

print("-" * 45)

for r in sorted(
    rows,
    key=lambda x: (x["group"], x["logical"]),
):
    print(
        f"{r['id']:5} "
        f"{r['group']:5} "
        f"{r['logical']:7} "
        f"{r['core']:5} "
        f"{r['efficiency']:5} "
        f"{r['parked']:7}"
    )


cores = defaultdict(list)

for r in rows:
    cores[(r["group"], r["core"])].append(
        r["logical"]
    )

print("\nSMT / physical-core mapping")

for key in sorted(cores):
    print(
        f"group={key[0]} "
        f"core={key[1]:2} "
        f"logical={sorted(cores[key])}"
    )

eff_classes = sorted(
    {r["efficiency"] for r in rows}
)

print("\nEfficiency classes:", eff_classes)
print("Physical cores:", len(cores))
print("Logical processors:", len(rows))
