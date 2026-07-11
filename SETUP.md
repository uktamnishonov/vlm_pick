# Setup — bringing `vlm_pick` up on a fresh Jetson

Everything this project depends on, where it comes from, and the three things that
**silently break on a stock Jetson**. Written from a working install (Jetson Orin Nano,
JetPack 6 / L4T R36.4.7, Ubuntu 22.04, Python 3.10.12).

---

## Read this first: the three landmines

A fresh `git clone` of the upstream dependencies **will not work**. In order of how badly
they'll waste your day:

| # | Problem | Symptom | Fix |
| --- | --- | --- | --- |
| 1 | **The Jetson has no USB-CAN driver.** Stock L4T ships none. | The adapter enumerates on USB but `can1` / `can2` never appear. `ip link` shows nothing. | Build `peak_usb.ko` from kernel sources — [below](#1-can-driver-must-be-built) |
| 2 | **Upstream `openarm` requires Python 3.11.** JetPack ships **3.10**. | `pip install -e .` refuses outright, or `ImportError: cannot import name 'StrEnum' from 'enum'` | Apply `patches/anvil-openarm-py310-canfd.patch` |
| 3 | **Upstream `openarm` opens CAN without FD.** The Damiao motors *always reply in FD framing*. | The worst one: **no error at all.** Commands appear to send, motors never respond, everything silently times out or reads zeros. | Same patch (`fd=True` on every `can.Bus`) |

**Landmine 3 is the cruel one.** The motors accept classic-CAN commands but reply in CAN-FD
frames, and a non-FD socket *silently discards* those replies. Nothing errors. The arm just
appears dead. If your other Jetson reads all-zero joint angles or times out on every call,
this is almost certainly why.

---

## Dependency map

| Component | Source | Pinned at | Notes |
| --- | --- | --- | --- |
| `openarm` (kinematics, URDF, Damiao driver) | https://github.com/anvil-robotics/openarm | `5007dd2` | **Needs our patch.** Editable install. Provides `openarm.kinematics.inverse.ikpy` and `urdf/openarm_bimanual.urdf`. |
| `openarm_can` (CAN SDK + CLI) | https://github.com/enactic/openarm_can | `b52805b` | C++ + Python bindings. Installed version `1.2.9`. |
| `CLI11` | https://github.com/CLIUtils/CLI11 | — | Build dependency of `openarm_can`. |
| `peak_usb.ko` | kernel.org linux **v5.15.148** `drivers/net/can/usb/peak_usb/` | — | Must match `uname -r` exactly. **Rebuild after any kernel/L4T upgrade.** |
| Python packages | PyPI | see `requirements.txt` | `depthai 3.7.1`, `numpy 2.2.6`, `scipy 1.15.3`, `ikpy 4.0.0`, `python-can 4.6.1`, `opencv-python`, `requests` |
| Gemini API | Google AI Studio | — | `GEMINI_API_KEY` in `.env` |

Hardware: OpenArm bimanual (8 Damiao motors/arm: 7 joints + gripper), **PEAK PCAN-USB Pro FD**
adapter (USB `0c72:0011`), OAK-D Pro W camera.

---

## 1. CAN driver (must be built)

Stock L4T has no USB-CAN drivers at all. Build the PEAK driver against **your exact running
kernel** (`uname -r` — ours is `5.15.148-tegra`, so we used kernel.org v5.15.148 sources):

```bash
# sources: linux-5.15.148/drivers/net/can/usb/peak_usb/
# build out-of-tree, then:
sudo cp peak_usb.ko /lib/modules/$(uname -r)/extra/
sudo depmod -a
sudo modprobe peak_usb
```

Verify: `modinfo peak_usb` should print `/lib/modules/<kernel>/extra/peak_usb.ko`.

Once installed it survives reboot and autoloads on hotplug. **It must be rebuilt after any
kernel or JetPack upgrade** — a version mismatch means `modprobe` fails and the CAN
interfaces vanish again.

## 2. Bring up the CAN interfaces (NOT persistent)

The Damiao motors run **CAN FD at 1 Mbit/s nominal / 5 Mbit/s data**:

```bash
sudo ip link set can1 up type can bitrate 1000000 dbitrate 5000000 fd on   # RIGHT arm
sudo ip link set can2 up type can bitrate 1000000 dbitrate 5000000 fd on   # LEFT arm
```

- `can1` = **right** arm, `can2` = **left** arm. `can0` is the Jetson's onboard mttcan
  controller and is **not** connected to the arm.
- **This does not survive a reboot.** Re-run it every boot, or write a systemd-networkd /
  udev rule. A "dead arm" after a reboot is usually just this.

Safe liveness probe — elicits a status reply with **no motion**:

```bash
cansend can1 001##1FFFFFFFFFFFFFFFD     # Damiao disable frame
```

> ⚠️ Never send `...FE` (re-zeroes the motor's flash-stored mechanical zero) or `...FC`
> (energizes the arm) as a "test".

## 3. Python environment

```bash
python3 -m venv ~/openarm/venv          # Python 3.10.12 on JetPack 6
source ~/openarm/venv/bin/activate
pip install -r requirements.txt
```

## 4. `openarm` from source — **with the patch**

This is the step that fails on a fresh machine.

```bash
git clone https://github.com/anvil-robotics/openarm.git ~/openarm/anvil_openarm
cd ~/openarm/anvil_openarm
git checkout 5007dd2                    # what we validated against

git apply /path/to/vlm_pick/patches/anvil-openarm-py310-canfd.patch

pip install -e .
```

The patch touches 12 files and does exactly two things:

**Python 3.10 compatibility** (upstream targets 3.11; JetPack ships 3.10):
- `pyproject.toml` — `requires-python >=3.11` → `>=3.10`, so pip will install at all
- `openarm/damiao/motor.py` — backports `enum.StrEnum` (3.11+ only)
- `openarm/netcan/client.py` — falls back to `typing_extensions.Self`

**CAN FD** — adds `fd=True` to every `can.Bus(...)` construction, in 9 files under
`openarm/damiao/`. Without this the motors' FD replies are silently dropped and the arm
looks dead.

If the patch fails to apply because upstream moved on, the changes are small and mechanical
— read the patch and redo them by hand.

## 5. `openarm_can`

```bash
git clone https://github.com/enactic/openarm_can.git ~/openarm/openarm_can
# needs CLI11: https://github.com/CLIUtils/CLI11
# build + install per its README; gives you openarm-can-cli et al.
```

## 6. Gemini key

```bash
cp .env.example .env      # add GEMINI_API_KEY from https://aistudio.google.com
```

---

## Verify, in this order

Do not skip ahead — each step isolates one layer.

```bash
# 1. Driver + interfaces exist
ip -details link show can1        # must show "can" and FD bitrates

# 2. Motors actually reply  (all-zeros = FD problem or bus down, NOT a real pose)
python ~/openarm/hello_openarm.py --once

# 3. Camera
python camera.py                  # expect depth coverage >20%, K printed

# 4. VLM
python vlm.py "the cup"           # expect a pixel

# 5. Full perception, no motion
python servo.py "Black coffee cup" --locate-only
```

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `can1` doesn't exist | No `peak_usb` driver, or adapter unplugged | §1, then `lsusb \| grep 0c72` |
| `can1` gone after reboot | `ip link` bring-up isn't persistent | Re-run §2 |
| Arm reads **all zeros**, no motion, no error | **Non-FD socket dropping FD replies** | Apply the patch (§4). Confirm `fd=True` reached `can.Bus`. |
| `ImportError: cannot import name 'StrEnum'` | Python 3.10 vs upstream's 3.11 | Apply the patch (§4) |
| `pip install -e .` refuses: requires-python | Same | Apply the patch (§4) |
| TX ok but RX 0 (`ip -s link show can1`) | Loose plug at the PEAK adapter — this bit us | Reseat the connector |
| Depth coverage ~1% | Over-aggressive stereo filtering | See `camera.py`; **extended disparity disables the median filter** |
| OAK crashes / USB brown-out | IR projector at full power | `setIrLaserDotProjectorIntensity(0.8)`, not 1.0 |
| "tape lost" mid-reach | HSV range wrong for your lighting/tape | Re-sample from a saved `servo_lost.jpg`; see `COLOR_RANGES` in `calibrate.py` |

## Safety

The arm moves on its own. Clear the workspace, hand on the e-stop. Ctrl+C parks and
disables; a `SIGKILL` does **not** — it leaves the arm energized mid-air, and `park_home.py`
brings it down.
