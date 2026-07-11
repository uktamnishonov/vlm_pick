#!/usr/bin/env python3
"""Hand-guided camera<->arm-base calibration for the RIGHT OpenArm (can1).

Torque stays OFF the whole time — you move the arm by hand. A brightly
colored tape/sticker must be wrapped on the RIGHT gripper fingertip.

For each pose: the camera finds the tape blob (HSV) and deprojects it with
stereo depth -> camera-frame point; motor angles are read over CAN and run
through FK -> base-frame point. A rigid Umeyama fit maps camera to base.
If the fit residual is high, all 2^7 joint-sign conventions are searched
automatically (a wrong sign warps the point cloud non-rigidly).

Usage:
    python calibrate.py --test-blob          # one frame, check tape detection
    python calibrate.py --poses 10           # full run (interactive)
    python calibrate.py --refit              # re-fit from saved samples.json
    python calibrate.py --color pink         # tape color: green|pink|orange|blue
"""
import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kin
from camera import OakCamera

SAMPLES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples.json")

# HSV ranges (H 0-179). Two ranges for red-ish hues that wrap around 0.
COLOR_RANGES = {
    "green": [((40, 80, 80), (85, 255, 255))],
    "pink": [((140, 60, 120), (175, 255, 255))],
    "orange": [((5, 100, 120), (22, 255, 255))],
    "blue": [((95, 100, 80), (125, 255, 255))],
    # Re-sampled from a real servo_lost.jpg (gripper reaching, tape dimmer/more
    # saturated than head-on): tape is H~28, S 90+, V 110+. Old range demanded
    # V>=195 & S<=110 and dropped the tape mid-reach (the phantom "lost"). This
    # range keeps it AND stays clean: H floor 22 rejects the reddish cardboard
    # (H 11-16), S floor 90 rejects wood table / white paper (S<=54). Verified
    # 326 tape px, 0 box/table/paper px.
    "yellow": [((22, 90, 90), (42, 255, 255))],
}
MIN_BLOB_AREA = 20  # px

def find_blob(bgr, color):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in COLOR_RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    # CLOSE only — an OPEN erases the tape when it's a thin edge-on sliver
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    big = max(contours, key=cv2.contourArea)
    if cv2.contourArea(big) < MIN_BLOB_AREA:
        return None, mask
    m = cv2.moments(big)
    return (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])), mask


def connect_right_arm():
    """Read-only CAN connection (openarm_can), torque never enabled."""
    import openarm_can as oa

    arm = oa.OpenArm("can1", True)
    arm.init_arm_motors(
        [oa.MotorType.DM8009] * 2 + [oa.MotorType.DM4340] * 2 + [oa.MotorType.DM4310] * 3,
        [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07],
        [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17],
    )
    arm.set_callback_mode_all(oa.CallbackMode.IGNORE)
    time.sleep(0.005)
    arm.recv_all(2000)
    arm.set_callback_mode_all(oa.CallbackMode.STATE)
    return arm


def read_joints(arm, n=5):
    """Median motor angles (rad, 7 joints) over n refreshes."""
    reads = []
    for _ in range(n):
        arm.refresh_all()
        time.sleep(0.003)
        arm.recv_all(1000)
        reads.append([m.get_position() for m in arm.get_arm().get_motors()])
        time.sleep(0.02)
    return np.median(np.array(reads), axis=0)


def show_debug(bgr, mask, blob=None, point3d=None):
    vis = bgr.copy()

    if blob is not None:
        cv2.drawMarker(
            vis, blob, (0, 0, 255),
            cv2.MARKER_CROSS, 25, 2
        )
        cv2.circle(vis, blob, 6, (0, 255, 0), -1)

    if point3d is not None:
        text = f"XYZ: {point3d[0]:.3f}, {point3d[1]:.3f}, {point3d[2]:.3f} m"
        cv2.putText(
            vis, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0), 2
        )

    cv2.imshow("Camera", vis)
    cv2.imshow("Mask", mask)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        raise KeyboardInterrupt

def capture_tape_point(cam, color, tries=8):
    """Median blob pixel + camera-frame 3D point over several frames."""
    hits, depths = [], []
    for _ in range(tries):
        bgr, depth = cam.read()
        blob, mask = find_blob(bgr, color)
        point = None
        if blob is not None:
            hits.append(blob)
            depths.append(depth)
            point = cam.point3d(blob[0], blob[1], depth, patch=7)
        show_debug(bgr, mask, blob, point)

    if len(hits) < 3:
        return None, None, bgr, mask
    uv = np.median(np.array(hits), axis=0).astype(int)
    p = cam.point3d(int(uv[0]), int(uv[1]), depths[-1], patch=7)
    if p is not None and p[2] < 0.25:
        print(f"  tape only {p[2]:.2f} m from camera — too close for reliable"
              " depth, keep it at least ~25 cm away")
        return None, None, bgr, mask
    if p is not None and p[2] > 1.0:
        print(f"  blob at {p[2]:.2f} m — that's beyond the workspace, a false "
              "detection on the background. Ignoring.")
        return None, None, bgr, mask
    return (int(uv[0]), int(uv[1])), p, bgr, mask


def _robust_fit(cam, motors, signs, min_keep=8, target=0.02):
    """Fit with fixed signs, iteratively dropping the single worst-residual
    pose until rms hits target or we reach min_keep. Bad poses (wrist-roll
    tape jumps, stray depth) get rejected instead of poisoning everything."""
    keep = list(range(len(cam)))
    best = None
    while True:
        R, t, d, rms, res = kin.fit_extended([cam[i] for i in keep],
                                             [motors[i] for i in keep], signs)
        best = (R, t, d, rms, res, list(keep))
        if rms <= target or len(keep) <= min_keep:
            return best
        keep.pop(int(np.argmax(res)))


def fit_and_report(samples):
    cam = [s["p_cam"] for s in samples]
    motors = [s["q_motor"] for s in samples]

    # Identity signs are physically confirmed (they matched the J2 joint-limit
    # convention on the 24 mm hand-guided fit). We NEVER flip them from noisy
    # data — a fit that only works with flipped signs means the DATA is bad
    # (usually wrist rotation), not the convention, and flipped signs would
    # drive the arm the wrong way. So: force identity, reject outliers.
    print(f"\nfitting {len(samples)} poses (identity signs, outlier-robust)...")
    signs = [1.0] * 7
    R, t, d, rms, res, keep = _robust_fit(cam, motors, signs)

    dropped = sorted(set(range(len(samples))) - set(keep))
    print(f"kept {len(keep)}/{len(samples)} poses"
          + (f", dropped outliers {[i+1 for i in dropped]}" if dropped else ""))
    print(f"signs {[int(x) for x in signs]}  rms {rms*1000:.1f} mm  "
          f"tape offset {np.round(np.asarray(d)*100,1)} cm")
    print("per-pose residuals (mm):", np.round(res * 1000, 1))

    if rms > 0.03:
        print("\n*** POOR FIT — do NOT trust this for grasping. Cause is almost")
        print("*** always WRIST ROTATION: the tape band shows a different face")
        print("*** to the camera as the wrist turns. REDO calibration keeping")
        print("*** the gripper/wrist FIXED, moving only shoulder + elbow.")
        print("*** (saving so you can still dry-run, but expect big misses)")

    kept_motors = [motors[i] for i in keep]
    base_pts = np.array([kin.fk_frame(kin.motor_to_urdf(q, signs))[:3, 3]
                         for q in kept_motors])
    path = kin.save_calibration(R, t, signs, np.array([cam[i] for i in keep]),
                                base_pts, rms, d)
    print(f"\nsaved {path}  (rms {rms*1000:.1f} mm, {len(keep)} poses)")
    print("workspace box (base frame):")
    print("  lo:", np.round(base_pts.min(0) - 0.12, 3))
    print("  hi:", np.round(base_pts.max(0) + 0.12, 3))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--poses", type=int, default=12)
    ap.add_argument("--color", default="yellow", choices=COLOR_RANGES)
    ap.add_argument("--interval", type=float, default=2.0,
                    help="seconds between auto-captures (default 2)")
    ap.add_argument("--test-blob", action="store_true")
    ap.add_argument("--refit", action="store_true")
    args = ap.parse_args()

    if args.refit:
        with open(SAMPLES_PATH) as f:
            fit_and_report(json.load(f))
        return

    cam = OakCamera()
    try:
        if args.test_blob:
            uv, p, bgr, mask = capture_tape_point(cam, args.color)
            cv2.imwrite("blob_mask.jpg", mask)
            if uv is None:
                print(f"no {args.color} blob found — see blob_mask.jpg")
            else:
                cv2.drawMarker(bgr, uv, (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
                cv2.imwrite("blob_hit.jpg", bgr)
                print(f"blob at {uv}, 3D cam point {None if p is None else np.round(p,3)} m")
                print("saved blob_hit.jpg / blob_mask.jpg")
            cv2.waitKey(0)
            return

        arm = connect_right_arm()
        # comms sanity check: a live arm never reads EXACTLY zero on all joints
        # (there is always a little encoder noise). Exact zeros = no CAN frames
        # → the bus is down or the arm is unpowered. Refuse rather than collect
        # garbage that silently ruins the fit.
        probe = read_joints(arm, n=3)
        if np.max(np.abs(probe)) < 1e-6:
            sys.exit(
                "\nARM NOT RESPONDING: all joints read exactly 0.000 — the CAN "
                "bus is almost certainly down.\nBring it up and retry:\n"
                "  sudo ip link set can1 up type can bitrate 1000000 "
                "dbitrate 5000000 fd on\nverify with:  ip -br link | grep can"
            )
        print(f"AUTO mode: collecting {args.poses} poses, hands-free.")
        print("Move the RIGHT arm to a new pose, hold still ~2 s — it captures")
        print("by itself. Use the SHOULDER and ELBOW to move it around the")
        print("workspace (vary height too). *** DO NOT TWIST THE WRIST OR")
        print("GRIPPER *** — hold the forearm, keep the tape facing the camera")
        print("the same way each time. Keep the tape 25-80 cm from the camera.\n")
        samples = []
        while len(samples) < args.poses:
            q1 = read_joints(arm, n=2)
            uv, p_cam, _, _ = capture_tape_point(cam, args.color, tries=4)
            q2 = read_joints(arm, n=2)
            if uv is None or p_cam is None:
                print("  ... waiting for tape in view", end="\r")
                time.sleep(0.8)
                continue
            if np.max(np.abs(np.degrees(q2 - q1))) > 1.5:
                print("  ... arm moving, hold still   ", end="\r")
                time.sleep(0.5)
                continue
            q = (q1 + q2) / 2.0
            if np.max(np.abs(q)) < 1e-6:
                print("  ... arm reading zeros (CAN down?) — skipping", end="\r")
                time.sleep(0.5)
                continue
            samples.append({"p_cam": p_cam.tolist(), "q_motor": q.tolist(),
                            "uv": list(uv)})
            with open(SAMPLES_PATH, "w") as f:
                json.dump(samples, f, indent=1)
            print(f"[{len(samples)}/{args.poses}] captured  blob {uv}  "
                  f"cam {np.round(p_cam, 3)}  joints {np.round(np.degrees(q), 0)}")
            time.sleep(args.interval)
        print()
        fit_and_report(samples)
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
