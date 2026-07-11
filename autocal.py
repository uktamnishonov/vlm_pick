#!/usr/bin/env python3
"""POWERED auto-calibration of the camera<->right-arm transform.

Why this exists: hand-guiding (calibrate.py) fails because a human cannot
hold the wrist steady, and the yellow tape is a BAND around the fingertip —
the instant the wrist rolls, a different face of the band faces the camera
and its detected centroid jumps. Three hand runs gave 18 / 24 / 41 mm.

This takes the human out of the loop. Now that the motor zeros are trued
(openarm-can-zero-position-calibration), FK is trustworthy, so the ROBOT
poses itself: it sweeps only the shoulder + elbow (J1, J2, J4) while the
motors HOLD the wrist joints (J3, J5, J6, J7) and the gripper at fixed
constants. The tape therefore presents the SAME face every pose, so its
offset from the TCP is a true constant and fit_extended nails it.

Each pose is screened through the column-clearance gate (target AND path)
before any motion, moves are velocity-capped POS_VEL, and the arm always
returns to a safe rest and disables on exit.

SAFETY: the arm moves ON ITS OWN. Clear the workspace, keep a hand on the
e-stop. Preview first with --dry-run (no motors touched).

Usage:
    python autocal.py --dry-run            # print the safe pose plan, no motion
    python autocal.py                      # run it (arm moves; collect + fit)
    python autocal.py --poses 14           # aim for more captures
    python autocal.py --refit              # re-fit from the saved samples.json
"""
import argparse
import asyncio
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kin
from arm_io import RightArm
from calibrate import SAMPLES_PATH, capture_tape_point, fit_and_report
from camera import OakCamera

# Fixed wrist + gripper (motor deg). J3,J5,J6,J7 held at 0 so the tape band
# never rolls; only J1,J2,J4 (shoulder yaw / shoulder pitch / elbow) sweep.
READY_DEG = [0, 30, 0, 90, 0, 0, 0]
GRIP_FIXED_RAD = 0.0                       # gripper closed & fixed all run

# Sweep grid (motor deg). J1 gives left-right (Y) spread for a well-conditioned
# rigid fit; J2/J4 give reach + height spread. Kept modest so the tape's
# viewing angle to the camera stays similar pose-to-pose.
J1_GRID = [-20, 0, 20, 40]
J2_GRID = [15, 30, 45]
J4_GRID = [70, 95, 115]

CAL_VELOCITY = 0.20                        # rad/s — slow, deliberate
CLEARANCE_MIN = kin.CLEARANCE_MIN_M        # 0.10 m to the central body column
REACH_MIN, REACH_MAX = 0.22, 0.62          # horizontal TCP dist from base (m)
HEIGHT_LO, HEIGHT_HI = -0.60, 0.15         # base-frame Y (up) band for the TCP


def gen_poses():
    """All grid poses as 7-joint motor-deg configs (wrist fixed at 0)."""
    poses = []
    for j1 in J1_GRID:
        for j2 in J2_GRID:
            for j4 in J4_GRID:
                poses.append([j1, j2, 0.0, j4, 0.0, 0.0, 0.0])
    return poses


def evaluate(q7_deg):
    """(safe, clearance_m, tcp_xyz) for a target config. Identity signs mean
    motor deg == URDF deg, so FK/clearance apply directly."""
    q7 = np.radians(q7_deg)
    clr = kin.min_column_clearance(q7)
    tcp = kin.fk(q7)
    reach = float(np.hypot(tcp[0], tcp[2]))      # X-Z is horizontal; Y is up
    safe = (clr >= CLEARANCE_MIN and REACH_MIN <= reach <= REACH_MAX
            and HEIGHT_LO <= tcp[1] <= HEIGHT_HI)
    return safe, clr, tcp


def path_clear(q_from_deg, q_to_deg, n=7):
    """Sample the straight joint-space path and require clearance the whole
    way — the target being clear is not enough if a distal link swings past
    the column mid-move."""
    a, b = np.radians(q_from_deg), np.radians(q_to_deg)
    for s in np.linspace(0.0, 1.0, n):
        if kin.min_column_clearance(a + (b - a) * s) < CLEARANCE_MIN:
            return False
    return True


def safe_poses():
    cand = gen_poses()
    safe = []
    for q in cand:
        ok, clr, tcp = evaluate(q)
        if ok:
            safe.append((q, clr, tcp))
    return safe, len(cand)


def _q8(q7_deg):
    return np.radians(q7_deg).tolist() + [GRIP_FIXED_RAD]


async def run(args):
    safe, n_total = safe_poses()
    print(f"{len(safe)}/{n_total} grid poses pass clearance + reach + height")
    for i, (q, clr, tcp) in enumerate(safe):
        print(f"  pose {i:2d}: J1{q[0]:+.0f} J2{q[1]:+.0f} J4{q[3]:+.0f}  "
              f"clr {clr*100:4.1f}cm  tcp {np.round(tcp, 3)}")
    if not safe:
        sys.exit("no safe poses — check grid ranges / clearance gate")

    if args.dry_run:
        print("\n--dry-run: no motion. Re-run without --dry-run to collect.")
        return

    cam = OakCamera()
    arm = RightArm(velocity=CAL_VELOCITY)
    samples = []
    try:
        await arm.connect()
        pose0 = await arm.read_pose()                # disabled read = CAN check
        if max(abs(x) for x in pose0[:7]) < 1e-6:
            sys.exit("arm reads all zeros — CAN down or unpowered; bring up can1")
        print("\nARM WILL MOVE ON ITS OWN in 3s — clear the workspace, "
              "keep the e-stop in reach (Ctrl+C aborts safely)")
        await asyncio.sleep(3)

        await arm.enable()
        await arm.move(_q8(READY_DEG), velocity=CAL_VELOCITY)
        prev = READY_DEG

        for i, (q, clr, tcp) in enumerate(safe):
            if len(samples) >= args.poses:
                break
            if not path_clear(prev, q):
                await arm.move(_q8(READY_DEG), velocity=CAL_VELOCITY)  # via rest
                prev = READY_DEG
                if not path_clear(prev, q):
                    print(f"  pose {i}: path not clear even via READY — skip")
                    continue
            await arm.move(_q8(q), velocity=CAL_VELOCITY)
            prev = q
            await asyncio.sleep(0.8)                  # let it fully settle

            actual = await arm.read_actual(_q8(q))    # real joint feedback
            uv, p_cam, _, _ = capture_tape_point(cam, args.color, tries=6)
            if uv is None or p_cam is None:
                print(f"  pose {i}: tape not clearly visible — skip")
                continue
            qj = list(actual[:7])
            samples.append({"p_cam": p_cam.tolist(), "q_motor": qj, "uv": list(uv)})
            with open(SAMPLES_PATH, "w") as f:
                json.dump(samples, f, indent=1)
            print(f"[{len(samples)}/{args.poses}] pose{i} blob{uv} "
                  f"cam {np.round(p_cam, 3)} joints {np.round(np.degrees(qj), 0)}")
    finally:
        try:
            if arm._enabled:
                await arm.move(_q8(READY_DEG), velocity=CAL_VELOCITY)
        except Exception:  # noqa: BLE001 — must still park/disable
            pass
        await arm.park_and_disable()
        arm.shutdown()
        cam.close()
        cv2.destroyAllWindows()

    print()
    if len(samples) >= 6:
        fit_and_report(samples)
    else:
        print(f"only {len(samples)} poses captured (need >=6). Likely the tape "
              "was out of view — check camera aim / tape, then rerun.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poses", type=int, default=12, help="target captures")
    ap.add_argument("--color", default="yellow")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the safe pose plan, touch no motors")
    ap.add_argument("--refit", action="store_true",
                    help="re-fit from the saved samples.json and exit")
    args = ap.parse_args()

    if args.refit:
        with open(SAMPLES_PATH) as f:
            fit_and_report(json.load(f))
        return

    t0 = time.time()
    asyncio.run(run(args))
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
