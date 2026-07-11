#!/usr/bin/env python3
"""Language-driven pick with the RIGHT OpenArm: 'pick up the red cup'.

Pipeline: OAK frame -> Gemini points at the object -> stereo depth
deprojects the pixel -> calibration transform to arm base frame ->
workspace check -> IK (gripper pointing down, tilted fallbacks) ->
velocity-capped primitive: approach above, descend, close, lift, put
back, release, retreat, park.

SAFETY: arm clamped down, workspace clear, e-stop in reach. Start with
--dry-run: it runs perception + IK and prints the full motion plan
without touching the motors.

Usage:
    python pick.py "the red cup" --dry-run
    python pick.py "the red cup"
    python pick.py --uv 320 240 --dry-run     # skip VLM, use a raw pixel
"""
import argparse
import asyncio
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kin
import vlm
from arm_io import GRIPPER_OPEN_DEG, LIMITS_RIGHT, RightArm, clamp8
from camera import OakCamera

APPROACH_M = 0.10   # pregrasp height above the target
GRASP_DROP_M = 0.015  # go slightly below the VLM center point
LIFT_M = 0.15
IK_TOL_M = 0.010
CLAMP_TOL_RAD = math.radians(1.0)


MAX_TILT_FROM_DOWN_DEG = 75  # steeper than this and a top grasp won't hold

# Elbow-bent IK seeds (deg) — ikpy is a local optimizer, zeros is singular.
IK_SEEDS_DEG = [
    [0, 40, 0, 80, 0, 0, 0],
    [30, 60, -20, 100, 0, 0, 0],
    [-30, 30, 20, 60, 0, 0, 0],
]


def _outward(p_base):
    """Unit horizontal vector from the arm base out toward the object."""
    horiz = p_base.copy()
    horiz[1] = 0.0
    n = np.linalg.norm(horiz)
    return horiz / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])


def grasp_orientations(p_base):
    """TOP grasp TCP z-axes: straight down, then tilting toward the shoulder."""
    down = -kin.UP_BASE
    h = -_outward(p_base)  # toward the shoulder
    cands = [down]
    for deg in (25, 45, 65):
        a = math.radians(deg)
        cands.append(down * math.cos(a) + h * math.sin(a))
    return cands


def side_orientations(p_base):
    """SIDE grasp TCP z-axes: horizontal (fingers pointing outward), then
    increasingly tilted downward so it still finds a hold if a fully level
    approach is out of the wrist's range."""
    h = _outward(p_base)
    down = -kin.UP_BASE
    cands = [h]
    for deg in (20, 40, 55):
        a = math.radians(deg)
        cands.append(h * math.cos(a) + down * math.sin(a))
    return cands


def _tcp_z_axis(q7):
    full = np.zeros(12)
    full[1:8] = q7
    return kin.chain().forward_kinematics(full)[:3, 2]


def solve_grasp(p_base, q7_init, drop=GRASP_DROP_M, mode="top"):
    """IK for pregrasp+grasp with matching orientation. Returns dict or None.

    mode="top":  approach from above, gripper pointing down (tilts allowed).
    mode="side": approach horizontally, gripper pointing outward at the
                 object's body — the natural, stable way to hold a cup.
    Tries several seeds; top mode also has a position-only fallback.
    """
    if mode == "side":
        h = _outward(p_base)
        grasp = p_base - kin.UP_BASE * drop        # a bit below the top, at the body
        pre = grasp - h * APPROACH_M               # backed off horizontally toward base
        z_candidates = side_orientations(p_base)
    else:
        pre = p_base + kin.UP_BASE * APPROACH_M
        grasp = p_base - kin.UP_BASE * drop
        z_candidates = grasp_orientations(p_base)

    seeds = [np.asarray(q7_init, dtype=float)] + [
        np.radians(s) for s in IK_SEEDS_DEG
    ]
    for z_axis in z_candidates:
        for seed in seeds:
            q_pre, e1 = kin.ik(pre, seed, z_axis)
            q_grasp, e2 = kin.ik(grasp, q_pre, z_axis)
            if e1 < IK_TOL_M and e2 < IK_TOL_M:
                return {"z_axis": z_axis, "pre": (pre, q_pre),
                        "grasp": (grasp, q_grasp),
                        "lift": grasp + kin.UP_BASE * LIFT_M, "mode": mode}
    if mode == "side":
        return None
    # Top last resort: position-only, accept if the wrist points down-ish.
    for seed in seeds:
        q_pre, e1 = kin.ik(pre, seed)
        q_grasp, e2 = kin.ik(grasp, q_pre)
        if e1 < IK_TOL_M and e2 < IK_TOL_M:
            z = _tcp_z_axis(q_grasp)
            tilt = math.degrees(math.acos(np.clip(-z @ kin.UP_BASE, -1, 1)))
            if tilt <= MAX_TILT_FROM_DOWN_DEG:
                return {"z_axis": z, "pre": (pre, q_pre),
                        "grasp": (grasp, q_grasp),
                        "lift": grasp + kin.UP_BASE * LIFT_M, "mode": mode}
    return None


def to_motor(q7_urdf, signs, gripper_deg):
    """URDF joints -> clamped 8-motor target; abort if clamping distorts it."""
    q8 = list(np.asarray(q7_urdf) * np.asarray(signs)) + [math.radians(gripper_deg)]
    clamped = clamp8(q8)
    worst = max(abs(a - b) for a, b in zip(q8, clamped))
    if worst > CLAMP_TOL_RAD:
        j = int(np.argmax([abs(a - b) for a, b in zip(q8, clamped)]))
        lo, hi = np.degrees(LIMITS_RIGHT[j])
        raise RuntimeError(
            f"target exceeds J{j+1} limit by {math.degrees(worst):.1f}deg "
            f"(limits {lo:.0f}..{hi:.0f}) — object out of comfortable reach"
        )
    return clamped


async def run(args):
    try:
        calib = kin.Calibration()
    except FileNotFoundError:
        sys.exit("no calibration.json — run calibrate.py first "
                 "(and rerun it whenever the camera moves)")
    print(f"calibration: rms {calib.rms*1000:.1f} mm, signs {calib.signs}")

    cam = OakCamera()
    try:
        bgr, depth = cam.read()
        if args.uv:
            u, v, label = args.uv[0], args.uv[1], "manual pixel"
        else:
            print(f"asking {vlm.DEFAULT_MODEL} to find: {args.object!r} ...")
            hit = vlm.locate(bgr, args.object)
            if hit is None:
                sys.exit("VLM: object not found in view")
            u, v, label = hit
        p_cam = cam.point3d(u, v, depth)
        if p_cam is None:
            for du, dv in [(0, 8), (0, -8), (8, 0), (-8, 0), (0, 16), (0, -16)]:
                p_cam = cam.point3d(u + du, v + dv, depth)
                if p_cam is not None:
                    break
        if p_cam is None:
            sys.exit(f"no valid depth around pixel ({u},{v})")

        cv2.drawMarker(bgr, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
        cv2.circle(bgr, (u, v), 12, (0, 255, 0), 2)
        cv2.imwrite("pick_target.jpg", bgr)

        p_base = calib.cam_to_base(p_cam)
        print(f"'{label}' pixel ({u},{v}) cam {np.round(p_cam,3)} "
              f"-> base {np.round(p_base,3)} m")
        if not calib.in_workspace(p_base):
            sys.exit(f"target outside calibrated workspace "
                     f"{np.round(calib.ws_lo,2)}..{np.round(calib.ws_hi,2)} — refusing")
    finally:
        cam.close()

    arm = RightArm(velocity=args.velocity)
    q7_now = np.zeros(7)
    if not args.dry_run:
        await arm.connect()
        pose = await arm.read_pose()
        q7_now = kin.motor_to_urdf(pose[:7], calib.signs)
        print("current joints (motor deg):", np.round(np.degrees(pose[:7]), 1))

    plan = solve_grasp(p_base, q7_now)
    if plan is None:
        arm.shutdown()
        sys.exit("IK found no reachable grasp (tried down + tilted) — move the "
                 "object closer to the arm")

    pre_t, q_pre = plan["pre"]
    grasp_t, q_grasp = plan["grasp"]
    q_lift, e3 = kin.ik(plan["lift"], q_grasp, plan["z_axis"])
    steps = [
        ("open gripper + approach", to_motor(q_pre, calib.signs, GRIPPER_OPEN_DEG), args.velocity),
        ("descend to grasp", to_motor(q_grasp, calib.signs, GRIPPER_OPEN_DEG), 0.2),
        ("CLOSE GRIPPER", None, None),
        ("lift", to_motor(q_lift, calib.signs, 0.0), 0.25),
        ("hold", None, None),
        ("lower back", to_motor(q_grasp, calib.signs, 0.0), 0.2),
        ("OPEN GRIPPER", None, None),
        ("retreat", to_motor(q_pre, calib.signs, GRIPPER_OPEN_DEG), args.velocity),
    ]

    print(f"\nplan (z-axis {np.round(plan['z_axis'],2)}, "
          f"pregrasp {np.round(pre_t,3)}, grasp {np.round(grasp_t,3)}):")
    for name, q8, v in steps:
        if q8 is not None:
            print(f"  {name:26s} v={v} deg={np.round(np.degrees(q8), 1)}")
        else:
            print(f"  {name}")

    if args.dry_run:
        print("\n--dry-run: no motion. Inspect pick_target.jpg and the plan above.")
        return

    print("\nARM WILL MOVE IN 3s — Ctrl+C to abort, keep e-stop in reach")
    await asyncio.sleep(3)
    try:
        await arm.enable()
        await arm.gripper(open_=True)
        for name, q8, v in steps:
            print(f"-> {name}")
            if name == "CLOSE GRIPPER":
                await arm.gripper(open_=False)
            elif name == "OPEN GRIPPER":
                await arm.gripper(open_=True)
            elif name == "hold":
                await asyncio.sleep(2.0)
            else:
                await arm.move(q8, velocity=v)
    finally:
        print("-> park + disable")
        await arm.park_and_disable()
        arm.shutdown()

    if not args.no_verify and not args.uv:
        try:
            cam = OakCamera()
            bgr, _ = cam.read()
            cam.close()
            cv2.imwrite("pick_result.jpg", bgr)
            ans = vlm.ask(bgr, f"Did the robot arm just move the {args.object}? "
                               f"Describe where the {args.object} is now.")
            print("VLM verdict:", ans)
        except Exception as e:  # noqa: BLE001 — verification is best-effort
            print(f"(verification skipped: {e})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("object", nargs="?", help="what to pick, e.g. 'the red cup'")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--velocity", type=float, default=0.35)
    ap.add_argument("--uv", type=int, nargs=2, metavar=("U", "V"),
                    help="skip the VLM, use this pixel")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    if not args.object and not args.uv:
        ap.error("give an object description or --uv U V")
    if args.object is None:
        args.object = "object"
    if not 0.1 <= args.velocity <= 0.8:
        ap.error("velocity must be 0.1..0.8 rad/s")
    t0 = time.time()
    asyncio.run(run(args))
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
