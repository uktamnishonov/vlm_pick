#!/usr/bin/env python3
"""One-shot language task for the RIGHT OpenArm: pick, and optionally place.

    python task.py --prompt "take the coffee cup and place it onto the white paper"
    python task.py --prompt "pick up the orange cube" --dry-run

Gemini reads the task and the camera image in ONE call, pointing at the
object to grasp and (if the task names one) the destination spot. Stereo
depth turns both pixels into 3D, the calibration maps them to the arm
base frame, IK plans a gripper-down grasp, and the arm executes:

  approach -> descend -> close -> lift -> [transit -> lower -> release]
  -> retreat -> park -> disable.

Without a destination in the prompt, the object is lifted, held, and put
back where it was. ALWAYS run --dry-run first after moving the camera or
recalibrating. Keep the e-stop in reach; motors disable on any exit.
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
from arm_io import GRIPPER_OPEN_DEG, RightArm
from camera import OakCamera
from pick import (APPROACH_M, IK_TOL_M, LIFT_M, grasp_orientations,
                  solve_grasp, to_motor)

RELEASE_M = 0.05  # open the gripper this high above the destination surface
WAYPOINT_M = 0.03  # subdivide precision moves into steps of this length

# RIGHT_SIDE_Q: arm raised and panned out to the robot's right, well clear
# (~18 cm) of the central body. Every task raises from rest to HERE first,
# then swings to the object; and reverses on the way down — so the big
# motions always happen out to the side, never across the center body.
RIGHT_SIDE_Q = np.radians([45.0, 20.0, 0.0, 95.0, 0.0, 0.0, 0.0])
# Seed pose for grasp IK (a natural front-facing reach) — just an optimizer
# hint, the arm never necessarily passes exactly through it.
READY_Q = np.radians([0.0, 30.0, 0.0, 90.0, 0.0, 0.0, 0.0])

_T0 = time.time()


def log(msg):
    print(f"[{time.time()-_T0:6.1f}s] {msg}", flush=True)


def joint_lerp(q_from, q_to, step_deg=12.0):
    """Straight line in JOINT space, one waypoint per ~step_deg of the
    largest-moving joint. Used for the raise-from-rest and lower-to-rest
    moves so they are slow, smooth, and mirror images of each other."""
    q_from = np.asarray(q_from, float)
    q_to = np.asarray(q_to, float)
    n = max(1, int(math.ceil(np.max(np.abs(np.degrees(q_to - q_from))) / step_deg)))
    return [q_from + (q_to - q_from) * (i / n) for i in range(1, n + 1)]


def chain_ik(p_from, p_to, q_seed, z_axis):
    """IK along a straight Cartesian line, one solve per ~3 cm.

    Joint-space interpolation between two distant IK solutions arcs in
    Cartesian space; dense waypoints keep descend/lift moves near-straight.
    Returns a list of 7-joint solutions or None.
    """
    p_from = np.asarray(p_from, float)
    p_to = np.asarray(p_to, float)
    n = max(1, int(math.ceil(np.linalg.norm(p_to - p_from) / WAYPOINT_M)))
    q = np.asarray(q_seed, float)
    out = []
    for i in range(1, n + 1):
        p = p_from + (p_to - p_from) * (i / n)
        q, e = kin.ik(p, q, z_axis)
        if e > IK_TOL_M:
            return None
        out.append(q)
    return out


def solve_place(p_place, q7_init):
    """IK for hover + release above the destination. Returns dict or None."""
    hover = p_place + kin.UP_BASE * (APPROACH_M + RELEASE_M)
    release = p_place + kin.UP_BASE * RELEASE_M
    seeds = [np.asarray(q7_init, dtype=float)]
    for z_axis in grasp_orientations(p_place):
        for seed in seeds:
            q_h, e1 = kin.ik(hover, seed, z_axis)
            q_r, e2 = kin.ik(release, q_h, z_axis)
            if e1 < IK_TOL_M and e2 < IK_TOL_M:
                return {"hover": (hover, q_h), "release": (release, q_r)}
    q_h, e1 = kin.ik(hover, seeds[0])
    q_r, e2 = kin.ik(release, q_h)
    if e1 < IK_TOL_M and e2 < IK_TOL_M:
        return {"hover": (hover, q_h), "release": (release, q_r)}
    return None


def deproject(cam, depth, u, v, what, half=40):
    p = cam.point3d(u, v, depth)
    if p is not None:
        return p
    # Dark/glossy objects (black cup lid) defeat stereo: no depth at the
    # pixel itself. Fall back to the nearest valid surface in a box around
    # it — an object's rim/edges usually return depth even when its face
    # doesn't, and the 15th percentile picks the object top, not the table.
    h, w = depth.shape
    box = depth[max(0, v - half):min(h, v + half),
                max(0, u - half):min(w, u + half)]
    valid = box[box > 0]
    if valid.size < 30:
        sys.exit(f"no valid depth around {what} pixel ({u},{v}) — "
                 "object too dark/reflective for stereo")
    z = float(np.percentile(valid, 15)) / 1000.0
    log(f"  ({what}: no depth on the surface itself, using nearest valid "
        f"depth in a {2*half}px box: {z:.3f} m)")
    fx, fy = cam.K[0, 0], cam.K[1, 1]
    cx, cy = cam.K[0, 2], cam.K[1, 2]
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])


def base_to_pixel(p_base, calib, K):
    """Project an arm-base 3D point back into the camera image (inverse of
    calib.cam_to_base), so planned waypoints can be drawn on the scene."""
    p_cam = calib.R.T @ (np.asarray(p_base, dtype=float) - calib.t)
    if p_cam[2] <= 1e-3:
        return None
    u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
    v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
    if not (0 <= u < 2000 and 0 <= v < 2000):
        return None
    return (int(round(u)), int(round(v)))


class Viz:
    """Live cv2 overlay of the planned arm path on the captured camera frame."""

    WIN = "OpenArm task (--show)"

    def __init__(self, bg, step_pixels, step_names):
        self.bg = bg
        self.step_pixels = step_pixels      # list per step: [pixel|None, ...]
        self.step_names = step_names
        self.path = [p for pix in step_pixels for p in pix if p]

    def render(self, active=None, wp=None, banner="", wait=1):
        img = self.bg.copy()
        # full planned path, faint
        for a, b in zip(self.path, self.path[1:]):
            cv2.line(img, a, b, (0, 150, 0), 1)
        for p in self.path:
            cv2.circle(img, p, 2, (0, 150, 0), -1)
        if active is not None and 0 <= active < len(self.step_pixels):
            pix = [p for p in self.step_pixels[active] if p]
            for p in pix:
                cv2.circle(img, p, 4, (0, 255, 255), -1)   # this step, yellow
            if pix:                                         # NEXT target
                t = pix[-1]
                cv2.circle(img, t, 13, (0, 255, 255), 2)
                cv2.putText(img, "NEXT", (t[0] + 15, t[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            if wp is not None and self.step_pixels[active][wp]:
                cv2.circle(img, self.step_pixels[active][wp], 7, (0, 0, 255), -1)  # arm now
        for txt, y, col in [(banner, 24, (0, 0, 0)), (banner, 23, (255, 255, 255))]:
            cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col,
                        2 if y == 24 else 1)
        cv2.imshow(self.WIN, img)
        cv2.waitKey(wait)

    @staticmethod
    def close():
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception:
            pass


async def run(args):
    log(f"TASK: {args.prompt!r}")
    try:
        calib = kin.Calibration()
    except FileNotFoundError:
        sys.exit("no calibration.json — run calibrate.py first "
                 "(and rerun it whenever the camera moves)")
    log(f"calibration loaded: accuracy ±{calib.rms*1000:.0f} mm "
        f"({'good' if calib.rms < 0.03 else 'POOR — supervise closely'})")

    # ---- perceive
    log("PHASE 1/4 PERCEPTION — starting camera (takes a few seconds)...")
    cam = OakCamera()
    try:
        bgr, depth = cam.read()
        log(f"frame captured, depth valid on {(depth > 0).mean()*100:.0f}% of pixels")
        log(f"asking {vlm.DEFAULT_MODEL} to find the targets...")
        parsed = vlm.parse_task(bgr, args.prompt)
        if parsed is None:
            sys.exit("VLM could not find the object for this task in view")
        pu, pv, plabel = parsed["pick"]
        log(f"VLM found pick target: {plabel!r} at pixel ({pu},{pv})")
        p_pick_cam = deproject(cam, depth, pu, pv, "pick")
        log(f"stereo depth: {plabel!r} is {p_pick_cam[2]:.2f} m from camera")
        annot = bgr.copy()
        cv2.drawMarker(annot, (pu, pv), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
        cv2.putText(annot, f"PICK: {plabel}", (pu + 14, pv), 0, 0.5, (0, 0, 255), 2)

        place = parsed["place"]
        p_place_cam = None
        if place is not None:
            du, dv, dlabel = place
            log(f"VLM found place target: {dlabel!r} at pixel ({du},{dv})")
            p_place_cam = deproject(cam, depth, du, dv, "place")
            cv2.drawMarker(annot, (du, dv), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 30, 3)
            cv2.putText(annot, f"PLACE: {dlabel}", (du + 14, dv), 0, 0.5, (255, 0, 0), 2)
        else:
            log("no place destination in the task — will lift and put back")
        cv2.imwrite("task_targets.jpg", annot)
        log("annotated view saved to task_targets.jpg")
        viz_bg = annot.copy()   # background for the --show overlay
        Kcam = cam.K
    finally:
        cam.close()

    # ---- geometry
    log("PHASE 2/4 GEOMETRY — converting camera points to arm coordinates")
    p_pick = calib.cam_to_base(p_pick_cam)
    log(f"pick point in arm base frame: {np.round(p_pick, 3)} m")
    if not calib.in_workspace(p_pick):
        sys.exit(f"pick target outside calibrated workspace "
                 f"{np.round(calib.ws_lo,2)}..{np.round(calib.ws_hi,2)} — refusing")
    p_place = None
    if p_place_cam is not None:
        p_place = calib.cam_to_base(p_place_cam)
        log(f"place point in arm base frame: {np.round(p_place, 3)} m "
            f"({(p_pick[1]-p_place[1])*100:.0f} cm below the pick point)")
        if not calib.in_workspace(p_place):
            sys.exit("place target outside calibrated workspace — refusing")
    log("both targets inside the calibrated workspace ✓")

    # ---- current pose
    arm = RightArm(velocity=args.velocity)
    q7_now = np.zeros(7)
    if not args.dry_run:
        log("connecting to right arm on can1...")
        await arm.connect()
        pose = await arm.read_pose()
        q7_now = kin.motor_to_urdf(pose[:7], calib.signs)
        log(f"all 8 motors answering; current joints (deg): "
            f"{np.round(np.degrees(pose[:7]), 0)}")

    log("PHASE 3/4 PLANNING — solving IK for every waypoint (~30 s)...")

    # ---- plan  (each motion step = list of clamped 8-motor waypoints)
    side = args.grasp == "side"
    log(f"grasp mode: {args.grasp.upper()}, "
        f"{args.grasp_drop*100:.1f} cm below the detected object top")
    # seed IK from the ready pose, not the current pose: execution passes
    # through READY_Q first, and a consistent seed keeps every run in the
    # same well-behaved configuration family
    g = solve_grasp(p_pick, READY_Q, drop=args.grasp_drop, mode=args.grasp)
    if g is None:
        arm.shutdown()
        sys.exit(f"IK: no reachable {args.grasp} grasp — move the object closer "
                 "to the arm" + (", or try --grasp top" if side else ""))
    pre_p, q_pre = g["pre"]
    grasp_p, q_grasp = g["grasp"]

    approach = chain_ik(pre_p, grasp_p, q_pre, g["z_axis"])
    up = chain_ik(grasp_p, g["lift"], q_grasp, g["z_axis"])
    if approach is None or up is None:
        arm.shutdown()
        sys.exit("IK: could not build a straight approach/lift path — "
                 "move the object closer to the arm")
    q_lift = up[-1]

    approach_label = "move beside object" if side else "position above object"
    movein_label = "move in to grasp" if side else "descend to grasp"
    # Each step carries its URDF-space joint waypoints so the clearance gate
    # can check them; motor_chain() converts to clamped 8-motor commands.
    raise_wp = joint_lerp(q7_now, RIGHT_SIDE_Q)      # rest -> up on the right
    to_obj = joint_lerp(RIGHT_SIDE_Q, q_pre)         # right -> in front of object
    steps = [
        ("raise to right side", raise_wp, GRIPPER_OPEN_DEG, args.velocity),
        (approach_label, to_obj, GRIPPER_OPEN_DEG, args.velocity),
        (movein_label, approach, GRIPPER_OPEN_DEG, 0.2),
        ("CLOSE GRIPPER", None, None, None),
        ("lift", up, 0.0, 0.25),
    ]
    if p_place is not None:
        pl = solve_place(p_place, q_lift)
        if pl is None:
            arm.shutdown()
            sys.exit("IK: destination not reachable — move it closer to the arm")
        hov_p, q_hov = pl["hover"]
        transit = chain_ik(g["lift"], hov_p, q_lift, None) or [q_hov]
        lower = chain_ik(hov_p, pl["release"][0], transit[-1], None)
        if lower is None:
            arm.shutdown()
            sys.exit("IK: could not build a straight lowering path at the destination")
        last_q = lower[0]
        steps += [
            ("transit to destination", transit, 0.0, 0.3),
            ("lower to release height", lower, 0.0, 0.2),
            ("OPEN GRIPPER", None, None, None),
            ("retreat up", [lower[0]], GRIPPER_OPEN_DEG, args.velocity),
        ]
    else:
        last_q = q_pre
        steps += [
            ("hold", None, None, None),
            ("lower back down", list(reversed(up))[1:] + [q_grasp], 0.0, 0.2),
            ("OPEN GRIPPER", None, None, None),
            ("retreat", [q_pre], GRIPPER_OPEN_DEG, args.velocity),
        ]
    # ending mirrors the start: swing back out to the right side, then a slow
    # controlled lowering straight down to the hanging base pose
    steps += [
        ("swing to right side", joint_lerp(last_q, RIGHT_SIDE_Q), GRIPPER_OPEN_DEG, args.velocity),
        ("lower to base pose", joint_lerp(RIGHT_SIDE_Q, np.zeros(7)), GRIPPER_OPEN_DEG, 0.3),
    ]

    # ---- SAFETY GATE: refuse if any waypoint swings near the central body
    worst_clear, worst_step = 1e9, None
    for name, qs, _grip, _v in steps:
        if qs is None:
            continue
        for q in qs:
            c = kin.min_column_clearance(q)
            if c < worst_clear:
                worst_clear, worst_step = c, name
    log(f"closest approach to central body: {worst_clear*100:.0f} cm "
        f"(at '{worst_step}')")
    if worst_clear < kin.CLEARANCE_MIN_M:
        arm.shutdown()
        sys.exit(f"REFUSING: step '{worst_step}' brings the arm within "
                 f"{worst_clear*100:.0f} cm of the central body (limit "
                 f"{kin.CLEARANCE_MIN_M*100:.0f} cm). "
                 + ("Try --grasp top, or " if side else "")
                 + "move the object farther from the robot's centerline.")

    # --show: project each step's TCP waypoints back into the camera image
    viz = None
    if args.show:
        step_pixels, step_names = [], []
        for name, qs, _grip, _v in steps:
            step_names.append(name)
            if qs is None:
                step_pixels.append([])
            else:
                step_pixels.append([base_to_pixel(kin.fk(q), calib, Kcam)
                                    for q in qs])
        viz = Viz(viz_bg, step_pixels, step_names)

    # convert URDF waypoints -> clamped 8-motor commands
    steps = [(name, (None if qs is None else [to_motor(q, calib.signs, grip)
                                             for q in qs]), v)
             for name, qs, grip, v in steps]

    tilt = math.degrees(math.acos(np.clip(-np.asarray(g["z_axis"]) @ kin.UP_BASE, -1, 1)))
    log(f"grasp solved: {args.grasp} approach, gripper {tilt:.0f}° from vertical "
        f"({'≈horizontal' if tilt > 60 else '≈top-down' if tilt < 30 else 'angled'})")
    log(f"plan ready — {len(steps)} steps:")
    for i, (name, wps, v) in enumerate(steps, 1):
        if wps is not None:
            print(f"    {i}. {name}  ({len(wps)} waypoint(s), "
                  f"{v} rad/s)", flush=True)
        else:
            print(f"    {i}. {name}", flush=True)

    if args.dry_run:
        if viz:
            log("DRY RUN — showing the planned path. Press any key in the "
                "window to close.")
            viz.render(banner="PLANNED PATH (dry run) - press a key", wait=0)
            Viz.close()
        else:
            log("DRY RUN — stopping here, no motion. Check task_targets.jpg.")
        return

    # ---- execute
    log("PHASE 4/4 EXECUTION — ARM WILL MOVE IN 3 SECONDS (Ctrl+C aborts)")
    if viz:
        viz.render(banner="starting in 3s...", wait=1)
    await asyncio.sleep(3)
    try:
        await arm.enable()
        log("motors enabled (position mode, velocity-capped)")
        await arm.gripper(open_=True)
        for i, (name, wps, v) in enumerate(steps, 1):
            banner = f"Step {i}/{len(steps)}: {name}"
            if name == "CLOSE GRIPPER":
                log(f"step {i}/{len(steps)}: closing gripper...")
                if viz:
                    viz.render(i - 1, banner=banner, wait=1)
                await arm.gripper(open_=False)
            elif name == "OPEN GRIPPER":
                log(f"step {i}/{len(steps)}: opening gripper...")
                if viz:
                    viz.render(i - 1, banner=banner, wait=1)
                await arm.gripper(open_=True)
            elif name == "hold":
                log(f"step {i}/{len(steps)}: holding 2 s...")
                if viz:
                    viz.render(i - 1, banner=banner, wait=1)
                await asyncio.sleep(2.0)
            else:
                log(f"step {i}/{len(steps)}: {name}")
                for j, q8 in enumerate(wps, 1):
                    if len(wps) > 1:
                        log(f"    waypoint {j}/{len(wps)}")
                    if viz:
                        viz.render(i - 1, wp=j - 1, banner=banner, wait=1)
                    await arm.move(q8, velocity=v, settle=0.15)
        log("task motion complete")
        if viz:
            viz.render(banner="DONE", wait=800)
    finally:
        log("returning to rest pose and disabling motors...")
        await arm.park_and_disable()
        arm.shutdown()
        Viz.close()
        log("motors disabled — arm is safe")

    # ---- verify
    try:
        log("VERIFY — taking a photo and asking the VLM how it went...")
        cam = OakCamera()
        bgr, _ = cam.read()
        cam.close()
        cv2.imwrite("task_result.jpg", bgr)
        verdict = vlm.ask(bgr, f'The robot was asked to: "{args.prompt}". '
                               "Did it succeed? Where are the objects now?")
        log(f"VLM verdict: {verdict}")
        log("result photo saved to task_result.jpg")
    except Exception as e:  # noqa: BLE001 — verification is best-effort
        log(f"(verification skipped: {e})")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--prompt", required=True,
                    help='e.g. "take the coffee cup and place it on the white paper"')
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show", action="store_true",
                    help="live cv2 window: camera view + planned arm path "
                         "projected onto the scene, highlighting the next spot")
    ap.add_argument("--velocity", type=float, default=0.35)
    ap.add_argument("--grasp", choices=("top", "side"), default="top",
                    help="top = approach from above (default, keeps the arm "
                         "clear of the central body); side = horizontal "
                         "approach (more stable hold, but blocked by the "
                         "clearance check when it would swing near the body)")
    ap.add_argument("--grasp-drop", type=float, default=0.045,
                    help="grasp this many meters below the object's detected "
                         "top (default 0.045)")
    args = ap.parse_args()
    if not 0.1 <= args.velocity <= 0.8:
        ap.error("velocity must be 0.1..0.8 rad/s")
    t0 = time.time()
    asyncio.run(run(args))
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
