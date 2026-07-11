#!/usr/bin/env python3
"""CALIBRATION-FREE image-plane visual servoing for the RIGHT OpenArm (can1).

Every one-shot calibration we fit (18 / 24 / 41 / 361 mm) was open-loop and
missed. The first servo attempt closed the loop but in camera *3D* space —
which needs reliable depth on BOTH the tape and the object. A black coffee
cup gives garbage stereo depth (glossy, absorbs the IR dots), so its 3D
point was ~0.5 m wrong and the loop chased a phantom.

So this servos in the IMAGE PLANE instead. Both the gripper tape and the
object are located as PIXELS (blob centroid / Gemini point) — pixels are
rock-solid even when depth is not. We drive the tape pixel onto the object
pixel with a 2x3 image Jacobian estimated live from probe moves. When they
coincide, the gripper is on the camera ray through the object, i.e. lined up
over it. No camera->base calibration, and no dependence on the object's
absolute depth for the aiming.

Wrist is LOCKED (only J1/J2/J4 move) so the tape shows one constant face.
Every step is clearance-, limit-, frame-edge- and stall-gated: it cannot
wander off frame or into the body, and an unreachable pixel stalls out
loudly instead of grinding.

Default run aligns the tape over the object and reports the residual pixel
error plus a depth readout (tape / object / table) — that proves the aim and
tells us what the descent has to work with. Depth-based descent + grasp is
the next stage (built once alignment is confirmed).

SAFETY: the arm moves ON ITS OWN, slowly (0.15 rad/s). Clear the workspace,
hand on the e-stop (Ctrl+C parks + disables). --locate-only touches nothing.

Usage:
    python servo.py "the red cup" --locate-only   # perception only, no motion
    python servo.py "the red cup"                 # align tape over it, report
    python servo.py --uv 300 220                  # skip VLM, aim at a pixel
"""
import argparse
import asyncio
import json
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kin
import vlm
from arm_io import LIMITS_RIGHT, RightArm, clamp8
from calibrate import COLOR_RANGES, find_blob
from camera import RGB_SIZE, OakCamera

# Fixed ABSOLUTE (J1,J2,J4) calibration grid (deg), wrist held at --wrist. Chosen
# so the gripper's 3D point cloud is strongly NON-planar (planarity ~0.42) —
# PnP camera recovery is then well-conditioned EVERY run, not dependent on how
# far alignment happened to sweep. Each J1 pairs an extended (low-J4) pose with a
# retracted (high-J4) one to span the toward/away-from-camera (depth) axis.
CALIB_POSES = [(-15, 25, 100), (-15, 0, 50), (0, 20, 95), (0, -8, 55),
               (18, 15, 90), (18, 5, 48), (-8, 10, 105), (10, 22, 60),
               (25, 0, 52), (-12, 18, 80), (8, -5, 70), (28, 12, 44)]
SERVO_CALIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "servo_calib.json")

# --- pose / joint layout ----------------------------------------------------
CONTROL = [0, 1, 3]                        # J1 yaw, J2 pitch, J4 elbow (0-based)
START_DEG = [-10, 25, 0, 85, 0, 0, 0]      # centered in the known-visible region
GRIP_SERVO_RAD = 0.0                       # gripper closed+fixed during servo

# Per-control-joint travel, for limit-aware weighting. J2 (pitch) has little
# downward room (floor -10 deg) while J1 (yaw) has 270 deg — so the servo must
# be told to spend J1/J4 before pinning J2.
CTRL_LO = np.array([LIMITS_RIGHT[j][0] for j in CONTROL])
CTRL_HI = np.array([LIMITS_RIGHT[j][1] for j in CONTROL])
CTRL_HALF = (CTRL_HI - CTRL_LO) / 2.0

# --- servo gains / limits ---------------------------------------------------
CAL_VELOCITY = 0.15                        # rad/s — slow and deliberate
PROBE_DEG = 4.0                            # bootstrap probe amplitude
STEP_MAX_DEG = 5.0                         # max change per joint per step
LAMBDA = 0.55                              # fraction of pixel error to close/step
MU_PX = 25.0                               # Levenberg-Marquardt damping (px^2)
CONVERGE_PX = 11                           # tape-vs-object pixel gap that counts
CONVERGE_HITS = 2                          # consecutive converged steps to stop
MAX_STEPS = 45
STALL_WINDOW = 6                           # steps
STALL_MIN_IMPROVE_PX = 6                   # <6 px progress over the window = stalled

# --- perception -------------------------------------------------------------
ROI_HALF = 90                              # px half-window for tape ROI search
FRAME_MARGIN = 28                          # px; tape closer than this to a border = stop
DEPTH_MIN, DEPTH_MAX = 0.20, 1.30          # m; plausible depth band
FRAME_W, FRAME_H = RGB_SIZE


# ---------------------------------------------------------------------------
# perception helpers
# ---------------------------------------------------------------------------
TAPE_MIN_AREA_ROI = 8   # px — inside a tracked ROI, accept a thinner sliver
                        # (edge-on band) than the full-frame threshold (20)


def blob_in(bgr, color, min_area):
    """Largest color blob centroid + mask, with a caller-set area floor."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in COLOR_RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask
    big = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(big) < min_area:
        return None, mask
    m = cv2.moments(big)
    return (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])), mask


def measure_tape(cam, color, center, tries=6, half=ROI_HALF):
    """Locate the tape as a pixel (and its depth, for readout) searching only an
    ROI around `center` (its last known pixel) so background yellow can't be
    mistaken for it. center=None searches the whole frame (first acquisition).
    Returns (uv float[2], depth_m or None) or (None, None)."""
    uvs, zs = [], []
    min_area = 20 if center is None else TAPE_MIN_AREA_ROI
    for _ in range(tries):
        bgr, depth = cam.read()
        if center is None:
            ox, oy, sub = 0, 0, bgr
        else:
            cx, cy = int(center[0]), int(center[1])
            x0 = max(0, cx - half); y0 = max(0, cy - half)
            x1 = min(FRAME_W, cx + half); y1 = min(FRAME_H, cy + half)
            ox, oy, sub = x0, y0, bgr[y0:y1, x0:x1]
        blob, _ = blob_in(sub, color, min_area)
        if blob is None:
            continue
        u, v = blob[0] + ox, blob[1] + oy
        uvs.append((u, v))
        # foreground read: the tape is a marker in front of whatever's behind
        # it, so take the near pixels (a dark cup behind gives bad depth).
        z = cam.depth_at(depth, u, v, patch=9, reduce="fg")
        if z is not None and DEPTH_MIN <= z <= DEPTH_MAX:
            zs.append(z)
    if len(uvs) < max(2, tries // 3):
        return None, None
    uv = np.median(np.array(uvs, dtype=float), axis=0)
    return uv, (float(np.median(zs)) if zs else None)


def depth_ring(cam, depth, u, v, r, n=12):
    """Robust median depth (m) on a ring of radius r around (u,v) — the table
    around an object reads cleanly even when the object itself (black/glossy)
    does not."""
    zs = []
    for k in range(n):
        a = 2 * math.pi * k / n
        z = cam.depth_at(depth, int(u + r * math.cos(a)), int(v + r * math.sin(a)), patch=5)
        if z is not None and DEPTH_MIN <= z <= DEPTH_MAX:
            zs.append(z)
    return float(np.median(zs)) if zs else None


def near_border(uv):
    return (uv[0] < FRAME_MARGIN or uv[0] > FRAME_W - FRAME_MARGIN or
            uv[1] < FRAME_MARGIN or uv[1] > FRAME_H - FRAME_MARGIN)


# --- live view --------------------------------------------------------------
SHOW = True
_view_ok = True


def show_view(cam, tape_uv, goal_uv, gap=None, status=""):
    """Draw the current tape (green) chasing the object (red cross) with the
    pixel gap, and imshow it. Self-disables if there's no display so the servo
    still runs headless."""
    global _view_ok
    if not (SHOW and _view_ok):
        return
    try:
        bgr, _ = cam.read()
        if goal_uv is not None:
            g = (int(goal_uv[0]), int(goal_uv[1]))
            cv2.drawMarker(bgr, g, (0, 0, 255), cv2.MARKER_CROSS, 28, 2)
            cv2.circle(bgr, g, 11, (0, 0, 255), 1)
        if tape_uv is not None:
            t = (int(tape_uv[0]), int(tape_uv[1]))
            cv2.circle(bgr, t, 7, (0, 255, 0), -1)
            if goal_uv is not None:
                cv2.line(bgr, t, g, (0, 255, 255), 1)
        txt = (f"gap {gap:.0f}px  {status}" if gap is not None else status).strip()
        if txt:
            cv2.putText(bgr, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
        cv2.imshow("servo — green=tape  red=target", bgr)
        cv2.waitKey(1)
    except Exception as e:  # noqa: BLE001 — view is best-effort
        _view_ok = False
        print(f"  (live view unavailable: {e}; continuing headless)")


# ---------------------------------------------------------------------------
# kinematics helpers (identity signs: motor angle == URDF angle)
# ---------------------------------------------------------------------------
def q8_of(q7):
    return clamp8(list(q7) + [GRIP_SERVO_RAD])


def clearance_ok(q7):
    return kin.min_column_clearance(q7) >= kin.CLEARANCE_MIN_M


def path_clear(q_from, q_to, n=6):
    a, b = np.asarray(q_from), np.asarray(q_to)
    return all(clearance_ok(a + (b - a) * s) for s in np.linspace(0, 1, n))


# ---------------------------------------------------------------------------
# servo primitives
# ---------------------------------------------------------------------------
async def goto(arm, q7, settle=0.6):
    """Move to a 7-joint config (caller guarantees the path is clear); return
    the actual joints read back (radians, 7)."""
    await arm.move(q8_of(q7), velocity=CAL_VELOCITY)
    await asyncio.sleep(settle)
    actual = await arm.read_actual(q8_of(q7))
    return np.array(actual[:7])


async def reacquire(cam, color, last_uv, attempts=3):
    """Fight to relocate the tape before declaring it lost: ROI at the last
    spot, then a widened ROI, then a full-frame search, retried a few times.
    A single noisy frame must not kill an entire run. Returns (uv, depth) or
    (None, None) only after all attempts genuinely fail."""
    for _ in range(attempts):
        for half in (ROI_HALF, ROI_HALF * 2):
            uv, z = measure_tape(cam, color, last_uv, half=half)
            if uv is not None:
                return uv, z
        uv, z = measure_tape(cam, color, None)       # tape may have left the ROI
        if uv is not None:
            return uv, z
        await asyncio.sleep(0.15)
    return None, None


async def bootstrap_px(arm, cam, color, q0, uv0):
    """Estimate the 2x3 image Jacobian d[u,v]/d[J1,J2,J4] from one small probe
    per control joint. Pixel-only — no depth. Returns J (2x3) or None."""
    probe = math.radians(PROBE_DEG)
    cols = []
    for j in CONTROL:
        qp = q0.copy(); qp[j] += probe; pj = probe
        if not (clearance_ok(qp) and path_clear(q0, qp)):
            qp = q0.copy(); qp[j] -= probe; pj = -probe
            if not (clearance_ok(qp) and path_clear(q0, qp)):
                print(f"  bootstrap: J{j+1} boxed in both directions — abort")
                return None
        await goto(arm, qp)
        uv, _ = measure_tape(cam, color, uv0)
        if uv is None:
            print(f"  bootstrap: lost tape probing J{j+1}")
            await goto(arm, q0)
            return None
        cols.append((uv - uv0) / pj)                 # px per rad
        await goto(arm, q0)                          # recenter each time
    return np.column_stack(cols)                     # 2x3


def joint_margins(q_ctrl):
    """Fractional distance of each control joint from its nearest limit: ~1 at
    mid-range, floored at 0.05 so a joint is discouraged — never fully frozen —
    near a limit."""
    return np.clip(np.minimum(q_ctrl - CTRL_LO, CTRL_HI - q_ctrl) / CTRL_HALF,
                   0.05, 1.0)


def servo_step_px(J, e, q_ctrl):
    """Weighted damped least-norm step for a 2D pixel error e. The redundant
    3rd DOF is resolved by joint-limit weighting: a joint near its limit gets a
    tiny weight, so the motion is routed into joints that still have travel
    (min dq^T W dq s.t. J dq = e, with W = 1/margin^2)."""
    Winv = np.diag(joint_margins(q_ctrl) ** 2)   # low mobility for near-limit joints
    dq = Winv @ J.T @ np.linalg.solve(J @ Winv @ J.T + MU_PX * np.eye(2), LAMBDA * e)
    m = np.max(np.abs(dq))
    lim = math.radians(STEP_MAX_DEG)
    if m > lim:
        dq *= lim / m
    return dq


def broyden(J, dq, dp):
    """Rank-1 Jacobian update from an observed (joint step -> pixel move)."""
    denom = float(dq @ dq)
    if denom < 1e-9:
        return J
    return J + np.outer(dp - J @ dq, dq) / denom


async def servo_align(arm, cam, color, q, uv, goal_uv, samples=None):
    """Drive the tape pixel uv onto goal_uv. Returns (status, q, uv). If a
    `samples` list is given, appends (q7, tape_uv) each step — the (FK-3D,
    pixel) pairs a later PnP calibration consumes."""
    J = await bootstrap_px(arm, cam, color, q, uv)
    if J is None:
        return "lost", q, uv
    uv2, z_tape = await reacquire(cam, color, uv)    # refresh after probes
    if uv2 is not None:
        uv = uv2

    err_hist, hits = [], 0
    for step in range(1, MAX_STEPS + 1):
        e = goal_uv - uv
        en = float(np.linalg.norm(e))
        err_hist.append(en)
        if samples is not None:
            samples.append((q.copy(), np.asarray(uv, float)))
        zs = f"{z_tape:.3f}m" if z_tape else "  ?  "
        print(f"  [{step:2d}] tape uv({uv[0]:.0f},{uv[1]:.0f}) d{zs} "
              f"gap {en:5.1f}px e({e[0]:+.0f},{e[1]:+.0f}) "
              f"joints {np.round(np.degrees(q[CONTROL]), 1)}")
        show_view(cam, uv, goal_uv, en, f"step {step}")

        if en < CONVERGE_PX:
            hits += 1
            if hits >= CONVERGE_HITS:
                return "converged", q, uv
        else:
            hits = 0
        if len(err_hist) > STALL_WINDOW and \
           err_hist[-STALL_WINDOW - 1] - en < STALL_MIN_IMPROVE_PX and en >= CONVERGE_PX:
            return "stalled", q, uv

        dq3 = servo_step_px(J, e, q[CONTROL])
        moved = False
        for scale in (1.0, 0.5, 0.25):
            qt = q.copy(); qt[CONTROL] += dq3 * scale
            if np.max(np.abs(np.array(q8_of(qt)[:7]) - qt)) > math.radians(1.0):
                continue                              # a joint limit clipped it
            if not (clearance_ok(qt) and path_clear(q, qt)):
                continue
            q_new = await goto(arm, qt)
            moved = True
            break
        if not moved:
            m = joint_margins(q[CONTROL])
            at_lim = [f"J{CONTROL[i]+1}" for i in range(len(CONTROL)) if m[i] <= 0.06]
            print(f"  step blocked (limits/clearance). control-joint margins "
                  f"{np.round(m, 2)}  at-limit {at_lim or 'none'}")
            return "stalled", q, uv

        uv_new, z_tape = await reacquire(cam, color, uv)
        if uv_new is None:
            bgr, _ = cam.read()
            _, mask = blob_in(bgr, color, 1)
            vis = bgr.copy()
            cv2.circle(vis, (int(uv[0]), int(uv[1])), ROI_HALF, (255, 0, 0), 2)
            cv2.imwrite("servo_lost.jpg", vis)
            cv2.imwrite("servo_lost_mask.jpg", mask)
            yellow_px = int((mask > 0).sum())
            print(f"  LOST: no tape found. yellow pixels in whole frame: "
                  f"{yellow_px}. saved servo_lost.jpg (blue circle = last known "
                  "spot) + servo_lost_mask.jpg — check if the tape is edge-on, "
                  "occluded, or truly gone.")
            return "lost", q, uv
        if near_border(uv_new):
            print(f"  tape at frame edge uv({uv_new[0]:.0f},{uv_new[1]:.0f}) — "
                  "stopping before it leaves view")
            return "edge", q_new, uv_new

        J = broyden(J, (q_new - q)[CONTROL], uv_new - uv)
        q, uv = q_new, uv_new

    return "maxsteps", q, uv


# ---------------------------------------------------------------------------
# depth-free hand-eye calibration (PnP on the gripper's own poses)
# ---------------------------------------------------------------------------
async def collect_spread(arm, cam, color, q0):
    """Visit the fixed CALIB_POSES grid (wide J2+J4 spread), holding the wrist,
    returning through the aligned pose between each so every move is
    clearance-checked. Records (q7, tape_uv) wherever the tape is visible."""
    samples = []
    for j1, j2, j4 in CALIB_POSES:
        qd = q0.copy()
        qd[0], qd[1], qd[3] = math.radians(j1), math.radians(j2), math.radians(j4)
        if np.max(np.abs(np.array(q8_of(qd)[:7]) - qd)) > math.radians(1.0):
            continue
        if not (clearance_ok(qd) and path_clear(q0, qd)):
            continue
        q = await goto(arm, qd)
        uv, _ = await reacquire(cam, color, None)     # full-frame re-acquire
        if uv is not None and not near_border(uv):
            samples.append((q.copy(), np.asarray(uv, float)))
            show_view(cam, uv, None, None, f"calib {len(samples)}")
        await goto(arm, q0)                           # back to aligned pose between
    return samples


def pnp_calibrate(samples, K):
    """Recover base->camera pose (R, t) + a bounded tape offset from (q7,
    tape_uv) pairs via reprojection minimization — NO depth used. Iteratively
    drops the worst-reprojection pose (a bad tape read) until the fit is clean,
    keeping the tape offset in the model so only true pixel outliers are cut.
    Returns (R, t, offset, rms_px, n_kept, n_total) or None."""
    from scipy.optimize import least_squares
    if len(samples) < 8:
        return None
    frames_all = [kin.fk_frame(q) for q, _ in samples]
    uv_all = np.array([p for _, p in samples], dtype=np.float64)
    tcp_all = np.array([f[:3, 3] for f in frames_all], dtype=np.float64)
    # RANSAC first: a robust initial pose that ignores gross outlier tape reads,
    # so the bounded refinement below starts near the truth (a plain solvePnP
    # gets dragged off by the outliers and the bounds then trap it). The 12 px
    # threshold tolerates the constant tape-offset while cutting real outliers.
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        tcp_all, uv_all, K, None, reprojectionError=12.0, iterationsCount=300,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inl is None or len(inl) < 6:
        ok, rvec, tvec = cv2.solvePnP(tcp_all, uv_all, K, None,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        idx = np.arange(len(samples))
    else:
        idx = inl.ravel()
    fr = [frames_all[i] for i in idx]
    uvi = uv_all[idx]

    def resid(xx):
        R, _ = cv2.Rodrigues(xx[:3]); t = xx[3:6]; off = xx[6:9]
        cam = np.array([R @ (f[:3, 3] + f[:3, :3] @ off) + t for f in fr])
        return np.concatenate([K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2] - uvi[:, 0],
                               K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2] - uvi[:, 1]])

    lo = np.concatenate([rvec.ravel() - 1.0, tvec.ravel() - 0.7, [-0.06] * 3])
    hi = np.concatenate([rvec.ravel() + 1.0, tvec.ravel() + 0.7, [0.06] * 3])
    x0 = np.concatenate([rvec.ravel(), tvec.ravel(), np.zeros(3)])
    sol = least_squares(resid, x0, bounds=(lo, hi), loss="soft_l1", f_scale=2.0)
    R, _ = cv2.Rodrigues(sol.x[:3])
    e = np.linalg.norm(resid(sol.x).reshape(2, -1), axis=0)
    return R, sol.x[3:6], sol.x[6:9], float(np.sqrt((e ** 2).mean())), len(idx), len(samples)


def deproject_to_base(u, v, depth, K, R, t):
    """Pixel (u,v) at camera-Z `depth` -> base-frame point, given base->camera (R,t)."""
    p_cam = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0]) * depth
    return R.T @ (p_cam - t)


GRASP_ABOVE = 0.05   # m; grab the cup body this far above the box-top surface
                     # (kept generous: the calibration depth axis is ~cm-loose,
                     # so bias UP — a miss-high is safe, ramming the box is not)


def box_plane_grasp(u, v, depth, cam, R, t, campos, grasp_above=GRASP_ABOVE):
    """Cup body center in base frame WITHOUT the cup's own (invalid) depth:
    intersect the cup-pixel ray with the box-top plane, then raise by
    `grasp_above`. The plane height comes from the reliable ring depths around
    the cup (the box top reads cleanly even when the black cup does not).
    Returns (grasp_base, y_box) or (None, y_box)."""
    K = cam.K
    ring = []
    for k in range(16):
        a = 2 * math.pi * k / 16
        ru, rv = int(u + 45 * math.cos(a)), int(v + 45 * math.sin(a))
        dz = cam.depth_at(depth, ru, rv, patch=5)
        if dz is not None and DEPTH_MIN <= dz <= DEPTH_MAX:
            ring.append(deproject_to_base(ru, rv, dz, K, R, t))
    if not ring:
        return None, None
    y_box = float(np.median([p[1] for p in ring]))
    # ray point at camera-Z z is campos + z*w (w = ray dir in base frame)
    w = R.T @ np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    if abs(w[1]) < 1e-6:
        return None, y_box
    s = (y_box + grasp_above - campos[1]) / w[1]     # intersect the grasp plane
    return campos + s * w, y_box


async def execute_grasp(arm, target, q_seed, mode, advance_m=0.03, velocity=0.2):
    """Pick the cup at base-frame `target`. Reuses pick.solve_grasp for the IK
    plan (approach direction + gripper orientation), then runs
    open -> approach -> descend -> close -> lift -> hold -> lower -> release ->
    retreat. Every planned pose is clearance-checked before any motion."""
    import pick
    from arm_io import GRIPPER_OPEN_DEG
    # try the requested approach first, then the other; keep the first whose
    # pregrasp/grasp/lift are ALL clear of the body column.
    chosen = None
    for m in (mode, "top" if mode == "side" else "side"):
        plan = pick.solve_grasp(target, q_seed, drop=0.0, mode=m)   # grasp AT the
        if plan is None:                                            # target height,
            continue                                               # not below (box safety)
        _, q_pre = plan["pre"]; _, q_grasp = plan["grasp"]
        q_lift, _ = kin.ik(plan["lift"], q_grasp, plan["z_axis"])
        cl = [kin.min_column_clearance(x) for x in (q_pre, q_grasp, q_lift)]
        if all(c >= kin.CLEARANCE_MIN_M for c in cl):
            chosen = (plan, q_pre, q_grasp, q_lift)
            break
        print(f"  {m} approach too close to column "
              f"(pre/grasp/lift {['%.0f' % (c*100) for c in cl]} cm) — trying next")
    if chosen is None:
        print("no clearance-safe grasp (side + top tried) — nudge the cup closer.")
        return
    plan, q_pre, q_grasp, q_lift = chosen
    signs = [1.0] * 7
    OPEN = -44.0                              # a touch wider than -42 (limit -45)
    # Forward nudge toward the cup before closing. The depth-free calibration is
    # good to ~1-2 cm and the cup gives no depth, so advancing a few cm in the
    # reach direction makes the fingers straddle it instead of stopping short.
    q_hold, q_lift2, adv = q_grasp, q_lift, None
    if advance_m > 0:
        grasp_pos = plan["grasp"][0]
        adv_pos = grasp_pos + pick._outward(grasp_pos) * advance_m
        q_adv, e_adv = kin.ik(adv_pos, q_grasp, plan["z_axis"])
        if e_adv < 0.012 and clearance_ok(q_adv):
            q_hold = q_adv
            q_lift2, _ = kin.ik(adv_pos + kin.UP_BASE * pick.LIFT_M, q_adv, plan["z_axis"])
            adv = advance_m
        else:
            print(f"  (skip advance: IK err {e_adv*1000:.0f}mm / clearance)")
    try:
        steps = [("open + approach", pick.to_motor(q_pre, signs, OPEN), velocity),
                 ("descend to grasp", pick.to_motor(q_grasp, signs, OPEN), 0.15)]
        if adv:
            steps.append((f"advance {adv*100:.0f}cm onto cup",
                          pick.to_motor(q_hold, signs, OPEN), 0.1))
        steps += [("CLOSE gripper", None, None),
                  ("lift", pick.to_motor(q_lift2, signs, 0.0), 0.2),
                  ("hold", None, None),
                  ("lower back", pick.to_motor(q_hold, signs, 0.0), 0.15),
                  ("OPEN gripper", None, None),
                  ("retreat", pick.to_motor(q_pre, signs, OPEN), velocity)]
    except RuntimeError as ex:
        print(f"grasp aborted (joint limit): {ex}")
        return
    print(f"\ngrasp plan — {plan['mode']} approach, z-axis {np.round(plan['z_axis'], 2)}:")
    for name, q8, v in steps:
        print(f"  {name}" if q8 is None
              else f"  {name:18s} deg {np.round(np.degrees(q8), 0)}")
    print("\nGRASP MOTION in 3s — Ctrl+C to abort, hand on the e-stop")
    await asyncio.sleep(3)
    await arm.gripper(open_=True)
    for name, q8, v in steps:
        print(f"-> {name}")
        if q8 is not None:
            await arm.move(q8, velocity=v)
        elif "CLOSE" in name:
            await arm.gripper(open_=False)
        elif "OPEN" in name:
            await arm.gripper(open_=True)
        else:
            await asyncio.sleep(2.0)             # hold
    print("grasp sequence complete.")


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
async def reach_visibility(arm, cam, color, start_deg, n=5, show=True):
    """Sweep START -> far reach holding this wrist; return (seen, total) tape
    visibility so a caller can compare wrist angles."""
    q = await goto(arm, np.radians(start_deg), settle=0.6)
    seen = total = 0
    for t in np.linspace(0.0, 1.0, n):
        j1, j2, j4 = -10 + 30 * t, 25 - 20 * t, 95 - 40 * t
        qd = list(start_deg); qd[0], qd[1], qd[3] = j1, j2, j4
        qn = np.radians(qd)
        if not (clearance_ok(qn) and path_clear(q, qn)):
            continue
        q = await goto(arm, qn)
        uv, _ = measure_tape(cam, color, None)
        total += 1; seen += int(uv is not None)
        if show:
            show_view(cam, uv, None, None,
                      f"wrist{start_deg[5]:+.0f} J1{j1:+.0f} "
                      f"{'VISIBLE' if uv is not None else 'LOST'}")
    return seen, total


async def wrist_scan(arm, cam, color, base_deg):
    """Try a range of fixed wrist-pitch bends and report where the tape stays
    visible across the reach — finds the right --wrist in one hands-free pass."""
    print("WRIST SCAN — trying J6 bends to keep the tape visible through the "
          "reach. Watch the live view; this takes a couple minutes.")
    results = []
    for w in (-30, -20, -10, 0, 10, 20, 30):
        sd = list(base_deg); sd[5] = w
        seen, total = await reach_visibility(arm, cam, color, sd, n=5)
        results.append((w, seen, total))
        print(f"  wrist J6={w:+3d}: tape visible {seen}/{total}")
    results.sort(key=lambda r: (-r[1], abs(r[0])))
    w, seen, total = results[0]
    print(f"\nBEST: wrist J6={w:+d}  ({seen}/{total} poses visible)")
    if seen >= total:
        print(f'Run:  python servo.py "<object>" --wrist {w}')
    else:
        print("No fixed wrist keeps the flat tape visible the whole way — it "
              "goes edge-on as the hand turns. The robust fix is a small colored "
              "BALL/blob on the fingertip (a sphere looks the same from every "
              f"angle). Best available for now is --wrist {w}.")
    await goto(arm, np.radians(base_deg), settle=0.6)


async def pose_preview(arm, cam, color, start_deg):
    """Sweep the reach envelope holding the wrist fixed, reporting where the
    tape stays visible. Use it to tune --wrist without running a full servo:
    the diagonal traces START -> the far reach where tracking was lost."""
    print("POSE PREVIEW — holding the wrist, sweeping into the reach. Watch the "
          "green dot: is the tape visible the WHOLE way?")
    q = await goto(arm, np.radians(start_deg), settle=0.8)
    seen = total = 0
    for t in np.linspace(0.0, 1.0, 7):
        j1, j2, j4 = -10 + 30 * t, 25 - 20 * t, 95 - 40 * t   # START -> far reach
        qd = list(start_deg); qd[0], qd[1], qd[3] = j1, j2, j4
        qn = np.radians(qd)
        if not (clearance_ok(qn) and path_clear(q, qn)):
            print(f"  J1{j1:+.0f} J4{j4:+.0f}: skipped (clearance)")
            continue
        q = await goto(arm, qn)
        uv, _ = measure_tape(cam, color, None)               # full-frame re-acquire
        vis = uv is not None
        total += 1; seen += int(vis)
        show_view(cam, uv, None, None,
                  f"J1{j1:+.0f} J4{j4:+.0f} {'VISIBLE' if vis else 'TAPE LOST'}")
        print(f"  J1{j1:+.0f} J2{j2:+.0f} J4{j4:+.0f}: "
              + (f"visible {tuple(int(x) for x in uv)}" if vis else "TAPE NOT VISIBLE"))
        await asyncio.sleep(0.5)
    print(f"\ntape visible in {seen}/{total} reach poses with wrist J6="
          f"{start_deg[5]:+.0f} deg."
          + ("  Good — try the servo with this --wrist." if seen == total else
             "  Try another --wrist value (e.g. +/-20, 30) until it's all VISIBLE."))
    await goto(arm, np.radians(start_deg), settle=0.6)


async def run(args):
    global SHOW
    SHOW = args.view
    cam = OakCamera()
    color = args.color
    # J6 (wrist pitch, index 5) held fixed all run — a bend keeps the tape's
    # broad face toward the camera so tracking survives the reach.
    start_deg = list(START_DEG)
    start_deg[5] = float(args.wrist)
    try:
        if args.pose or args.scan:
            arm = RightArm(velocity=CAL_VELOCITY)
            try:
                await arm.connect()
                pose0 = await arm.read_pose()
                if max(abs(x) for x in pose0[:7]) < 1e-6:
                    sys.exit("arm reads all zeros — CAN down/unpowered; bring up can1")
                print("\nARM WILL MOVE ON ITS OWN in 3s — clear the workspace, hand "
                      "on the e-stop (Ctrl+C parks + disables)")
                await asyncio.sleep(3)
                await arm.enable()
                if args.scan:
                    await wrist_scan(arm, cam, color, start_deg)
                else:
                    await pose_preview(arm, cam, color, start_deg)
            finally:
                print("-> park + disable")
                try:
                    await arm.park_and_disable()
                finally:
                    arm.shutdown()
            return

        # --- perception: object -> pixel (+ depth readout, not used for aiming) --
        bgr, depth = cam.read()
        if args.uv:
            u, v, label = args.uv[0], args.uv[1], "manual pixel"
        else:
            print(f"asking {vlm.DEFAULT_MODEL} to find: {args.object!r} ...")
            try:
                hit = vlm.locate(bgr, args.object)
            except vlm.GeminiError as e:
                sys.exit(f"VLM error: {e}\nUsually transient — just re-run. If the "
                         "OAK camera also crashed, unplug/replug it first.")
            if hit is None:
                sys.exit("VLM: object not found in view")
            u, v, label = hit
        goal_uv = np.array([u, v], dtype=float)
        z_obj = cam.depth_at(depth, u, v, patch=11)
        z_tab = depth_ring(cam, depth, u, v, r=45)
        vis = bgr.copy()
        cv2.drawMarker(vis, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
        cv2.circle(vis, (u, v), 12, (0, 255, 0), 2)
        cv2.imwrite("servo_target.jpg", vis)
        print(f"'{label}' at pixel ({u},{v})  depth: object "
              f"{'%.3f' % z_obj if z_obj else 'INVALID'} m, table-ring "
              f"{'%.3f' % z_tab if z_tab else 'INVALID'} m  (saved servo_target.jpg)")
        show_view(cam, None, goal_uv, None, f"target: {label}")

        if args.locate_only:
            if SHOW and _view_ok:
                print("--locate-only: showing target — press any key in the view to exit")
                try:
                    cv2.waitKey(0)
                except Exception:
                    pass
            print("--locate-only: no motion.")
            return

        if args.regrasp:
            # fast path: reuse a saved calibration, skip the align + wiggle
            try:
                with open(SERVO_CALIB) as f:
                    cd = json.load(f)
            except FileNotFoundError:
                sys.exit("no saved calibration (servo_calib.json) — run --grasp "
                         "once first. Do NOT move the camera between runs.")
            R = np.array(cd["R"]); t = np.array(cd["t"]); campos = -R.T @ t
            arm = RightArm(velocity=CAL_VELOCITY)
            try:
                await arm.connect()
                pose0 = await arm.read_pose()
                if max(abs(x) for x in pose0[:7]) < 1e-6:
                    sys.exit("arm reads all zeros — CAN down/unpowered; bring up can1")
                print(f"\n--regrasp: reusing saved calibration (rms {cd.get('rms')}). "
                      "ARM MOVES in 3s — clear workspace, e-stop ready.")
                await asyncio.sleep(3)
                await arm.enable()
                q = await goto(arm, np.radians(start_deg), settle=0.8)
                bgr, depth = cam.read()
                grasp, y_box = box_plane_grasp(u, v, depth, cam, R, t, campos)
                if grasp is None:
                    print("couldn't read the box-top plane — no valid ring depth.")
                    return
                q7g, ikerr = kin.ik(grasp, q[:7]); clr = kin.min_column_clearance(q7g)
                print(f"cup -> base {np.round(grasp, 3)}  Y_box {y_box:.3f}m  "
                      f"IK err {ikerr*1000:.0f}mm  clr {clr*100:.0f}cm")
                if ikerr < 0.02 and grasp[0] > 0.1 and clr >= 0.10:
                    await execute_grasp(arm, grasp, q[:7], args.grasp_mode, args.advance)
                else:
                    print("target not reachable — recalibrate with --grasp (cup moved?).")
            finally:
                print("-> park + disable")
                try:
                    await arm.park_and_disable()
                finally:
                    arm.shutdown()
            return

        arm = RightArm(velocity=CAL_VELOCITY)
        try:
            await arm.connect()
            pose0 = await arm.read_pose()
            if max(abs(x) for x in pose0[:7]) < 1e-6:
                sys.exit("arm reads all zeros — CAN down/unpowered; bring up can1")
            print("\nARM WILL MOVE ON ITS OWN in 3s — clear the workspace, hand on "
                  "the e-stop (Ctrl+C parks + disables)")
            await asyncio.sleep(3)
            await arm.enable()
            # Rest->START is autocal's proven first move; the hanging rest sits near
            # the column by design, so we don't path-gate it. Every step below IS.
            q = await goto(arm, np.radians(start_deg), settle=0.8)

            uv, z_tape = measure_tape(cam, color, None)      # first acquisition
            if uv is None:
                sys.exit("tape not visible at the start pose — check camera aim / "
                         "tape, or nudge START_DEG so the gripper is centered")
            print(f"tape acquired at uv({uv[0]:.0f},{uv[1]:.0f})  depth "
                  f"{'%.3f' % z_tape if z_tape else '?'} m")
            show_view(cam, uv, goal_uv, float(np.linalg.norm(goal_uv - uv)), "acquired")

            samples = []
            status, q, uv = await servo_align(arm, cam, color, q, uv, goal_uv, samples)
            gap = float(np.linalg.norm(goal_uv - uv))
            print(f"\nalign: {status}   final pixel gap {gap:.1f}px  "
                  f"tape uv({uv[0]:.0f},{uv[1]:.0f})  object uv({u},{v})")
            show_view(cam, uv, goal_uv, gap, status)

            if status != "converged":
                print(f"\ndid not converge ({status}). Paste this log.")
                return
            if not args.calib:
                bgr, depth = cam.read()
                z_tab2 = depth_ring(cam, depth, u, v, r=45)
                print(f"\nALIGNMENT OK (table-ring "
                      f"{'%.3f' % z_tab2 if z_tab2 else '?'} m). Re-run with --calib "
                      "to compute the grasp calibration.")
                return

            # --- depth-free hand-eye calibration from the gripper's own poses ---
            print("\ncollecting reach-varied poses (breaks the planar trajectory so "
                  "PnP is unambiguous)...")
            samples += await collect_spread(arm, cam, color, q)
            print(f"calibrating from {len(samples)} (pose, pixel) pairs — NO depth...")
            calib = pnp_calibrate(samples, cam.K)
            if calib is None:
                print("calibration failed — too few visible poses.")
                return
            R, t, off, rms, n_kept, n_tot = calib
            campos = -R.T @ t
            tcpc = np.array([kin.fk(s[0]) for s in samples])
            sv = np.linalg.svd(tcpc - tcpc.mean(0), compute_uv=False)
            planar = float(sv[2] / sv[0]) if sv[0] > 1e-9 else 0.0
            bgr, depth = cam.read()
            grasp, y_box = box_plane_grasp(u, v, depth, cam, R, t, campos)
            print(f"\n=== CALIBRATION ({n_kept}/{n_tot} poses kept) ===")
            print(f"  reproj rms {rms:.1f} px   planarity {planar:.2f} (want >0.15)   "
                  f"tape-offset {np.round(off * 100, 1)} cm")
            print(f"  camera position (base): {np.round(campos, 3)} m")
            if grasp is None:
                print("  couldn't read the box-top plane (no valid ring depth).")
                return
            reach = float(np.hypot(grasp[0], grasp[2]))
            q7g, ikerr = kin.ik(grasp, q[:7])
            clr = kin.min_column_clearance(q7g)
            print(f"  box-top height Y_box {y_box:.3f} m")
            print(f"  GRASP target (ray x box-plane +{GRASP_ABOVE*100:.0f}cm) -> "
                  f"base {np.round(grasp, 3)}  (reach {reach:.2f} m; want +X, -Y)")
            print(f"  grasp IK: err {ikerr * 1000:.0f} mm  clr {clr * 100:.0f} cm")
            # rms<6 cleanly separates the good runs (1-3px) from the bad ones
            # (>=9px); planarity guards conditioning; the rest guard reachability.
            sane = (rms < 6 and planar > 0.15 and grasp[0] > 0.1
                    and grasp[1] < 0.05 and ikerr < 0.02 and clr >= 0.10)
            if not sane:
                print("  OFF — paste this and I'll adjust before any grasp motion.")
                return
            print("  OK — calibration sane + cup reachable.")
            with open(SERVO_CALIB, "w") as f:            # cache for --regrasp
                json.dump({"R": R.tolist(), "t": t.tolist(),
                           "K": cam.K.tolist(), "rms": rms}, f)
            if args.grasp:
                await execute_grasp(arm, grasp, q[:7], args.grasp_mode, args.advance)
            else:
                print("  Re-run with --grasp to pick it up, or --regrasp to skip "
                      "the calibration wiggle next time.")
        finally:
            print("-> park + disable")
            try:
                await arm.park_and_disable()
            finally:
                arm.shutdown()
    finally:
        try:
            cam.close()
        except Exception:  # noqa: BLE001 — cleanup must not mask the real error
            pass
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("object", nargs="?", help="what to aim at, e.g. 'the red cup'")
    ap.add_argument("--color", default="yellow", help="gripper tape color")
    ap.add_argument("--locate-only", action="store_true",
                    help="run perception only, print the object pixel, no motion")
    ap.add_argument("--no-view", dest="view", action="store_false",
                    help="disable the live camera window (headless / over SSH)")
    ap.add_argument("--wrist", type=float, default=0.0, metavar="DEG",
                    help="fixed J6 wrist-pitch bend held all run, to keep the "
                         "tape facing the camera (try +/-20..30)")
    ap.add_argument("--pose", action="store_true",
                    help="preview: sweep the reach envelope at this --wrist and "
                         "report where the tape stays visible (no servo)")
    ap.add_argument("--scan", action="store_true",
                    help="hands-free: try a range of wrist bends and report which "
                         "keeps the tape visible through the reach (no servo)")
    ap.add_argument("--calib", action="store_true",
                    help="after aligning, collect reach-varied poses and compute a "
                         "depth-free PnP hand-eye calibration; report the cup's "
                         "base position + reachability (no grasp motion)")
    ap.add_argument("--grasp", action="store_true",
                    help="calibrate then actually pick the cup (implies --calib)")
    ap.add_argument("--grasp-mode", choices=("side", "top"), default="side",
                    help="grasp approach: side (fingers around the body, default) "
                         "or top (descend from above)")
    ap.add_argument("--advance", type=float, default=0.03, metavar="M",
                    help="forward nudge toward the cup before closing, meters "
                         "(default 0.03; raise if it closes short, e.g. 0.05)")
    ap.add_argument("--regrasp", action="store_true",
                    help="reuse the saved calibration (servo_calib.json) and grasp "
                         "immediately — skips align + wiggle (camera must not move)")
    ap.add_argument("--uv", type=int, nargs=2, metavar=("U", "V"),
                    help="skip the VLM, aim at this object pixel")
    args = ap.parse_args()
    if args.grasp:
        args.calib = True                       # grasp needs the calibration
    if not args.pose and not args.scan and not args.object and not args.uv:
        ap.error("give an object description or --uv U V (or use --pose/--scan)")
    if args.object is None:
        args.object = "object"
    t0 = time.time()
    asyncio.run(run(args))
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
