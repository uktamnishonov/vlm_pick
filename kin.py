#!/usr/bin/env python3
"""Kinematics + camera<->arm-base calibration math for the right OpenArm.

Frame notes (verified via FK on the shipped URDF):
  - base frame = openarm_right_link0; the arm hangs along -Y at the zero
    pose, so +Y(base) points up (opposite gravity) when the arm is mounted
    upright.
  - TCP = gripper tool center, 8 cm past the hand link (chain's
    last_link_vector).

Motor angles are assumed to equal URDF right-arm joint values (J2's
asymmetric limits match in both conventions). calibrate.py verifies this
empirically and, if the rigid-fit residual is high, searches per-joint sign
flips to find the true mapping — the winning signs are stored in
calibration.json and applied everywhere via motor_to_urdf().
"""
import json
import math
import os

import numpy as np
from openarm.kinematics.inverse.ikpy import IkpyInverseKinematics

CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
UP_BASE = np.array([0.0, 1.0, 0.0])  # +Y = up in the right-arm base frame

# The two arms mount 6.2 cm apart on a central body column. In the RIGHT
# arm's base frame that column sits at (X=0, Z=-0.031) and runs vertically
# (along Y). Distal links swinging back toward it are what hit the metal —
# clearance is the horizontal distance from a link to this axis.
COLUMN_XZ = np.array([0.0, -0.031])
CLEARANCE_MIN_M = 0.10  # refuse any pose bringing a distal link closer than this


def min_column_clearance(q7_urdf):
    """Smallest horizontal distance from any distal link (elbow→TCP) to the
    central body column, for a 7-joint config. Below CLEARANCE_MIN_M is a
    collision risk with the body between the two arms."""
    full = np.zeros(12)
    full[1:8] = np.asarray(q7_urdf, dtype=float)
    frames = chain().forward_kinematics(full, full_kinematics=True)
    # links 0-2 are the shoulder housing, structurally next to the column;
    # only distal links (3+) can swing into it
    return min(
        float(np.hypot(f[0, 3] - COLUMN_XZ[0], f[2, 3] - COLUMN_XZ[1]))
        for f in frames[3:]
    )

_solver = None


def solver():
    global _solver
    if _solver is None:
        _solver = IkpyInverseKinematics()
    return _solver


def chain():
    return solver()._right_chain


def motor_to_urdf(q7_motor, signs=None):
    q = np.asarray(q7_motor, dtype=float)
    if signs is not None:
        q = q * np.asarray(signs, dtype=float)
    return q


def fk(q7_urdf):
    """TCP position (base frame, meters) for 7 URDF joint angles (rad)."""
    full = np.zeros(12)
    full[1:8] = q7_urdf
    return chain().forward_kinematics(full)[:3, 3]


def ik(target_pos, q7_init, z_axis=None):
    """Solve 7 joint angles (rad). If z_axis given, align TCP z to it
    (orientation_mode='Z'); else position-only. Returns (q7, fk_err_m)."""
    # clip the seed into the URDF bounds — a motor resting at -0.0001 rad on
    # a joint bounded [0, ...] makes scipy reject the initial guess outright
    lo = np.array([l.bounds[0] for l in chain().links[1:8]]) + 1e-6
    hi = np.array([l.bounds[1] for l in chain().links[1:8]]) - 1e-6
    full = np.zeros(12)
    full[1:8] = np.clip(q7_init, lo, hi)
    if z_axis is not None:
        q = chain().inverse_kinematics(
            target_position=np.asarray(target_pos, dtype=float),
            target_orientation=np.asarray(z_axis, dtype=float),
            orientation_mode="Z",
            initial_position=full,
        )
    else:
        q = chain().inverse_kinematics(
            target_position=np.asarray(target_pos, dtype=float),
            initial_position=full,
        )
    q7 = q[1:8]
    err = float(np.linalg.norm(fk(q7) - np.asarray(target_pos, dtype=float)))
    return q7, err


def umeyama(src, dst):
    """Rigid transform (R, t) minimizing ||R @ src + t - dst||. Nx3 each."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    cov = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    t = mu_d - R @ mu_s
    return R, t


def fit_residuals(R, t, src, dst):
    pred = (R @ np.asarray(src).T).T + t
    return np.linalg.norm(pred - np.asarray(dst), axis=1)


def search_signs(cam_pts, motor_angles):
    """Try all 2^7 per-joint sign flips; return (best_signs, best_rms, per_sign_rms).

    A wrong sign warps the FK point cloud non-rigidly, so the correct
    convention is the one whose Umeyama fit has the lowest residual.
    """
    cam_pts = np.asarray(cam_pts)
    best = (None, np.inf)
    results = []
    for mask in range(128):
        signs = [(-1.0 if mask >> j & 1 else 1.0) for j in range(7)]
        base_pts = np.array([fk(motor_to_urdf(q, signs)) for q in motor_angles])
        R, t = umeyama(cam_pts, base_pts)
        rms = float(np.sqrt((fit_residuals(R, t, cam_pts, base_pts) ** 2).mean()))
        results.append((signs, rms))
        if rms < best[1]:
            best = (signs, rms)
    results.sort(key=lambda x: x[1])
    return best[0], best[1], results[:5]


def fk_frame(q7_urdf):
    """Full 4x4 TCP transform (base frame) for 7 URDF joint angles."""
    full = np.zeros(12)
    full[1:8] = q7_urdf
    return chain().forward_kinematics(full)


def _rodrigues(r):
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def fit_extended(cam_pts, motor_angles, signs):
    """Jointly fit T_base_cam AND the tape's offset from the TCP.

    The calibration tape sits on a fingertip, not exactly at the TCP; with
    varying wrist angles that unknown offset dominates the residual of a
    rigid-only fit. Model: R@p_cam + t = T_tcp(q) @ [d, 1]. 9 params, needs
    scipy. Returns (R, t, d_local, rms, residuals).
    """
    from scipy.optimize import least_squares

    cam_pts = np.asarray(cam_pts, dtype=float)
    frames = [fk_frame(motor_to_urdf(q, signs)) for q in motor_angles]
    tcp = np.array([f[:3, 3] for f in frames])
    R0, t0 = umeyama(cam_pts, tcp)

    def unpack(x):
        return _rodrigues(x[:3]), x[3:6], x[6:9]

    def resid(x):
        R, t, d = unpack(x)
        pred_tape = np.array([f[:3, 3] + f[:3, :3] @ d for f in frames])
        r = ((R @ cam_pts.T).T + t - pred_tape).ravel()
        return np.concatenate([r, 0.05 * d])  # mild prior: tape near TCP

    # start rotation vector from R0
    from numpy.linalg import svd
    A = (R0 - R0.T) / 2
    rv = np.array([A[2, 1], A[0, 2], A[1, 0]])
    s = np.linalg.norm(rv)
    ang = math.asin(min(1.0, max(-1.0, s)))
    if np.trace(R0) < 1:  # obtuse rotation: fall back to identity start
        ang, rv, s = 0.0, np.zeros(3), 1.0
    x0 = np.concatenate([rv / s * ang if s > 1e-9 else np.zeros(3), t0, np.zeros(3)])
    # Bound the tape offset to a physically real magnitude (<=8 cm from TCP):
    # without this the optimizer invents a 1 m lever arm that overfits noisy
    # (wrist-rotated) poses to a meaningless low residual.
    lo = np.array([-3.2, -3.2, -3.2, -2, -2, -2, -0.08, -0.08, -0.08])
    hi = np.array([3.2, 3.2, 3.2, 2, 2, 2, 0.08, 0.08, 0.08])
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
    sol = least_squares(resid, x0, loss="soft_l1", f_scale=0.02, bounds=(lo, hi))
    R, t, d = unpack(sol.x)
    pred = np.array([f[:3, 3] + f[:3, :3] @ d for f in frames])
    res = np.linalg.norm((R @ cam_pts.T).T + t - pred, axis=1)
    return R, t, d, float(np.sqrt((res**2).mean())), res


def search_signs_extended(cam_pts, motor_angles, top_n=5):
    """Full 2^7 sign search using the tape-offset model. Returns best 5."""
    results = []
    for mask in range(128):
        signs = [(-1.0 if mask >> j & 1 else 1.0) for j in range(7)]
        try:
            R, t, d, rms, res = fit_extended(cam_pts, motor_angles, signs)
        except Exception:
            continue
        # penalize physically absurd tape offsets (tape is ON the fingertip)
        score = rms + max(0.0, np.linalg.norm(d) - 0.06) * 0.5
        results.append((score, rms, signs, R, t, d, res))
    results.sort(key=lambda r: r[0])
    return results[:top_n]


def save_calibration(R, t, signs, cam_pts, base_pts, rms, tape_offset=None):
    pts = np.asarray(base_pts)
    lo, hi = pts.min(0) - 0.12, pts.max(0) + 0.12
    data = {
        "R_base_cam": np.asarray(R).tolist(),
        "t_base_cam": np.asarray(t).tolist(),
        "joint_signs": list(signs),
        "rms_m": float(rms),
        "n_points": len(pts),
        "workspace_lo": lo.tolist(),
        "workspace_hi": hi.tolist(),
        "tape_offset_m": None if tape_offset is None else np.asarray(tape_offset).tolist(),
    }
    with open(CALIB_PATH, "w") as f:
        json.dump(data, f, indent=2)
    return CALIB_PATH


class Calibration:
    def __init__(self, path=CALIB_PATH):
        with open(path) as f:
            d = json.load(f)
        self.R = np.array(d["R_base_cam"])
        self.t = np.array(d["t_base_cam"])
        self.signs = d["joint_signs"]
        self.rms = d["rms_m"]
        self.ws_lo = np.array(d["workspace_lo"])
        self.ws_hi = np.array(d["workspace_hi"])

    def cam_to_base(self, p_cam):
        return self.R @ np.asarray(p_cam, dtype=float) + self.t

    def in_workspace(self, p_base):
        return bool(np.all(p_base >= self.ws_lo) and np.all(p_base <= self.ws_hi))
