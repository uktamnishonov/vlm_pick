# vlm_pick — language-driven picking on an OpenArm, without trusting depth

Tell a bimanual [OpenArm](https://github.com/enactic/openarm) what to pick up in plain
English. A VLM finds the object, the arm servos its gripper onto it **in the camera
image**, and only then does it work out the geometry needed to close its fingers.

```bash
python servo.py "Black coffee cup" --wrist 40 --grasp
```

Hardware: OpenArm (right arm, CAN), OAK-D Pro W stereo camera, Jetson Orin Nano.

---

## Why this exists

The obvious pipeline is: calibrate the camera to the arm once, ask a VLM for the object's
pixel, read the object's depth, convert to arm coordinates, move there.

It doesn't work on a black coffee cup.

1. **Open-loop calibration is unforgiving.** Four one-shot hand-eye fits gave 18 / 24 / 41
   / 361 mm of residual error. Whatever error the calibration has, the gripper inherits —
   there is no feedback to correct it.
2. **Stereo depth on a glossy black object is garbage.** The cup absorbs the IR dot
   pattern. Its depth reads ~0.5 m when it is really ~1.0 m away. Feeding that into the
   calibration poisons the fit, and feeding it into the grasp aims at empty air.

So this repo is built around one idea: **use the camera only for what it's good at.**

| Signal | Trustworthy? | Used for |
| --- | --- | --- |
| Pixels | Yes | Aiming, and calibration |
| Joint encoders | Yes | The 3D half of calibration |
| Depth on matte cardboard | Mostly | One number: the height of the surface the cup stands on |
| Depth on a glossy black cup | **No** | **Nothing. Never read.** |

## How it works

**1. Gemini locates the object** (`vlm.py`) — one image in, one pixel out.

**2. The gripper servos onto that pixel** (`servo.py`) — the gripper wears a strip of
yellow tape. A blob tracker gives its pixel; Gemini gave the cup's pixel. The arm drives
one onto the other using a 2×3 image Jacobian (∂pixel/∂joint), bootstrapped from three
4° probe moves and refined every step with a Broyden rank-1 update.

*No camera calibration is loaded, and no depth is read.* Redundancy in the 3 control
joints is resolved by joint-limit weighting (`W = 1/margin²`) so motion routes into the
yaw joint, which has 270° of travel, instead of pinning the pitch joint, which has 35°.
Converges to <11 px.

**3. Alignment is not enough** — and this is the crux. Hold your thumb up so it covers the
moon: perfectly aligned, and you are not touching the moon. Pixel alignment puts the
gripper somewhere on the camera ray through the cup, but the camera cannot tell whether
it is 15 cm short or 15 cm past. Two of three dimensions come free; the third decides
whether the fingers close on the cup or on air.

**4. So the robot calibrates the camera using itself as the measuring stick.** It visits a
fixed 12-pose grid and photographs its own tape at each. Forward kinematics gives the
tape's exact 3D position from the encoders; the blob tracker gives its pixel. Twelve
(3D, pixel) pairs are exactly the input for **PnP** — *"given known 3D landmarks and
where they landed in this photo, where was the camera?"* — solved with
`cv2.solvePnPRansac` plus a bounded refinement that also fits the tape's offset from the
tool center point.

**The camera is never asked how far away anything is.** 3D comes from encoders, 2D from
pixels. That is what makes this calibration depth-free. Typical reprojection error:
**1–4 px**.

Two failure modes this guards against, both of which bit us:
- *Planar degeneracy.* If the 12 landmarks lie roughly in a plane, PnP has two valid
  solutions — the real one and a mirror. It once picked the mirror and confidently placed
  the cup behind the robot. The pose grid is chosen for a strongly non-planar spread
  (planarity ≈ 0.42), reported and gated every run.
- *Outliers.* One background yellow object mistaken for the tape drags a plain fit off.
  RANSAC keeps the pose the majority of landmarks agree on.

**5. The cup's 3D position, without the cup's depth** — shoot a ray from the camera
through the cup's pixel and intersect it with the *box-top plane*, whose height comes from
a ring of 16 pixels on the cardboard **around** the cup. Cardboard is matte and reads
cleanly; the cup does not. The box tells us the range the cup refuses to.

**6. Grasp** — open wide → approach → descend → advance onto the cup → close → lift.
Every planned pose is checked for body-column clearance before any motion.

## Status

| Stage | State |
| --- | --- |
| VLM object grounding | works |
| Image-plane visual servoing | works — converges to 2–8 px |
| Depth-free PnP hand-eye calibration | works — 1–4 px reprojection |
| Cup → base coordinates, IK, clearance | works |
| Grasp close | **in progress** — executes end-to-end; tuning the final approach |

## Layout

| File | Role |
| --- | --- |
| `servo.py` | **Main entry point.** Servoing, PnP calibration, grasp. |
| `camera.py` | OAK-D: undistorted RGB + depth aligned to it (DepthAI v3 `ImageAlign`). |
| `vlm.py` | Gemini grounding — image + description → pixel. |
| `kin.py` | FK / IK, clearance checks. |
| `arm_io.py` | Damiao CAN motor I/O, joint limits, always-disable safety wrapper. |
| `pick.py` | Grasp planning (side / top approach, IK seeds). |
| `park_home.py` | Recovers an arm left energized after a killed run. |
| `calibrate.py` | Legacy hand-guided calibration + the HSV blob tracker still used. |
| `task.py`, `autocal.py` | The superseded open-loop pipeline, kept for reference. |

## Setup

**See [SETUP.md](SETUP.md) — a fresh clone of the upstream dependencies will not work.**
The Jetson ships no USB-CAN driver, upstream `openarm` requires Python 3.11 while JetPack
ships 3.10, and it opens CAN without FD framing — which makes the motors *silently* stop
replying, with no error at all. `patches/anvil-openarm-py310-canfd.patch` fixes the latter
two.

```bash
pip install -r requirements.txt        # plus openarm_can + openarm from source
cp .env.example .env                   # add your GEMINI_API_KEY
```

Stick yellow tape on the gripper. If your lighting or tape differs, re-sample the HSV
range in `calibrate.py` — it is the single most common cause of "tape lost".

```bash
python servo.py "Black coffee cup" --locate-only    # perception only, no motion
python servo.py "Black coffee cup" --wrist 40       # align the gripper over it
python servo.py "Black coffee cup" --wrist 40 --grasp
python servo.py "Black coffee cup" --regrasp        # reuse saved calibration, skip the wiggle
```

## Safety

**The arm moves on its own.** Clear the workspace and keep a hand on the e-stop. Motion is
slow (0.15 rad/s), every step is clearance/limit/frame-edge/stall gated, and Ctrl+C parks
and disables. If a run is killed with `SIGKILL` the arm stays energized — run
`park_home.py` to bring it down.

The camera must not move after calibration. `servo_calib.json` is only valid for one
camera placement, which is why it is gitignored rather than shipped.
