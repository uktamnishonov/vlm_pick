#!/usr/bin/env python3
"""OAK-D Pro W capture: undistorted RGB + stereo depth aligned to RGB.

Pipeline (DepthAI v3): mono pair -> StereoDepth -> ImageAlign (to the
undistorted RGB output) -> Sync. depth[v, u] is then the Z-depth (mm) of
RGB pixel (u, v), and because the RGB stream is undistorted on-device,
point3d() is a plain pinhole deprojection with the intrinsics the device
attaches to every frame.

Self-test:  python camera.py
"""
import time

import cv2
import depthai as dai
import numpy as np

RGB_SIZE = (640, 480)
MONO_SIZE = (640, 400)


class OakCamera:
    def __init__(self, fps=15, warmup_s=6.0):
        self._device = dai.Device()
        self._pipeline = dai.Pipeline(self._device)

        color = self._pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        mono_l = self._pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        mono_r = self._pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        stereo = self._pipeline.create(dai.node.StereoDepth)
        # The OLD config was the bare DEFAULT preset + extendedDisparity, and
        # extended disparity DISABLES the on-device median filter (the "1520
        # exceeds 1024" spam) -> raw noisy disparity. Fix ONLY the safe,
        # parameter-free wins here (they can't gut coverage):
        #  - drop extended disparity  -> median filter re-enabled; min depth
        #    ~0.35 m is still well inside our 0.4-1.2 m workspace
        #  - median 7x7               -> smooths the disparity
        #  - LR-check                 -> rejects left/right-inconsistent matches
        #  - subpixel                 -> finer depth steps at range
        # (Heavier post-processing — temporal/spatial — needs tuned alpha/delta
        # or it zeros the map; added separately once base coverage is confirmed.)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        stereo.initialConfig.setMedianFilter(
            dai.StereoDepthConfig.MedianFilter.KERNEL_7x7)
        mono_l.requestOutput(MONO_SIZE, fps=fps).link(stereo.left)
        mono_r.requestOutput(MONO_SIZE, fps=fps).link(stereo.right)

        rgb_out = color.requestOutput(RGB_SIZE, fps=fps, enableUndistortion=True)

        align = self._pipeline.create(dai.node.ImageAlign)
        stereo.depth.link(align.input)
        rgb_out.link(align.inputAlignTo)

        sync = self._pipeline.create(dai.node.Sync)
        rgb_out.link(sync.inputs["rgb"])
        align.outputAligned.link(sync.inputs["depth"])
        self._queue = sync.out.createOutputQueue(maxSize=2, blocking=False)

        self._pipeline.start()

        # Active stereo: the IR dot projector paints texture for matching.
        # (Flood light is intentionally OFF — it washes out the dot pattern and
        # HURTS coverage; it only helps plain low-light RGB, not stereo.)
        # Kept at 0.8, not max: full power draws more current and the OAK was
        # crashing on this Jetson's USB — 0.8 was stable across earlier runs.
        try:
            self._device.setIrLaserDotProjectorIntensity(0.8)
        except Exception:
            pass

        # Warm up until depth flows AND auto-exposure has settled (>=3 s).
        self.K = None
        t0 = time.time()
        cov = 0.0
        while time.time() - t0 < warmup_s:
            got = self._queue.tryGet()
            if got is not None:
                cov = (got["depth"].getFrame() > 0).mean()
                if self.K is None:
                    self._grab_intrinsics(got["rgb"])
                if cov > 0.2 and time.time() - t0 >= 3.0:
                    break
            time.sleep(0.02)
        self.warmup_coverage = cov

    def _grab_intrinsics(self, rgb_msg):
        try:
            self.K = np.array(
                rgb_msg.getTransformation().getIntrinsicMatrix(), dtype=np.float64
            )
        except Exception:
            self.K = np.array(
                self._device.readCalibration().getCameraIntrinsics(
                    dai.CameraBoardSocket.CAM_A, *RGB_SIZE
                ),
                dtype=np.float64,
            )

    def read(self):
        """Return (bgr uint8 HxWx3, depth_mm uint16 HxW), synced pair."""
        grp = None
        while grp is None:
            grp = self._queue.tryGet()
            if grp is None:
                time.sleep(0.005)
        if self.K is None:
            self._grab_intrinsics(grp["rgb"])
        return grp["rgb"].getCvFrame(), grp["depth"].getFrame()

    def depth_at(self, depth_mm, u, v, patch=11, reduce="median"):
        """Valid depth (meters) in a patch around (u, v), or None.

        reduce='median' — a surface's depth (default).
        reduce='fg'     — foreground: a low percentile, so a small marker in
                          FRONT of a bad-depth background (yellow tape over a
                          dark cup) reports its OWN depth instead of the
                          polluted background median.
        """
        h, w = depth_mm.shape
        r = patch // 2
        window = depth_mm[max(0, v - r):min(h, v + r + 1),
                          max(0, u - r):min(w, u + r + 1)]
        valid = window[window > 0]
        if valid.size < 5:
            return None
        val = np.percentile(valid, 20) if reduce == "fg" else np.median(valid)
        return float(val) / 1000.0

    def point3d(self, u, v, depth_mm=None, patch=11):
        """Deproject undistorted RGB pixel (u, v) to camera-frame XYZ meters."""
        if depth_mm is None:
            _, depth_mm = self.read()
        z = self.depth_at(depth_mm, u, v, patch)
        if z is None:
            return None
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])

    def close(self):
        try:
            self._pipeline.stop()
        except Exception:
            pass
        self._device.close()


if __name__ == "__main__":
    cam = OakCamera()
    print("warmup depth coverage: %.0f%%" % (cam.warmup_coverage * 100))
    print("K:\n", np.round(cam.K, 1))
    t0 = time.time()
    for _ in range(10):
        bgr, depth = cam.read()
    print(f"10 synced frames in {time.time()-t0:.2f}s")
    h, w = depth.shape
    cx, cy = w // 2, h // 2
    p = cam.point3d(cx, cy, depth)
    print(f"depth coverage: {(depth > 0).mean()*100:.0f}%")
    print(f"center pixel ({cx},{cy}) -> {p if p is None else np.round(p, 3)} m")
    vis = cv2.applyColorMap(
        cv2.convertScaleAbs(np.clip(depth, 0, 4000), alpha=255.0 / 4000), cv2.COLORMAP_JET
    )
    vis[depth == 0] = 0
    cv2.imwrite("probe_rgb.jpg", bgr)
    cv2.imwrite("probe_depth.jpg", vis)
    print("saved probe_rgb.jpg / probe_depth.jpg")
    cam.close()
