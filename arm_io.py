#!/usr/bin/env python3
"""RIGHT OpenArm (can1) command wrapper with safety rails.

Built on anvil_openarm's damiao package (same stack as demo_motions.py):
POS_VEL control so every move is velocity-capped by the motor firmware.
Joint targets are clamped to MOTOR_CONFIGS right-arm limits. Motors are
ALWAYS disabled on exit — including Ctrl+C and exceptions.

Angles at this layer are MOTOR angles in radians (7 joints + gripper).
kin.motor_to_urdf()/signs handle the URDF convention elsewhere.
"""
import asyncio
import math

import can as pycan

from openarm.bus import Bus
from openarm.damiao import Arm, ControlMode, Motor
from openarm.damiao.config import MOTOR_CONFIGS
from openarm.damiao.detect import detect_motors

GRIPPER_OPEN_DEG = -42.0
GRIPPER_CLOSED_DEG = 0.0

# (lo, hi) radians per motor, right-arm convention, straight from config.
LIMITS_RIGHT = [
    (math.radians(c.min_angle_right), math.radians(c.max_angle_right))
    for c in MOTOR_CONFIGS
]


def clamp8(q8):
    return [max(lo, min(hi, q)) for q, (lo, hi) in zip(q8, LIMITS_RIGHT)]


class RightArm:
    def __init__(self, channel="can1", velocity=0.35):
        self.channel = channel
        self.velocity = velocity  # rad/s cap for normal moves
        self.bus = None
        self.arm = None
        self._enabled = False

    async def connect(self, attempts=4):
        self.bus = pycan.Bus(channel=self.channel, interface="socketcan", fd=True)
        # A motor can answer a beat late on the very first transaction (e.g. the
        # gripper right after it was disabled by another tool). Retry the scan
        # before giving up — a single slow reply must not abort a calibration.
        missing = None
        for i in range(attempts):
            detected = {m.slave_id for m in detect_motors(self.bus, range(1, 9), timeout=0.5)}
            missing = [c.name for c in MOTOR_CONFIGS if c.slave_id not in detected]
            if not missing:
                break
            await asyncio.sleep(0.3)
        if missing:
            self.bus.shutdown()
            self.bus = None
            raise RuntimeError(f"{self.channel}: motors not responding after "
                               f"{attempts} scans: {missing}")
        motors = [
            Motor(Bus(self.bus), slave_id=c.slave_id, master_id=c.master_id,
                  motor_type=c.type)
            for c in MOTOR_CONFIGS
        ]
        self.arm = Arm(motors)
        return self

    async def read_pose(self):
        """Motor angles (rad, 8 values) WITHOUT energizing: the Damiao
        disable frame echoes state and is a no-op on a disabled motor."""
        states = await self.arm.disable()
        self._enabled = False
        return [s.position for s in states]

    async def enable(self):
        await self.arm.enable()
        await self.arm.set_control_mode(ControlMode.POS_VEL)
        self._enabled = True

    async def move(self, q8_rad, velocity=None, wait=True, settle=0.5):
        """Clamped, velocity-capped move; optionally block until arrival."""
        if not self._enabled:
            raise RuntimeError("arm not enabled")
        v = velocity or self.velocity
        target = clamp8(list(q8_rad))
        states = await self.arm.control_pos_vel(position=target, velocity=v)
        if wait:
            here = [s.position for s in states]
            dt = max(abs(a - b) for a, b in zip(here, target)) / v + settle
            await asyncio.sleep(dt)
        return target

    async def read_actual(self, hold_q8):
        """Actual motor positions (rad, 8) while ENABLED. Re-asserts the given
        hold target and returns the feedback the motors echo — their measured
        position, which after settling is where the arm truly is. Used by
        autocal to log the real joint angles for FK (never the raw command)."""
        if not self._enabled:
            raise RuntimeError("arm not enabled")
        states = await self.arm.control_pos_vel(
            position=clamp8(list(hold_q8)), velocity=self.velocity)
        return [s.position for s in states]

    async def gripper(self, open_, velocity=1.0):
        """Open/close only the gripper (motor 8)."""
        target = math.radians(GRIPPER_OPEN_DEG if open_ else GRIPPER_CLOSED_DEG)
        from openarm.damiao.encoding import PosVelControlParams

        await self.arm.motors[7].control_pos_vel(
            PosVelControlParams(position=target, velocity=velocity)
        )
        await asyncio.sleep(1.2)

    async def park_and_disable(self, park_pose=None, velocity=0.3):
        """Slow return to hanging rest, then disable. Never raises."""
        try:
            if self._enabled:
                rest = park_pose if park_pose is not None else [0.0] * 8
                pose = await self.arm.control_pos_vel(
                    position=clamp8(rest), velocity=velocity
                )
                dt = max(abs(s.position - r) for s, r in zip(pose, rest)) / velocity
                await asyncio.sleep(dt + 1.0)
        except BaseException as e:  # noqa: BLE001 — disable must still run
            print(f"park interrupted ({type(e).__name__}); disabling now")
        finally:
            try:
                if self.arm is not None:
                    await self.arm.disable()
            except Exception as e:  # noqa: BLE001
                print(f"disable failed: {e}")
            self._enabled = False

    def shutdown(self):
        if self.bus is not None:
            self.bus.shutdown()
            self.bus = None
