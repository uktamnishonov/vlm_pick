#!/usr/bin/env python3
"""Bring the RIGHT OpenArm slowly back to the hanging base pose and disable it.

Use this whenever the arm is left energized in mid-air — e.g. a run was killed
(SIGTERM/SIGKILL) so its park-and-disable cleanup never ran and the motors are
still holding their last target.

    python park_home.py
"""
import asyncio

from arm_io import RightArm


async def main():
    arm = RightArm(velocity=0.12)          # slow, deliberate
    await arm.connect()
    print("connected. Re-asserting control and lowering to base pose in 2s — "
          "keep the workspace clear, hand on the e-stop.")
    await asyncio.sleep(2)
    await arm.enable()                     # re-take control of the held motors
    await arm.park_and_disable(velocity=0.12)   # move to [0]*8, then torque off
    arm.shutdown()
    print("arm at base pose, torque disabled.")


if __name__ == "__main__":
    asyncio.run(main())
