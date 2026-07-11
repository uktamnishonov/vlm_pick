#!/usr/bin/env python3
"""Gemini grounding client: image + object description -> pixel coordinates.

Uses Gemini's native pointing: it returns [y, x] normalized to 0-1000,
which we scale back to pixels. Requires GEMINI_API_KEY in the environment.
Model defaults to gemini-2.5-flash, override with GEMINI_MODEL.

Self-test:  python vlm.py "the red cup" [image.jpg]
"""
import base64
import json
import os
import re
import sys
import time

import cv2
import requests

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
# tried in order when a model is overloaded/rate-limited (429/5xx)
FALLBACK_MODELS = [
    DEFAULT_MODEL,
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
]

POINT_PROMPT = (
    'Point to the {obj} in the image. Answer with JSON only: '
    '{{"found": true/false, "point": [y, x], "label": "<what you found>"}} '
    "where point coordinates are normalized to 0-1000 ([0,0] is top-left). "
    'If the object is not visible, return {{"found": false}}.'
)


class GeminiError(RuntimeError):
    pass


def _loads(text):
    """Parse JSON from a model reply, salvaging the first complete object/array
    if the model wrapped it in prose or emitted trailing junk (responseMimeType
    usually prevents this, but not always). Raises GeminiError if nothing
    parseable is found — callers treat that as 'no result', never a crash."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if starts:
        start = min(starts)
        openc = text[start]
        closec = "}" if openc == "{" else "]"
        depth = 0
        in_str = esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == openc:
                depth += 1
            elif c == closec:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    raise GeminiError(f"model returned unparseable JSON: {text[:200]!r}")


def _api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(env_path):
            for line in open(env_path):
                if line.strip().startswith("GEMINI_API_KEY"):
                    key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        raise GeminiError("GEMINI_API_KEY not set (env var or vlm_pick/.env)")
    return key


def _generate_once(parts, model, timeout=30):
    url = f"{API_ROOT}/models/{model}:generateContent"
    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.0,
        },
    }
    r = requests.post(
        url, params={"key": _api_key()}, json=body, timeout=timeout
    )
    if r.status_code != 200:
        raise GeminiError(f"HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise GeminiError(f"unexpected response: {json.dumps(data)[:300]}")


def _generate(parts, model=None, timeout=30):
    """Try the requested model, then fall back through FALLBACK_MODELS on
    overload/rate-limit (429/5xx) so one busy model doesn't stop the robot."""
    chain = [model] if model else []
    chain += [m for m in FALLBACK_MODELS if m not in chain]
    last = None
    for i, m in enumerate(chain):
        try:
            text = _generate_once(parts, m, timeout)
            if i > 0:
                print(f"  (vlm: {chain[0]} unavailable, used {m})", flush=True)
            return text
        except (GeminiError, requests.RequestException) as e:
            msg = str(e)
            last = e
            transient = any(c in msg for c in
                            ("HTTP 429", "HTTP 500", "HTTP 503", "HTTP 502",
                             "UNAVAILABLE", "overloaded", "timed out", "Timeout"))
            if not transient:
                raise
            time.sleep(1.0)
    raise GeminiError(f"all models unavailable, last error: {last}")


def _jpeg_part(bgr):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise GeminiError("jpeg encode failed")
    return {
        "inlineData": {
            "mimeType": "image/jpeg",
            "data": base64.b64encode(buf.tobytes()).decode(),
        }
    }


def locate(bgr, description, model=DEFAULT_MODEL):
    """Find `description` in a BGR image. Returns (u, v, label) or None."""
    text = _generate(
        [_jpeg_part(bgr), {"text": POINT_PROMPT.format(obj=description)}], model
    )
    try:
        ans = _loads(text)
    except GeminiError:
        # last-ditch: pull the point coords straight out of malformed JSON
        # (Gemini occasionally emits a stray quote / trailing junk)
        m = re.search(r'"point"\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)', text)
        if not m:
            raise
        ans = {"found": True, "point": [float(m.group(1)), float(m.group(2))]}
    if isinstance(ans, list):
        ans = ans[0] if ans else {}
    if not ans.get("found") or "point" not in ans:
        return None
    y, x = ans["point"][:2]
    h, w = bgr.shape[:2]
    u = int(round(x / 1000.0 * w))
    v = int(round(y / 1000.0 * h))
    u = min(max(u, 0), w - 1)
    v = min(max(v, 0), h - 1)
    return u, v, ans.get("label", description)


TASK_PROMPT = (
    'You are the perception module of a robot arm with a two-finger gripper. '
    'Task given by the operator: "{task}". Look at the image and identify: '
    "(1) the single object the arm must pick up, and (2) the destination "
    "spot where it must be placed, if the task specifies one. "
    'Return JSON only: {{"pick": {{"point": [y, x], "label": "<object>"}}, '
    '"place": {{"point": [y, x], "label": "<destination>"}}}} — '
    "set \"place\" to null if the task has no destination. Coordinates are "
    "normalized to 0-1000, [0,0] is top-left. Point at the CENTER of the "
    "object and the CENTER of the destination spot."
)


def parse_task(bgr, task, model=DEFAULT_MODEL):
    """One call: task string + image -> pick target and optional place target.

    Returns {"pick": (u, v, label), "place": (u, v, label) | None} or None.
    """
    text = _generate(
        [_jpeg_part(bgr), {"text": TASK_PROMPT.format(task=task)}], model
    )
    ans = _loads(text)
    if isinstance(ans, list):
        ans = ans[0] if ans else {}
    h, w = bgr.shape[:2]

    def to_px(node):
        if not node or "point" not in node:
            return None
        y, x = node["point"][:2]
        u = min(max(int(round(x / 1000.0 * w)), 0), w - 1)
        v = min(max(int(round(y / 1000.0 * h)), 0), h - 1)
        return u, v, node.get("label", "?")

    pick = to_px(ans.get("pick"))
    if pick is None:
        return None
    return {"pick": pick, "place": to_px(ans.get("place"))}


def ask(bgr, question, model=DEFAULT_MODEL):
    """Free-form visual question (used for success verification)."""
    text = _generate(
        [_jpeg_part(bgr), {"text": question + ' Answer with JSON: {"answer": "..."}'}],
        model,
    )
    return _loads(text).get("answer", "")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    desc = sys.argv[1]
    if len(sys.argv) > 2:
        img = cv2.imread(sys.argv[2])
        if img is None:
            sys.exit(f"cannot read {sys.argv[2]}")
    else:
        from camera import OakCamera

        cam = OakCamera()
        img, _ = cam.read()
        cam.close()
    hit = locate(img, desc)
    print("model:", DEFAULT_MODEL)
    if hit is None:
        print(f"'{desc}' not found")
    else:
        u, v, label = hit
        print(f"'{label}' at pixel ({u}, {v})")
        cv2.drawMarker(img, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
        cv2.circle(img, (u, v), 12, (0, 255, 0), 2)
        cv2.imwrite("vlm_hit.jpg", img)
        print("saved vlm_hit.jpg")
