"""
generate_test_images.py
FullWorks C-UASC - Vision Team

Makes a set of synthetic "aerial" test images that look like the 2027 C-UASC targets:
  - Drop target:      red/white bullseye (~5 m across)
  - Recovery target:  green X on a white square with a thin black outline
It also adds things that should NOT be detected (red car, green bushes, GCP flags)
so we can see if the detector gets fooled.

The true target centers are saved to test_images/ground_truth.json so the detector's
center estimate can be checked in pixels.

It also saves printable targets (printable_bullseye.png, printable_green_x.png)
for the live webcam demo.

Run:  python generate_test_images.py
"""

import json
import os
import random

import cv2
import numpy as np

W, H = 1280, 720
OUT_DIR = "test_images"
RNG_SEED = 7

# Colors in BGR (OpenCV order)
RED = (40, 30, 210)
WHITE = (235, 235, 235)
GREEN = (60, 175, 40)
BLACK = (20, 20, 20)


# ---------------------------------------------------------------- backgrounds
def background(kind, rng):
    """Ground texture: desert (Mojave), grass, or asphalt."""
    base = {
        "desert": (120, 160, 190),
        "grass": (55, 110, 75),
        "asphalt": (85, 85, 88),
    }[kind]
    img = np.full((H, W, 3), base, np.float32)
    # large blotches + fine grain
    low = cv2.resize(rng.normal(0, 1, (H // 40, W // 40)).astype(np.float32), (W, H),
                     interpolation=cv2.INTER_CUBIC)
    img += low[..., None] * 16                       # brightness patches (same in all channels)
    img += rng.normal(0, 3, (H, W, 3))               # a little color grain
    img += rng.normal(0, 7, (H, W, 1))               # brightness grain
    if kind == "desert":  # small rocks / scrub
        for _ in range(250):
            c = (int(rng.integers(0, W)), int(rng.integers(0, H)))
            shade = int(rng.integers(60, 110))
            cv2.circle(img, c, int(rng.integers(1, 5)), (shade, shade + 10, shade + 20), -1)
    if kind == "asphalt":  # painted line
        y = int(rng.integers(100, H - 100))
        cv2.line(img, (0, y), (W, y + int(rng.integers(-80, 80))), (200, 220, 230), 6)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- targets (drawn flat, then warped)
def draw_bullseye(size):
    """Square canvas with a red/white bullseye. Returns (bgr, alpha)."""
    canvas = np.zeros((size, size, 3), np.uint8)
    alpha = np.zeros((size, size), np.uint8)
    c = size // 2
    r = size // 2 - 1
    cv2.circle(alpha, (c, c), r, 255, -1)
    n_rings = 7  # red, white, red, white, red, white, red center
    for i in range(n_rings):
        rr = int(r * (n_rings - i) / n_rings)
        cv2.circle(canvas, (c, c), rr, RED if i % 2 == 0 else WHITE, -1, cv2.LINE_AA)
    return canvas, alpha


def draw_green_x(size):
    """White square, thin black outline, big green X."""
    canvas = np.full((size, size, 3), WHITE, np.uint8)
    alpha = np.full((size, size), 255, np.uint8)
    t = max(2, size // 60)
    cv2.rectangle(canvas, (0, 0), (size - 1, size - 1), BLACK, t)
    m = int(size * 0.12)
    w = max(3, size // 9)
    cv2.line(canvas, (m, m), (size - m, size - m), GREEN, w, cv2.LINE_AA)
    cv2.line(canvas, (size - m, m), (m, size - m), GREEN, w, cv2.LINE_AA)
    return canvas, alpha


def paste_warped(img, patch, alpha, center, angle, tilt, rng):
    """Rotate + apply a perspective 'camera tilt', then paste. Returns true center (x, y)."""
    s = patch.shape[0]
    src = np.float32([[0, 0], [s, 0], [s, s], [0, s]])
    # rotation about the patch center
    rot = cv2.getRotationMatrix2D((s / 2, s / 2), angle, 1.0)
    pts = np.hstack([src, np.ones((4, 1), np.float32)]) @ rot.T
    # perspective squash (top edge narrower => camera not pointing straight down)
    k = tilt * s
    pts[0, 0] += k * 0.5
    pts[1, 0] -= k * 0.5
    pts[0, 1] += k * 0.3
    pts[1, 1] += k * 0.3
    pts += np.float32(center) - np.float32([s / 2, s / 2])
    M = cv2.getPerspectiveTransform(src, pts.astype(np.float32))
    warped = cv2.warpPerspective(patch, M, (W, H))
    a = cv2.warpPerspective(alpha, M, (W, H)).astype(np.float32)[..., None] / 255.0
    img[:] = (img * (1 - a) + warped * a).astype(np.uint8)
    true_c = cv2.perspectiveTransform(np.float32([[[s / 2, s / 2]]]), M)[0, 0]
    corners = pts
    x0, y0 = corners.min(0)
    x1, y1 = corners.max(0)
    return [float(true_c[0]), float(true_c[1])], [float(x0), float(y0), float(x1), float(y1)]


# ---------------------------------------------------------------- distractors
def add_distractors(img, rng, which):
    if "red_car" in which:  # solid red rectangle (car / tarp / shirt)
        x, y = int(rng.integers(50, W - 200)), int(rng.integers(50, H - 120))
        cv2.rectangle(img, (x, y), (x + 120, y + 60), (35, 35, 190), -1)
        cv2.rectangle(img, (x + 25, y + 10), (x + 95, y + 50), (60, 50, 60), -1)  # windshield
    if "bushes" in which:
        for _ in range(int(rng.integers(4, 9))):
            c = (int(rng.integers(0, W)), int(rng.integers(0, H)))
            cv2.circle(img, c, int(rng.integers(15, 45)), (50, 140, 45), -1)
    if "gcp" in which:  # ground control point flag: black X on white
        x, y = int(rng.integers(50, W - 120)), int(rng.integers(50, H - 120))
        cv2.rectangle(img, (x, y), (x + 70, y + 70), WHITE, -1)
        cv2.line(img, (x + 8, y + 8), (x + 62, y + 62), BLACK, 8)
        cv2.line(img, (x + 62, y + 8), (x + 8, y + 62), BLACK, 8)
    if "red_circle" in which:  # solid red disc, e.g. a cooler lid - round but NOT a bullseye
        c = (int(rng.integers(100, W - 100)), int(rng.integers(100, H - 100)))
        cv2.circle(img, c, int(rng.integers(30, 60)), (30, 30, 200), -1)


# ---------------------------------------------------------------- camera effects
def camera_effects(img, rng, brightness=1.0, blur=0, noise=0, shadow=False, jpeg=90):
    out = img.astype(np.float32) * brightness
    if shadow:  # big soft shadow (drone/cloud)
        mask = np.zeros((H, W), np.float32)
        cv2.ellipse(mask, (int(rng.integers(300, 900)), int(rng.integers(200, 500))),
                    (350, 200), float(rng.integers(0, 180)), 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 60)[..., None]
        out *= 1 - 0.45 * mask
    if noise:
        out += rng.normal(0, noise, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if blur:  # motion blur
        k = np.zeros((blur, blur), np.float32)
        k[blur // 2, :] = 1.0 / blur
        out = cv2.filter2D(out, -1, k)
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------- scenes
# Each scene: what targets, how big (pixels ~ altitude), conditions.
SCENES = [
    # name,                   bg,        targets,              size, tilt, effects,                                  distractors
    ("01_bullseye_desert_low",  "desert",  ["bullseye"],          420, 0.00, {},                                       []),
    ("02_bullseye_desert_high", "desert",  ["bullseye"],          110, 0.00, {},                                       []),
    ("03_bullseye_grass",       "grass",   ["bullseye"],          260, 0.10, {},                                       ["bushes"]),
    ("04_bullseye_tilted",      "asphalt", ["bullseye"],          300, 0.35, {},                                       []),
    ("05_bullseye_red_car",     "desert",  ["bullseye"],          200, 0.05, {},                                       ["red_car"]),
    ("06_bullseye_red_disc",    "asphalt", ["bullseye"],          180, 0.00, {},                                       ["red_circle"]),
    ("07_bullseye_dim",         "desert",  ["bullseye"],          240, 0.10, {"brightness": 0.55},                     []),
    ("08_bullseye_bright",      "desert",  ["bullseye"],          240, 0.00, {"brightness": 1.35},                     []),
    ("09_bullseye_shadow",      "grass",   ["bullseye"],          280, 0.05, {"shadow": True},                         ["bushes"]),
    ("10_bullseye_motion_blur", "desert",  ["bullseye"],          240, 0.05, {"blur": 15},                             []),
    ("11_bullseye_noisy",       "asphalt", ["bullseye"],          160, 0.05, {"noise": 18, "jpeg": 55},                []),
    ("12_bullseye_edge",        "desert",  ["bullseye_edge"],     260, 0.00, {},                                       []),
    ("13_greenx_desert",        "desert",  ["green_x"],           300, 0.05, {},                                       ["gcp"]),
    ("14_greenx_grass",         "grass",   ["green_x"],           240, 0.10, {},                                       ["bushes"]),
    ("15_greenx_small",         "asphalt", ["green_x"],           90,  0.00, {},                                       ["gcp"]),
    ("16_both_targets",         "desert",  ["bullseye", "green_x"], 220, 0.10, {},                                     ["gcp", "red_car"]),
    ("17_no_target_desert",     "desert",  [],                    0,   0.00, {},                                       ["red_car", "gcp"]),
    ("18_no_target_grass",      "grass",   [],                    0,   0.00, {},                                       ["bushes", "red_circle"]),
]


def make_printables():
    b, _ = draw_bullseye(1000)
    pb = np.full((1100, 1100, 3), 255, np.uint8)
    pb[50:1050, 50:1050][b.sum(2) > 0] = b[b.sum(2) > 0]
    cv2.imwrite("printable_bullseye.png", pb)
    g, _ = draw_green_x(1000)
    cv2.imwrite("printable_green_x.png", g)


def main():
    rng = np.random.default_rng(RNG_SEED)
    random.seed(RNG_SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    truth = {}
    for name, bg, targets, size, tilt, fx, distract in SCENES:
        img = background(bg, rng)
        add_distractors(img, rng, distract)
        objs = []
        used = []
        for t in targets:
            s = size if t != "green_x" or len(targets) == 1 else int(size * 0.9)
            if t == "bullseye_edge":  # partly out of frame
                center = (W - s // 3, H // 2)
            else:
                for _ in range(100):
                    center = (int(rng.integers(s // 2 + 20, W - s // 2 - 20)),
                              int(rng.integers(s // 2 + 20, H - s // 2 - 20)))
                    if all(np.hypot(center[0] - u[0], center[1] - u[1]) > s for u in used):
                        break
            used.append(center)
            if t.startswith("bullseye"):
                patch, alpha = draw_bullseye(s)
                label = "bullseye"
            else:
                patch, alpha = draw_green_x(s)
                label = "green_x"
            angle = float(rng.uniform(0, 360))
            c, box = paste_warped(img, patch, alpha, center, angle, tilt, rng)
            objs.append({"label": label, "center": c, "bbox": box,
                         "partly_out_of_frame": t == "bullseye_edge"})
        img = camera_effects(img, rng, **fx)
        path = os.path.join(OUT_DIR, name + ".jpg")
        cv2.imwrite(path, img)
        truth[name + ".jpg"] = {"background": bg, "conditions": fx, "distractors": distract,
                                "targets": objs}
        print("wrote", path)
    with open(os.path.join(OUT_DIR, "ground_truth.json"), "w") as f:
        json.dump(truth, f, indent=2)
    make_printables()
    print("wrote ground_truth.json and printable targets")


if __name__ == "__main__":
    main()
