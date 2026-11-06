"""
detect_targets.py
FullWorks C-UASC - Vision Team - Part A: Static target detection

Pipeline (for each image):
  1. Load the image
  2. Convert BGR -> HSV and make color masks (red for the bullseye, green for the X)
  3. Clean the mask (blur + morphology) and find contours (connected shapes)
  4. Check each shape:
       Bullseye  -> round (fits an ellipse well) AND has rings (only part of it is red)
       Green X   -> skinny "X" shape (low solidity) AND sits on a white square
  5. Draw a bounding box and the estimated center in image (pixel) coordinates
  6. Compare across all test images against ground_truth.json (center error in px)

Run on the test set:     python detect_targets.py
Run on one image:        python detect_targets.py --image path/to/photo.jpg
Show the color masks:    python detect_targets.py --save-masks

Image coordinates: origin (0,0) is the TOP-LEFT pixel, x goes right, y goes down.
We also report the offset from the image center (dx, dy), which is what the drone
would use later to steer over the target.
"""

import argparse
import csv
import glob
import json
import os
import time

import cv2
import numpy as np

# ------------------------------------------------------------------ tunable settings
# HSV ranges in OpenCV: H 0-179, S 0-255, V 0-255. Red wraps around 0, so it needs two ranges.
RED_LOW_1, RED_HIGH_1 = (0, 90, 60), (10, 255, 255)
RED_LOW_2, RED_HIGH_2 = (165, 90, 60), (179, 255, 255)
GREEN_LOW, GREEN_HIGH = (40, 90, 60), (85, 255, 255)
WHITE_LOW, WHITE_HIGH = (0, 0, 150), (179, 60, 255)

MIN_AREA_FRAC = 0.0003      # ignore shapes smaller than this fraction of the image (noise)
ELLIPSE_FIT_MIN = 0.80      # contour area / fitted-ellipse area (1.0 = perfect ellipse)
RING_RED_FRAC = (0.35, 0.85)  # a bullseye is only partly red; a solid red disc is ~1.0
MIN_RING_CROSSINGS = 4      # red/white edges crossed going from center to edge (bullseye has 6)
X_SOLIDITY = (0.20, 0.65)   # area / convex hull area: an X is "skinny", a bush is ~0.9+
X_WHITE_AROUND_MIN = 0.45   # fraction of the X's bounding box that must be white


# ------------------------------------------------------------------ helpers
def color_masks(bgr):
    """Return dict of binary masks (255 = pixel matches the color)."""
    blur = cv2.GaussianBlur(bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, RED_LOW_1, RED_HIGH_1) | cv2.inRange(hsv, RED_LOW_2, RED_HIGH_2)
    green = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
    white = cv2.inRange(hsv, WHITE_LOW, WHITE_HIGH)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, k)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, k)
    return {"red": red, "green": green, "white": white}


def fill_holes(mask):
    """Fill the inside of every outer contour, so the white rings of a bullseye become solid."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, cnts, -1, 255, -1)
    return out


# ------------------------------------------------------------------ bullseye
def find_bullseyes(bgr, masks):
    h, w = bgr.shape[:2]
    min_area = MIN_AREA_FRAC * h * w
    red = masks["red"]
    # close small gaps so the outer red ring is one connected loop, then fill it in
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(red, cv2.MORPH_CLOSE, k)
    solid = fill_holes(closed)
    cnts, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    found = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or len(c) < 5:
            continue
        (ex, ey), (ma, mi), ang = cv2.fitEllipse(c)
        ell_area = np.pi * ma * mi / 4
        fit = area / ell_area if ell_area > 0 else 0
        # Is it round? (handles camera tilt because tilted circles look like ellipses)
        if not (ELLIPSE_FIT_MIN <= fit <= 1.15):
            continue
        # Does it have rings? Look at how much of the filled shape is actually red.
        region = np.zeros_like(red)
        cv2.drawContours(region, [c], -1, 255, -1)
        red_frac = cv2.countNonZero(red & region) / max(1, cv2.countNonZero(region))
        if not (RING_RED_FRAC[0] <= red_frac <= RING_RED_FRAC[1]):
            continue
        # Bullseye signature: walking outward from the middle we should cross several
        # red <-> not-red ring edges. A red car / solid red disc gives 0-2 crossings.
        rings = count_ring_crossings(red, (ex, ey), (ma, mi), ang)
        if rings < MIN_RING_CROSSINGS:
            continue
        # Center: the small red dot in the middle is the true center, even when the camera
        # is tilted (a tilted circle's ellipse center is slightly off from the real center).
        cx, cy, center_method = find_center_dot(red, c, (ex, ey), area)
        x, y, bw, bh = cv2.boundingRect(c)
        score = (max(0.0, 1 - abs(1 - fit) * 3)
                 * max(0.0, 1 - abs(red_frac - 0.57) * 2)
                 * min(1.0, rings / 6))
        found.append({
            "label": "bullseye",
            "bbox": [x, y, bw, bh],
            "center": [float(cx), float(cy)],
            "confidence": round(float(score), 3),
            "debug": {"ellipse_fit": round(fit, 3), "red_fraction": round(red_frac, 3),
                      "ring_crossings": rings, "center_method": center_method},
        })
    return found


def count_ring_crossings(red, center, axes, angle, n_rays=16):
    """Cast rays from the center to the edge of the ellipse; return median # of color changes."""
    h, w = red.shape
    ex, ey = center
    a, b = axes[0] / 2, axes[1] / 2
    t = np.deg2rad(angle)
    counts = []
    for k in range(n_rays):
        phi = 2 * np.pi * k / n_rays
        # point on the ellipse edge in direction phi
        px = ex + a * np.cos(phi) * np.cos(t) - b * np.sin(phi) * np.sin(t)
        py = ey + a * np.cos(phi) * np.sin(t) + b * np.sin(phi) * np.cos(t)
        n = int(max(20, np.hypot(px - ex, py - ey)))
        xs = np.clip(np.linspace(ex, px, n).astype(int), 0, w - 1)
        ys = np.clip(np.linspace(ey, py, n).astype(int), 0, h - 1)
        if not (0 <= px < w and 0 <= py < h):
            continue  # ray leaves the image (target partly out of frame)
        vals = red[ys, xs] > 0
        counts.append(int(np.count_nonzero(vals[1:] != vals[:-1])))
    return int(np.median(counts)) if counts else 0


def find_center_dot(red, outer_contour, ellipse_center, outer_area):
    """Find the small solid red dot at the middle of the bullseye. Fallback: ellipse center."""
    region = np.zeros_like(red)
    cv2.drawContours(region, [outer_contour], -1, 255, -1)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(red & region)
    best, best_d = None, 1e9
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if not (0.002 * outer_area < a < 0.08 * outer_area):
            continue
        d = np.hypot(cents[i][0] - ellipse_center[0], cents[i][1] - ellipse_center[1])
        if d < best_d:
            best, best_d = i, d
    if best is not None and best_d < 0.25 * np.sqrt(outer_area):
        return float(cents[best][0]), float(cents[best][1]), "center_dot"
    return float(ellipse_center[0]), float(ellipse_center[1]), "ellipse"


# ------------------------------------------------------------------ green X
def find_green_x(bgr, masks):
    h, w = bgr.shape[:2]
    min_area = MIN_AREA_FRAC * h * w * 0.5
    green_all, white = masks["green"], masks["white"]
    # The recovery X is always printed on a white square. Find white squares first and only
    # look for green INSIDE them - otherwise grass/bushes can merge with the X.
    sq_cnts, _ = cv2.findContours(fill_holes(white), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = np.zeros_like(white)
    for s in sq_cnts:
        if cv2.contourArea(s) < min_area * 2:
            continue
        approx = cv2.approxPolyDP(s, 0.04 * cv2.arcLength(s, True), True)
        if 4 <= len(approx) <= 6:
            cv2.drawContours(squares, [s], -1, 255, -1)
    squares = cv2.erode(squares, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    green = green_all & squares
    cnts, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = area / hull_area if hull_area > 0 else 1
        if not (X_SOLIDITY[0] <= solidity <= X_SOLIDITY[1]):
            continue  # too solid -> bush / grass patch
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / bh if bh else 0
        if not (0.5 <= aspect <= 2.0):
            continue
        # The recovery X sits on a white square: check the non-green part of the box is white
        box_white = white[y:y + bh, x:x + bw]
        box_green = green[y:y + bh, x:x + bw]
        non_green = cv2.countNonZero(255 - box_green)
        white_frac = cv2.countNonZero(box_white & (255 - box_green)) / max(1, non_green)
        if white_frac < X_WHITE_AROUND_MIN:
            continue
        M = cv2.moments(c)
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]  # X is symmetric -> centroid = crossing point
        score = max(0.0, 1 - abs(solidity - 0.4) * 2) * min(1.0, white_frac / 0.8)
        found.append({
            "label": "green_x",
            "bbox": [x, y, bw, bh],
            "center": [float(cx), float(cy)],
            "confidence": round(float(score), 3),
            "debug": {"solidity": round(solidity, 3), "white_around": round(white_frac, 3)},
        })
    return found


# ------------------------------------------------------------------ main detect + draw
def detect(bgr):
    """Return a list of detections. Each has label, bbox [x,y,w,h], center [x,y], confidence."""
    masks = color_masks(bgr)
    dets = find_bullseyes(bgr, masks) + find_green_x(bgr, masks)
    h, w = bgr.shape[:2]
    for d in dets:
        d["offset_from_image_center"] = [round(d["center"][0] - w / 2, 1),
                                         round(d["center"][1] - h / 2, 1)]
    dets.sort(key=lambda d: -d["confidence"])
    return dets, masks


COLORS = {"bullseye": (255, 200, 0), "green_x": (255, 0, 255)}


def draw(bgr, dets, extra_text=None):
    out = bgr.copy()
    h, w = out.shape[:2]
    # image center crosshair (where the camera is pointing)
    cv2.drawMarker(out, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 30, 1)
    for d in dets:
        x, y, bw, bh = d["bbox"]
        cx, cy = d["center"]
        col = COLORS[d["label"]]
        cv2.rectangle(out, (x, y), (x + bw, y + bh), col, 3)
        cv2.drawMarker(out, (int(cx), int(cy)), (0, 255, 255), cv2.MARKER_CROSS, 24, 3)
        cv2.line(out, (w // 2, h // 2), (int(cx), int(cy)), (0, 255, 255), 1, cv2.LINE_AA)
        txt = f"{d['label']} ({cx:.0f},{cy:.0f}) conf {d['confidence']:.2f}"
        ty = y - 10 if y > 30 else y + bh + 25
        cv2.putText(out, txt, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, txt, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    if extra_text:
        cv2.putText(out, extra_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, extra_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ------------------------------------------------------------------ evaluation across images
def match(dets, truths, max_dist_frac=0.5):
    """Match detections to ground-truth targets. Returns (tp list with errors, fp count, fn count)."""
    used = set()
    tp = []
    for t in truths:
        best, best_d = None, 1e9
        x0, y0, x1, y1 = t["bbox"]
        size = max(x1 - x0, y1 - y0)
        for i, d in enumerate(dets):
            if i in used or d["label"] != t["label"]:
                continue
            dist = np.hypot(d["center"][0] - t["center"][0], d["center"][1] - t["center"][1])
            if dist < best_d and dist < max_dist_frac * size:
                best, best_d = i, dist
        if best is not None:
            used.add(best)
            tp.append({"label": t["label"], "err_px": best_d, "size_px": size,
                       "partly_out_of_frame": t.get("partly_out_of_frame", False)})
    fn = len(truths) - len(tp)
    fp = len(dets) - len(used)
    return tp, fp, fn


def run_folder(folder, out_dir, save_masks=False):
    os.makedirs(out_dir, exist_ok=True)
    gt_path = os.path.join(folder, "ground_truth.json")
    gt = json.load(open(gt_path)) if os.path.exists(gt_path) else {}
    files = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))
    rows, thumbs = [], []
    totals = {"tp": 0, "fp": 0, "fn": 0, "errs": []}

    for f in files:
        name = os.path.basename(f)
        img = cv2.imread(f)
        t0 = time.perf_counter()
        dets, masks = detect(img)
        ms = (time.perf_counter() - t0) * 1000

        truths = gt.get(name, {}).get("targets", [])
        tp, fp, fn = match(dets, truths) if name in gt else ([], 0, 0)
        totals["tp"] += len(tp); totals["fp"] += fp; totals["fn"] += fn
        totals["errs"] += [t["err_px"] for t in tp]

        result = "PASS" if (fp == 0 and fn == 0) else "FAIL"
        if name not in gt:
            result = "n/a"
        err_txt = ", ".join(f"{t['err_px']:.1f}" for t in tp) or "-"
        rows.append({
            "image": name,
            "expected": len(truths),
            "detected": len(dets),
            "labels": ";".join(d["label"] for d in dets) or "none",
            "centers_px": ";".join(f"({d['center'][0]:.1f},{d['center'][1]:.1f})" for d in dets) or "-",
            "offset_from_center_px": ";".join(f"({d['offset_from_image_center'][0]},{d['offset_from_image_center'][1]})" for d in dets) or "-",
            "center_error_px": err_txt,
            "false_pos": fp,
            "missed": fn,
            "time_ms": round(ms, 1),
            "result": result,
        })
        vis = draw(img, dets, f"{name}  [{result}]  {ms:.0f} ms")
        cv2.imwrite(os.path.join(out_dir, name.rsplit(".", 1)[0] + "_detected.jpg"), vis)
        if save_masks:
            m = np.hstack([masks["red"], masks["green"], masks["white"]])
            cv2.imwrite(os.path.join(out_dir, name.rsplit(".", 1)[0] + "_masks_red_green_white.png"), m)
        thumbs.append(cv2.resize(vis, (480, 270)))
        print(f"{name:32s} expected {len(truths)} found {len(dets)}  err(px) {err_txt:14s} {ms:6.1f} ms  {result}")

    # results table
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)

    # contact sheet of all annotated images
    while len(thumbs) % 4:
        thumbs.append(np.zeros_like(thumbs[0]))
    sheet = np.vstack([np.hstack(thumbs[i:i + 4]) for i in range(0, len(thumbs), 4)])
    cv2.imwrite(os.path.join(out_dir, "all_results_grid.jpg"), sheet)

    # summary
    if gt:
        n_true = totals["tp"] + totals["fn"]
        prec = totals["tp"] / max(1, totals["tp"] + totals["fp"])
        rec = totals["tp"] / max(1, n_true)
        errs = np.array(totals["errs"]) if totals["errs"] else np.array([0.0])
        passed = sum(r["result"] == "PASS" for r in rows)
        summary = {
            "images": len(rows),
            "images_passed": passed,
            "targets_in_ground_truth": n_true,
            "targets_found": totals["tp"],
            "missed": totals["fn"],
            "false_positives": totals["fp"],
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "center_error_px_mean": round(float(errs.mean()), 2),
            "center_error_px_max": round(float(errs.max()), 2),
            "avg_time_ms": round(float(np.mean([r["time_ms"] for r in rows])), 1),
        }
        json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
        print("\nSUMMARY")
        for k, v in summary.items():
            print(f"  {k:26s} {v}")


def main():
    ap = argparse.ArgumentParser(description="FullWorks C-UASC static target detection")
    ap.add_argument("--folder", default="test_images", help="folder of test images")
    ap.add_argument("--out", default="results", help="where to save annotated images + results.csv")
    ap.add_argument("--image", help="run on a single image and show it in a window")
    ap.add_argument("--save-masks", action="store_true", help="also save the red/green/white masks")
    a = ap.parse_args()

    if a.image:
        img = cv2.imread(a.image)
        if img is None:
            raise SystemExit(f"Could not read {a.image}")
        dets, _ = detect(img)
        for d in dets:
            print(json.dumps({k: v for k, v in d.items() if k != "debug"}))
        if not dets:
            print("No target found")
        out = draw(img, dets)
        cv2.imwrite("detected_" + os.path.basename(a.image), out)
        cv2.imshow("detection (press any key)", out)
        cv2.waitKey(0)
    else:
        run_folder(a.folder, a.out, a.save_masks)


if __name__ == "__main__":
    main()
