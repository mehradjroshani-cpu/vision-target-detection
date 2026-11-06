"""
live_demo.py
FullWorks C-UASC - Vision Team - live demonstration

Webcam mode (default): point a laptop/USB camera at printable_bullseye.png or
printable_green_x.png (on a phone screen or printed on paper).
    python live_demo.py
    python live_demo.py --camera 1        # use a second camera

Slideshow mode (no camera needed - good backup for the meeting):
    python live_demo.py --images test_images

Keys:  q = quit   m = show/hide color masks   s = save snapshot
       (slideshow) any other key = next image
"""

import argparse
import glob
import os
import time

import cv2
import numpy as np

from detect_targets import detect, draw


def mask_view(frame, masks):
    """Small strip showing the red / green / white masks under the frame."""
    h, w = frame.shape[:2]
    tiles = []
    for name in ("red", "green", "white"):
        m = cv2.cvtColor(masks[name], cv2.COLOR_GRAY2BGR)
        m = cv2.resize(m, (w // 3, h // 3))
        cv2.putText(m, name + " mask", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        tiles.append(m)
    strip = np.hstack(tiles)
    strip = cv2.resize(strip, (w, strip.shape[0]))
    return np.vstack([frame, strip])


def info_lines(dets):
    if not dets:
        return "no target"
    d = dets[0]
    return (f"{d['label']}  center=({d['center'][0]:.0f},{d['center'][1]:.0f})  "
            f"offset=({d['offset_from_image_center'][0]:+.0f},{d['offset_from_image_center'][1]:+.0f}) px")


def run_camera(cam_index):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {cam_index}. Try --camera 1, or use --images test_images")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    show_masks, fps, t_prev = False, 0.0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets, masks = detect(frame)
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - t_prev))
        t_prev = now
        out = draw(frame, dets, f"{fps:4.1f} FPS | {info_lines(dets)}")
        if show_masks:
            out = mask_view(out, masks)
        cv2.imshow("FullWorks target detection - live (q to quit)", out)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("m"):
            show_masks = not show_masks
        if k == ord("s"):
            fn = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(fn, out)
            print("saved", fn)
    cap.release()
    cv2.destroyAllWindows()


def run_slideshow(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))
    if not files:
        raise SystemExit(f"No images in {folder}")
    show_masks, i = False, 0
    while True:
        img = cv2.imread(files[i % len(files)])
        t0 = time.perf_counter()
        dets, masks = detect(img)
        ms = (time.perf_counter() - t0) * 1000
        out = draw(img, dets, f"{os.path.basename(files[i % len(files)])} | {ms:.0f} ms | {info_lines(dets)}")
        if show_masks:
            out = mask_view(out, masks)
        cv2.imshow("FullWorks target detection - slideshow (q to quit, any key = next)", out)
        k = cv2.waitKey(0) & 0xFF
        if k == ord("q"):
            break
        if k == ord("m"):
            show_masks = not show_masks
            continue
        if k == ord("s"):
            cv2.imwrite(f"snapshot_{i}.jpg", out)
            continue
        i += 1
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--images", help="run as a slideshow over this folder instead of the webcam")
    a = ap.parse_args()
    if a.images:
        run_slideshow(a.images)
    else:
        run_camera(a.camera)
