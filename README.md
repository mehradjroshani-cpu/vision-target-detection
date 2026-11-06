# FullWorks C-UASC — Vision Team, Part A: Static Target Detection

**Milestone:** Target detection demo
**Deliverables:** ✅ initial detection pipeline · ✅ collection of test images · ✅ live demonstration

## What it does
Finds the two ground targets from the **2027 C-UASC draft rules** in a still image:

| Target | Mission | What it looks like (2027 Rules Rev 1) |
|---|---|---|
| Red bullseye | Package Drop (beanbag) + Package Delivery | ~5 m red/white bullseye |
| Green X | Package Recovery | Green X on a white square with a thin black outline |

For each target it draws a **bounding box**, marks the **center in image coordinates**
(pixels, origin top-left), and reports the **offset from the image center**, which
the drone will later use to steer over the target before dropping.

## Files
| File | Purpose |
|---|---|
| `detect_targets.py` | The pipeline. Runs on the whole test set and writes results |
| `generate_test_images.py` | Builds the 18 test images + `ground_truth.json` + printable targets |
| `live_demo.py` | Live webcam demo (or slideshow if there's no camera) |
| `test_images/` | Test dataset (18 images + true target centers) |
| `results/` | Annotated images, `results.csv`, `summary.json`, `all_results_grid.jpg`, color masks |
| `printable_bullseye.png`, `printable_green_x.png` | Hold up to the webcam for the live demo |

## How to run
```bash
pip install -r requirements.txt

python detect_targets.py                         # whole test set -> results/
python detect_targets.py --image my_photo.jpg    # one image
python live_demo.py                              # webcam; press m for masks, q to quit
python live_demo.py --images test_images         # backup demo, no camera needed
```

## How the pipeline works
1. **Load** the image (OpenCV, BGR).
2. **Color masks:** blur, convert to HSV, threshold for red (two hue ranges because red wraps around 0), green and white.
3. **Clean up:** morphological open/close to remove specks and join ring pieces; fill holes so the bullseye becomes one solid shape.
4. **Shape checks** to reject look-alikes:
   - **Bullseye:** must fit an ellipse (works when the camera is tilted), be only partly red (a solid red disc is ~100% red), and show **≥4 red↔white ring crossings** along rays cast from its middle. A red car or red disc gets 0–2.
   - **Green X:** find white squares first, then look for green **inside** them only (so grass can't merge with the X). The green shape must be "skinny" (solidity 0.2–0.65); a bush is ~0.9+.
5. **Center:** bullseye = the small red center dot (stays accurate under camera tilt; falls back to the ellipse center). Green X = centroid of the X, which is where the arms cross.
6. **Compare across images:** each detection is matched to `ground_truth.json`, and the script logs center error, misses, false positives and run time.

## Results (18 test images)
| Metric | Value |
|---|---|
| Images passed | **18 / 18** |
| Targets found | 17 / 17 (recall 1.00) |
| False positives | 0 (precision 1.00) |
| Mean center error | **0.9 px** (max 5.8 px, on a tilted green X) |
| Speed | ~21 ms per 1280×720 image (~45 FPS on CPU) |

Test conditions covered: desert, grass and asphalt ground; high and low altitude
(target 90–420 px wide); camera tilt; dim, bright and shadowed light; motion blur;
heavy noise and JPEG compression; a target half out of frame; both targets in one image;
and decoys (red car, solid red disc, green bushes, black-X GCP flags), including two images with no target at all.

**First run vs. final:** the first version passed 14/18. A red car was detected as a bullseye,
grass merged with the green X, and camera tilt caused a 25 px center error. The fixes were the
ring-crossing test, searching for the X only inside white squares, and using the center dot.

## Presentation talking points
1. **What we accomplished:** a working detector for both 2027 targets with bounding box + center in pixels; an 18-image test set with known answers; and a live webcam demo.
2. **Show our work:** run `live_demo.py` with the printed target, then show `results/all_results_grid.jpg` and the table above.
3. **Biggest technical challenge:** telling the real target apart from look-alikes (red objects, grass) and keeping the center accurate when the camera is tilted. Also, **our test images are synthetic**: real aerial photos will have glare, color shift and real clutter, so 18/18 here does not mean 18/18 in the field.
4. **Proposed next step:**
   - Print a full-size or scaled bullseye and photograph it from a drone or a ladder at 25–60 ft to build a **real image dataset**, then re-tune the HSV thresholds.
   - Run on **live video** from the drone camera (Hardware team: which camera and companion computer?).
   - Convert the pixel center into a **GPS position / steering command** using altitude, camera field of view and drone heading (needed for Package Drop and Target Localization).
   - Start on the **Target Localization** task (5–10 objects, ≥25 ft altitude), which likely needs a trained model such as YOLO once the object types are published.

## Tuning
All thresholds are at the top of `detect_targets.py` (`RED_LOW_1`, `GREEN_LOW`, `MIN_RING_CROSSINGS`, …).
Run with `--save-masks` to see what each color mask picks up when tuning on real photos.

*Source for target descriptions: 2027 C-UASC Rules Rev 1 (draft), calstatela.edu.*
