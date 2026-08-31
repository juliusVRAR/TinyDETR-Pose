# pose_diagnostics

Offline diagnostics for the rotation bottleneck in your LW-DETR + PoET-style 6D pose model on YCB-V.

The whole idea: **dump records once** from your existing PoET evaluation loop, then run
all analyses offline from the dump. Nothing here depends on your exact codebase except the
~10-line hook in `collect_records.py`.

## Dependencies
```
numpy, matplotlib           # required
scipy                       # optional (Spearman); falls back to a numpy rank-corr
trimesh, pillow             # only for overlay_gt.py (loading .ply + image)
```

---

## STEP 0 - dump records (do this ONCE, twice)

Open `collect_records.py`, copy the `RecordCollector` usage into your PoET eval loop
right where you already compute the per-class rotation error. For each matched
(prediction, GT) instance call `rec.add(...)`, then `rec.save(path)` at the end.

Run your evaluation **twice**:

| run        | output            | image list                                   |
|------------|-------------------|----------------------------------------------|
| validation | `records_val.npz` | your normal test / keyframe split            |
| train      | `records_train.npz` | a **subset of the TRAINING sequences**     |

### CRITICAL rules for the TRAIN run (or the diagnostic is meaningless)
1. **Use the EVAL transform = NO training augmentation.** You want to measure how well
   the model FIT the clean data, not augmented data.
2. **Use the same box source on both runs** - ideally **GT boxes on train AND val**, so
   detection quality does not confound the pose gap.
3. **Subsample.** ~300-500 instances each for `037_scissors` and `003_cracker_box`
   is plenty. You do not need the full train set.
4. Use a **fixed** subset, not live training batches.

---

## STEP 1 - Q1: train-vs-val per-class (find frame-bug vs capacity)
```
python per_class_table.py records_train.npz records_val.npz --id-mode index1
```
Reads the naive geodesic error per class for both dumps and prints the gap.

**Decision rule (focus on 037_scissors, 003_cracker_box):**
- LOW on train, HIGH on val  -> memorization / generalization gap
  -> levers: capacity (scale backbone), viewpoint diversity, allocentric.
- HIGH on train too          -> the model cannot even FIT it
  -> frame/convention bug or symmetry-gradient conflict (scaling will NOT help).

---

## STEP 2 - Q2: error histogram + error axis
```
python error_histogram.py records_val.npz --classes 037_scissors 003_cracker_box --id-mode index1
```
For each class: histogram of per-instance naive geodesic error (red line at 126.5 deg =
mean error of a RANDOM rotation) and the dominant **error-rotation axis**.

**Read the shape:**
- bimodal, peak ~0 AND peak ~180  -> near-symmetry flip -> capacity / resolution.
- single tight peak at a fixed nonzero angle, consistent error axis -> **constant offset
  = per-object frame bug** (very fixable).
- broad / ~uniform around 126.5    -> class not learned -> capacity / data.

---

## STEP 3 - Q4: scatter rot-error vs distance from principal point
```
python scatter_pp.py records_val.npz --id-mode index1
```
Non-symmetric instances only. Prints Spearman rho and saves a scatter + radial-bin means.
- rho > 0 (clear positive)  -> egocentric-from-local-features is hurting -> switch to
  **allocentric** (see `ego_allo.py`).
- rho ~ 0                   -> radial position is not the driver; look elsewhere.

---

## STEP 4 - GT consistency: train R == eval R ?
```
python check_gt_consistency.py     # fill in the two dataset constructors first
```
Loads the SAME frame from your TRAIN dataset and your EVAL dataset, pulls R_gt/t_gt,
and reports: allclose? transpose? axis-swap/sign-flip? This is the direct test for
"is the GT rotation the same source in both pipelines".

Also answers ego-vs-allo: if `x_cam = R_gt @ x_model + t_gt` projects ONTO the object
(use overlay_gt.py), your label is **egocentric** (the raw BOP/YCB label). Allocentric is
something you must construct explicitly - if you never wrote that code, you are on ego.

---

## STEP 5 - Q3: overlay GT (and prediction) on the image
```
python overlay_gt.py records_val.npz --models-dir /path/to/ycbv/models \
       --id-mode index1 --model-unit-to-m 0.001 --n 6
```
Projects CAD points with (R_gt, t_gt, K) in GREEN and (R_pred, t_pred) in RED.
- GT green not on the object -> K / 640x480->640x640 padding mismatch, or a frame bug.
- GT green correct, red far off -> pure pose error (expected for the bad classes).

Note on padding: overlay on the ORIGINAL (unpadded) image with the ORIGINAL K. Padding
the bottom/right of a 640x480 image to 640x640 does NOT change K (fx,fy,cx,cy). If you
ALSO resized or center-padded, K changed and that is a prime suspect - reproduce both.

---

## File map
```
common.py               shared: class names, 6D->R, geodesic, id mapping, loaders
collect_records.py      the hook you paste into your eval loop  (EDIT/USE)
per_class_table.py      Q1 train-vs-val table
error_histogram.py      Q2 histograms + error axis
scatter_pp.py           Q4 scatter + Spearman
check_gt_consistency.py  GT source check (EDIT: dataset constructors)
overlay_gt.py           Q3 reprojection overlay
ego_allo.py             ego<->allo conversion for the next step
```
