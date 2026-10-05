# airseekers-yolo-retrain

Capture frames from an Airseekers mower's cameras, get bounding boxes from a
local vision LLM, convert them to a YOLO dataset, train, and export back to a
Rockchip `.rknn` that the ROS 2 `det_ros` node can load.

Everything runs on **your LAN host**. The mower is only read over HTTP, and
nothing is installed or written on it. Its HTTP camera server must be up,
though: on the vendor ROS 1 firmware `web_video_server` (port 8181) is always
running; on the ROS 2 port it is off until you launch
`cameras.launch.py web_video_server:=true` (port 8080).

---

## The one thing to read first

**The `.rknn` files that shipped with the mower are inference-only.**

`best_large_0208.rknn` and friends are int8 blobs already compiled for the
RK3588's NPU. There are no weights inside them to fine-tune from — they are a
compiled graph plus quantised activation ranges. You cannot resume training from
a `.rknn`, and there is no `yolo detect train model=*.rknn`.

So:

| To do this | Start from |
|---|---|
| Train a model of the **same class** as the vendor's (22 classes, same order, 640×480 portrait input, 9-output head, rk3588 int8) | the matching **ultralytics checkpoint** — `yolov8n.pt` for the small variant, `yolov8s.pt` if you have the VRAM and the latency headroom |
| Fine-tune the **vendor's actual model** | their **original PyTorch `.pt`**, the one they trained before compiling. If you have it, this is strictly better than a COCO checkpoint: the backbone already knows your lawn, your lighting, your mower's deck height. |
| Any of the above starting from the `.rknn` | **not possible.** |

**Pseudo-labelling with the shipped `.rknn` as a teacher is NOT implemented
here.** It is described in the design doc (see References) and would be the best
substitute for the vendor `.pt`, but running an `.rknn` on x86 needs the
rknn-toolkit2 simulator and is not wired into this pipeline. The provenance file
already has a `source` field (`llm_suggestion` | `human`, with `pseudo`
reserved) so teacher labels can be added later without changing the format.

Both 04 and 05 refuse a `.rknn` on `--weights` with an explanation rather than
failing somewhere deep in torch.

---

## The det_ros contract (what the exported model must match)

From `airseekers-decompile/ros2_stack/src/det_ros` and
`src/mower_rknn/mower_rknn/preprocess.py`:

| | |
|---|---|
| Input | `images`, NCHW `[1,3,640,480]` — **640 tall, 480 wide**. Metadata: float32, range [0,1] via mean `[0,0,0]` / std `[255,255,255]`, rgb2bgr False |
| Runtime preprocessing | letterbox: scale `min(480/w, 640/h)`, centre, **pad 0 (black)** to 480×640; BGR→RGB; uint8 NHWC; no normalisation. det.yaml `img_size: [480, 640]` is `[W, H]` |
| Outputs | 9 tensors, `[box, cls, score_sum]` per stride: `[1,64,80,60] [1,22,80,60] [1,1,80,60] [1,64,40,30] [1,22,40,30] [1,1,40,30] [1,64,20,15] [1,22,20,15] [1,1,20,15]`. box = DFL, reg_max 16; cls already sigmoid (airockchip head); score_sum ignored |
| Quantisation | asymmetric int8, per-**layer**, target rk3588, rknn-toolkit2 2.3.0 |
| Classes | 22, order = `pipeline.yaml classes.names` = `det.yaml classes` = `labels.py BEST_LARGE_CLASSES` |
| Cameras | ROS 2 det_ros reads **only** `/left_oa_camera/image_raw` and `/right_oa_camera/image_raw` (GC2093, 1920×1080). The vendor ROS 1 node also read `/vio/right/image_raw` (640×480) |

A 1920×1080 OA frame therefore reaches the model as a 480×270 image with
185-pixel black bands above and below. That is why the OA cameras are the
primary capture source, and why the int8 calibration set is letterboxed the same
way (05).

---

## The flow

```
  01_capture.py      mower MJPEG / snapshot ──┐
                     (profile ros1 | ros2)    ├──>  frames/<session>/<cam>/*.jpg
                     local folder          ───┘     frames/<session>/manifest.jsonl
                                                        │
                                                        ▼
  02_llm_label.py    LM Studio (OpenAI-compatible, vision model, LAN)
                     ──> labels/<session>/raw/*.json   (reply + sent image size)
                     ──> labels/<session>/index.jsonl
                                                        │
          validate · OOV → review · per-frame coord space · rescale by sent size
                                                        ▼
  03_to_yolo.py      ──> dataset/images/{train,val}/*.jpg   (hardlinked, rebuilt each run)
                     ──> dataset/labels/{train,val}/*.txt   (YOLO format)
                     ──> dataset/review/                     (needs a human; hand labels go here)
                     ──> dataset/data.yaml                   (22 classes, exact order)
                     ──> dataset/provenance.jsonl            (audit trail)
                                                        │
                                                        ▼
  04_train.sh        ultralytics train (imgsz 640) + val → runs/<name>/weights/best.pt
                     (ROCm or CUDA if torch sees a GPU, else CPU)
                                                        │
                                                        ▼
  05_export_rknn.sh  best.pt → airockchip fork export(format='rknn', imgsz=[640,480])
                     → ONNX contract check → letterboxed calibration
                     → rknn-toolkit2 2.3.0 (int8, per-layer) → best_22cls.rknn
                     + .rknn metadata self-check
```

---

## Setup

```bash
git clone <this repo> airseekers-yolo-retrain
cd airseekers-yolo-retrain

python3 -m venv .venv && source .venv/bin/activate
pip install pyyaml requests pillow          # runtime (01-03, calibration)
pip install ultralytics                     # 04 (torch comes with it)
pip install pytest                          # tests/

# 05 exports with the Rockchip fork, next to this repo by default
# (export.airockchip_repo: ../ultralytics_yolov8). It is put on PYTHONPATH
# for the export only, so it does not replace your pip ultralytics.
git clone https://github.com/airockchip/ultralytics_yolov8 ../ultralytics_yolov8
```

For the `.rknn` conversion step, on an **x86_64** host:

```bash
pip install rknn-toolkit2==2.3.0            # the version the shipped models used
```

Edit `config/pipeline.yaml`. The values you must check before anything works:

```yaml
mower:
  profile: ros1            # ros1 = vendor firmware (8181) | ros2 = the ROS 2 port (8080)
  host: 192.168.1.105
  base_url: null           # null -> http://<host>:<profile port>; set to override
```

Every value can also be overridden per run without editing the file:

```bash
python3 scripts/01_capture.py --set capture.duration_s=30 --set mower.base_url=http://10.0.0.5:8181
```

Run the unit tests (no GPU, no network, no mower):

```bash
python3 -m pytest -q tests
```

---

## 01 — Capture frames

The mower publishes its camera topics through `web_video_server`, which turns
any image topic into HTTP MJPEG or single JPEG snapshots. You do **not** need
ROS on your host, and you do **not** touch `/dev/video*` — the camera drivers
own those devices and a second `VIDIOC_STREAMON` on the same node fails with
`EBUSY`. The HTTP path sidesteps that entirely.

Two profiles (`mower.profile`, or `--profile`):

| | `ros1` (vendor firmware) | `ros2` (the ROS 2 port) |
|---|---|---|
| HTTP port | 8181, always on | 8080, **off by default** — `ros2 launch mower_bringup cameras.launch.py web_video_server:=true` |
| Cameras in the preset | `left_oa`, `right_oa` (primary, 1 Hz), `front` = `/vio/right` (0.5 Hz), `front_left` = `/vio/left` (0.25 Hz), `rear` (0.25 Hz) | `left_oa`, `right_oa` (primary, 1 Hz), `rear` (0.25 Hz) |
| Gating | OA + rear need `rosservice call /<cam>/start_capture "{}"` | no start_capture services |
| RTSP | — | 8554 via mediamtx (`docker/docker-compose.video.yml`), for viewers; 01 reads HTTP |

Check what's answering first. `--probe` fetches one frame per camera through
the exact URL capture would use (streams are read until the first JPEG, then
closed):

```bash
python3 scripts/01_capture.py --probe
python3 scripts/01_capture.py --probe --profile ros2
```

```
probing http://192.168.1.105:8181  (profile ros1)
  OK    left_oa      /left_oa_camera/image_raw  1920x1080  271 KiB  88 ms  [snapshot]
  EMPTY right_oa     /right_oa_camera/image_raw  no JPEG after 5031 ms  [snapshot]
         -> topic silent, or the camera needs: rosservice call /right_oa_camera/start_capture '{}'
         -> (mower_cam_keeper.service normally does this when a viewer subscribes)
         -> check the topic on the mower: rostopic hz /right_oa_camera/image_raw
  ...
```

(The numbers above are illustrative; the line format is what the script prints.)

Then capture:

```bash
# every camera in the profile's preset, 60 s, rates from the config
python3 scripts/01_capture.py

# just what det_ros consumes
python3 scripts/01_capture.py --cameras left_oa,right_oa --duration 120

# explicit base URL + topics
python3 scripts/01_capture.py \
  --base-url http://<mower-ip>:8181 \
  --topic left_oa=/left_oa_camera/image_raw \
  --topic rear=/rear_camera/image_raw

# any MJPEG server at all
python3 scripts/01_capture.py --url front='http://<host>:8181/stream?topic=/vio/right/image_raw&type=mjpeg'

# no mower: import frames you already have
python3 scripts/01_capture.py --folder ~/pics/lawn --rate 2
```

`--base-url` applies to every source (stream URLs, snapshot URLs and the
probe). Stream-mode and snapshot-mode cameras run together in one session, one
thread per camera, so a silent snapshot topic cannot stall the others.

Output:

```
work/frames/20261005-142530/
├── left_oa/000001_1772...jpg
├── right_oa/...
└── manifest.jsonl
```

The manifest is not decoration. Later steps need to know which camera a frame
came from (03 records it in provenance, 05 stratifies calibration by it):

```json
{"seq": 1, "camera": "left_oa", "topic": "/left_oa_camera/image_raw",
 "stamp_ns": 1772130150123456789, "width": 1920, "height": 1080,
 "bytes": 277321, "phash": "3f9c1a77e0d24b51", "session": "20261005-142530",
 "source": "snapshot", "path": "work/frames/20261005-142530/left_oa/000001_1772...jpg"}
```

01 does not record the mower's pose. If you add an `odom: [x, y]` field to the
manifest yourself, 03 can split by pose cell (see 03).

**Notes**

- **Don't leave the rear camera streaming while docked** on the vendor
  firmware: `mower_charge` reads `/rear_camera/image_raw` for dock detection.
  That's why `rear` is capped at 0.25 Hz in both presets.
- **Near-duplicate rejection is on** (`phash_min_distance: 6`). A parked mower
  emits identical frames at the full topic rate and would otherwise write
  hundreds of copies of the same picture. After `stop_after_dup_run` (5)
  consecutive rejects a camera is stopped — the scene has stopped changing.
  This applies to stream and snapshot cameras alike (not to `--folder` imports,
  where dedupe is off).
- **Disk is capped.** Capture stops when the *session* (all cameras together)
  reaches `quota_bytes` (2 GiB) and prunes the oldest sessions when all
  sessions exceed `total_quota_bytes` (12 GiB). It never raises. The mower is
  not written to; these limits protect the host.
- **Capture parked or docked, blade off.** A 30-minute stationary harvest of one
  lawn is worth more than 30 minutes of driving.

---

## 02 — Suggested boxes from a local VLM

LM Studio (or anything OpenAI-compatible: vLLM, llama.cpp's server, Ollama's
`/v1`).

On the host that runs LM Studio: **Developer tab → Start Server**, load a
*vision* model. The default port is **1234** (`http://127.0.0.1:1234/v1`).

```bash
# what is being served?
python3 scripts/02_llm_label.py --images work/frames/20261005-142530 --health

# label the session
python3 scripts/02_llm_label.py --images work/frames/20261005-142530

# explicit model, more parallelism
python3 scripts/02_llm_label.py --images work/frames/<session> \
    --model qwen2.5-vl-7b-instruct --concurrency 4
```

Output:

```
work/labels/20261005-142530/
├── raw/000001_1772...json      the request, the full raw reply, sent/orig image size
└── index.jsonl                 one row per frame
```

**Verify one frame before running a batch.** Vision models disagree about
coordinate space, and a wrong assumption here silently corrupts every label:

```bash
python3 scripts/02_llm_label.py --images work/frames/<session> --limit 1 \
    --print-response
```

You asked for 0–1000 normalised `[x1,y1,x2,y2]`. Confirm the reply is in that
space and that the boxes land on the objects. `03_to_yolo.py` also infers the
space per response and records the decision in `provenance.jsonl`, but that is
a safety net, not a substitute for looking.

Why these choices:

- **Closed vocabulary.** The prompt enumerates the 22 names. Common synonyms
  are mapped first (`rock`→`stone`, `trash can`/`bin`→`trashbin`,
  `toy car`→`toycars`, `can`→`cans`, `ball`→`sports ball`, `log`/`plank`→`wood`,
  `tree trunk`/`stump`→`trunk`, `charging station`→`dock`,
  `plant pot`→`potted plant`, …; extend with `llm.class_aliases`). Anything
  still outside the vocabulary is stripped from `boxes` and listed in
  `out_of_vocabulary`, and **03 sends that frame to review** — the object is
  still in the picture and must not train as background.
- **`temperature: 0`, one image per request.** Otherwise boxes drift between
  runs, re-running the batch silently changes the dataset, and your training
  result isn't reproducible.
- **Images are downscaled to `llm.max_edge` (1024) before sending**, and the
  size actually sent is recorded (`sent_width`/`sent_height`) next to the
  original. A model that answers in pixels answers in the *sent* image's
  pixels; 03 rescales by original/sent.
- **Every reply is kept**, including failures and empties. The raw JSON is the
  audit trail; a bad batch can be re-run (`--retry-errors`) or hand-fixed
  without re-capturing, which is the expensive step.
- **Empty is a real answer.** `{"boxes": []}` means "nothing from the
  vocabulary". Those frames become negatives, and a detector trained only on
  positives invents obstacles in clean grass.
- If the server rejects `response_format` with HTTP 400, 02 retries without it
  and, once that works, stops sending it for the rest of the run.

If more than half the frames come back empty, the model probably isn't seeing
the images at all (a text-only model returns `{"boxes": []}` forever). The script
warns when it sees that. Ctrl-C cancels the queued frames and keeps what is done.

---

## 03 — Validate and convert to YOLO

```bash
python3 scripts/03_to_yolo.py \
    --labels work/labels/20261005-142530 \
    --images work/frames/20261005-142530
```

The summary looks like this (numbers illustrative, format exact):

```
=== validation summary ===
frames seen      : 412
  accepted       : 361  (58 negatives)
  review queue   : 51
  skipped        : 0
  partial accepts: 22  (some boxes dropped, rest kept)

review reasons:
     31  at least one kept box in the 0.35-0.55 confidence band
     12  out-of-vocabulary class
      6  request failed
      2  no usable box survived

coordinate space used (space, decided by):
    280  0-1000   session
     81  0-1000   frame
         Verify one frame by eye. A wrong space scales every box and the model
         still trains 'fine' — pass --coord-space to override.

per-class box counts (dataset is NOT balanced by design — recall-critical
classes need more examples than exotic ones):
   0 person             34  ##################################
   1 dog                 9  #########
   ...
  21 dock                0
```

This step is the gate, and it exists because a suggestion that is *slightly*
wrong is worse than no label — it trains confidently in the wrong direction. It:

- sends frames whose reply contained an **out-of-vocabulary class** to review
- **resolves the coordinate space per frame**: each reply is classified against
  the size of the image the LLM was sent (0–1000 / 0–1 / pixels); a frame that
  settles it on its own keeps its verdict; ambiguous frames take the majority of
  their session (or 0–1000 if nothing in the session is decisive);
  `--coord-space` overrides everything. Scale factors are always built from
  each frame's own size, so 640×480 and 1920×1080 frames can share a session
- rescales **pixel** replies by original/sent size (recorded by 02; inferred from
  `llm.max_edge` for older raw files, flagged `sent_size_inferred`)
- normalises xyxy into YOLO `cx cy w h`
- drops boxes covering >85 % of the frame or <0.05 %, and boxes below `min_conf`
- `bad_box_policy: lenient` (default): a frame where some boxes were dropped
  keeps the valid ones and is **accepted** (a partial accept, with a `note` in
  provenance). `strict`: any dropped box sends the whole frame to review
- routes a frame with a *kept* box in the `min_conf`–`review_conf_max` band,
  too many boxes, an unparseable reply or a failed request to `review/`
- keeps negatives as empty `.txt` files
- splits **by session (and pose cell, if the manifest has `odom`), never per
  frame**

That last one matters more than it sounds. At 1 Hz, the same rock is in twelve
consecutive frames. A per-frame split puts near-identical images in both train
and val and the resulting mAP is fiction. 01 does not write `odom`, so in
practice the split is **by session**: capture several sessions. If there are
fewer than two buckets (one session, no pose data), 03 falls back to a
per-frame split and says so loudly, rather than putting 100 % of the data in val.

`images/` and `labels/` are **emptied and rebuilt on every run**, so a frame
cannot linger in a split (or survive in a split after moving to review) from a
previous run. In `review/`, generated `*.reason.json` files and frame copies are
refreshed each run; hand labels are never deleted, not even by `--clean`
(which additionally removes everything else in the dataset dir).

Output:

```
work/dataset/
├── images/train/*.jpg        hardlinks — no duplicate bytes on disk
├── labels/train/*.txt        "<class_id> cx cy w h", normalised
├── review/                   frames + a .reason.json each; your hand labels
├── data.yaml                 22 classes, exact names, exact order
├── provenance.jsonl          frame → camera, model, coord space + how decided, source
└── summary.json              counts and thresholds
```

### Hand-labelling the review queue

**Read `review/` before training.** Label those frames and put the labels **in
`work/dataset/review/`**, one file per frame, named after the image stem:

- `<stem>.txt` — YOLO format as exported by CVAT ("YOLO 1.1") or labelme:
  `<class_id> cx cy w h`, normalised. If the export's `classes.txt` /
  `obj.names` is in the same folder, ids are remapped **by name** onto the 22
  classes (a name outside the vocabulary is an error); without it the ids must
  already be this pipeline's ids (0–21). An empty file means "no objects" (a
  human-confirmed negative).
- `<stem>.label.txt` — same format, this pipeline's older convention.
- `<stem>.json` — the prompt's schema, `{"boxes": [{"class": "dock", "box":
  [x1, y1, x2, y2]}]}` in 0–1000; `class` names are mapped to ids, `class_id`
  may be given instead.

Then re-run with `--include-review`. A hand label replaces the LLM suggestion
for that frame (provenance `source: human`); a malformed one (bad id, unknown
name, non-normalised box) sends the frame back to review with the reason. An
LLM suggestion is not ground truth; the `--include-review` path is what makes a
mixed dataset honest.

---

## 04 — Train

```bash
bash scripts/04_train.sh
bash scripts/04_train.sh --epochs 300 --batch 16
bash scripts/04_train.sh --weights /path/to/vendor.pt    # if you have it
bash scripts/04_train.sh --device cpu --epochs 2         # smoke test
bash scripts/04_train.sh --resume runs/det_22cls/weights/last.pt
```

`runs/<name>/weights/best.pt`, plus `results.csv` and the usual plots.

### Backends (AMD RX 7900 XTX / RDNA3 first)

`04_train.sh` runs `scripts/_backend.py` and tells you what it found:
`rocm`, `cuda`, `rocm-unavailable` or `cpu`. GPU availability is checked
first and `torch.version.hip` only tells ROCm from CUDA, so an NVIDIA build is
not mistaken for "no GPU".

1. **ROCm PyTorch → it just works.** `device=0` with no special flags;
   ultralytics shells out to torch.
   ```bash
   pip install torch torchvision --index-url \
     https://download.pytorch.org/whl/rocm6.2
   python3 -c "import torch; print(torch.version.hip, torch.cuda.is_available())"
   ```

2. **Two env vars, both set automatically by `04_train.sh`:**
   - `HSA_OVERRIDE_GFX_VERSION=11.0.0` — some ROCm builds ship kernels only up
     to gfx1030 and fail to load on gfx1100. Forgetting this is the most common
     RDNA3 failure.
   - `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` — the stock caching
     allocator fragment-errors above batch 8.
   You must be in the `video` and `render` groups. RDNA3 does **not** need the
   old `amdgpu-pro` install.

3. **AMP defaults to off on ROCm.** RDNA3 has no native FP16 accumulate in the
   kernels ROCm ships, so AMP frequently yields NaN losses. `04_train.sh` sets
   `amp=False` automatically and prints why. `--amp true` overrides. On CUDA
   the ultralytics default (AMP on) is kept.

4. **CPU fallback.** If torch sees no GPU, `device=cpu` trains correctly and
   slowly — expect 20–40× the wall clock. Fine for `--epochs 1` smoke tests and
   for confirming `data.yaml` loads. It is not fine for a real run.

5. **Batch 32** at 640 fits in 16 GB with AMP off. Drop to 16 on OOM (ROCm
   surfaces it as an allocator message, not a Python exception).

### `imgsz`: square for training, portrait for export

The ultralytics trainer calls `check_imgsz(imgsz, max_dim=1)`, which turns any
`[h, w]` into `max(h, w)` with a warning — **training is always square**. So
`train.imgsz` is `640`. YOLOv8 is fully convolutional; the det_ros geometry,
NCHW `[1,3,640,480]` (640 tall, 480 wide, feature maps 80×60 / 40×30 / 20×15),
is fixed at **export** time by `export.imgsz: [640, 480]` in 05.

One mismatch remains and is worth knowing: ultralytics letterboxes with grey
(114) padding, while det_ros pads with **black (0)** and a 1920×1080 OA frame
arrives with 185-px black bands top and bottom. The calibration set (05) is
letterboxed the det_ros way; training data is not. Check field behaviour near
the bands before trusting val mAP.

---

## 05 — Export ONNX, convert to `.rknn`

```bash
bash scripts/05_export_rknn.sh                      # airockchip export + check + int8

# float A/B, no quantisation
bash scripts/05_export_rknn.sh --dtype fp

# per-channel int8 A/B (the shipped model is per-layer)
bash scripts/05_export_rknn.sh --quant-method channel

# ONNX only (export + contract check), no toolkit needed
bash scripts/05_export_rknn.sh --skip-rknn

# conversion only, on the x86 host, from an ONNX made elsewhere
bash scripts/05_export_rknn.sh --onnx-in work/models/best_22cls.onnx

# metadata self-check on anything
bash scripts/05_export_rknn.sh --verify-only work/models/best_22cls.rknn
```

Paths given on the command line are relative to your current directory;
paths in the config are relative to the repository root.

### Export: the airockchip fork

The 9-output head comes from
[airockchip/ultralytics_yolov8](https://github.com/airockchip/ultralytics_yolov8).
The fork has no stand-alone `export.py`; its RKNN path is the normal exporter
with `format='rknn'` (`Exporter.export_rknn`: the Detect head returns, per
stride, the raw DFL box tensor, the sigmoid class tensor and a clamped
class-sum, and the ONNX is written next to the weights with opset 12 and input
name `images`). Its `RKOPT_README.md` says to run
`ultralytics/engine/exporter.py` with `PYTHONPATH=./`. 05 does the equivalent
through the Python API, with the fork first on `PYTHONPATH`:

```
PYTHONPATH=<fork> python -c "from ultralytics import YOLO;
  YOLO('<abs best.pt>').export(format='rknn', imgsz=[640,480], opset=12)"
```

and moves the `.onnx` to `work/models/best_22cls.onnx`. It refuses to run if the
imported `ultralytics` is not the fork. `--exporter ultralytics` (stock
`format=onnx`) is kept for comparison only: it produces a single `[1,26,6300]`
output and fails the next step.

### ONNX contract check

Before the toolkit sees anything, `scripts/_onnx_contract.py` checks the ONNX:
input `[1,3,640,480]`, 9 outputs with the shapes in the contract table above,
`nc` (22) class channels. A mismatch stops 05. (It needs the `onnx` package,
which rknn-toolkit2 installs; with `--skip-rknn` and no `onnx`, the check is
skipped with a message.)

### Conversion: x86_64 only

`rknn-toolkit2` is an **x86_64-only compiler**. The RK3588 runs the *runtime*
(`librknnrt` / `rknn-toolkit-lite2`), not the compiler. Convert on a desktop or
in a container; use `--onnx-in` there to skip the export.

Quantisation is passed explicitly to match the shipped model:
`quantized_dtype='w8a8'` (asymmetric int8), `quantized_method='layer'`,
`quantized_algorithm='normal'`, mean `[0,0,0]`, std `[255,255,255]`. The
toolkit's own default is per-channel; build a `--quant-method channel` variant
and A/B both against `--dtype fp` if int8 accuracy disappoints.

### Version pinning — check this before converting

The vendor tree ships two `librknnrt` copies (2.1.0 and 2.2.0) and the shipped
models were compiled with rknn-toolkit2 **2.3.0**. 05 refuses to convert when
the installed toolkit differs from `export.rknn.toolkit_version` /
`--rknn-toolkit`; `--reinstall-toolkit` pip-installs exactly that version. Find
out which runtime the mower actually loads:

```bash
ldd <det_ros binary or the python env's librknnrt> | grep rknnrt
strings <that librknnrt.so> | grep -m1 'librknnrt version'
```

If your model fails `rknn_init` with a version error, step the toolkit down
(`--rknn-toolkit 2.1.0 --reinstall-toolkit`) before touching anything else.

### Preprocessing is ABI too

`mean [0,0,0]` / `std [255,255,255]` are baked into the shipped models'
quantization table and the conversion keeps those exact values. det_ros feeds
uint8 RGB without normalisation and lets the runtime apply them. If anything
preprocesses differently, every box is subtly wrong and it looks like the
retrain didn't work.

### Calibration

int8 ranges are calibrated on `calib_count` (300) frames from these cameras
(`calib_from: dataset` = `<convert.dataset_dir>/images`, or `frames` =
`paths.frames_dir`; both from the config). The set is **stratified by the
manifest's `camera`**, weighted by `export.rknn.calib_camera_weights` (default
`left_oa: 0.4, right_oa: 0.4, "*": 0.2` — det_ros only sees the OA cameras), and
every image is **pre-letterboxed to 480×640 with black padding** exactly like
`mower_rknn.preprocess.letterbox` into a temporary directory, so the toolkit
never stretches a 1920×1080 frame. Cameras with no frames donate their share;
frames without a manifest entry count as camera `unknown`.

### Output contract self-check

Run automatically at the end of 05, and on demand with `--verify-only`. It reads
the `.rknn` metadata and checks:

- input `1 × 3 × 640 × 480`
- **nine** output tensors at 80×60, 40×30, 20×15
- 22 score channels at each scale
- quantization mean/std

A deviation is an **export bug**, not a tuning problem. Go back to 05; don't
"try it and see".

---

## QA gate

Before any model reaches the mower:

| Check | Threshold |
|---|---|
| mAP50 on the val split | ≥ vendor baseline + 2 pts |
| mAP50-95 | ≥ baseline |
| Per-class recall, `person`/`dog`/`cat`/`hedgehog`/`rabbit` | ≥ 0.95 each, **day and night** |
| Per-class recall, static classes | ≥ 0.70 |
| False positives per 1000 frames, clean lawn | ≤ baseline |
| int8 vs fp box agreement | ≥ 0.97 at IoU>0.5; mean score delta ≤ 0.05 |
| Class names and order in `data.yaml` | exact match, 22 entries |
| Exported input geometry | `1×3×640×480`, exact |

A caveat worth internalising: **your val labels are LLM suggestions**, so mAP
measures "did the model fit the suggestions", not "is the model correct". Hand-
labelled holdout frames from the `review/` queue are the only real metric. Get
the baseline by scoring the current node's own behaviour (the annotated image,
`/ai/det/image_annotated` on ROS 2) on a fixed frame set by hand — a rough
baseline beats none.

Day and night is not optional. The mower ships a `light_ai` classifier precisely
because lighting changes behaviour; a model that only passes at noon has not
passed.

---

## Layout

```
airseekers-yolo-retrain/
├── README.md
├── config/pipeline.yaml        one file for the whole pipeline (incl. mower profiles)
├── scripts/
│   ├── common.py               config, paths, profiles, hashing, MJPEG, manifest io
│   ├── 01_capture.py           mower MJPEG/snapshot or a local folder
│   ├── 02_llm_label.py         LM Studio / OpenAI-compatible vision labelling
│   ├── 03_to_yolo.py           validate + convert to YOLO + data.yaml + hand labels
│   ├── 04_train.sh             ultralytics train + val (ROCm notes)
│   ├── 05_export_rknn.sh       airockchip export + ONNX check + rknn-toolkit2 + self-check
│   ├── _backend.py             torch backend detection for 04
│   ├── _calib.py               camera-stratified, letterboxed calibration set for 05
│   ├── _onnx_contract.py       ONNX input/output contract check for 05
│   ├── _yolo_train.py          library-API fallback for the `yolo` CLI
│   └── _yolo_export.py         library-API fallback for the stock exporter
├── tests/                      pytest unit tests (no GPU, no network)
└── .gitignore                  work/, runs/, wandb/, *.pt, *.onnx, *.rknn, ...
```

Everything produced goes under `work/` and `runs/`, both gitignored. Nothing
here writes to the mower.

## Safety and privacy

- **Blade off** for any capture or test session with people or pets in frame.
- Frames from a home lawn contain houses, plates, neighbours and pets. Keep the
  LLM endpoint on your own LAN. `work/` (frames, labels, dataset) is gitignored
  for exactly this reason — do not commit it.
- LM Studio has no authentication. Bind it to the LAN interface only and restrict
  the port with a firewall rule, the way you would for the mower's
  8181/8080/8554.
- Ultralytics (and the airockchip fork) is **AGPL-3.0**; `rknn_model_zoo` is
  Apache-2.0. Personal use on your own mower is fine. Redistribution is a
  different question.
- Don't `apt upgrade` the mower on the vendor firmware. The closed binaries need
  Ubuntu 20.04 / Noetic / the vendor OpenCV 4.2 ABI.
- On the vendor firmware an OTA replaces `/workspace`, so it replaces any
  swapped model. Keep the rollback copy of the original `.rknn` on `/userdata`,
  which survives a rootfs reflash (the ROS 2 det_ros loads from
  `/userdata/ros2/models/` by default).

## References

The background survey behind this pipeline lives in the sibling
`airseekers-decompile` tree — read-only, useful when a number here needs
sourcing:

- `ros2_stack/src/det_ros/` — the ROS 2 node, `config/det.yaml`, `labels.py`,
  `yolo_postprocess.py` (the 9-output decode)
- `ros2_stack/src/mower_rknn/mower_rknn/preprocess.py` — letterbox (pad 0) and
  RGB NHWC conversion
- `ros2_stack/launch/cameras.launch.py` — OA/rear cameras (1920×1080),
  `web_video_server:=true` on 8080
- `docs/streaming-and-telemetry.md` — vendor camera map, MJPEG/RTSP setup
- `mower_docs/03-yolo-perception.md` — the model I/O contract
- `ros2_stack/docs/yolo_retrain_pipeline.md` — the design document this
  implementation follows, including the (not implemented here) pseudo-labelling
  with the vendor `.rknn` as teacher, the QA gate rationale, disk budget, the
  on-device acceptance run
- `ros2_port_handoff/13_open_source_upstreams/rknn_model_zoo/examples/yolov8/`
  — `convert.py` and the post-processing reference

---

## CHANGELOG

### 2026-10-06 — code-review fixes

**Critical**

- **05 export rewritten (C1).** The old `export.py --input_size` call does not
  exist in airockchip/ultralytics_yolov8; the fork exports through
  `YOLO(pt).export(format='rknn')`. 05 now runs exactly that with the fork on
  `PYTHONPATH`, `imgsz=[640,480]`, opset 12, moves the `.onnx` into
  `work/models/`, resolves `--weights` to an absolute path (absolute paths used
  to break), makes `airockchip` the default exporter, and checks the ONNX
  (`_onnx_contract.py`: 9 outputs, input `[1,3,640,480]`) before conversion.
  The stock exporter's single `[1,26,6300]` head can no longer reach the toolkit.
- **Hand labels actually reach the dataset (C2).** `--include-review` used to
  build hand-label outcomes without an image (dropped by assembly), could write
  a class *name* where the id goes, looked in the wrong directories and only for
  `<stem>.label.txt`. Now: labels are read from `dataset/review/` (and
  `<labels>/review/`), `<stem>.txt` (CVAT/labelme), `<stem>.label.txt` and
  `<stem>.json` are accepted, names map to ids, unknown/out-of-range ids are
  rejected to review, the frame image is attached (falling back to the review
  copy), and provenance records `source: human`.

**High**

- **Lenient partial accepts are accepts (H1).** Previously every partial accept
  was silently routed to review. The dropped-box note is stored separately.
- **Coordinate space per frame (H2).** Only the space *name* is resolved per
  session (for ambiguous frames); scale factors come from each frame's own size,
  an unambiguous per-frame verdict wins, and the 0–1 space now actually scales
  to the frame (it used a factor of 1.0). The garbled "split" warning is fixed.
- **Pixel replies rescaled by sent size (H3).** 02 records the size of the image
  it sent (`llm.max_edge` downscale); 03 maps pixel boxes by original/sent
  (inferred from `llm.max_edge` for older raw files). 1920×1080 frames used to
  get boxes ~0.53× too small.
- **Out-of-vocabulary frames go to review (H4);** 02 maps common synonyms onto
  the vocabulary first (`DEFAULT_ALIASES`, `llm.class_aliases`).
- **Capture plumbing (H5).** `--base-url` now reaches probe and snapshot URLs;
  stream and snapshot cameras run together (one thread per camera) instead of a
  single snapshot camera forcing everything into snapshot mode; `--probe` reads
  streams with a streaming GET to the first JPEG instead of hanging.
- **OA cameras are primary (H6).** Config, calibration weights and docs follow
  the ROS 2 det_ros, which reads only the 1920×1080 OA cameras with a pad-0
  letterbox.

**Medium**

- **imgsz (M1).** `train.imgsz: 640` (ultralytics coerces tuples to square);
  new `export.imgsz: [640, 480]`. The 114-vs-0 padding difference is documented.
- **Calibration (M2).** Roots come from the config; stratified by manifest
  camera with weights; images pre-letterboxed (pad 0) into a temp dir.
- **Toolkit (M3).** A version mismatch is an error (`--reinstall-toolkit`
  installs the pinned one); `quantized_dtype/method/algorithm` are passed
  explicitly (`w8a8`, `layer`, `normal`); `--quant-method channel` for A/B.
- **`--onnx-in` (M4)** replaces the invalid `--skip-rknn=false` advice; there is
  no re-export fallback.
- **Dataset rebuild (M5).** `images/` and `labels/` are always emptied (no split
  leak); `--clean` never deletes hand labels in `review/`.
- **04 (M6).** Backend detection checks GPU availability before HIP (CUDA builds
  were classified as CPU); `--resume` passes `resume=True model=<pt>` (the
  fallback also used to pass the literal `train` token as an argument).
- **Pseudo-labelling (M7)** is documented as not implemented; `source` in
  provenance is ready for it.
- **Mower profiles (M8).** `mower.profile: ros1|ros2` with per-profile port,
  camera presets and "camera silent" hints; `--profile` on 01.

**Low**

- 01: stream-mode dedupe stop (`stop_after_dup_run`) implemented; a silent
  snapshot camera no longer stalls the others; the quota is per session, not per
  camera; unused imports removed.
- 02: `--limit` now limits (it was ignored); unused `--rate` flag and imports
  removed; `llm.view_hint` replaces the missing `mower.front_camera_hint`;
  Ctrl-C cancels queued frames; the response_format 400 fallback is remembered;
  pretty-printed JSON is no longer flagged as "not pure".
- 03: unused imports removed; README no longer claims a verified pose split
  (01 never writes odom).
- 04/05: `--help` works when the script is run by relative path; 05's header
  describes the metadata (NCHW float32) correctly; the verify exit code is
  reported correctly (it used to print `$?` of the `if`).
- README: five cameras / profiles, "conversion step", LM Studio port 1234, what
  must run on the mower, sample output matches the code.
- Added `tests/` (pytest).
