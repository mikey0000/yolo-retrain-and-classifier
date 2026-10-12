#!/usr/bin/env bash
# 04_train.sh — ultralytics train + val on the dataset built by 03_to_yolo.py
#
# ---------------------------------------------------------------------------
# The .rknn question, once, because it comes up every time:
#
#   The shipped best_large_0208.rknn is an INFERENCE-ONLY ARTEFACT. It is an
#   int8 blob already compiled for the RK3588 NPU. You cannot fine-tune from
#   it: there are no weights in it, only quantised activations and a compiled
#   graph. `yolo detect train model=best_large_0208.rknn` does not exist.
#
#   To train a model of the SAME CLASS as the vendor's (22 classes, same order,
#   640x480, 9-output head, rk3588 int8), you start from the matching
#   ultralytics checkpoint — yolov8n.pt for the small variant, yolov8s.pt if
#   you have the VRAM and the latency headroom — and you reproduce the export
#   and conversion path in 05_export_rknn.sh.
#
#   The only exception: if you have the vendor's ORIGINAL PyTorch checkpoint
#   (the .pt they trained before compiling), that is a legitimate fine-tuning
#   start and is strictly better than a public COCO checkpoint, because the
#   backbone already knows the lawn, the lighting and the mower's deck height.
#   Supply it with --weights /path/to/vendor.pt and the rest of this pipeline is
#   unchanged. Without it, yolov8n.pt is the honest starting point.
# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
# AMD RX 7900 XTX (RDNA3, gfx1100) notes
#
# Device detection, best to worst:
#
#   0. CUDA PyTorch (NVIDIA) is detected too and used as device 0, AMP on.
#
#   1. ROCm PyTorch. Install the ROCm build of PyTorch, then:
#        python3 -c "import torch; print(torch.version.hip, torch.cuda.is_available())"
#      If `torch.version.hip` prints a version and `is_available()` is True,
#      ultralytics sees it as CUDA — `device=0` just works, no special flags.
#      pip install ultralytics is enough; ultralytics shells out to torch.
#
#   2. ROCm needs these, and forgetting them is the single most common
#      ROCm-on-RDNA3 failure:
#        * HSA_OVERRIDE_GFX_VERSION=11.0.0  — some ROCm builds ship kernels
#          compiled only up to gfx1030 and fail to load on gfx1100. This
#          variable forces the gfx11 code path. Export it before running.
#        * PYTORCH_HIP_ALLOC_CONF=expandable_segments:True — the stock caching
#          allocator fragmentation-errors on RDNA3 at batch > 8.
#        * The user must be in the `video` and `render` groups.
#          (RDNA3 does NOT need the old `amdgpu-pro` install.)
#
#   3. FP16 (AMP). RDNA3 has no native FP16 accumulate in the kernels ROCm
#      builds by default, so AMP frequently produces NaN losses on this card.
#      This script sets amp=False automatically when ROCm is detected unless
#      you pass --amp. If you see "loss is NaN", that is why.
#
#   4. CPU fallback. If torch sees no GPU, `device=cpu` trains correctly and
#      slowly. Expect roughly 20-40x the wall-clock of a 7900 XTX. On a 22
#      class, ~2k frame dataset at 640x480 that is hours per run, not minutes.
#      Use it for smoke tests (--epochs 1) and for confirming data.yaml loads.
#
# Batch size: 32 at 640x480 fits in 16 GB of VRAM with AMP off. Start there and
# drop to 16 if you hit OOM; ROCm OOM surfaces as an "out of memory" from the
# allocator, not a Python exception.
#
# ---------------------------------------------------------------------------
# imgsz: training is SQUARE. The ultralytics trainer runs
# check_imgsz(imgsz, max_dim=1), which turns any [h, w] into max(h, w) with a
# warning, so a tuple here is silently ignored. Train at imgsz=640 (default);
# YOLOv8 is fully convolutional, and the det_ros geometry — NCHW [1,3,640,480],
# 640 TALL x 480 wide — is fixed at EXPORT time by 05 (export.imgsz).
#
# Letterbox padding differs: ultralytics pads with grey (114) in training,
# det_ros (mower_rknn.preprocess.letterbox) pads with black (0) and feeds the
# 1920x1080 OA frames scaled to 480x270 with 185 px bands top and bottom. The
# model sees black bands in the field that it never saw in training; keep
# that in mind when reading field results (and see README "Preprocessing").
#
# Usage:
#   bash scripts/04_train.sh                              # config defaults
#   bash scripts/04_train.sh --epochs 300 --batch 16
#   bash scripts/04_train.sh --weights /path/to/vendor.pt
#   bash scripts/04_train.sh --device cpu --epochs 2      # smoke test
#   bash scripts/04_train.sh --resume runs/det_22cls/weights/last.pt
#   bash scripts/04_train.sh --dry-run                    # print, do not run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# Prefer the repo's own venv, so these steps work without
# `source .venv/bin/activate` first. PYTHON in the environment still wins.
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON=python3
  fi
fi

CONFIG="config/pipeline.yaml"
WEIGHTS=""
DATA=""
IMGSZ=""
EPOCHS=""
PATIENCE=""
BATCH=""
DEVICE=""
WORKERS=""
PROJECT=""
NAME=""
AMP=""
CACHE=""
RESUME=""
DRY_RUN=0
NO_VAL=0
EXTRA=()

usage() {
  # Reprint this file's own header comment block as the help text.
  sed -n '2,/^set -euo/{/^set -euo/d;p}' "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")" \
    | sed 's/^# \{0,1\}//' | grep -v '^$'
  cat <<'USAGE'

Options:
  --config FILE     pipeline.yaml (default: config/pipeline.yaml)
  --weights PT      starting checkpoint (default: train.weights, e.g. yolov8n.pt)
  --data YAML       dataset (default: train.data)
  --imgsz N         square train size (default: train.imgsz = 640; a "H,W"
                    tuple is coerced to max(H,W) by ultralytics anyway)
  --epochs N        (default: train.epochs)
  --patience N      early-stop patience
  --batch N         batch size
  --device DEV      auto | 0 | 0,1 | cpu
  --workers N       dataloader workers
  --project DIR     ultralytics project dir
  --name NAME       run name
  --amp MODE        auto | true | false
  --cache MODE      false | ram | disk
  --resume PT       resume an interrupted run from its weights/last.pt
  --no-val          skip the validation pass afterwards
  --dry-run         print the commands, run nothing
  -h, --help        this text
  --                pass the rest straight to `yolo detect train`
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)    CONFIG="$2"; shift 2 ;;
    --weights)   WEIGHTS="$2"; shift 2 ;;
    --data)      DATA="$2"; shift 2 ;;
    --imgsz)     IMGSZ="$2"; shift 2 ;;
    --epochs)    EPOCHS="$2"; shift 2 ;;
    --patience)  PATIENCE="$2"; shift 2 ;;
    --batch)     BATCH="$2"; shift 2 ;;
    --device)    DEVICE="$2"; shift 2 ;;
    --workers)   WORKERS="$2"; shift 2 ;;
    --project)   PROJECT="$2"; shift 2 ;;
    --name)      NAME="$2"; shift 2 ;;
    --amp)       AMP="$2"; shift 2 ;;
    --cache)     CACHE="$2"; shift 2 ;;
    --resume)    RESUME="$2"; shift 2 ;;
    --no-val)    NO_VAL=1; shift ;;
    --dry-run)   DRY_RUN=1; shift ;;
    --)          shift; EXTRA+=("$@"); break ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# --- read the config ------------------------------------------------------
command -v "$PYTHON" >/dev/null 2>&1 || { echo "ERROR: python not found ($PYTHON)" >&2; exit 1; }
"$PYTHON" -c "import yaml" 2>/dev/null || {
  echo "ERROR: pyyaml is required to read $CONFIG.  ./setup.sh, or: pip install pyyaml" >&2; exit 1; }

cfg() {  # cfg <dotted.key> [default]
  "$PYTHON" - "$CONFIG" "$1" "${2-}" <<'PY'
import sys, yaml
path, key, default = sys.argv[1], sys.argv[2], sys.argv[3]
node = yaml.safe_load(open(path)) or {}
for part in key.split('.'):
    if not isinstance(node, dict) or part not in node:
        print(default); raise SystemExit
    node = node[part]
if isinstance(node, bool):
    print('true' if node else 'false')
elif isinstance(node, (list, tuple)):
    print(','.join(str(v) for v in node))
elif node is None:
    print(default)
else:
    print(node)
PY
}

# --- environment detection ------------------------------------------------
DEVICE_MODE="${DEVICE:-$(cfg train.device auto)}"
AMP_MODE="${AMP:-$(cfg train.amp auto)}"

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"

echo "=== environment ==="
# _backend.py prints a human line, then "backend=<rocm|cuda|rocm-unavailable|cpu>".
# Availability is checked before torch.version.hip: a CUDA build has hip=None
# and must not be mistaken for "no GPU".
if TORCH_INFO=$("$PYTHON" "$SCRIPT_DIR/_backend.py" 2>/dev/null); then
  echo "$(sed '$d' <<<"$TORCH_INFO")"
  BACKEND="$(tail -n1 <<<"$TORCH_INFO" | sed 's/^backend=//')"
else
  echo "torch is not importable — the training step will not work until it is."
  BACKEND="missing"
fi
echo "backend: $BACKEND"

if [[ "$BACKEND" == "rocm" ]]; then
  # RDNA3 FP16 is the classic silent failure; default AMP off unless asked.
  if [[ "$AMP_MODE" == "auto" ]]; then
    AMP_MODE="false"
    echo "AMP: disabled automatically (RDNA3 FP16 commonly yields NaN losses)."
    echo "     Override with --amp true if your ROCm build handles it."
  fi
  case "$DEVICE_MODE" in
    auto) DEVICE_MODE="0" ;;
  esac
elif [[ "$BACKEND" == "cuda" ]]; then
  case "$DEVICE_MODE" in
    auto) DEVICE_MODE="0" ;;
  esac
  [[ "$AMP_MODE" == "auto" ]] && AMP_MODE=""   # ultralytics default (AMP on) is fine on CUDA
elif [[ "$BACKEND" == "cpu" ]]; then
  case "$DEVICE_MODE" in
    auto|0|0,1|"0, 1") DEVICE_MODE="cpu" ;;
  esac
  echo "note: torch sees no GPU. Training on CPU is correct but slow"
  echo "      (expect 20-40x the wall clock). Install a ROCm torch build to use"
  echo "      the RX 7900 XTX: https://rocm.docs.amd.com/install-on-linux"
elif [[ "$BACKEND" == "rocm-unavailable" || "$BACKEND" == "missing" ]]; then
  [[ "$DEVICE_MODE" == "auto" ]] && DEVICE_MODE="cpu"
fi
if [[ "$BACKEND" == "rocm-unavailable" ]]; then
  echo "WARNING: torch was built with HIP but reports no usable device." >&2
  echo "         Check: rocm-smi, your membership of the video/render groups," >&2
  echo "         and HSA_OVERRIDE_GFX_VERSION (currently $HSA_OVERRIDE_GFX_VERSION)." >&2
fi

# --- resolve training parameters -----------------------------------------
WEIGHTS="${WEIGHTS:-$(cfg train.weights yolov8n.pt)}"
DATA="${DATA:-$(cfg train.data work/dataset/data.yaml)}"
IMGSZ="${IMGSZ:-$(cfg train.imgsz 640)}"
if [[ "$IMGSZ" == *,* ]]; then
  echo "note: train imgsz '$IMGSZ' is a tuple; ultralytics trains square at max(h,w)."
  echo "      The 640x480 geometry is applied at export (05, export.imgsz)."
fi
EPOCHS="${EPOCHS:-$(cfg train.epochs 150)}"
PATIENCE="${PATIENCE:-$(cfg train.patience 25)}"
BATCH="${BATCH:-$(cfg train.batch 32)}"
WORKERS="${WORKERS:-$(cfg train.workers 8)}"
PROJECT="${PROJECT:-$(cfg train.project runs)}"
NAME="${NAME:-$(cfg train.name det_22cls)}"
CACHE="${CACHE:-$(cfg train.cache false)}"
AMP_ARG=""
[[ "$AMP_MODE" == "true" ]] && AMP_ARG="amp=True"
[[ "$AMP_MODE" == "false" ]] && AMP_ARG="amp=False"
VAL_AFTER="$(cfg train.val true)"
VAL_SPLIT="$(cfg train.val_split val)"
VAL_BATCH="$(cfg train.val_batch 16)"

# Refuse the obvious mistake loudly rather than training a model that cannot
# be deployed: a .rknn is a compiled artefact, not a checkpoint.
case "$WEIGHTS" in
  *.rknn)
    cat >&2 <<'MSG'
ERROR: --weights points at a .rknn file.

A .rknn is a compiled, quantized, inference-only blob for the Rockchip NPU.
There are no weights in it to fine-tune from. Use one of:

  --weights yolov8n.pt          the matching ultralytics architecture (small)
  --weights yolov8s.pt          the matching ultralytics architecture (large)
  --weights /path/to/vendor.pt  ONLY if you have the vendor's original
                                PyTorch checkpoint, not just their .rknn

If you have only the .rknn, it could in principle serve as a *teacher* for
pseudo-labelling, but this pipeline does not implement that (see README).
MSG
    exit 2 ;;
esac

if [[ -f "$DATA" ]]; then
  echo "data.yaml : $DATA"
else
  echo "ERROR: $DATA not found. Run scripts/03_to_yolo.py first." >&2; exit 1
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "=== dry run — nothing executed ==="
fi

# --- build the command ----------------------------------------------------
YOLO=(yolo detect train
      "model=$WEIGHTS"
      "data=$DATA"
      "imgsz=$IMGSZ"
      "epochs=$EPOCHS"
      "patience=$PATIENCE"
      "batch=$BATCH"
      "device=$DEVICE_MODE"
      "workers=$WORKERS"
      "project=$PROJECT"
      "name=$NAME")
[[ -n "$AMP_ARG" ]] && YOLO+=("$AMP_ARG")
[[ "$CACHE" != "false" ]] && YOLO+=("cache=$CACHE")
YOLO+=("${EXTRA[@]+"${EXTRA[@]}"}")

# Ultralytics ships a `yolo` console script; fall back to the library API when
# only the library is installed. Both are stable; the CLI is just less quoting.
# Resume takes no other settings: the run directory already holds them.
if [[ -n "$RESUME" ]]; then
  [[ -f "$RESUME" ]] || { echo "ERROR: --resume $RESUME not found" >&2; exit 1; }
  YOLO=(yolo detect train "resume=True" "model=$RESUME")
fi
# key=value arguments only (drop "yolo detect train") for _yolo_train.py.
YOLO_KV=("${YOLO[@]:3}")

if command -v yolo >/dev/null 2>&1; then
  RUN_CMD=("${YOLO[@]}")
elif "$PYTHON" -c "import ultralytics" 2>/dev/null; then
  RUN_CMD=("$PYTHON" "$REPO_ROOT/scripts/_yolo_train.py" "${YOLO_KV[@]}")
elif [[ "$DRY_RUN" -eq 1 ]]; then
  # A dry run is about inspecting the command, so report the canonical form and
  # the missing dependency rather than bailing out.
  RUN_CMD=("${YOLO[@]}")
  echo "note: ultralytics is not installed here; the command below is what would run." >&2
else
  echo "ERROR: ultralytics is not installed.  pip install ultralytics" >&2
  exit 1
fi

echo
echo "=== training ==="
echo "command   : ${RUN_CMD[*]}"
echo "weights   : $WEIGHTS"
echo "device    : $DEVICE_MODE (backend $BACKEND)"
echo "imgsz     : $IMGSZ   (square training; 640x480 portrait is set at export, 05)"
echo "project   : $PROJECT / $NAME"
echo
echo "This writes to $PROJECT/$NAME/weights/best.pt"
echo "Note: ultralytics is AGPL-3.0. Personal use on your own mower is fine;"
echo "      redistributing a model trained with it is a different question."
echo

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

mkdir -p "$PROJECT"
"${RUN_CMD[@]}"

BEST="$PROJECT/$NAME/weights/best.pt"
if [[ ! -f "$BEST" ]]; then
  # Ultralytics sometimes renames the dir when name is taken.
  BEST="$(ls -1dt "$PROJECT"/${NAME}*/weights/best.pt 2>/dev/null | head -1 || true)"
fi

echo
echo "=== done ==="
if [[ -z "$BEST" || ! -f "$BEST" ]]; then
  echo "WARNING: no best.pt found under $PROJECT/$NAME/weights/. Check the log above."
  exit 1
fi
echo "best.pt   : $BEST  ($(du -h "$BEST" | cut -f1))"
RUN_DIR="$(dirname "$(dirname "$BEST")")"
echo "metrics   : $RUN_DIR/results.csv"
echo "plots     : $RUN_DIR/results.png"

if [[ "$NO_VAL" -eq 0 && "$VAL_AFTER" == "true" ]]; then
  echo
  echo "=== validation pass ==="
  # `val` reports on the val split only, which is the number you actually care
  # about. mAP here is measured against LLM suggestions, so read it as "did the
  # model fit the suggestions", not "is the model correct". A human-labelled
  # holdout is the only real metric — see README.
  if command -v yolo >/dev/null 2>&1; then
    yolo detect val "model=$BEST" "data=$DATA" "split=$VAL_SPLIT" \
      "imgsz=$IMGSZ" "batch=$VAL_BATCH" "device=$DEVICE_MODE" \
      "project=$PROJECT" "name=${NAME}_val"
  else
    "$PYTHON" -c "
import sys
from ultralytics import YOLO
sz = [int(x) for x in sys.argv[4].split(',')]
YOLO(sys.argv[1]).val(data=sys.argv[2], split=sys.argv[3], imgsz=max(sz),
                      batch=int(sys.argv[5]), device=sys.argv[6])
" "$BEST" "$DATA" "$VAL_SPLIT" "$IMGSZ" "$VAL_BATCH" "$DEVICE_MODE"
  fi
fi

echo
echo "=== QA gate before this model touches the mower ==="
echo "  1. mAP50 on the val split vs the vendor baseline (README 'QA gate')"
echo "  2. per-class recall for person/dog/cat/hedgehog/rabbit, day AND night"
echo "  3. false positives per 1000 frames on clean lawn"
echo "  4. class names and order in data.yaml == the vendor list, exactly"
echo "  5. int8 vs fp32 box agreement after conversion (05_export_rknn.sh)"
echo
echo "next: bash scripts/05_export_rknn.sh --weights $BEST"
