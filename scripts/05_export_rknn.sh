#!/usr/bin/env bash
# 05_export_rknn.sh — export the trained .pt to ONNX, then convert to int8 .rknn
#
# Two steps, on two different machines potentially:
#
#   1. ONNX export. Runs anywhere with torch + a checkout of the airockchip
#      fork (github.com/airockchip/ultralytics_yolov8).
#   2. rknn-toolkit2 conversion. This is x86_64 Linux ONLY — it does not run on
#      the RK3588 (that machine runs the *runtime*, rknn-toolkit-lite2, not the
#      compiler). Re-run on the x86 host with --onnx-in <the .onnx> to do just
#      this step.
#
# ---------------------------------------------------------------------------
# The contract this has to satisfy (ROS 2 det_ros, best_large_0208.rknn)
#
#   input   images : [1, 3, 640, 480] NCHW in the model metadata — 640 TALL,
#                    480 wide. Metadata dtype float32, range [0,1] via
#                    mean [0,0,0] / std [255,255,255], rgb2bgr False.
#                    At runtime det_ros feeds uint8 NHWC RGB (letterboxed, pad
#                    0, no normalisation) and librknnrt does the conversion.
#   outputs        : 9 tensors = 3 strides x (box [1,64,h,w] DFL reg_max 16,
#                    cls [1,22,h,w] sigmoid in-graph, score_sum [1,1,h,w])
#                    at h x w = 80x60, 40x30, 20x15
#   quantisation   : asymmetric int8, per-LAYER, target rk3588, toolkit 2.3.0
#
# How it is produced: the airockchip fork's exporter with format='rknn',
#     PYTHONPATH=<fork> python -c "from ultralytics import YOLO;
#         YOLO('<abs best.pt>').export(format='rknn', imgsz=[640,480], opset=12)"
# which writes <best>.onnx next to the checkpoint (the fork's RKOPT_README
# describes the same exporter, run as ultralytics/engine/exporter.py). Stock
# ultralytics (--exporter ultralytics) gives ONE [1,26,6300] output and fails
# the ONNX check below — it is kept only for comparison.
#
# ---------------------------------------------------------------------------
# Version pinning (check this BEFORE converting anything)
#
# The vendor tree ships two librknnrt copies (2.1.0 and 2.2.0) and the shipped
# models were compiled with rknn-toolkit2 2.3.0. Compile with a toolkit the
# runtime will accept. On the mower:
#
#   ldd <det_ros binary or python env> | grep rknnrt
#   strings <that librknnrt.so> | grep -m1 'librknnrt version'
#
# If the installed rknn-toolkit2 differs from --rknn-toolkit, this script
# stops; --reinstall-toolkit pip-installs exactly that version first.
#
# ---------------------------------------------------------------------------
# Calibration: frames from these cameras, weighted toward the OA cameras det_ros
# actually reads (export.rknn.calib_camera_weights, by manifest camera), each
# pre-letterboxed to 480x640 with black padding exactly like
# mower_rknn.preprocess.letterbox, so the int8 ranges see the black bands.
#
# ---------------------------------------------------------------------------
# Usage:
#   bash scripts/05_export_rknn.sh                       # config defaults
#   bash scripts/05_export_rknn.sh --weights runs/det_22cls/weights/best.pt
#   bash scripts/05_export_rknn.sh --dtype fp             # no quantisation, for A/B
#   bash scripts/05_export_rknn.sh --quant-method channel # per-channel A/B
#   bash scripts/05_export_rknn.sh --skip-rknn            # ONNX only
#   bash scripts/05_export_rknn.sh --onnx-in work/models/best_22cls.onnx  # convert only
#   bash scripts/05_export_rknn.sh --verify-only /path/to.rknn
#   bash scripts/05_export_rknn.sh --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ORIG_PWD="$PWD"
cd "$REPO_ROOT"

CONFIG="config/pipeline.yaml"
WEIGHTS=""
ONNX_OUT=""
ONNX_IN=""
OPSET=""
SIMPLIFY=""
HALF=""
DYNAMIC=""
EXPORTER=""
AIROCKCHIP_REPO=""
TARGET_PLATFORM=""
DTYPE=""
TOOLKIT_VER=""
REINSTALL_TOOLKIT=0
QUANT_METHOD=""
CALIB_COUNT=""
CALIB_FROM=""
RKNN_OUT=""
MEAN=""
STD=""
VERIFY=1
SKIP_RKNN=0
VERIFY_ONLY=""
DRY_RUN=0
EXTRA=()

usage() {
  # Reprint this file's own header comment block as the help text.
  sed -n '2,/^set -euo/{/^set -euo/d;p}' "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")" \
    | sed 's/^# \{0,1\}//' | grep -v '^$'
  cat <<'USAGE'

Options (paths given here are relative to your current directory; paths in
the config are relative to the repository root):
  --config FILE           pipeline.yaml (default: config/pipeline.yaml)
  --weights PT            trained checkpoint (default: export.weights)
  --onnx-out PATH         output ONNX path (default: paths.models_dir/onnx_name)
  --onnx-in PATH          skip the export; convert this existing ONNX
  --opset N               ONNX opset for --exporter ultralytics (the fork's rknn
                          path always uses 12)
  --simplify BOOL         onnx-simplifier, --exporter ultralytics only
  --half BOOL             fp16 export, --exporter ultralytics only
  --dynamic BOOL          dynamic axes, --exporter ultralytics only
  --exporter NAME         airockchip | ultralytics (default: export.exporter)
  --airockchip-repo DIR   checkout of airockchip/ultralytics_yolov8
  --target-platform NAME  rk3588 (etc.)
  --dtype NAME            i8 | fp
  --quant-method NAME     layer | channel (default: export.rknn.quantized_method)
  --rknn-toolkit VER      required rknn-toolkit2 version (default: 2.3.0)
  --reinstall-toolkit     pip-install exactly that version if another is installed
  --calib-count N         calibration images (default: 300)
  --calib-from NAME       dataset | frames
  --mean "a,b,c"          mean values (default: 0,0,0 — do not change casually)
  --std "a,b,c"           std values (default: 255,255,255 — same)
  --rknn-out PATH         output .rknn path
  --skip-rknn             export (and check) the ONNX only; no toolkit needed
  --no-verify             skip the ONNX contract check and the .rknn self-check
  --verify-only PATH      run the metadata self-check on an existing .rknn and exit
  --dry-run               print what would run
  --                      pass the rest to the exporter (key=value)
USAGE
}

# A path from the command line: absolute, or relative to where the user ran us.
cli_path() { [[ "$1" = /* ]] && printf '%s\n' "$1" || printf '%s\n' "$ORIG_PWD/$1"; }
# A path from the config: absolute, or relative to the repository root.
cfg_abs()  { [[ "$1" = /* ]] && printf '%s\n' "$1" || printf '%s\n' "$REPO_ROOT/$1"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)            CONFIG="$(cli_path "$2")"; shift 2 ;;
    --weights)           WEIGHTS="$(cli_path "$2")"; shift 2 ;;
    --onnx-out)          ONNX_OUT="$(cli_path "$2")"; shift 2 ;;
    --onnx-in)           ONNX_IN="$(cli_path "$2")"; shift 2 ;;
    --opset)             OPSET="$2"; shift 2 ;;
    --simplify)          SIMPLIFY="$2"; shift 2 ;;
    --half)              HALF="$2"; shift 2 ;;
    --dynamic)           DYNAMIC="$2"; shift 2 ;;
    --exporter)          EXPORTER="$2"; shift 2 ;;
    --airockchip-repo)   AIROCKCHIP_REPO="$(cli_path "$2")"; shift 2 ;;
    --target-platform)   TARGET_PLATFORM="$2"; shift 2 ;;
    --dtype)             DTYPE="$2"; shift 2 ;;
    --quant-method)      QUANT_METHOD="$2"; shift 2 ;;
    --rknn-toolkit)      TOOLKIT_VER="$2"; shift 2 ;;
    --reinstall-toolkit) REINSTALL_TOOLKIT=1; shift ;;
    --calib-count)       CALIB_COUNT="$2"; shift 2 ;;
    --calib-from)        CALIB_FROM="$2"; shift 2 ;;
    --mean)              MEAN="$2"; shift 2 ;;
    --std)               STD="$2"; shift 2 ;;
    --rknn-out)          RKNN_OUT="$(cli_path "$2")"; shift 2 ;;
    --skip-rknn)         SKIP_RKNN=1; shift ;;
    --verify-only)       VERIFY_ONLY="$(cli_path "$2")"; shift 2 ;;
    --no-verify)         VERIFY=0; shift ;;
    --dry-run)           DRY_RUN=1; shift ;;
    --)                  shift; EXTRA+=("$@"); break ;;
    -h|--help)           usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# Prefer the repo's own venvs so 05 works without activating one first:
# .venv-export (CPU torch 2.2 + rknn-toolkit2 2.3.0, which cannot share a venv
# with the training torch), then .venv. PYTHON in the environment still wins.
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$REPO_ROOT/.venv-export/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv-export/bin/python"
  elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON=python3
  fi
fi
command -v "$PYTHON" >/dev/null 2>&1 || { echo "ERROR: python not found ($PYTHON)" >&2; exit 1; }
"$PYTHON" -c "import yaml" 2>/dev/null || {
  echo "ERROR: pyyaml required.  ./setup.sh, or: pip install pyyaml" >&2; exit 1; }

cfg() {
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
elif isinstance(node, dict):
    import json
    print(json.dumps(node))
elif isinstance(node, (list, tuple)):
    # Flatten [[0,0,0]] / [0,0,0] to the comma form the toolkit API takes.
    flat = []
    for v in node:
        if isinstance(v, (list, tuple)):
            flat.extend(str(x) for x in v)
        else:
            flat.append(str(v))
    print(','.join(flat))
elif node is None:
    print(default)
else:
    print(node)
PY
}

# ---------------------------------------------------------------------------
# verify-only: the output contract self-check
# ---------------------------------------------------------------------------
verify_rknn() {
  local model="$1"
  if [[ ! -f "$model" ]]; then
    echo "ERROR: $model not found" >&2; return 1
  fi
  echo "=== output contract self-check: $model ==="
  echo
  # Parsed with Python, not grep. The metadata is a Python-repr dict on one
  # long line, and its tensor names are not uniformly numeric — the score
  # channels are named 'onnx::ReduceSum_968' — so a numeric-name grep silently
  # finds only 6 of the 9 outputs and reports a false mismatch.
  "$PYTHON" - "$model" <<'PY'
import ast
import re
import subprocess
import sys

path = sys.argv[1]
try:
    blob = subprocess.run(["strings", "-a", path], capture_output=True,
                          text=True, check=True).stdout
except (OSError, subprocess.CalledProcessError) as exc:
    sys.exit(f"  ERROR: cannot read strings from {path}: {exc}")

# The attrs block holds every input/output tensor description.
line = next((ln for ln in blob.splitlines()
             if ln.startswith("{'attrs'") and "'is_output'" in ln), None)
if line is None:
    print("  ERROR: no rknn v2 'attrs' metadata found in this file.")
    print("         It may not be an rknn v2 container at all.")
    sys.exit(1)

# Take the whole line; the repr is valid Python as far as the tensor dict goes.
start = line.index("{'attrs'")
depth, in_str, esc = 0, False, False
for i in range(start, len(line)):
    ch = line[i]
    if in_str:
        if esc:
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == "'":
            in_str = False
        continue
    if ch == "'":
        in_str = True
    elif ch == "{":
        depth += 1
    elif ch == "}":
        depth -= 1
        if depth == 0:
            blob_attrs = line[start:i + 1]
            break
else:
    sys.exit("  ERROR: could not delimit the attrs dictionary.")

try:
    attrs = ast.literal_eval(blob_attrs)["attrs"]
except (SyntaxError, ValueError, KeyError) as exc:
    sys.exit(f"  ERROR: could not parse the attrs dictionary: {exc}")

ins = {k: v for k, v in attrs.items() if not v.get("is_output")}
outs = {k: v for k, v in attrs.items() if v.get("is_output")}

fail = 0
print("-- input tensor (expect 1 x 3 x 640 x 480) --")
for name, meta in ins.items():
    print(f"  {name}: shape={meta.get('shape')} layout={meta.get('layout')} "
          f"dtype={meta.get('dtype')} range={meta.get('range')} "
          f"mean={meta.get('mean')} std={meta.get('std')}")
    shape = list(meta.get("shape") or [])
    if shape[:4] != [1, 3, 640, 480]:
        print(f"  FAIL: input shape is {shape}, expected [1, 3, 640, 480].")
        print("        640 is the HEIGHT and 480 the WIDTH (portrait canvas from a")
        print("        landscape stream). A 640x640 export will not load in det_ros.")
        fail = 1
    if list(meta.get("mean") or []) != [0, 0, 0] or \
            list(meta.get("std") or []) != [255, 255, 255]:
        print(f"  WARN: mean/std is {meta.get('mean')}/{meta.get('std')}, the shipped "
              "models use [0,0,0]/[255,255,255].")
        print("        If det_ros does not compensate, every box is subtly wrong.")

print()
print("-- output tensors (expect 9: 80x60, 40x30, 20x15 x boxes/scores/sum) --")
by_idx = sorted(outs.items(), key=lambda kv: kv[1].get("idx", 0))
for name, meta in by_idx:
    print(f"  [{meta.get('idx')}] {name:<22} shape={meta.get('shape')}")

n = len(outs)
if n != 9:
    print(f"  FAIL: {n} output tensors, expected 9.")
    print("        det_ros's post-processing is written against the Rockchip")
    print("        3-scale x (boxes, scores, score-sum) head. Re-export with:")
    print("          --exporter airockchip")
    print("        (https://github.com/airockchip/ultralytics_yolov8)")
    fail = 1
else:
    shapes = [list(m.get("shape") or []) for _n, m in by_idx]
    expect = {
        0: [1, 64, 80, 60], 1: [1, 22, 80, 60], 2: [1, 1, 80, 60],
        3: [1, 64, 40, 30], 4: [1, 22, 40, 30], 5: [1, 1, 40, 30],
        6: [1, 64, 20, 15], 7: [1, 22, 20, 15], 8: [1, 1, 20, 15],
    }
    for i, want in expect.items():
        got = shapes[i] if i < len(shapes) else None
        if got != want:
            print(f"  FAIL: output {i} is {got}, expected {want}.")
            print("        This is the 22-CLASS (det_ros large) contract. A smaller")
            print("        model legitimately has fewer score channels — the vendor's")
            print("        best_small_0208.rknn has 1. If you are targeting the small")
            print("        model, expect [1,1,...] there instead.")
            fail = 1
            break
    else:
        print("  OK: 9 outputs, three scales, 22 score channels at the largest.")
        print("      Matches the shipped best_large_0208.rknn layout.")

print()
print("-- compiler version --")
for ln in blob.splitlines():
    if "compiler version" in ln:
        print(f"  {ln.strip()}")
        break
else:
    print("  (not recorded in this file)")

print()
if fail:
    print("  RESULT: contract NOT met. If this is a model you just built, do not")
    print("          deploy it — the export path is wrong, and the fix is upstream,")
    print("          not a tuning problem.")
    print("          If this is a shipped vendor model you are only inspecting, the")
    print("          mismatch may just mean it is a different model (different class")
    print("          count or task) than the 22-class det_ros detector.")
else:
    print("  RESULT: contract met. Still compare against the shipped model:")
    print("          bash scripts/05_export_rknn.sh --verify-only <vendor>.rknn")
sys.exit(fail)
PY
}

if [[ -n "$VERIFY_ONLY" ]]; then
  rc=0
  verify_rknn "$VERIFY_ONLY" || rc=$?
  exit "$rc"
fi

# ---------------------------------------------------------------------------
# resolve parameters
# ---------------------------------------------------------------------------
WEIGHTS="${WEIGHTS:-$(cfg_abs "$(cfg export.weights runs/det_22cls/weights/best.pt)")}"
MODELS_DIR="$(cfg_abs "$(cfg paths.models_dir work/models)")"
ONNX_OUT="${ONNX_OUT:-${MODELS_DIR}/$(cfg export.onnx_name best_22cls.onnx)}"
OPSET="${OPSET:-$(cfg export.opset 12)}"
SIMPLIFY="${SIMPLIFY:-$(cfg export.simplify true)}"
HALF="${HALF:-$(cfg export.half false)}"
DYNAMIC="${DYNAMIC:-$(cfg export.dynamic false)}"
EXPORTER="${EXPORTER:-$(cfg export.exporter airockchip)}"
AIROCKCHIP_REPO="${AIROCKCHIP_REPO:-$(cfg_abs "$(cfg export.airockchip_repo ../ultralytics_yolov8)")}"
TARGET_PLATFORM="${TARGET_PLATFORM:-$(cfg export.rknn.target_platform rk3588)}"
DTYPE="${DTYPE:-$(cfg export.rknn.dtype i8)}"
TOOLKIT_VER="${TOOLKIT_VER:-$(cfg export.rknn.toolkit_version 2.3.0)}"
QUANT_DTYPE="$(cfg export.rknn.quantized_dtype w8a8)"
QUANT_METHOD="${QUANT_METHOD:-$(cfg export.rknn.quantized_method layer)}"
QUANT_ALGO="$(cfg export.rknn.quantized_algorithm normal)"
CALIB_COUNT="${CALIB_COUNT:-$(cfg export.rknn.calib_count 300)}"
CALIB_FROM="${CALIB_FROM:-$(cfg export.rknn.calib_from dataset)}"
CALIB_WEIGHTS="$(cfg export.rknn.calib_camera_weights '{}')"
MEAN="${MEAN:-$(cfg export.rknn.mean_values 0,0,0)}"
STD="${STD:-$(cfg export.rknn.std_values 255,255,255)}"
RKNN_OUT="${RKNN_OUT:-${MODELS_DIR}/$(cfg export.rknn.output_name best_22cls.rknn)}"
DATASET_DIR="$(cfg_abs "$(cfg convert.dataset_dir work/dataset)")"
FRAMES_DIR="$(cfg_abs "$(cfg paths.frames_dir work/frames)")"
NC="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
sys.path.insert(0, "scripts")
from common import class_names, load_config
print(len(class_names(load_config(sys.argv[1]))))
PY
)"

# export.imgsz is [height, width] of the exported graph (det_ros: 640 x 480).
IMGSZ="$(cfg export.imgsz '640,480')"
IFS=',' read -r IMG_H IMG_W <<< "$IMGSZ"
if [[ -z "${IMG_W:-}" ]]; then
  echo "ERROR: export.imgsz must be [height, width], got '$IMGSZ'" >&2; exit 2
fi

case "$QUANT_METHOD" in layer|channel) ;; *)
  echo "ERROR: --quant-method must be layer or channel, got '$QUANT_METHOD'" >&2; exit 2 ;;
esac

if [[ -n "$ONNX_IN" ]]; then
  [[ -f "$ONNX_IN" ]] || { echo "ERROR: --onnx-in $ONNX_IN not found" >&2; exit 1; }
  ONNX_OUT="$ONNX_IN"
else
  if [[ "$WEIGHTS" == *.rknn ]]; then
    cat >&2 <<'MSG'
ERROR: --weights points at a .rknn.

There is nothing to export FROM a .rknn: it is already compiled, quantized for
the NPU, and inference-only. Export needs a PyTorch checkpoint.

  --weights runs/<name>/weights/best.pt     what 04_train.sh produced
  --weights /path/to/vendor.pt              the vendor's original .pt, if you have it

If all you have is the shipped .rknn, you cannot re-export or fine-tune it.
MSG
    exit 2
  fi
  if [[ "$WEIGHTS" == *.onnx ]]; then
    echo "ERROR: --weights is an .onnx; use --onnx-in to convert an existing ONNX." >&2
    exit 2
  fi
  if [[ ! -f "$WEIGHTS" ]]; then
    echo "ERROR: weights not found: $WEIGHTS" >&2
    echo "       Run scripts/04_train.sh first." >&2
    exit 1
  fi
  WEIGHTS="$(realpath "$WEIGHTS")"
fi

echo "=== parameters ==="
if [[ -n "$ONNX_IN" ]]; then
  echo "onnx in   : $ONNX_IN  (export skipped)"
else
  echo "weights   : $WEIGHTS"
  echo "onnx      : $ONNX_OUT"
  echo "exporter  : $EXPORTER"
fi
echo "imgsz     : ${IMG_H}x${IMG_W} (h x w — det_ros input [1,3,640,480])"
echo "rknn      : $RKNN_OUT  platform=$TARGET_PLATFORM dtype=$DTYPE"
echo "quant     : $QUANT_DTYPE  method=$QUANT_METHOD  algorithm=$QUANT_ALGO"
echo "calib     : $CALIB_COUNT frames from $CALIB_FROM, camera weights $CALIB_WEIGHTS"
echo "mean/std  : $MEAN / $STD"
echo "classes   : $NC"
echo

# ---------------------------------------------------------------------------
# step 1: ONNX export
# ---------------------------------------------------------------------------
export_onnx() {
  # Both exporters write <weights stem>.onnx next to the checkpoint. Remove a
  # stale one first so an export that silently failed cannot pass as success.
  local produced="${WEIGHTS%.*}.onnx"
  rm -f "$produced"
  case "$EXPORTER" in
    airockchip)
      local repo="$AIROCKCHIP_REPO"
      if [[ ! -d "$repo/ultralytics/engine" ]]; then
        echo "ERROR: airockchip fork not found at $repo" >&2
        echo "  git clone https://github.com/airockchip/ultralytics_yolov8 $repo" >&2
        echo "  (or pass --airockchip-repo DIR / set export.airockchip_repo)" >&2
        exit 1
      fi
      repo="$(realpath "$repo")"
      echo "=== exporting via airockchip fork (9-output head) ==="
      echo "fork      : $repo"
      echo "call      : YOLO('$WEIGHTS').export(format='rknn', imgsz=[$IMG_H, $IMG_W], opset=12)"
      echo
      # The fork must shadow any pip-installed ultralytics: PYTHONPATH first.
      # Its export_rknn() hard-codes opset 12 and input name 'images'.
      (
        cd "$repo"
        PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - \
          "$WEIGHTS" "$IMG_H" "$IMG_W" "${EXTRA[@]+"${EXTRA[@]}"}" <<'PY'
import sys
import ultralytics
from ultralytics import YOLO
from ultralytics.engine.exporter import Exporter

if not hasattr(Exporter, "export_rknn"):
    sys.exit(f"ERROR: {ultralytics.__file__} is not the airockchip fork (no "
             "Exporter.export_rknn). Check --airockchip-repo.")
print(f"ultralytics {ultralytics.__version__} from {ultralytics.__file__}")
weights, h, w = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
import yaml  # cast "key=value" extras the way the yolo CLI would (True, 12, ...)
extra = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in sys.argv[4:])}
YOLO(weights).export(format="rknn", imgsz=[h, w], opset=12, **extra)
PY
      )
      ;;
    ultralytics)
      echo "=== exporting via stock ultralytics (single decoded head) ==="
      echo "NOTE: this does NOT produce the 9-output head det_ros expects; the"
      echo "      ONNX check below will stop here. Comparison use only."
      echo
      local args=("model=$WEIGHTS" "format=onnx" "imgsz=$IMG_H,$IMG_W" "opset=$OPSET")
      [[ "$SIMPLIFY" == "true" ]] && args+=("simplify=True")
      [[ "$HALF" == "true" ]] && args+=("half=True")
      [[ "$DYNAMIC" == "true" ]] && args+=("dynamic=True")
      args+=("${EXTRA[@]+"${EXTRA[@]}"}")
      if "$PYTHON" -c "import ultralytics" 2>/dev/null; then
        "$PYTHON" "$REPO_ROOT/scripts/_yolo_export.py" "${args[@]}"
      else
        echo "ERROR: ultralytics is not installed.  pip install ultralytics" >&2
        exit 1
      fi
      ;;
    *)
      echo "ERROR: unknown --exporter '$EXPORTER' (airockchip | ultralytics)" >&2
      exit 2 ;;
  esac
  if [[ ! -f "$produced" ]]; then
    echo "ERROR: export finished but $produced is not there; check the log above." >&2
    exit 1
  fi
  if [[ "$(realpath "$produced")" != "$(realpath -m "$ONNX_OUT")" ]]; then
    mv -f "$produced" "$ONNX_OUT"
    echo "moved $produced -> $ONNX_OUT"
  fi
}

check_onnx() {
  # 0 = contract met, 1 = violated, 3 = could not check (no onnx package).
  echo "=== ONNX contract check: $ONNX_OUT ==="
  local rc=0
  "$PYTHON" "$REPO_ROOT/scripts/_onnx_contract.py" "$ONNX_OUT" \
    --nc "$NC" --imgsz "$IMG_H,$IMG_W" || rc=$?
  return "$rc"
}

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "=== dry run ==="
  if [[ -z "$ONNX_IN" ]]; then
    echo "would export $WEIGHTS -> $ONNX_OUT via $EXPORTER (imgsz=[$IMG_H,$IMG_W])"
  fi
  echo "would check the ONNX: input [1,3,$IMG_H,$IMG_W], 9 outputs, $NC classes"
  if [[ "$SKIP_RKNN" -eq 0 ]]; then
    echo "would letterbox $CALIB_COUNT calibration frames to ${IMG_W}x${IMG_H} (pad 0)"
    echo "would run rknn-toolkit2 $TOOLKIT_VER to build $RKNN_OUT ($DTYPE, $QUANT_METHOD)"
  fi
  exit 0
fi

mkdir -p "$(dirname "$ONNX_OUT")" "$(dirname "$RKNN_OUT")"
if [[ -z "$ONNX_IN" ]]; then
  export_onnx
fi
echo "onnx ready : $ONNX_OUT ($(du -h "$ONNX_OUT" | cut -f1))"
echo

ONNX_CHECKED=0
if [[ "$VERIFY" -eq 1 ]]; then
  rc=0; check_onnx || rc=$?
  if [[ "$rc" -eq 1 ]]; then
    echo >&2
    echo "The ONNX does not match the det_ros contract. Not converting." >&2
    echo "Re-export with --exporter airockchip (or pass --no-verify to override)." >&2
    exit 1
  elif [[ "$rc" -eq 0 ]]; then
    ONNX_CHECKED=1
  fi
  echo
fi

if [[ "$SKIP_RKNN" -eq 1 ]]; then
  echo "=== ONNX only (--skip-rknn) ==="
  echo "Convert on an x86_64 host with:"
  echo "  bash scripts/05_export_rknn.sh --onnx-in $ONNX_OUT"
  exit 0
fi

# ---------------------------------------------------------------------------
# step 2: rknn-toolkit2 (x86_64 only), exact version
# ---------------------------------------------------------------------------
echo "=== rknn-toolkit2 ==="
ARCH="$(uname -m)"
if [[ "$ARCH" != "x86_64" ]]; then
  cat >&2 <<MSG
WARNING: rknn-toolkit2 is an x86_64-only compiler and this host is $ARCH.

  The RK3588 (and any other ARM box) runs the RUNTIME (rknn-toolkit-lite2 /
  librknnrt), not the compiler. Copy $ONNX_OUT to an x86_64 Linux host (or
  container) with rknn-toolkit2==$TOOLKIT_VER and run:

    bash scripts/05_export_rknn.sh --onnx-in <that .onnx>
MSG
  echo >&2
  read -r -p "Continue anyway and try? [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]] || exit 1
fi

toolkit_version() {
  "$PYTHON" -c "from importlib.metadata import version; print(version('rknn-toolkit2'))" \
    2>/dev/null || echo none
}
INSTALLED_RKNN="$(toolkit_version)"
if [[ "$INSTALLED_RKNN" == "none" || ( "$INSTALLED_RKNN" != "$TOOLKIT_VER" && "$REINSTALL_TOOLKIT" -eq 1 ) ]]; then
  echo "installing rknn-toolkit2==$TOOLKIT_VER (installed: $INSTALLED_RKNN) ..."
  "$PYTHON" -m pip install --force-reinstall "rknn-toolkit2==$TOOLKIT_VER" || {
    echo "ERROR: could not install rknn-toolkit2==$TOOLKIT_VER on this host." >&2
    exit 1; }
  INSTALLED_RKNN="$(toolkit_version)"
fi
if [[ "$INSTALLED_RKNN" != "$TOOLKIT_VER" ]]; then
  cat >&2 <<MSG
ERROR: rknn-toolkit2 $INSTALLED_RKNN is installed, but $TOOLKIT_VER is required.

  A model compiled by a different toolkit may fail rknn_init on the mower, or
  load and behave differently. Either:
    --reinstall-toolkit                 pip-install exactly $TOOLKIT_VER here
    --rknn-toolkit $INSTALLED_RKNN       if you have checked the mower's librknnrt
                                        accepts models from that version
MSG
  exit 1
fi
echo "toolkit   : rknn-toolkit2 $INSTALLED_RKNN"

if [[ "$VERIFY" -eq 1 && "$ONNX_CHECKED" -eq 0 ]]; then
  # rknn-toolkit2 depends on onnx, so the check can run now if it could not before.
  rc=0; check_onnx || rc=$?
  if [[ "$rc" -ne 0 ]]; then
    echo "ERROR: ONNX contract check failed or could not run (rc=$rc). Not converting." >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# step 3: calibration set (letterboxed like det_ros), then conversion
# ---------------------------------------------------------------------------
CALIB_LIST="$(dirname "$RKNN_OUT")/calib_list.txt"
CALIB_DIR=""
if [[ "$DTYPE" != "fp" ]]; then
  CALIB_DIR="$(mktemp -d "$(dirname "$RKNN_OUT")/calib.XXXXXX")"
  trap 'rm -rf "$CALIB_DIR"' EXIT
  echo
  "$PYTHON" "$REPO_ROOT/scripts/_calib.py" \
    --source "$CALIB_FROM" --dataset-dir "$DATASET_DIR" --frames-dir "$FRAMES_DIR" \
    --count "$CALIB_COUNT" --weights "$CALIB_WEIGHTS" --size "$IMG_W,$IMG_H" \
    --out-dir "$CALIB_DIR" --list "$CALIB_LIST"
fi

echo
echo "=== rknn-toolkit2 conversion ==="
DO_QUANT="True"
[[ "$DTYPE" == "fp" ]] && DO_QUANT="False"

"$PYTHON" - "$ONNX_OUT" "$RKNN_OUT" "$TARGET_PLATFORM" "$DO_QUANT" \
            "$CALIB_LIST" "$MEAN" "$STD" "$QUANT_DTYPE" "$QUANT_METHOD" "$QUANT_ALGO" <<'PY'
"""Thin rknn-toolkit2 wrapper.

The model zoo's convert.py hard-codes its calibration path and mean/std, so
running it unmodified would calibrate on COCO with the wrong preprocessing.
Quantisation settings are passed explicitly: the shipped model is asymmetric
int8 per-LAYER, while the toolkit default is per-channel.
"""
import sys
from rknn.api import RKNN

(onnx_path, rknn_path, platform, do_quant, calib, mean_csv, std_csv,
 q_dtype, q_method, q_algo) = sys.argv[1:11]
quant = do_quant.lower() in ("true", "1", "yes")

def triplet(text):
    return [[float(v) for v in text.split(",")]]

rknn = RKNN(verbose=True)
print("--> config")
ret = rknn.config(mean_values=triplet(mean_csv), std_values=triplet(std_csv),
                  target_platform=platform, quantized_dtype=q_dtype,
                  quantized_method=q_method, quantized_algorithm=q_algo)
if ret != 0:
    sys.exit("config() failed")

print("--> load_onnx")
ret = rknn.load_onnx(model=onnx_path)
if ret != 0:
    sys.exit("load_onnx() failed")

print("--> build")
kwargs = {"do_quantization": quant}
if quant:
    kwargs["dataset"] = calib
ret = rknn.build(**kwargs)
if ret != 0:
    sys.exit("build() failed")

print("--> export_rknn")
ret = rknn.export_rknn(rknn_path)
if ret != 0:
    sys.exit("export_rknn() failed")
rknn.release()
print("wrote", rknn_path)
PY

echo
echo "=== done ==="
echo "onnx  : $ONNX_OUT ($(du -h "$ONNX_OUT" | cut -f1))"
echo "rknn  : $RKNN_OUT ($(du -h "$RKNN_OUT" | cut -f1))"
[[ -n "$CALIB_DIR" ]] && echo "calib : $CALIB_LIST (letterboxed copies removed on exit)"

if [[ "$VERIFY" -eq 1 ]]; then
  echo
  rc=0
  verify_rknn "$RKNN_OUT" || rc=$?
  if [[ "$rc" -ne 0 ]]; then
    echo
    echo "The metadata check did not pass (exit $rc)." >&2
    echo "Do NOT deploy this model. Fix the export path first." >&2
    exit 1
  fi
fi

cat <<'NEXT'

=== next: QA before the mower ===
  1. Compare the block above against the vendor's best_large_0208.rknn field by
     field. Input 1x3x640x480, nine outputs at 80x60 / 40x30 / 20x15, 22 score
     channels. Any deviation is an export bug, not a tuning problem.
  2. int8 vs fp agreement on the val set: >= 0.97 IoU>0.5 box match, mean
     score delta <= 0.05. Quantisation drift is the usual silent regression.
     Build a --dtype fp model to compare against, and a --quant-method channel
     model if per-layer int8 loses accuracy.
  3. On the mower, run the new .rknn through the same path det_ros uses, from a
     scratch location, before touching the deployed model. Check:
       cat /sys/kernel/debug/rknpu/load        (needs debugfs)
  4. Keep the original: cp the vendor .rknn to /userdata before any swap, and
     write the rollback as a script. An OTA replaces /workspace, so the only
     copy that survives is the one on /userdata.
NEXT
