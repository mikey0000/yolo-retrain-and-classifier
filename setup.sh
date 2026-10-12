#!/usr/bin/env bash
# setup.sh — one command to get this repo ready to run (Linux, macOS, WSL).
#
#   ./setup.sh                 venv + everything for 01..04 + airockchip fork
#   ./setup.sh --minimal       runtime only (01-03): pyyaml, requests, pillow
#   ./setup.sh --rknn          also install rknn-toolkit2 (05; x86_64 only)
#   ./setup.sh --check         report what is installed, change nothing
#   ./setup.sh --help          all options
#
# It uses uv when uv is on your PATH (interpreter + installs), and falls back
# to python3 -m venv + pip when it is not. Idempotent: run it again after a
# git pull. Everything it installs lives in .venv/ (gitignored) and
# ../ultralytics_yolov8 (a sibling checkout, per config/pipeline.yaml
# export.airockchip_repo).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MINIMAL=0
WANT_RKNN=0
WANT_FORK=1
WANT_TESTS=1
CHECK_ONLY=0
FORK_DIR=""
PYTHON_MIN=(3 10)

usage() {
  sed -n '2,/^set -euo/{/^set -euo/d;p}' "$REPO_ROOT/setup.sh" \
    | sed 's/^# \{0,1\}//' | grep -v '^$'
  cat <<'USAGE'

Options:
  --minimal        only pyyaml/requests/pillow (01-03); no torch, no fork
  --rknn           also install rknn-toolkit2==<export.rknn.toolkit_version>
                   (the .rknn compiler; x86_64 Linux only, including x86 WSL)
  --no-fork        do not clone airockchip/ultralytics_yolov8 (05 export)
  --no-tests       do not run tests/ at the end
  --fork-dir DIR   where to clone the airockchip fork
                   (default: export.airockchip_repo from config/pipeline.yaml)
  --check          diagnose an existing install and exit
  -h, --help       this text

Environment:
  PYTHON=python3   interpreter to build the venv from (default: python3 when
                   it is usable). With uv you can pass a version instead,
                   e.g. PYTHON=3.12 — uv installs it if it is missing.
                   An existing .venv is always reused; delete it to rebuild.
USAGE
}

die() { echo "ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --minimal)   MINIMAL=1; shift ;;
    --rknn)      WANT_RKNN=1; shift ;;
    --no-fork)   WANT_FORK=0; shift ;;
    --no-tests)  WANT_TESTS=0; shift ;;
    --fork-dir)  FORK_DIR="$2"; shift 2 ;;
    --check)     CHECK_ONLY=1; shift ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done

PY_BIN="${PYTHON:-python3}"
VENV_PY="$REPO_ROOT/.venv/bin/python"
PYTHON_MIN_STR="${PYTHON_MIN[0]}.${PYTHON_MIN[1]}"

# --- host detection --------------------------------------------------------
IS_WSL=0
if [[ -n "${WSL_DISTRO_NAME:-}${WSL_INTEROP:-}" ]] || \
   grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; then
  IS_WSL=1
fi
ARCH="$(uname -m)"

py_version() { "$1" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || true; }
py_at_least() {  # py_at_least <exe> <major> <minor>
  "$1" -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= ($(echo "$2 $3" | tr ' ' ',')) else 1)" 2>/dev/null
}

# Read one dotted key out of config/pipeline.yaml (uses the venv's pyyaml).
cfg_get() {  # cfg_get <key> <default>
  "$VENV_PY" - "$REPO_ROOT/config/pipeline.yaml" "$1" "$2" <<'PY' 2>/dev/null \
    || printf '%s\n' "$2"
import sys, yaml
path, key, default = sys.argv[1], sys.argv[2], sys.argv[3]
node = yaml.safe_load(open(path)) or {}
for part in key.split('.'):
    if not isinstance(node, dict) or part not in node:
        print(default); raise SystemExit
    node = node[part]
if isinstance(node, bool):
    print('true' if node else 'false')
elif node is None:
    print(default)
else:
    print(node)
PY
}

# Resolve to an absolute path lexically, so it prints cleanly and works even
# when the target does not exist yet.
abspath() {  # abspath <path>
  local d b
  d="$(dirname "$1")"; b="$(basename "$1")"
  (cd "$d" 2>/dev/null && printf '%s/%s\n' "$PWD" "$b") || printf '%s\n' "$1"
}

# ---------------------------------------------------------------------------
# --check: diagnose, change nothing
# ---------------------------------------------------------------------------
run_check() {
  local rc=0 ver
  echo "=== setup check: $REPO_ROOT ==="
  if ! command -v "$PY_BIN" >/dev/null 2>&1; then
    echo "  python      MISSING ($PY_BIN not found)  -> sudo apt install python3 python3-venv"
    return 1
  fi
  ver="$(py_version "$PY_BIN")"
  if py_at_least "$PY_BIN" "${PYTHON_MIN[@]}"; then
    printf '  %-13s %s\n' "python" "$ver  (>= $PYTHON_MIN_STR required)"
  else
    printf '  %-13s %s\n' "python" "$ver  TOO OLD (need >= $PYTHON_MIN_STR)"; rc=1
  fi

  if [[ ! -x "$VENV_PY" ]]; then
    echo "  venv         MISSING  -> ./setup.sh"
    return 1
  fi
  printf '  %-13s .venv (%s)\n' "venv" "$(py_version "$VENV_PY")"

  check_mod() {  # check_mod <pip-name> <label> <required?>
    ver="$("$VENV_PY" -c "import importlib.metadata as m; print(m.version('$1'))" 2>/dev/null || true)"
    if [[ -n "$ver" ]]; then
      printf '  %-13s %-14s %s\n' "$2" "$ver" OK
    elif [[ "${3:-0}" == 1 ]]; then
      printf '  %-13s %-14s %s\n' "$2" MISSING "-> ./setup.sh"; rc=1
    else
      printf '  %-13s %-14s %s\n' "$2" "-" "not installed (optional)"
    fi
  }
  check_mod pyyaml     pyyaml     1
  check_mod requests   requests   1
  check_mod pillow     pillow     1
  check_mod ultralytics ultralytics "$([[ $MINIMAL -eq 1 ]] && echo 0 || echo 1)"
  check_mod pytest     pytest     0
  check_mod rknn-toolkit2 rknn-toolkit2 0

  # ffmpeg is a system package, not a pip one: RTSP capture (01 --rtsp /
  # mower.transport: rtsp) and video-frame extraction both shell out to it.
  if command -v ffmpeg >/dev/null 2>&1; then
    printf '  %-13s %-14s %s\n' "ffmpeg" \
      "$(ffmpeg -version 2>/dev/null | head -n1 | awk '{print $3}')" \
      "OK  (RTSP capture)"
  else
    printf '  %-13s %-14s %s\n' "ffmpeg" MISSING \
      "optional — needed for --rtsp (sudo apt install ffmpeg)"
  fi

  # Which torch backend 04 would use.
  if "$VENV_PY" -c "import torch" 2>/dev/null; then
    local backend
    backend="$("$VENV_PY" "$REPO_ROOT/scripts/_backend.py" 2>/dev/null | tail -n1 || true)"
    printf '  %-13s %s\n' "torch" "${backend#backend=}"
  else
    printf '  %-13s %s\n' "torch" "not installed (needed for 04/05)"
  fi

  # The airockchip fork, at whatever path the config points at.
  local fork="${FORK_DIR:-$REPO_ROOT/$(cfg_get export.airockchip_repo ../ultralytics_yolov8)}"
  [[ "$fork" = /* ]] || fork="$REPO_ROOT/$fork"
  fork="$(abspath "$fork")"
  if [[ -d "$fork/ultralytics/engine" ]]; then
    printf '  %-13s %s\n' "fork" "$fork  OK"
  else
    printf '  %-13s %s\n' "fork" "MISSING at $fork -> ./setup.sh"; rc=1
  fi

  if "$VENV_PY" -c "import pytest" 2>/dev/null; then
    echo
    "$VENV_PY" -m pytest -q tests || rc=1
  fi

  echo
  if [[ $rc -eq 0 ]]; then
    echo "OK — next: source .venv/bin/activate, then python3 scripts/01_capture.py --probe"
  else
    echo "NOT READY — run ./setup.sh (see --help for options)."
  fi
  return "$rc"
}

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  if ! command -v "$PY_BIN" >/dev/null 2>&1 && [[ -x "$VENV_PY" ]]; then
    PY_BIN="$VENV_PY"
  fi
  run_check
  exit $?
fi

# ---------------------------------------------------------------------------
# 1/5 — python
# ---------------------------------------------------------------------------
HAS_UV=0
command -v uv >/dev/null 2>&1 && HAS_UV=1

echo "=== 1/5 python ==="
# PY_BIN  = interpreter to validate and report ("" = let uv supply one)
# PY_SPEC = what to hand to `uv venv --python`
if [[ -n "${PYTHON:-}" ]]; then
  if command -v "$PYTHON" >/dev/null 2>&1; then
    PY_SPEC="$PYTHON"
    PY_BIN="$PYTHON"
    py_at_least "$PY_SPEC" "${PYTHON_MIN[@]}" \
      || die "python $(py_version "$PY_SPEC") (from PYTHON=) is too old; need >= $PYTHON_MIN_STR"
  elif [[ $HAS_UV -eq 1 ]]; then
    # A version request such as PYTHON=3.12 — uv installs it if needed.
    PY_SPEC="$PYTHON"
    PY_BIN=""
    echo "PYTHON=$PYTHON is not a command — uv will provide it"
  else
    die "PYTHON=$PYTHON not found.
  Give a command on PATH (PYTHON=python3.12), or install uv
  (https://docs.astral.sh/uv/) and setup.sh can fetch the interpreter itself."
  fi
elif command -v python3 >/dev/null 2>&1 && py_at_least python3 "${PYTHON_MIN[@]}"; then
  PY_BIN=python3
  PY_SPEC=python3
elif [[ $HAS_UV -eq 1 ]]; then
  PY_BIN=""
  PY_SPEC=3.12
  echo "no usable python3 on PATH — uv will provide python $PY_SPEC"
else
  die "python3 >= $PYTHON_MIN_STR not found.
  Debian/Ubuntu/WSL: sudo apt install python3 python3-venv python3-pip
  ...or install uv (https://docs.astral.sh/uv/): setup.sh will then fetch a Python itself."
fi

# Rockchip ships rknn-toolkit2 wheels only up to cp312 (2.3.0: cp36..cp312),
# so a newer interpreter cannot install it and 05 would have no compiler. With
# uv available and no interpreter pinned by hand, use 3.12 for --rknn.
if [[ "$WANT_RKNN" -eq 1 && -z "${PYTHON:-}" && $HAS_UV -eq 1 && -n "$PY_BIN" ]] && \
   ! "$PY_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] <= (3, 12) else 1)' 2>/dev/null; then
  echo "note: --rknn needs rknn-toolkit2, whose wheels stop at python 3.12 —"
  echo "      a fresh .venv will be built on a uv-managed python 3.12 instead."
  PY_SPEC=3.12
  PY_BIN=""
fi
echo "uv         : $( [[ $HAS_UV -eq 1 ]] && echo yes || echo 'no — falling back to venv + pip' )"
if [[ -n "$PY_BIN" ]]; then
  echo "interpreter: $PY_BIN ($(py_version "$PY_BIN"))"
else
  echo "interpreter: uv-managed python $PY_SPEC"
fi
echo "host       : $(uname -s) $ARCH$( [[ $IS_WSL -eq 1 ]] && echo ' (WSL)' )"

if [[ -n "$PY_BIN" ]]; then
  VENV_HINT="Debian/Ubuntu/WSL: sudo apt install python3-venv   (specifically python3.$("$PY_BIN" -c 'import sys; print(sys.version_info[1])')-venv)"
else
  VENV_HINT="install uv: https://docs.astral.sh/uv/"
fi

# ---------------------------------------------------------------------------
# 2/5 — virtualenv
# ---------------------------------------------------------------------------
echo
echo "=== 2/5 virtualenv ==="
if [[ -x "$VENV_PY" ]] && "$VENV_PY" -c 'import sys' 2>/dev/null; then
  echo ".venv already exists ($(py_version "$VENV_PY")) — reusing it"
  # If an interpreter was requested explicitly, say so when the venv differs.
  REQ=""
  if [[ -n "$PY_BIN" ]]; then
    REQ="$(py_version "$PY_BIN" | cut -d. -f1,2)"
  elif [[ "$PY_SPEC" =~ ^[0-9]+\.[0-9]+ ]]; then
    REQ="$PY_SPEC"
  fi
  if [[ -n "$REQ" && "$REQ" != "$(py_version "$VENV_PY" | cut -d. -f1,2)" ]]; then
    echo "WARNING: PYTHON=$PY_SPEC was requested, but the existing .venv is"
    echo "         python $(py_version "$VENV_PY") and is being reused. Delete .venv to rebuild it."
  fi
else
  rm -rf "$REPO_ROOT/.venv"
  if [[ $HAS_UV -eq 1 ]]; then
    echo "uv venv --python $PY_SPEC --seed"
    uv venv -q --python "$PY_SPEC" --seed "$REPO_ROOT/.venv" \
      || die "uv could not create .venv.
  $VENV_HINT"
  elif ! VENV_OUT="$("$PY_BIN" -m venv "$REPO_ROOT/.venv" 2>&1)"; then
    # Common on minimal images: the stdlib venv module exists but ensurepip
    # (python3.XX-venv) does not.
    rm -rf "$REPO_ROOT/.venv"
    die "could not create .venv:
$VENV_OUT
  $VENV_HINT
  ...or install uv (https://docs.astral.sh/uv/) and re-run; setup.sh will use it."
  fi
  echo "created .venv ($(py_version "$VENV_PY"))"
fi
"$VENV_PY" -c "import sys; sys.exit(0 if sys.version_info[:2] >= ($(echo "${PYTHON_MIN[@]}" | tr ' ' ',')) else 1)" \
  || die ".venv was built with python $(py_version "$VENV_PY"), too old for this repo (>= $PYTHON_MIN_STR).
  Delete .venv and re-run ./setup.sh with a newer PYTHON=python3.12"

# How we install into the venv: uv when it is present, pip otherwise.
if [[ $HAS_UV -eq 1 ]]; then
  PIP_MODE=uv
elif "$VENV_PY" -m pip --version >/dev/null 2>&1; then
  PIP_MODE=pip
else
  die ".venv has no pip.
  $VENV_HINT"
fi
pip_install() {  # pip_install <args...>   (quiet; -q is added here)
  if [[ "$PIP_MODE" == pip ]]; then "$VENV_PY" -m pip install -q "$@"
  else uv pip install -q --python "$VENV_PY" "$@"; fi
}

# ---------------------------------------------------------------------------
# 3/5 — dependencies
# ---------------------------------------------------------------------------
echo
echo "=== 3/5 dependencies ==="
echo "installer  : $PIP_MODE"
if [[ "$PIP_MODE" == pip ]]; then pip_install --upgrade pip; fi
pip_install -r requirements.txt
echo "runtime     : pyyaml, requests, pillow"

if [[ "$MINIMAL" -eq 0 ]]; then
  pip_install ultralytics pytest
  echo "training    : ultralytics (pulls torch), pytest"
fi

if [[ "$WANT_RKNN" -eq 1 ]]; then
  if [[ "$ARCH" != "x86_64" ]]; then
    die "rknn-toolkit2 is an x86_64-only compiler and this host is $ARCH.
  Build the .onnx here (05 --skip-rknn) and convert on an x86_64 Linux host
  or x86 WSL; see README 'Conversion: x86_64 only'."
  fi
  TK_VER="$(cfg_get export.rknn.toolkit_version 2.3.0)"
  pip_install --force-reinstall "rknn-toolkit2==$TK_VER" || die "could not install rknn-toolkit2==$TK_VER (05 can retry with --reinstall-toolkit).
  Most often the venv's python is newer than the wheels Rockchip publishes
  (they stop at cp312); this venv is python $(py_version "$VENV_PY").
    with uv:    rm -rf .venv && PYTHON=3.12 ./setup.sh --rknn     (uv fetches 3.12)
    without uv: install python3.12 + python3.12-venv, then
                rm -rf .venv && PYTHON=python3.12 ./setup.sh --rknn
  Or leave this venv alone and convert on an x86_64 host that has the toolkit:
                bash scripts/05_export_rknn.sh --skip-rknn    (export here)
                bash scripts/05_export_rknn.sh --onnx-in ...   (convert there)"
  echo "conversion  : rknn-toolkit2 $TK_VER"
fi

# ---------------------------------------------------------------------------
# 4/5 — airockchip fork (05's export path)
# ---------------------------------------------------------------------------
echo
echo "=== 4/5 airockchip fork ==="
if [[ "$MINIMAL" -eq 1 || "$WANT_FORK" -eq 0 ]]; then
  echo "skipped (--minimal/--no-fork). 05_export_rknn.sh will need it later:"
  echo "  git clone https://github.com/airockchip/ultralytics_yolov8 ../ultralytics_yolov8"
else
  FORK_PATH="${FORK_DIR:-$REPO_ROOT/$(cfg_get export.airockchip_repo ../ultralytics_yolov8)}"
  [[ "$FORK_PATH" = /* ]] || FORK_PATH="$REPO_ROOT/$FORK_PATH"
  FORK_PATH="$(abspath "$FORK_PATH")"
  if [[ -d "$FORK_PATH/ultralytics/engine" ]]; then
    echo "already cloned: $FORK_PATH"
  else
    command -v git >/dev/null 2>&1 || die "git not found (needed to clone the fork).
  Debian/Ubuntu/WSL: sudo apt install git"
    echo "cloning https://github.com/airockchip/ultralytics_yolov8 -> $FORK_PATH"
    git clone -q --depth 1 https://github.com/airockchip/ultralytics_yolov8 "$FORK_PATH"
  fi
  echo "05 will put it first on PYTHONPATH for the export only (it does not"
  echo "replace the pip ultralytics installed above)."
fi

# ---------------------------------------------------------------------------
# 5/5 — tests
# ---------------------------------------------------------------------------
echo
echo "=== 5/5 tests ==="
if [[ "$WANT_TESTS" -eq 0 ]]; then
  echo "skipped (--no-tests)"
elif "$VENV_PY" -c "import pytest" 2>/dev/null; then
  "$VENV_PY" -m pytest -q tests
else
  echo "skipped (pytest not installed — it comes with the default, non-minimal setup)"
fi

# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
cat <<'NEXT'

=== ready ===
Everything lives in .venv/ (gitignored). Nothing is written to the mower.

Next steps:
  1. source .venv/bin/activate        # once per shell
  2. edit config/pipeline.yaml        # mower.host / mower.profile, llm.base_url
  3. python3 scripts/01_capture.py --probe        # is the mower answering?
  4. python3 scripts/02_llm_label.py --images work/frames/<session> --health

Re-check any time with:  ./setup.sh --check
Options:                 ./setup.sh --help
NEXT

if [[ $IS_WSL -eq 1 ]]; then
  cat <<'WSL'

=== WSL notes ===
  - Keep the repo on the Linux filesystem (~/git/...), NOT under /mnt/c:
    /mnt/c is much slower for pip/torch and does not support hardlinks, so
    03's dataset assembly degrades to full copies.
  - GPU: NVIDIA works in WSL2 using the Windows driver (do NOT install a
    Linux NVIDIA driver inside WSL). AMD ROCm does not work under WSL, so 04
    falls back to CPU there — train on native Linux for a real run.
  - LM Studio on Windows binds 127.0.0.1, which WSL2's NAT cannot reach.
    Either enable mirrored networking (Windows 11: .wslconfig ->
    networkingMode=mirrored, then 127.0.0.1:1234 works as-is) or bind LM
    Studio to 0.0.0.0 and point llm.base_url at the Windows host IP.
WSL
fi
