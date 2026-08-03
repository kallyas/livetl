#!/usr/bin/env bash
#
# livetl on Google Colab (tested against a T4: ~12.7 GB system RAM, ~15 GB VRAM).
#
# In a Colab cell:
#
#   !git clone https://github.com/YOU/transcribe-audio.git
#   %cd transcribe-audio
#   !bash colab.sh setup
#   !bash colab.sh run https://twitch.tv/somechannel --source ja --target en
#
# Runtime > Change runtime type > T4 GPU first, or this falls back to CPU and
# will not keep up with a live stream.
#
# Unlike livetl.sh this deliberately does NOT create a venv. Colab already
# ships torch built against its exact driver; a venv would download a second
# multi-gigabyte torch and risk a CUDA mismatch. Everything installs into the
# runtime's own Python instead.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PY="${LIVETL_PYTHON:-python3}"
RED=$'\033[31m'; YEL=$'\033[33m'; GRN=$'\033[32m'; DIM=$'\033[2m'; OFF=$'\033[0m'
[ -t 2 ] || { RED=""; YEL=""; GRN=""; DIM=""; OFF=""; }

die()  { printf '%s\n' "${RED}error:${OFF} $*" >&2; exit 1; }
warn() { printf '%s\n' "${YEL}warn:${OFF} $*" >&2; }
info() { printf '%s\n' "${DIM}==>${OFF} $*" >&2; }

on_colab() { [ -n "${COLAB_RELEASE_TAG:-}" ] || [ -d /content ]; }

# CTranslate2 loads cuBLAS and cuDNN by dlopen at runtime. On Colab those live
# inside the pip nvidia-* packages rather than a system CUDA install, so
# without this it reports 0 GPUs and silently falls back to CPU.
cuda_lib_path() {
    "$PY" - <<'EOF' 2>/dev/null || true
import os, importlib
paths = []
for mod in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
    try:
        paths.append(os.path.dirname(importlib.import_module(mod).__file__))
    except Exception:
        pass
print(":".join(paths))
EOF
}

export_cuda_libs() {
    local extra; extra="$(cuda_lib_path)"
    if [ -n "$extra" ]; then
        export LD_LIBRARY_PATH="${extra}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
}

# ------------------------------------------------------------------- setup

cmd_setup() {
    on_colab || warn "this does not look like Colab; livetl.sh is the general launcher"

    if ! command -v ffmpeg >/dev/null 2>&1; then
        info "installing ffmpeg"
        apt-get -qq update >/dev/null 2>&1 || true
        apt-get -qq install -y ffmpeg >/dev/null 2>&1 || die "could not install ffmpeg"
    fi

    # torch and transformers are preinstalled and intentionally left alone --
    # pip skips anything already satisfied, so torch is never re-resolved.
    info "installing dependencies (leaving Colab's torch untouched)"
    "$PY" -m pip install -q --upgrade \
        faster-whisper \
        silero-vad \
        streamlink \
        yt-dlp \
        websockets \
        rich \
        sentencepiece \
        sacremoses >&2

    # transformers 4.56 renamed from_pretrained's torch_dtype to dtype. The
    # code supports both, so an older preinstalled version is fine and is not
    # upgraded -- upgrading it on Colab tends to break other preinstalled bits.
    info "installing livetl"
    "$PY" -m pip install -q -e . --no-deps >&2

    export_cuda_libs
    printf '%s\n' "${GRN}ready${OFF}" >&2
    cmd_doctor
}

# ------------------------------------------------------------------ doctor

cmd_doctor() {
    export_cuda_libs
    printf 'python      : %s\n' "$("$PY" -V 2>&1)"
    printf 'ffmpeg      : %s\n' "$(command -v ffmpeg || echo "${RED}MISSING${OFF}")"
    if command -v nvidia-smi >/dev/null 2>&1; then
        printf 'gpu         : %s\n' \
            "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
               | paste -sd'; ' - || echo 'nvidia-smi failed')"
    else
        printf 'gpu         : %s\n' "${YEL}none - set Runtime > Change runtime type > GPU${OFF}"
    fi
    "$PY" - <<'EOF'
import importlib
for mod, label in [("torch", "torch"), ("transformers", "transformers"),
                   ("faster_whisper", "faster-whisper"), ("silero_vad", "silero-vad"),
                   ("streamlink", "streamlink"), ("livetl", "livetl")]:
    try:
        m = importlib.import_module(mod)
        print(f"{label:12}: {getattr(m, '__version__', 'ok')}")
    except ImportError as exc:
        print(f"{label:12}: MISSING ({exc})")
try:
    import ctranslate2
    n = ctranslate2.get_cuda_device_count()
    print(f"{'ctranslate2':12}: {ctranslate2.__version__}, {n} CUDA device(s)"
          + ("" if n else "  <-- will run on CPU, too slow for live"))
except Exception as exc:
    print(f"{'ctranslate2':12}: {exc}")
EOF
}

# --------------------------------------------------------------------- run

cmd_run() {
    export_cuda_libs

    "$PY" -c "import livetl" 2>/dev/null || die "run 'bash colab.sh setup' first"

    local gpu=0
    "$PY" -c "import ctranslate2,sys; sys.exit(0 if ctranslate2.get_cuda_device_count() else 1)" \
        2>/dev/null && gpu=1

    local defaults=()
    if [ "$gpu" = "1" ]; then
        # 15 GB of VRAM comfortably holds large-v3-turbo (~3 GB in float16)
        # alongside NLLB, and the quality gap over `small` is large.
        defaults=(--asr faster --asr-device cuda --compute-type float16
                  --model "${MODEL:-large-v3-turbo}" --mt-device cuda)
    else
        warn "no CUDA device visible - falling back to a small CPU model."
        warn "Runtime > Change runtime type > T4 GPU, then rerun."
        defaults=(--asr faster --asr-device cpu --model "${MODEL:-small}")
    fi

    # Colab kills a session that produces no output for too long, and there is
    # no terminal, so keep captions flowing to stdout and log them too.
    defaults+=(--log "${LOG_FILE:-captions.jsonl}")

    info "livetl ${defaults[*]} $*"
    exec "$PY" -m livetl "${defaults[@]}" "$@"
}

# ------------------------------------------------------------------ overlay

cmd_overlay() {
    cat <<'EOF'
The overlay binds a local port, and Colab has no inbound networking. Run
livetl with the overlay enabled:

    !bash colab.sh run <URL> --source ja --overlay --overlay-host 0.0.0.0 &

then, in a separate Python cell, open Colab's authenticated port proxy:

    from google.colab.output import eval_js
    print(eval_js('google.colab.kernel.proxyPort(8765)'))

Open that URL. The proxy does forward websockets, but it is the flakiest
part of running this on Colab -- if captions never appear in the overlay,
they are still going to the cell output and to captions.jsonl, which is
the more reliable way to read them here.
EOF
}

# -------------------------------------------------------------------- main

case "${1:-}" in
    setup)   shift; cmd_setup "$@" ;;
    doctor)  shift; cmd_doctor "$@" ;;
    run)     shift; cmd_run "$@" ;;
    overlay) shift; cmd_overlay "$@" ;;
    ""|-h|--help)
        sed -n '3,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        printf '\ncommands: setup | doctor | run <URL> [args] | overlay\n'
        ;;
    *)       cmd_run "$@" ;;
esac
