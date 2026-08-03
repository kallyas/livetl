#!/usr/bin/env bash
#
# livetl launcher: sets up the environment on first run, then translates a
# live stream. Safe to re-run; setup is idempotent.
#
#   ./livetl.sh setup                       install venv + deps for this machine
#   ./livetl.sh doctor                      check the install and report hardware
#   ./livetl.sh https://twitch.tv/xyz       run (setup happens automatically)
#   ./livetl.sh --retry https://twitch.tv/x add an outer restart loop (rarely needed)
#
# Defaults live in livetl.env (see livetl.env.example). Anything after the URL
# is passed straight through to the Python CLI, and wins over livetl.env.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

VENV=".venv"
PY="$VENV/bin/python"
RED=$'\033[31m'; YEL=$'\033[33m'; GRN=$'\033[32m'; DIM=$'\033[2m'; OFF=$'\033[0m'
[ -t 2 ] || { RED=""; YEL=""; GRN=""; DIM=""; OFF=""; }

CHILD=0

# Forward the signal to the running child, then leave. Without this a
# supervisor's SIGTERM would kill the wrapper and orphan the Python process.
on_signal() {
    trap - INT TERM
    [ "$CHILD" -ne 0 ] && kill -TERM "$CHILD" 2>/dev/null || true
    [ "$CHILD" -ne 0 ] && wait "$CHILD" 2>/dev/null || true
    printf '%s\n' "${DIM}==>${OFF} stopped" >&2
    exit 0
}

die()  { printf '%s\n' "${RED}error:${OFF} $*" >&2; exit 1; }
warn() { printf '%s\n' "${YEL}warn:${OFF} $*" >&2; }
info() { printf '%s\n' "${DIM}==>${OFF} $*" >&2; }

# ---------------------------------------------------------------- platform

# Echoes the uv extras this machine should install.
detect_extras() {
    local os arch
    os="$(uname -s)"; arch="$(uname -m)"
    if [ "$os" = "Darwin" ] && [ "$arch" = "arm64" ]; then
        echo "--extra mlx --extra localmt"
    elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        echo "--extra cuda --extra localmt"
    else
        echo "--extra ct2 --extra localmt"
    fi
}

find_uv() {
    if command -v uv >/dev/null 2>&1; then command -v uv; return; fi
    for c in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        [ -x "$c" ] && { echo "$c"; return; }
    done
    return 1
}

# Prefer a real 3.12: broadest wheel coverage for ctranslate2/torch/mlx.
find_python() {
    for c in "${LIVETL_PYTHON:-}" /usr/local/bin/python3.12 /opt/homebrew/bin/python3.12 \
             "$(command -v python3.12 2>/dev/null || true)" \
             "$(command -v python3.11 2>/dev/null || true)"; do
        [ -n "$c" ] && [ -x "$c" ] && { echo "$c"; return; }
    done
    return 1
}

# ------------------------------------------------------------------- setup

cmd_setup() {
    command -v ffmpeg >/dev/null 2>&1 || die \
        "ffmpeg not found. macOS: brew install ffmpeg | Debian: sudo apt install -y ffmpeg"

    local uv
    if ! uv="$(find_uv)"; then
        info "installing uv"
        curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 \
            || die "could not install uv; see https://docs.astral.sh/uv/"
        uv="$(find_uv)" || die "uv installed but not found on PATH"
    fi

    if [ ! -x "$PY" ]; then
        local py
        py="$(find_python)" || die \
            "no Python 3.11/3.12 found. Set LIVETL_PYTHON=/path/to/python3.12"
        info "creating $VENV from $py"
        "$uv" venv --python "$py" "$VENV" >&2
    fi

    local extras; extras="$(detect_extras)"
    info "installing dependencies ($extras)"
    # shellcheck disable=SC2086
    VIRTUAL_ENV="$PWD/$VENV" "$uv" sync $extras >&2
    printf '%s\n' "${GRN}ready${OFF}" >&2
}

# ------------------------------------------------------------------ doctor

cmd_doctor() {
    printf 'platform    : %s %s\n' "$(uname -s)" "$(uname -m)"
    printf 'cpu cores   : %s\n' "$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo '?')"
    if [ "$(uname -s)" = "Darwin" ]; then
        printf 'memory      : %s GB\n' "$(( $(sysctl -n hw.memsize) / 1073741824 ))"
        printf 'chip        : %s\n' "$(sysctl -n machdep.cpu.brand_string)"
    else
        printf 'memory      : %s\n' "$(free -h 2>/dev/null | awk '/^Mem:/{print $2}' || echo '?')"
    fi
    printf 'ffmpeg      : %s\n' "$(command -v ffmpeg || echo "${RED}MISSING${OFF}")"
    printf 'venv        : %s\n' "$([ -x "$PY" ] && "$PY" -V || echo "${YEL}not created${OFF}")"

    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        printf 'gpu         : %s\n' \
            "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | paste -sd'; ' -)"
    else
        printf 'gpu         : %s\n' "none detected (nvidia-smi absent)"
    fi
    printf 'recommended : uv sync %s\n' "$(detect_extras)"

    [ -x "$PY" ] || { warn "run './livetl.sh setup' first"; return; }
    printf '\nbackends:\n'
    "$PY" - <<'EOF'
import importlib
for mod, label in [("mlx_whisper","mlx-whisper (Apple GPU ASR)"),
                   ("faster_whisper","faster-whisper (CPU/CUDA ASR)"),
                   ("transformers","transformers (local MT)"),
                   ("silero_vad","silero-vad"),
                   ("streamlink","streamlink"),
                   ("yt_dlp","yt-dlp")]:
    try:
        importlib.import_module(mod); print(f"  ok      {label}")
    except ImportError:
        print(f"  missing {label}")
try:
    import ctranslate2 as c
    print(f"  cuda devices visible to ctranslate2: {c.get_cuda_device_count()}")
except Exception:
    pass
EOF
}

# --------------------------------------------------------------------- run

cmd_run() {
    [ -x "$PY" ] || cmd_setup

    # livetl.env supplies defaults; explicit CLI args below still win because
    # argparse takes the last occurrence of a repeated option.
    local defaults=()
    if [ -f livetl.env ]; then
        # errexit is disabled around the source deliberately: bash 3.2 (still
        # the system bash on macOS) does not suppress it inside a sourced file
        # even in an `if` condition, so a malformed config would kill the
        # script at 127 with only bash's cryptic error. The classic mistake is
        # an unquoted value containing spaces, which bash reads as a command.
        local rc=0
        set +e
        set -a
        # shellcheck source=/dev/null
        . ./livetl.env
        rc=$?
        set +a
        set -e
        [ "$rc" -eq 0 ] || die "livetl.env failed to load (exit $rc).
       Values containing spaces must be quoted, e.g.
       EXTRA_ARGS=\"--timing --soft-max 4\""
        [ -n "${SOURCE_LANG:-}" ] && defaults+=(--source "$SOURCE_LANG")
        [ -n "${TARGET_LANG:-}" ] && defaults+=(--target "$TARGET_LANG")
        [ -n "${MODEL:-}" ]       && defaults+=(--model "$MODEL")
        [ -n "${MT:-}" ]          && defaults+=(--mt "$MT")
        [ -n "${OBS_FILE:-}" ]    && defaults+=(--obs-file "$OBS_FILE")
        [ -n "${LOG_FILE:-}" ]    && defaults+=(--log "$LOG_FILE")
        if [ "${OVERLAY:-0}" = "1" ]; then
            defaults+=(--overlay --overlay-host "${OVERLAY_HOST:-127.0.0.1}"
                       --overlay-port "${OVERLAY_PORT:-8765}")
        fi
        # shellcheck disable=SC2206
        [ -n "${EXTRA_ARGS:-}" ] && defaults+=(${EXTRA_ARGS})
    fi

    # STREAM_URL lets a service unit run `livetl.sh run` with no arguments.
    if [ "$#" -eq 0 ] && [ -n "${STREAM_URL:-}" ]; then
        set -- "$STREAM_URL"
    fi

    if [ "$RETRY" != "1" ]; then
        exec "$PY" -m livetl ${defaults[@]+"${defaults[@]}"} "$@"
    fi

    # Safety net for a hard crash. Normal stream drops are handled inside the
    # Python process by --reconnect, which keeps the models loaded; restarting
    # here costs a full model reload, so it should stay rare.
    local delay=5 max=120
    trap on_signal INT TERM
    while true; do
        # Backgrounded + wait, not a plain foreground call: bash defers traps
        # until a foreground child exits, so `systemctl stop` (SIGTERM) would
        # otherwise hang until the stream ended on its own.
        "$PY" -m livetl ${defaults[@]+"${defaults[@]}"} "$@" &
        CHILD=$!
        wait "$CHILD" && delay=5 || true
        CHILD=0
        info "livetl exited; restarting in ${delay}s (ctrl-c to stop)"
        sleep "$delay" &
        CHILD=$!
        wait "$CHILD" || true
        CHILD=0
        delay=$(( delay * 2 )); [ "$delay" -gt "$max" ] && delay="$max"
    done
}

# -------------------------------------------------------------------- main

RETRY=0
args=()
for a in "$@"; do
    case "$a" in
        --retry) RETRY=1 ;;
        *) args+=("$a") ;;
    esac
done
set -- ${args[@]+"${args[@]}"}

case "${1:-}" in
    setup)  shift; cmd_setup "$@" ;;
    doctor) shift; cmd_doctor "$@" ;;
    run)    shift; cmd_run "$@" ;;
    ""|-h|--help)
        sed -n '3,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        [ -x "$PY" ] && { echo; "$PY" -m livetl --help; } || true
        ;;
    *)      cmd_run "$@" ;;
esac
