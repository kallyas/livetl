"""Audio capture: live stream URL / local file / input device -> 16 kHz mono float32 frames."""

from __future__ import annotations

import collections
import logging
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
# Silero VAD requires exactly 512 samples per frame at 16 kHz (32 ms).
FRAME_SAMPLES = 512
FRAME_BYTES = FRAME_SAMPLES * 2  # s16le


def _which(name: str) -> str:
    # Console scripts installed into the active venv live next to the
    # interpreter, and that directory is not on PATH unless the venv was
    # "activated" -- which it isn't when invoked as .venv/bin/python -m livetl.
    local = os.path.join(os.path.dirname(sys.executable), name)
    if os.access(local, os.X_OK):
        return local
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name!r} not found on PATH or in {os.path.dirname(sys.executable)}.")
    return path


def _have(name: str) -> bool:
    try:
        _which(name)
        return True
    except RuntimeError:
        return False


def _drain(pipe, sink: collections.deque, tag: str) -> None:
    """Consume a child's stderr so its pipe never fills and blocks the child."""
    try:
        for raw in iter(pipe.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                sink.append(f"[{tag}] {line}")
                log.debug("%s: %s", tag, line)
    except Exception:
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _stream_puller(url: str, tool: str, quality: str) -> list[str]:
    if tool == "streamlink":
        cmd = [
            _which("streamlink"),
            "--stdout",
            "--quiet",
            "--retry-streams", "3",
            "--retry-max", "0",
        ]
        if "twitch.tv" in url:
            # Sit at the live edge instead of the default ~6s back, and skip
            # ad segments that would otherwise be transcribed as content.
            cmd += ["--twitch-low-latency", "--twitch-disable-ads"]
        cmd += [url, quality]
        return cmd
    if tool == "yt-dlp":
        return [
            _which("yt-dlp"),
            "--quiet", "--no-warnings", "--no-part",
            "-f", "bestaudio/worstvideo+bestaudio/worst",
            "-o", "-",
            url,
        ]
    raise ValueError(f"unknown puller {tool!r}")


def _ffmpeg_decode(input_arg: list[str], realtime_pace: bool) -> list[str]:
    cmd = [
        _which("ffmpeg"),
        "-hide_banner", "-loglevel", "warning",
        # NOT -fflags nobuffer: it discards already-demuxed packets and
        # silently eats the first seconds of audio (measured: 3.2s of a
        # 10.4s file). Live-edge latency is the puller's job, not ffmpeg's.
        "-flags", "low_delay",
        "-probesize", "64k",
    ]
    if realtime_pace:
        # Only for local files: pace decoding at 1x so a file behaves like a
        # live source instead of flooding the pipeline instantly.
        cmd += ["-re"]
    cmd += input_arg
    cmd += [
        "-vn", "-sn", "-dn",
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE), "-ac", "1",
        "pipe:1",
    ]
    return cmd


@dataclass
class CaptureSpec:
    source: str
    puller: str = "auto"          # auto | streamlink | yt-dlp
    quality: str = "audio_only,worst,best"
    device: bool = False          # source is an avfoundation device index


class AudioCapture:
    """Iterable of float32 frames in [-1, 1], FRAME_SAMPLES long.

    Spawns at most two children (puller -> ffmpeg) and yields decoded PCM.
    Use as a context manager so the children are always reaped.
    """

    def __init__(self, spec: CaptureSpec):
        self.spec = spec
        self._procs: list[subprocess.Popen] = []
        self._threads: list[threading.Thread] = []
        self._errors: collections.deque[str] = collections.deque(maxlen=40)
        self._stop = threading.Event()

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "AudioCapture":
        spec = self.spec
        if spec.device:
            ff = _ffmpeg_decode(["-f", "avfoundation", "-i", f":{spec.source}"], False)
            self._spawn_ffmpeg(ff, stdin=None)
        elif spec.source.startswith(("http://", "https://")):
            tool = spec.puller
            if tool == "auto":
                tool = "streamlink" if _have("streamlink") else "yt-dlp"
            puller = subprocess.Popen(
                _stream_puller(spec.source, tool, spec.quality),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            self._track(puller, tool)
            ff = _ffmpeg_decode(["-i", "pipe:0"], False)
            self._spawn_ffmpeg(ff, stdin=puller.stdout)
            # ffmpeg owns the read end now; drop ours so EOF propagates.
            puller.stdout.close()
        else:
            ff = _ffmpeg_decode(["-i", spec.source], realtime_pace=True)
            self._spawn_ffmpeg(ff, stdin=None)
        return self

    def _spawn_ffmpeg(self, cmd: list[str], stdin) -> None:
        proc = subprocess.Popen(
            cmd, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
        )
        self._track(proc, "ffmpeg")
        self._ff = proc

    def _track(self, proc: subprocess.Popen, tag: str) -> None:
        self._procs.append(proc)
        t = threading.Thread(target=_drain, args=(proc.stderr, self._errors, tag), daemon=True)
        t.start()
        self._threads.append(t)

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._stop.set()
        for proc in reversed(self._procs):
            if proc.poll() is None:
                proc.terminate()
        for proc in reversed(self._procs):
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    @property
    def recent_errors(self) -> str:
        return "\n".join(self._errors)

    # -- data --------------------------------------------------------------

    def frames(self):
        """Yield float32 frames until the source ends or close() is called."""
        buf = bytearray(FRAME_BYTES)
        view = memoryview(buf)
        stdout = self._ff.stdout
        while not self._stop.is_set():
            filled = 0
            while filled < FRAME_BYTES:
                n = stdout.readinto(view[filled:])
                if not n:  # EOF
                    if filled == 0:
                        if self._ff.poll() is not None:
                            self._raise_if_failed()
                        return
                    # Zero-pad the ragged tail rather than discarding it, so
                    # the last words of a stream still reach the recognizer.
                    view[filled:] = b"\x00" * (FRAME_BYTES - filled)
                    pcm = np.frombuffer(bytes(buf), dtype="<i2")
                    yield pcm.astype(np.float32) / 32768.0
                    return
                filled += n
            pcm = np.frombuffer(bytes(buf), dtype="<i2")
            yield pcm.astype(np.float32) / 32768.0

    def _raise_if_failed(self) -> None:
        bad = [p for p in self._procs if p.poll() not in (0, None)]
        if bad:
            raise RuntimeError(
                "capture failed (exit codes: "
                + ", ".join(str(p.returncode) for p in bad)
                + ")\n"
                + self.recent_errors
            )
