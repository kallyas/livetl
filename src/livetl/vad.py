"""Turn a stream of VAD-scored frames into utterance-sized audio segments.

Whisper is a 30-second-window model, not a streaming model. Feeding it fixed
timer-based chunks slices words in half and hands the translator sentence
fragments. Cutting on speech pauses instead gives whole clauses, which is both
what Whisper was trained on and what MT needs to produce a sane translation.
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass

import numpy as np

from .capture import FRAME_SAMPLES, SAMPLE_RATE

FRAME_MS = FRAME_SAMPLES / SAMPLE_RATE * 1000.0  # 32 ms


@dataclass
class Segment:
    audio: np.ndarray
    final: bool
    started_at: float          # monotonic clock when speech began
    seq: int                   # utterance counter; partials share their final's seq

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE


@dataclass
class SegmenterConfig:
    threshold: float = 0.5           # speech probability to open a segment
    min_silence_ms: int = 550        # trailing quiet before we close it
    speech_pad_ms: int = 250         # keep this much audio either side
    min_speech_ms: int = 350         # drop anything shorter (coughs, clicks)
    # Latency tracks utterance length, so a streamer on a monologue would sit
    # at 10s+ before seeing a caption. Past soft_max_s, settle for a much
    # shorter pause -- still a real boundary, just a comma instead of a period.
    soft_max_s: float = 6.0
    soft_silence_ms: int = 240
    max_segment_s: float = 12.0      # hard cut for someone who never pauses
    carry_overlap_ms: int = 200      # audio replayed after a hard cut
    partials: bool = False
    partial_interval_s: float = 1.2
    partial_min_s: float = 1.0


class Segmenter:
    """Feed frames in, get Segments out. Stateful; not thread-safe."""

    def __init__(self, vad, cfg: SegmenterConfig | None = None):
        self.vad = vad
        self.cfg = cfg or SegmenterConfig()
        pad_frames = max(1, int(self.cfg.speech_pad_ms / FRAME_MS))
        self._pre: collections.deque[np.ndarray] = collections.deque(maxlen=pad_frames)
        self._buf: list[np.ndarray] = []
        self._triggered = False
        self._silence_frames = 0
        self._started_at = 0.0
        self._last_partial = 0.0
        self._seq = 0

    def _close(self) -> Segment | None:
        audio = np.concatenate(self._buf) if self._buf else np.empty(0, np.float32)
        self._buf = []
        self._triggered = False
        self._silence_frames = 0
        self._pre.clear()
        if len(audio) < self.cfg.min_speech_ms / 1000.0 * SAMPLE_RATE:
            return None
        self._seq += 1
        return Segment(audio, final=True, started_at=self._started_at, seq=self._seq)

    def feed(self, frame: np.ndarray):
        """Process one frame; yields zero or more Segments."""
        cfg = self.cfg
        prob = self.vad(frame)
        now = time.monotonic()

        if not self._triggered:
            self._pre.append(frame)
            if prob >= cfg.threshold:
                self._triggered = True
                self._started_at = now - len(self._pre) * FRAME_MS / 1000.0
                self._buf = list(self._pre)
                self._pre.clear()
                self._last_partial = 0.0
            return

        self._buf.append(frame)
        # Hysteresis: it takes a clearly lower score to count as silence than it
        # took to trigger, so a wavering score mid-word doesn't split the segment.
        if prob < cfg.threshold - 0.15:
            self._silence_frames += 1
        else:
            self._silence_frames = 0

        duration = len(self._buf) * FRAME_MS / 1000.0
        needed_silence = (
            cfg.soft_silence_ms if duration >= cfg.soft_max_s else cfg.min_silence_ms
        )
        if self._silence_frames * FRAME_MS >= needed_silence:
            keep = max(0, len(self._buf) - self._silence_frames + int(cfg.speech_pad_ms / FRAME_MS))
            self._buf = self._buf[:keep]
            seg = self._close()
            if seg:
                yield seg
            return

        if duration >= cfg.max_segment_s:
            overlap = int(cfg.carry_overlap_ms / FRAME_MS)
            tail = self._buf[-overlap:] if overlap else []
            seg = self._close()
            # Continuous speech: reopen immediately, replaying a little audio so
            # a word straddling the cut still lands whole in one of the halves.
            self._triggered = True
            self._buf = list(tail)
            self._started_at = now
            self._last_partial = 0.0
            if seg:
                yield seg
            return

        # Paced on audio duration, not wall clock: the two coincide on a live
        # source, but wall clock silently disables partials whenever input
        # arrives faster than realtime (a file, or a burst after a stall).
        if (
            cfg.partials
            and duration >= cfg.partial_min_s
            and duration - self._last_partial >= cfg.partial_interval_s
        ):
            self._last_partial = duration
            yield Segment(
                np.concatenate(self._buf),
                final=False,
                started_at=self._started_at,
                seq=self._seq + 1,
            )

    def flush(self) -> Segment | None:
        """Close out whatever is buffered (call at end of stream)."""
        return self._close() if self._triggered else None


def load_vad():
    """Return a callable frame -> speech probability.

    The model is recurrent, so it is deliberately never reset between
    utterances: its carried state is what lets it hold a noise estimate
    across a whole stream. score.reset exists for tests replaying fixtures.
    """
    import torch
    from silero_vad import load_silero_vad

    model = load_silero_vad()
    torch.set_num_threads(1)  # tiny model; threading costs more than it saves

    def score(frame: np.ndarray) -> float:
        with torch.no_grad():
            return float(model(torch.from_numpy(frame), SAMPLE_RATE).item())

    score.reset = model.reset_states  # type: ignore[attr-defined]
    return score
