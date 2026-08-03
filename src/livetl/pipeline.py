"""Stage orchestration.

    capture ──frames──▶ [VAD segmenter] ──segments──▶ [ASR] ──text──▶ [MT] ──▶ sinks
     thread              (in capture thread,           thread          thread
                          ~0.3 ms/frame)

Separate threads because each stage blocks on something different (I/O, GPU,
network) and the GIL is released inside all three. The queues between them are
bounded: if ASR falls behind realtime the pipeline sheds partials, then oldest
finals, rather than growing an unbounded backlog and drifting further behind
the live edge with every second.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass

from .asr import Transcript, is_junk
from .capture import AudioCapture, CaptureSpec
from .sinks import Caption, Sink
from .vad import Segment, Segmenter, SegmenterConfig, load_vad

log = logging.getLogger(__name__)

_STOP = object()


@dataclass
class Stats:
    segments: int = 0
    dropped: int = 0
    junk: int = 0
    errors: int = 0
    reconnects: int = 0
    asr_ms_total: int = 0
    mt_ms_total: int = 0
    audio_s_total: float = 0.0

    def summary(self) -> str:
        n = max(1, self.segments)
        rtf = (self.asr_ms_total / 1000.0) / max(0.01, self.audio_s_total)
        return (
            f"{self.segments} utterances · avg asr {self.asr_ms_total // n}ms · "
            f"avg mt {self.mt_ms_total // n}ms · asr RTF {rtf:.2f}x · "
            f"{self.dropped} dropped · {self.junk} filtered · {self.errors} errors"
            + (f" · {self.reconnects} reconnects" if self.reconnects else "")
        )


class Pipeline:
    def __init__(
        self,
        capture_spec: CaptureSpec,
        asr,
        mt,
        sink: Sink,
        *,
        source_lang: str | None = None,
        target_lang: str = "en",
        whisper_task: str = "transcribe",
        seg_cfg: SegmenterConfig | None = None,
        asr_queue_size: int = 8,
        mt_queue_size: int = 16,
        no_speech_max: float = 0.75,
        reconnect: bool = False,
        reconnect_max_delay: float = 60.0,
    ):
        self.capture_spec = capture_spec
        self.asr = asr
        self.mt = mt
        self.sink = sink
        self.source_lang = None if source_lang in (None, "auto") else source_lang
        self.target_lang = target_lang
        self.whisper_task = whisper_task
        self.seg_cfg = seg_cfg or SegmenterConfig()
        self.no_speech_max = no_speech_max
        self.reconnect = reconnect
        self.reconnect_max_delay = reconnect_max_delay
        self.stats = Stats()

        self._seg_q: queue.Queue = queue.Queue(maxsize=asr_queue_size)
        self._txt_q: queue.Queue = queue.Queue(maxsize=mt_queue_size)
        self._stop = threading.Event()
        self._capture: AudioCapture | None = None
        self._fatal: BaseException | None = None

    # -- public ------------------------------------------------------------

    def run(self) -> None:
        workers = [
            threading.Thread(target=self._asr_worker, name="asr", daemon=True),
            threading.Thread(target=self._mt_worker, name="mt", daemon=True),
        ]
        for w in workers:
            w.start()
        try:
            self._capture_loop()
        finally:
            # A stream that ended on its own still has audio in flight, so let
            # the workers finish it. A user-requested stop should be immediate:
            # they want the process gone, not another 20s of backlog.
            interrupted = self._stop.is_set()
            self._stop.set()
            self._seg_q.put(_STOP)
            for w in workers:
                w.join(timeout=5.0 if interrupted else None)
            if self._capture:
                self._capture.close()
        if self._fatal:
            raise self._fatal

    def warmup(self) -> None:
        """Load/compile models before the stream starts.

        Without this the first real utterance pays for a multi-gigabyte
        download plus lazy graph construction, by which point the bounded
        queues have already started shedding audio.
        """
        import numpy as np

        silence = np.zeros(16_000, dtype=np.float32)
        try:
            self.asr.transcribe(silence, self.source_lang, self.whisper_task)
        except Exception:
            log.debug("ASR warmup failed (harmless)", exc_info=True)

        if getattr(self.mt, "needs_source_lang", False):
            # Any source language exercises the same weights, and it must
            # differ from the target or translate() short-circuits and the
            # first real caption eats the cold-start cost (measured: 12s).
            src = self.source_lang or ("es" if self.target_lang != "es" else "en")
            try:
                self.mt.translate("Hello.", src, self.target_lang)
            except Exception:
                log.debug("MT warmup failed (harmless)", exc_info=True)

    def stop(self) -> None:
        self._stop.set()
        if self._capture:
            self._capture.close()

    # -- stage 1: capture + VAD --------------------------------------------

    def _capture_loop(self) -> None:
        vad = load_vad()
        seg = Segmenter(vad, self.seg_cfg)
        delay = 2.0
        while not self._stop.is_set():
            got_audio = False
            try:
                got_audio = self._capture_once(seg)
            except RuntimeError as exc:
                # stop() terminates the child processes, so a shutdown always
                # surfaces here as a capture error. Don't report our own
                # teardown as a failure.
                if self._stop.is_set():
                    return
                if not self.reconnect:
                    raise
                log.warning("capture failed: %s", str(exc).splitlines()[0])
            if not self.reconnect or self._stop.is_set():
                return
            # A run that actually carried audio means the source is healthy and
            # merely ended (stream over, ad break, blip), so start over from a
            # short delay. Repeated no-data failures mean nobody is streaming;
            # back off instead of hammering the host.
            delay = 2.0 if got_audio else min(delay * 2, self.reconnect_max_delay)
            self.stats.reconnects += 1
            log.info("reconnecting in %.0fs", delay)
            self._stop.wait(delay)

    def _capture_once(self, seg: Segmenter) -> bool:
        """One capture session. Returns whether any audio was received."""
        got_audio = False
        with AudioCapture(self.capture_spec) as cap:
            self._capture = cap
            for frame in cap.frames():
                if self._stop.is_set():
                    break
                got_audio = True
                for s in seg.feed(frame):
                    self._offer(s)
            tail = seg.flush()
            if tail:
                self._offer(tail)
        return got_audio

    def _offer(self, s: Segment) -> None:
        """Enqueue a segment, shedding load instead of falling behind live."""
        try:
            self._seg_q.put_nowait(s)
            return
        except queue.Full:
            pass
        if not s.final:
            self.stats.dropped += 1
            return
        # Finals matter; make room by discarding the oldest partial, or failing
        # that the oldest final. Dropping the *oldest* keeps us near live.
        drained = []
        try:
            while True:
                drained.append(self._seg_q.get_nowait())
        except queue.Empty:
            pass
        keep = [x for x in drained if x is _STOP or x.final]
        self.stats.dropped += len(drained) - len(keep)
        if len(keep) >= self._seg_q.maxsize:
            self.stats.dropped += 1
            keep = keep[1:]
        for x in keep + [s]:
            try:
                self._seg_q.put_nowait(x)
            except queue.Full:
                self.stats.dropped += 1

    # -- stage 2: ASR ------------------------------------------------------

    def _asr_worker(self) -> None:
        while True:
            item = self._seg_q.get()
            # Only the sentinel ends the loop. Breaking on _stop would discard
            # everything still queued the instant capture ends, silently losing
            # audio that was already recorded; an interrupted run is bounded by
            # the join timeout in run() instead.
            if item is _STOP:
                break
            seg: Segment = item
            t0 = time.monotonic()
            try:
                tr: Transcript = self.asr.transcribe(
                    seg.audio, self.source_lang, self.whisper_task
                )
            except Exception:
                log.exception("ASR failed on %.1fs of audio", seg.duration)
                self.stats.errors += 1
                continue
            asr_ms = int((time.monotonic() - t0) * 1000)

            if seg.final:
                self.stats.segments += 1
                self.stats.asr_ms_total += asr_ms
                self.stats.audio_s_total += seg.duration
            if tr.no_speech >= self.no_speech_max or is_junk(tr.text):
                if seg.final:
                    self.stats.junk += 1
                log.debug("filtered %r (no_speech=%.2f)", tr.text, tr.no_speech)
                continue
            self._push_text(seg, tr, asr_ms)
        self._txt_q.put(_STOP)

    def _push_text(self, seg: Segment, tr: Transcript, asr_ms: int) -> None:
        try:
            self._txt_q.put_nowait((seg, tr, asr_ms))
        except queue.Full:
            if seg.final:
                try:
                    self._txt_q.get_nowait()
                    self._txt_q.put_nowait((seg, tr, asr_ms))
                    self.stats.dropped += 1
                except (queue.Empty, queue.Full):
                    self.stats.dropped += 1
            else:
                self.stats.dropped += 1

    # -- stage 3: MT + output ---------------------------------------------

    def _mt_worker(self) -> None:
        while True:
            item = self._txt_q.get()
            if item is _STOP:
                break
            seg, tr, asr_ms = item
            src_lang = (tr.language or self.source_lang or "").lower()
            t0 = time.monotonic()
            try:
                target = self.mt.translate(tr.text, src_lang, self.target_lang)
            except Exception as exc:
                log.warning("translation failed (%s); showing source text", exc)
                self.stats.errors += 1
                target = tr.text
            mt_ms = int((time.monotonic() - t0) * 1000)
            if seg.final:
                self.stats.mt_ms_total += mt_ms

            self.sink.emit(
                Caption(
                    seq=seg.seq,
                    final=seg.final,
                    source=tr.text,
                    target=target,
                    src_lang=src_lang,
                    tgt_lang=self.target_lang,
                    asr_ms=asr_ms,
                    mt_ms=mt_ms,
                    lag_ms=int((time.monotonic() - seg.started_at) * 1000),
                    audio_s=round(seg.duration, 2),
                )
            )
