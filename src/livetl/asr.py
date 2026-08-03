"""Speech recognition backends.

Two backends, same interface:
  mlx     - Apple Silicon GPU via mlx-whisper. Fastest on M-series.
  faster  - CTranslate2 int8 on CPU. Portable fallback.

Both can also run Whisper's built-in ``task="translate"``, which goes straight
to English without a separate MT stage (English target only).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

# fp16 by default. The q4 builds are ~2x smaller but measurably worse at
# accents and proper nouns, which is exactly what streams are full of; the
# q4 aliases are kept for machines that genuinely can't spare the memory.
MLX_MODELS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "tiny-q4": "mlx-community/whisper-tiny-mlx-q4",
    "base-q4": "mlx-community/whisper-base-mlx-q4",
    "small-q4": "mlx-community/whisper-small-mlx-q4",
    "medium-q4": "mlx-community/whisper-medium-mlx-q4",
}

# Whisper emits these from silence, music and applause. On a stream with a
# background music bed that is most of what you would otherwise "translate".
_HALLUCINATIONS = {
    "thank you", "thanks for watching", "thank you for watching",
    "thanks for watching!", "please subscribe", "subscribe to my channel",
    "you", "bye", "bye.", "okay", "ok", "so", "yeah", "hmm", "mm",
    "ご視聴ありがとうございました", "ありがとうございました",
    "字幕by邹á", "字幕", "视频字幕", "다음 영상에서 만나요",
    "sous-titrage société radio-canada", "amara.org", "www.amara.org",
    "untertitel der amara.org-community",
}
_MUSIC = re.compile(r"^[\s\[\(♪♫*_-]*(music|applause|音楽|拍手|silence)[\s\]\)♪♫*_-]*$", re.I)
_PUNCT_ONLY = re.compile(r"^[\W_]+$", re.UNICODE)
# "haha haha haha haha" / "。。。。" - decoder loops that survived temperature fallback
_REPEAT = re.compile(r"^(.{1,20}?)\1{3,}$", re.S)


@dataclass
class Transcript:
    text: str
    language: str
    no_speech: float = 0.0


def is_junk(text: str) -> bool:
    t = text.strip()
    if not t or _PUNCT_ONLY.match(t) or _MUSIC.match(t):
        return True
    if t.lower().strip(" .!?…") in _HALLUCINATIONS:
        return True
    return bool(_REPEAT.match(t.replace(" ", "")))


class MLXWhisper:
    name = "mlx"

    def __init__(self, model: str):
        import mlx_whisper  # noqa: F401  (import here so the dep stays optional)

        self.repo = MLX_MODELS.get(model, model)
        self._mod = mlx_whisper
        self._model = None

    def _detect_language(self, audio: np.ndarray) -> str:
        """Detect the spoken language explicitly.

        mlx_whisper.transcribe() decodes in the right language but reports
        the DecodingOptions default ("en") in its result dict, so trusting
        that field hands every MT backend a bogus source language and the
        translation silently no-ops.
        """
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.decoding import detect_language
        from mlx_whisper.transcribe import ModelHolder

        # Share transcribe()'s cached instance. Calling load_model() directly
        # would hold a second full copy of the weights, at a different dtype.
        if self._model is None:
            self._model = ModelHolder.get_model(self.repo, mx.float16)
        # Pad the *audio* to a full 30s window, not the mel: zero-padding a
        # log-mel produces values that never occur in a real spectrogram.
        mel = log_mel_spectrogram(audio, n_mels=self._model.dims.n_mels, padding=N_SAMPLES)
        # Whisper exposes no .dtype; the encoder weights are the source of truth
        # and mel must match them or the encoder errors on a dtype mismatch.
        dtype = self._model.encoder.conv1.weight.dtype
        mel = pad_or_trim(mel, N_FRAMES, axis=-2).astype(dtype)
        _, probs = detect_language(self._model, mel)
        # Returns a bare dict for unbatched mel, a list of dicts when batched.
        table = probs[0] if isinstance(probs, list) else probs
        return max(table, key=table.get)

    def transcribe(self, audio: np.ndarray, language: str | None, task: str) -> Transcript:
        detected = language or self._detect_language(audio)
        out = self._mod.transcribe(
            audio,
            path_or_hf_repo=self.repo,
            language=detected,
            task=task,
            temperature=(0.0, 0.2, 0.4),
            condition_on_previous_text=False,  # stops cross-utterance loops
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            fp16=True,
        )
        segs = out.get("segments") or []
        no_speech = max((s.get("no_speech_prob", 0.0) for s in segs), default=0.0)
        return Transcript(out.get("text", "").strip(), detected, no_speech)


class FasterWhisper:
    name = "faster"

    def __init__(self, model: str, compute_type: str = "auto", device: str = "auto"):
        from faster_whisper import WhisperModel

        if device == "auto":
            device = "cuda" if _cuda_available() else "cpu"
        if compute_type == "auto":
            # float16 on a GPU; int8 is ~4x faster than float32 on CPU and the
            # accuracy loss is well under the noise floor of live stream audio.
            compute_type = "float16" if device == "cuda" else "int8"
        self.device, self.compute_type = device, compute_type
        self.model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, language: str | None, task: str) -> Transcript:
        segments, info = self.model.transcribe(
            audio,
            language=language,
            task=task,
            beam_size=1,               # realtime: greedy is ~2x faster, barely worse
            temperature=[0.0, 0.2, 0.4],
            condition_on_previous_text=False,
            vad_filter=False,          # we already ran Silero upstream
            no_speech_threshold=0.6,
        )
        parts, no_speech = [], 0.0
        for s in segments:
            parts.append(s.text)
            no_speech = max(no_speech, getattr(s, "no_speech_prob", 0.0))
        return Transcript("".join(parts).strip(), info.language or language or "", no_speech)


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def load_asr(backend: str, model: str, device: str = "auto", compute_type: str = "auto"):
    """backend: auto | mlx | faster"""
    if backend in ("auto", "mlx"):
        try:
            engine = MLXWhisper(model)
            log.info("ASR: mlx-whisper (%s)", engine.repo)
            return engine
        except ImportError:
            if backend == "mlx":
                raise RuntimeError("mlx-whisper not installed: uv sync --extra mlx") from None
            log.info("mlx-whisper unavailable, falling back to faster-whisper")
    engine = FasterWhisper(model, compute_type, device)
    log.info("ASR: faster-whisper (%s, %s %s)", model, engine.compute_type, engine.device)
    return engine
