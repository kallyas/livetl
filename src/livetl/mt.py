"""Translation backends.

  marian  - Helsinki-NLP opus-mt, one small model per language pair (~300 MB).
            Fastest local option; best default on a memory-tight machine.
  nllb    - facebook/nllb-200-distilled-600M. 200 languages in one model,
            better quality, ~2.5 GB resident.
  google  - Cloud Translation API v2. Needs GOOGLE_TRANSLATE_API_KEY.
  whisper - No MT stage; Whisper's own translate task already produced English.
  none    - Pass the transcript through untranslated.
"""

from __future__ import annotations

import logging
import os
import threading

import requests

from . import langs

log = logging.getLogger(__name__)

# Greedy decoding on a disfluent ASR transcript sends these models into
# degenerate loops ("a year to a year to a year ..." was reproduced live on
# ja->en). no_repeat_ngram_size is what actually breaks the cycle; the
# repetition penalty just discourages it earlier.
GEN_KWARGS = dict(
    num_beams=1,
    max_new_tokens=256,
    no_repeat_ngram_size=4,
    repetition_penalty=1.15,
)


def _dtype_kwarg() -> str:
    """Name of from_pretrained's dtype argument for the installed transformers.

    It was renamed torch_dtype -> dtype in 4.56. from_pretrained takes
    **kwargs, so passing the wrong one raises nothing at all -- it lands in
    the config and the model quietly loads at full precision, doubling
    memory. Colab still ships 4.x, so both spellings have to be supported.
    """
    from transformers import __version__ as ver

    try:
        major, minor = (int(part) for part in ver.split(".")[:2])
    except ValueError:
        return "dtype"
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


def _tune_generation(model):
    """Clear the config's max_length so max_new_tokens isn't 'conflicting'.

    Both being set is harmless -- max_new_tokens wins -- but transformers
    warns about it on every single call, once per caption.
    """
    model.generation_config.max_length = None
    return model


class Passthrough:
    name = "none"
    needs_source_lang = False

    def translate(self, text: str, src: str | None, tgt: str) -> str:
        return text


class WhisperNative(Passthrough):
    """Whisper already translated to English upstream; nothing left to do."""

    name = "whisper"


class Google:
    name = "google"
    needs_source_lang = False
    ENDPOINT = "https://translation.googleapis.com/language/translate/v2"

    def __init__(self, api_key: str | None = None, timeout: float = 5.0):
        self.key = api_key or os.environ.get("GOOGLE_TRANSLATE_API_KEY", "")
        if not self.key:
            raise RuntimeError(
                "--mt google needs an API key: export GOOGLE_TRANSLATE_API_KEY=..., "
                "or pass --google-key."
            )
        self.timeout = timeout
        self.session = requests.Session()

    def translate(self, text: str, src: str | None, tgt: str) -> str:
        payload = {"q": text, "target": tgt, "format": "text"}
        if src and src != "auto":
            payload["source"] = src
        r = self.session.post(
            self.ENDPOINT, params={"key": self.key}, json=payload, timeout=self.timeout
        )
        if r.status_code != 200:
            raise RuntimeError(f"Google Translate {r.status_code}: {r.text[:200]}")
        return r.json()["data"]["translations"][0]["translatedText"]


class Marian:
    name = "marian"
    needs_source_lang = True

    def __init__(self, device: str = "auto"):
        import torch

        self.torch = torch
        self.device = _pick_device(device, torch)
        self._pairs: dict[tuple[str, str], tuple] = {}
        self._lock = threading.Lock()

    def _load(self, src: str, tgt: str):
        key = (src, tgt)
        with self._lock:
            if key in self._pairs:
                return self._pairs[key]
            from transformers import MarianMTModel, MarianTokenizer

            repo = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
            log.info("loading %s on %s", repo, self.device)
            try:
                tok = MarianTokenizer.from_pretrained(repo)
                model = _tune_generation(MarianMTModel.from_pretrained(repo)).to(self.device).eval()
            except Exception as exc:
                raise RuntimeError(
                    f"No Marian model for {src}->{tgt} ({repo}). Not every pair "
                    f"exists; use --mt nllb or --mt google for this one."
                ) from exc
            self._pairs[key] = (tok, model)
            return self._pairs[key]

    def translate(self, text: str, src: str | None, tgt: str) -> str:
        if not src or src == "auto":
            raise RuntimeError("marian needs a concrete source language")
        if src == tgt:
            return text
        tok, model = self._load(src, tgt)
        batch = tok([text], return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with self.torch.no_grad():
            out = model.generate(**batch, **GEN_KWARGS)
        return tok.decode(out[0], skip_special_tokens=True)


class NLLB:
    name = "nllb"
    needs_source_lang = True

    def __init__(self, model: str = "facebook/nllb-200-distilled-600M", device: str = "auto"):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = _pick_device(device, torch)
        log.info("loading %s on %s", model, self.device)
        self.tok = AutoTokenizer.from_pretrained(model)
        dtype = {_dtype_kwarg(): torch.float16 if self.device != "cpu" else torch.float32}
        self.model = _tune_generation(
            AutoModelForSeq2SeqLM.from_pretrained(model, **dtype)
        ).to(self.device).eval()
        self._name = model

    def translate(self, text: str, src: str | None, tgt: str) -> str:
        if not src or src == "auto":
            raise RuntimeError("nllb needs a concrete source language")
        if src == tgt:
            return text
        self.tok.src_lang = langs.to_flores(src)
        batch = self.tok([text], return_tensors="pt", truncation=True, max_length=512).to(self.device)
        bos = self.tok.convert_tokens_to_ids(langs.to_flores(tgt))
        with self.torch.no_grad():
            out = self.model.generate(**batch, forced_bos_token_id=bos, **GEN_KWARGS)
        return self.tok.batch_decode(out, skip_special_tokens=True)[0]


def _pick_device(requested: str, torch) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_mt(backend: str, *, google_key: str | None = None, device: str = "auto",
            nllb_model: str = "facebook/nllb-200-distilled-600M"):
    if backend == "none":
        return Passthrough()
    if backend == "whisper":
        return WhisperNative()
    if backend == "google":
        return Google(google_key)
    if backend == "marian":
        return Marian(device)
    if backend == "nllb":
        return NLLB(nllb_model, device)
    raise ValueError(f"unknown --mt backend {backend!r}")
