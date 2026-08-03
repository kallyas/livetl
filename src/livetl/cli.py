"""Command line entry point."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys

from . import langs
from .capture import CaptureSpec
from .pipeline import Pipeline
from .sinks import ConsoleSink, Fanout, FileSink, JsonlSink, WebSocketSink
from .vad import SegmenterConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="livetl",
        description="Realtime speech translation for live streams.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  livetl https://twitch.tv/somechannel --source ja --target en
  livetl https://youtube.com/watch?v=XXXX --source es --target en --mt google
  livetl https://twitch.tv/ch --target de --mt nllb --overlay --obs-file subs.txt
  livetl clip.mp4 --source fr --target en          # test on a local file
  livetl --list-devices                            # then: livetl 1 --device
""",
    )
    p.add_argument("source", nargs="?", help="stream URL, local file, or device index with --device")

    g = p.add_argument_group("languages")
    g.add_argument("--source-lang", "--source", dest="source_lang", default="auto",
                   help="spoken language (ISO-639-1) or 'auto' (default: auto)")
    g.add_argument("--target", default="en", help="translate into this language (default: en)")

    g = p.add_argument_group("speech recognition")
    g.add_argument("--asr", default="auto", choices=["auto", "mlx", "faster"],
                   help="auto picks mlx-whisper on Apple Silicon, else faster-whisper")
    g.add_argument("--model", default="small",
                   help="tiny|base|small|medium|large-v3|large-v3-turbo, or an explicit repo id")
    g.add_argument("--asr-device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="faster-whisper only; auto uses CUDA when visible (default: auto)")
    g.add_argument("--compute-type", default="auto",
                   help="faster-whisper quantization: float16 (GPU), int8, int8_float16 …")

    g = p.add_argument_group("translation")
    g.add_argument("--mt", default="auto",
                   choices=["auto", "marian", "nllb", "google", "whisper", "none"],
                   help="auto = marian with a known --source, else nllb. "
                        "'whisper' uses Whisper's own translate task (English only, lower quality)")
    g.add_argument("--google-key", help="overrides $GOOGLE_TRANSLATE_API_KEY")
    g.add_argument("--nllb-model", default="facebook/nllb-200-distilled-600M")
    g.add_argument("--mt-device", default="auto", choices=["auto", "mps", "cuda", "cpu"])

    g = p.add_argument_group("segmentation / latency")
    g.add_argument("--min-silence", type=int, default=550, metavar="MS",
                   help="pause length that ends an utterance (default: 550)")
    g.add_argument("--soft-max", type=float, default=6.0, metavar="S",
                   help="past this, cut on a much shorter pause (default: 6)")
    g.add_argument("--max-segment", type=float, default=12.0, metavar="S",
                   help="hard cut for continuous speech (default: 12)")
    g.add_argument("--vad-threshold", type=float, default=0.5)
    g.add_argument("--partials", action="store_true",
                   help="show interim text mid-utterance (lower perceived latency, ~2x ASR load)")

    g = p.add_argument_group("output")
    g.add_argument("--quiet", action="store_true", help="no console captions")
    g.add_argument("--no-source", action="store_true", help="hide original-language text")
    g.add_argument("--timing", action="store_true", help="per-utterance latency breakdown")
    g.add_argument("--overlay", action="store_true", help="serve a browser/OBS overlay")
    g.add_argument("--overlay-port", type=int, default=8765)
    g.add_argument("--overlay-host", default="127.0.0.1")
    g.add_argument("--obs-file", metavar="PATH", help="rolling text file for an OBS text source")
    g.add_argument("--obs-lines", type=int, default=2)
    g.add_argument("--log", metavar="PATH", help="append captions as JSONL")

    g = p.add_argument_group("capture")
    g.add_argument("--device", action="store_true", help="treat SOURCE as an avfoundation device index")
    g.add_argument("--list-devices", action="store_true", help="list audio input devices and exit")
    g.add_argument("--reconnect", dest="reconnect", action="store_true", default=None,
                   help="keep retrying the source, models stay loaded (default: on for URLs)")
    g.add_argument("--no-reconnect", dest="reconnect", action="store_false",
                   help="exit when the stream ends or fails")
    g.add_argument("--puller", default="auto", choices=["auto", "streamlink", "yt-dlp"])
    g.add_argument("--quality", default="audio_only,worst,best",
                   help="streamlink quality fallback list (default: audio_only,worst,best)")

    p.add_argument("-v", "--verbose", action="count", default=0)
    return p


def list_devices() -> int:
    import subprocess

    print("Audio input devices (use the index with --device):\n")
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True, text=True,
    )
    audio = False
    for line in proc.stderr.splitlines():
        if "AVFoundation audio devices" in line:
            audio = True
            continue
        if "AVFoundation video devices" in line:
            audio = False
            continue
        if audio and "] [" in line:
            print("  " + line.split("] ", 1)[-1])
    print(
        "\nTo capture what your speakers are playing you need a loopback device\n"
        "(`brew install blackhole-2ch`, then route the browser's output to it)."
    )
    return 0


def resolve_mt(args) -> tuple[str, str]:
    """Return (mt_backend, whisper_task).

    Deliberately does NOT default to Whisper's built-in translate task even
    for English targets. That task is a soft instruction small models often
    ignore outright (measured: whisper-small fp16 returned the untranslated
    source), and when it is honoured the output is clearly worse than a real
    MT model. It stays available behind an explicit --mt whisper.
    """
    if args.mt == "auto":
        # Marian is one model per pair, so it needs a known source language.
        # With auto-detect, NLLB's single 200-language model is the only
        # local option that can handle whatever Whisper reports.
        return ("marian", "transcribe") if args.source_lang != "auto" else ("nllb", "transcribe")
    if args.mt == "whisper":
        if args.target != "en":
            raise SystemExit(
                f"--mt whisper only produces English; --target is {args.target!r}. "
                f"Use --mt marian, nllb or google."
            )
        return "whisper", "translate"
    return args.mt, "transcribe"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)],
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    for noisy in ("urllib3", "httpx", "huggingface_hub", "filelock", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if not args.verbose:
        # HF's download bars scribble over the caption stream
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    if args.list_devices:
        return list_devices()
    if not args.source:
        build_parser().print_usage(sys.stderr)
        print("livetl: error: SOURCE is required", file=sys.stderr)
        return 2

    mt_backend, whisper_task = resolve_mt(args)

    if mt_backend == "marian" and args.source_lang == "auto":
        print(
            "note: --mt marian needs a fixed source language (one model per pair).\n"
            "      Pass e.g. --source ja, or use --mt nllb / --mt google with auto-detect.",
            file=sys.stderr,
        )
        return 2

    from .asr import load_asr
    from .mt import load_mt

    print(
        f"livetl: {langs.name(args.source_lang) if args.source_lang != 'auto' else 'auto-detect'}"
        f" -> {langs.name(args.target)}  ·  asr={args.asr}/{args.model}  ·  mt={mt_backend}",
        file=sys.stderr,
    )

    try:
        asr = load_asr(args.asr, args.model, args.asr_device, args.compute_type)
        mt = load_mt(
            mt_backend,
            google_key=args.google_key,
            device=args.mt_device,
            nllb_model=args.nllb_model,
        )
    except Exception as exc:
        print(f"livetl: {exc}", file=sys.stderr)
        return 1

    sinks = []
    if not args.quiet:
        sinks.append(ConsoleSink(not args.no_source, args.timing, args.partials))
    if args.overlay:
        sinks.append(WebSocketSink(args.overlay_host, args.overlay_port))
        print(
            f"overlay: http://{args.overlay_host}:{args.overlay_port}/"
            f"  (add ?source=1&lines=3&size=40 to taste)",
            file=sys.stderr,
        )
    if args.obs_file:
        sinks.append(FileSink(args.obs_file, args.obs_lines, not args.no_source))
    if args.log:
        sinks.append(JsonlSink(args.log))
    sink = Fanout(sinks)

    # A live URL can drop, go offline or blip and should be waited on; a local
    # file that reached its end is simply done.
    is_stream = args.source.startswith(("http://", "https://")) or args.device
    reconnect = is_stream if args.reconnect is None else args.reconnect

    pipe = Pipeline(
        CaptureSpec(args.source, args.puller, args.quality, args.device),
        asr,
        mt,
        sink,
        source_lang=args.source_lang,
        target_lang=args.target,
        whisper_task=whisper_task,
        seg_cfg=SegmenterConfig(
            threshold=args.vad_threshold,
            min_silence_ms=args.min_silence,
            soft_max_s=args.soft_max,
            max_segment_s=args.max_segment,
            partials=args.partials,
        ),
        reconnect=reconnect,
    )

    signal.signal(signal.SIGINT, lambda *_: pipe.stop())
    signal.signal(signal.SIGTERM, lambda *_: pipe.stop())

    print("loading models…", file=sys.stderr)
    pipe.warmup()
    print("listening. ctrl-c to stop.\n", file=sys.stderr)

    code = 0
    try:
        pipe.run()
    except RuntimeError as exc:
        print(f"\nlivetl: {exc}", file=sys.stderr)
        code = 1
    except KeyboardInterrupt:
        pass
    finally:
        sink.close()
        print(f"\n{pipe.stats.summary()}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
