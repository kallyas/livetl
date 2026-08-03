# livetl

Realtime speech translation for live streams. Point it at a Twitch or YouTube
URL and get translated subtitles in your terminal, in a browser overlay, or in
an OBS text source.

Everything can run locally (Whisper + Marian/NLLB). Google Cloud Translation is
available as a drop-in alternative for the translation stage.

> [!WARNING]
> **This is a personal project. It may be broken at any given moment.**
>
> It works on the setup it was built on and has been tested end to end there,
> but it is not a maintained product: no releases, no support, no stability
> guarantees, and no promise that any of it still works tomorrow. Expect to
> read and edit the source to get it running on your machine.
>
> It leans on several moving parts it does not control — Twitch and YouTube
> stream endpoints, streamlink/yt-dlp, ffmpeg, and Hugging Face model repos.
> Any of those changing will break it, and site changes routinely do.
>
> The translations themselves are machine output over machine transcription:
> errors compound across the two stages, and casual speech, slang and proper
> nouns come out worst. Do not rely on it for anything that matters. See
> [Known limitations](#known-limitations) for the specifics that are already
> understood.

```
streamlink/yt-dlp ─▶ ffmpeg ─▶ 16kHz PCM ─▶ Silero VAD ─▶ Whisper ─▶ MT ─▶ sinks
   (pull stream)     (decode)   (frames)   (segment on     (ASR)   (local  (console/
                                            pauses)                 /Google) OBS/web)
```

## Quick start

`livetl.sh` installs everything on first run (uv, venv, the right extras for
your hardware) and then runs. You only need `ffmpeg` on PATH first:
`brew install ffmpeg` / `sudo apt install -y ffmpeg`.

```sh
./livetl.sh https://twitch.tv/somechannel --source ja --target en
```

Other entry points:

```sh
./livetl.sh setup      # install/refresh deps only
./livetl.sh doctor     # what's installed, what hardware, what's recommended
./livetl.sh --help
```

Put defaults in `livetl.env` (copy `livetl.env.example`) so you don't retype
them; anything on the command line overrides the file. Values containing
spaces must be quoted:

```sh
STREAM_URL=https://twitch.tv/somechannel
SOURCE_LANG=ja
TARGET_LANG=en
EXTRA_ARGS="--timing --soft-max 4"
```

Models download on first use (~500 MB Whisper, ~300 MB per Marian pair,
~2.4 GB for NLLB) and are cached in `~/.cache/huggingface`.

## Use

```sh
# Spanish YouTube live -> English, via Google Translate
export GOOGLE_TRANSLATE_API_KEY=...
./livetl.sh 'https://youtube.com/watch?v=XXXX' --source es --mt google

# Browser/OBS overlay + a text file for an OBS text source
./livetl.sh https://twitch.tv/ch --source ja --overlay --obs-file subs.txt

# Try it on a local file first
./livetl.sh clip.mp4 --source fr --target en --timing
```

`--source` is optional but strongly recommended — see *Language detection* below.

To bypass the wrapper entirely: `.venv/bin/python -m livetl <same args>`.

### Reconnecting

For live URLs the process reconnects on its own and **keeps the models
loaded**, so a stream ending, an ad break or a network blip costs about a
second rather than a full model reload. If the channel is offline it simply
waits, and starts captioning when the broadcast begins. Disable with
`--no-reconnect`; local files always exit at EOF.

`./livetl.sh --retry` adds an outer restart loop for hard crashes. It is not
needed under systemd, which already supervises.

## Choosing backends

Measured on an M3 (8 GB), `small` model, per utterance, one backend at a time
with models already downloaded:

| Stage | Backend | Speed | Notes |
|---|---|---|---|
| ASR | `--asr mlx` | ~0.4-0.6 s (RTF 0.08x) | Apple GPU. Default on Apple Silicon. |
| ASR | `--asr faster` | ~1.5 s (RTF 0.38x) | CPU int8. Portable fallback. |
| MT | `--mt marian` | ~0.2-0.7 s | One ~300 MB model per pair. Fastest, needs `--source`. |
| MT | `--mt nllb` | ~0.9-1.1 s | 200 languages in one 2.4 GB model. Better on ja→en. |
| MT | `--mt google` | network RTT | Best quality, costs money, sends audio transcripts off-machine. |
| MT | `--mt whisper` | free | See warning below. |

`--mt auto` (the default) picks Marian when you pass `--source`, NLLB otherwise.

Bigger Whisper models are the main quality lever: `--model medium` roughly
doubles ASR time but is noticeably better on accents and proper nouns. On 8 GB,
`small` plus NLLB is about the practical ceiling before swapping.

### Don't rely on `--mt whisper`

Whisper can translate to English itself (`task=translate`), which sounds like it
should replace the MT stage. In testing it was unreliable: `whisper-small` fp16
**ignored the instruction entirely** and returned untranslated Spanish, and the
q4 build produced "Welcome to the Diary of the Huy" for "bienvenidos al directo
de hoy". It is English-only and lower quality than a real MT model. Kept as an
option, not a default.

## Latency

End-to-end lag is roughly **utterance length + 1.4 s**. The fixed part is the
pause detection (~0.55 s) plus ASR and MT; the variable part is that a sentence
cannot be translated until it has been spoken.

That's the tradeoff for cutting on speech pauses instead of a fixed timer.
Whisper is a 30-second-window model, not a streaming one — fixed chunks slice
words in half and hand the translator sentence fragments, which wrecks
translation quality far more than a second of latency does.

Knobs, roughly in order of usefulness:

- `--soft-max 4` — after this many seconds of continuous speech, accept a much
  shorter pause (240 ms) as a cut point. This is what keeps monologues from
  sitting at 10 s+. Default 6.
- `--partials` — show interim text mid-utterance. Big drop in *perceived*
  latency, roughly doubles ASR load.
- `--min-silence 400` — shorter pause ends an utterance. Too low fragments
  sentences mid-clause.
- `--model tiny` / `base` — faster, meaningfully worse.

## Output sinks

All can be combined.

- **Console** (default) — source line dim, translation bright. `--timing` adds a
  per-utterance latency breakdown, `--no-source` hides the original.
- **`--overlay`** — serves `http://127.0.0.1:8765/`. Use as an OBS Browser
  Source, or just open it. Query params: `?source=1` show original text,
  `&lines=3` how many stay on screen, `&size=40` font px, `&bg=1` solid
  background, `&hold=9` seconds before fading. Reconnects on its own.
- **`--obs-file subs.txt`** — rolling plain text, written atomically so OBS
  never reads a half-written file. Point a Text source at it with "read from
  file".
- **`--log out.jsonl`** — every caption with timings, for tuning after the fact.

## Running on a server

Everything is headless already — no display, no audio device needed, since the
stream is pulled over HTTP and decoded to PCM in memory.

```sh
sudo cp contrib/livetl.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now livetl
journalctl -u livetl -f
```

The unit reads `livetl.env` from its `WorkingDirectory`, so set `STREAM_URL`
there. Point `HF_HOME` at a directory the service user owns — otherwise every
restart re-downloads the models.

### Sizing

The ASR stage dominates; MT is a rounding error next to it. The number that
matters is **RTF** (seconds of compute per second of audio) — anything at or
above 1.0x means the pipeline can't keep up and starts shedding audio.

| Workload | Spec | Flags |
|---|---|---|
| 1 stream, cheapest | 4 vCPU (AVX2), 8 GB RAM, no GPU | `--asr faster --model small --source xx --mt marian` |
| 1–2 streams, better quality | 8 vCPU, 16 GB RAM, no GPU | `--model medium` |
| 3+ streams, or `large-v3` | NVIDIA T4/L4 16 GB, 8 vCPU, 16 GB RAM | `--asr faster --asr-device cuda --model large-v3-turbo` |

Notes on each axis:

- **GPU.** This is the one upgrade that changes what's possible. `large-v3-turbo`
  in float16 occupies ~1.6 GB of VRAM and runs several times faster than `small`
  does on CPU, so a single 16 GB T4 comfortably hosts several streams at better
  quality than any CPU config. Install with `uv sync --extra cuda --extra localmt`
  (CTranslate2 4.x needs CUDA 12 + cuDNN 9; the extra pulls matching pip wheels).
  `--asr-device auto` uses CUDA whenever CTranslate2 can see a device.
- **CPU.** Without a GPU, budget ~4 modern cores per stream for `small` int8 and
  expect RTF in the 0.3–0.5x range. AVX2 is effectively required; AVX-512 helps.
  Core *count* matters more than clock — CTranslate2 threads well.
- **RAM.** 8 GB runs Whisper `small` + Marian. NLLB adds ~2.4 GB resident and is
  what pushes a small box into swap. 16 GB if you want `medium` or NLLB.
- **Disk.** A few GB for the model cache; on a fresh box the first start spends
  most of its time downloading, not translating.
- **Network.** Trivial for Twitch — the default quality list prefers
  `audio_only` (~160 kbps). YouTube has no audio-only HLS rendition, so a video
  stream gets pulled and discarded; keep `--quality worst` there if bandwidth
  is metered.
- **Multiple streams.** Run one process per stream with its own
  `--overlay-port`. They share the on-disk model cache but not memory, so
  multiply RAM per instance — the GPU path is far more economical here.

### Google Colab

`colab.sh` is a separate launcher for Colab, which is different enough from a
normal server to need its own path. Set **Runtime > Change runtime type > T4
GPU** first, then:

```python
!git clone https://github.com/YOU/transcribe-audio.git
%cd transcribe-audio
!bash colab.sh setup
!bash colab.sh run https://twitch.tv/somechannel --source ja --target en
```

`bash colab.sh doctor` reports what's installed and, importantly, whether
CTranslate2 can actually see the GPU.

What it does differently from `livetl.sh`:

- **No venv.** Colab already ships torch built against its exact driver.
  Creating a venv would pull a second multi-gigabyte torch and risk a CUDA
  mismatch, so dependencies install into the runtime's own Python and torch and
  transformers are left untouched.
- **Sets `LD_LIBRARY_PATH` for CTranslate2.** It dlopens cuBLAS and cuDNN at
  runtime, and on Colab those live inside the pip `nvidia-*` packages rather
  than a system CUDA install. Without this it reports zero GPUs and silently
  runs on CPU — which will not keep up with a live stream.
- **Defaults to `large-v3-turbo` in float16.** ~15 GB of VRAM holds it (~3 GB)
  alongside NLLB with room to spare, and the quality gain over `small` is
  large. Override with `MODEL=medium bash colab.sh run ...`. With no GPU
  visible it warns and drops to a small CPU model.
- **Always writes `captions.jsonl`**, since a Colab cell is a worse place to
  read scrollback than a file.

#### Getting the most out of the T4

The instinct is to worry about the 15 GB of VRAM, but that is not the
constraint — `large-v3` in float16 is about 3 GB. The constraint is that this
pipeline is **latency-bound, not throughput-bound**: it handles one utterance
at a time and then waits for the speaker, so a T4 running `large-v3-turbo`
sits idle most of the time. Using the card fully means spending that idle
capacity on accuracy.

Roughly in order of value per unit of idle GPU:

```python
!bash colab.sh run https://twitch.tv/CHANNEL --source ja --target en \
    --model large-v3 --beam-size 5 \
    --mt nllb --nllb-model facebook/nllb-200-distilled-1.3B --mt-device cuda \
    --partials --soft-max 4 --timing
```

- **`--model large-v3`** instead of turbo. Turbo has 4 decoder layers against
  32 and gives most of the quality for a fraction of the cost — a good trade
  when compute is scarce, which here it isn't. The gap shows up on exactly what
  streams are full of: accents, proper nouns, non-English.
- **`--beam-size 5`.** Greedy decoding is the default because it is 2-3x
  cheaper, and that matters on CPU. On an idle GPU it is the cheapest accuracy
  you can buy. faster-whisper only — mlx-whisper has no beam decoder.
- **A larger MT model.** `nllb-200-distilled-1.3B` is ~2.6 GB in float16 and
  clearly better than the 600M default on casual speech; both still fit
  alongside `large-v3`.
- **`--partials`** roughly doubles ASR load to show interim text. On a T4 that
  load is free, and it is the single biggest improvement to how responsive the
  captions *feel*.
- **`--soft-max 4`** cuts monologues into captions sooner. Costs nothing.

T4-specific notes:

- Turing has strong INT8 tensor cores, so **`--compute-type int8_float16` is
  often faster than `float16` on a T4** at negligible quality cost. Worth
  measuring both — it is one flag.
- Turing predates bfloat16. `--compute-type bfloat16` will not work; that is
  Ampere and newer.
- **Watch RTF in the `--timing` output.** It is seconds of compute per second
  of audio. Keep it under ~0.5 so bursts have headroom; as it approaches 1.0
  the pipeline starts shedding audio, which shows up as `dropped` in the exit
  summary. Turn the knobs above up until RTF stops being comfortable.
- **Several streams at once** is the other way to use the card: one process
  each, different `--overlay-port`. Each loads its own copy of the weights, so
  `large-v3` at ~3 GB means three or four concurrent streams on 15 GB.
- Point `HF_HOME` at mounted Drive, or every fresh runtime re-downloads
  several gigabytes before it says a word.

Free-tier T4s are preemptible and time-capped, so treat a long session as
something that will be interrupted rather than something that will keep
running.

The overlay is the awkward part: Colab has no inbound networking, so it has to
go through the authenticated port proxy. `bash colab.sh overlay` prints the
exact steps. The proxy does forward websockets but it is the flakiest piece
here — if captions never show up in the overlay, they are still in the cell
output and in `captions.jsonl`.

Colab is fine for trying this out or translating a stream you're watching now.
It is not somewhere to run it continuously: sessions are killed after a period
of inactivity and capped at a handful of hours, and the models re-download on
every fresh runtime unless you point `HF_HOME` at your mounted Drive.

### Exposing the overlay

`--overlay-host 0.0.0.0` serves it on all interfaces. **There is no
authentication**, so put it behind a firewall, a reverse proxy with auth, or
just tunnel it: `ssh -L 8765:127.0.0.1:8765 user@server`.

## Capturing system audio instead

Pulling the stream directly is the default: no extra software, works headless,
clean audio. The downside is that your `livetl` copy and your browser sit at
slightly different points in the live stream, so subtitles won't line up with
what you're watching.

For perfectly synced subtitles, capture what your speakers are playing:

```sh
.venv/bin/python -m livetl --list-devices
.venv/bin/python -m livetl 4 --device --source ja    # 4 = a loopback device
```

This needs a loopback audio device (BlackHole, eqMac, Loopback) with your
browser's output routed through it. It also works for anything else that plays
audio — Zoom, a local video file, DRM-protected players streamlink can't touch.

## Known limitations

- **Per-utterance language detection is unreliable on short clips.** Whisper
  decides from a single 30 s window, so a 1.5 s utterance can be misidentified,
  and a wrong guess sends the wrong source language to the translator. Pass
  `--source` whenever you know it. (Synthetic TTS is especially bad: both
  Whisper backends independently called macOS `say` Spanish "English" at 0.9
  confidence, while real speech detects fine.)
- **Marian doesn't cover every pair.** `Helsinki-NLP/opus-mt-{src}-{tgt}` has to
  exist; if it doesn't you get a clear error pointing at `--mt nllb`/`google`.
- **Whisper hallucinates on silence and music.** A blocklist filters the usual
  output ("Thank you", "♪", "Subscribe", subtitle-site credits) plus decoder
  repetition loops; the filtered count shows in the exit summary. A music-heavy
  stream will still produce occasional nonsense.
- **Machine translation of casual speech is rough.** ASR disfluencies compound
  with MT, and streamer slang is not in these models' training data. Prefer NLLB
  or Google for Japanese.
- **8 GB is tight, and NLLB is what makes it tight.** Whisper `small` + NLLB
  (2.4 GB) runs, but under memory pressure ASR degraded from RTF 0.08x to
  1.24x — i.e. slower than realtime, at which point the pipeline starts
  shedding audio. On 8 GB prefer `--source xx --mt marian`, and treat
  `--mt nllb` as the quality option to reach for when Marian lacks the pair.
  `medium` + NLLB will swap.

## Layout

| File | Role |
|---|---|
| `capture.py` | stream/file/device → 16 kHz mono float32 frames |
| `vad.py` | Silero VAD → utterance-sized segments |
| `asr.py` | Whisper backends, language detection, hallucination filter |
| `mt.py` | Marian / NLLB / Google / passthrough |
| `pipeline.py` | threads, bounded queues, load shedding |
| `sinks.py` | console, OBS file, JSONL, websocket overlay |
| `cli.py` | argument parsing and wiring |
