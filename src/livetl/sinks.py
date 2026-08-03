"""Output sinks: terminal, OBS text file, JSONL log, websocket overlay."""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Caption:
    seq: int
    final: bool
    source: str
    target: str
    src_lang: str = ""
    tgt_lang: str = ""
    asr_ms: int = 0
    mt_ms: int = 0
    lag_ms: int = 0          # speech onset -> caption ready
    audio_s: float = 0.0
    t: float = field(default_factory=time.time)


class Sink:
    def emit(self, cap: Caption) -> None: ...
    def close(self) -> None: ...


class ConsoleSink(Sink):
    def __init__(self, show_source: bool = True, timing: bool = False, partials: bool = False):
        from rich.console import Console

        self.console = Console()
        self.show_source = show_source
        self.timing = timing
        # A partial is an in-place preview and only makes sense on a terminal.
        # In a notebook cell, a pipe or a log file there is nothing to redraw,
        # so Live appends instead of overwriting and every partial lands as its
        # own line, burying the finished captions in hundreds of near-identical
        # ones. Fall back to finals only.
        self._live = None
        if partials and self.console.is_terminal:
            from rich.live import Live

            self._live = Live(console=self.console, refresh_per_second=8, transient=True)
            self._live.start()
        elif partials:
            log.info("partials hidden: stdout is not a terminal (finals still shown)")

    def emit(self, cap: Caption) -> None:
        from rich.text import Text

        if not cap.final:
            if self._live:
                self._live.update(Text(f"  … {cap.target or cap.source}", style="dim italic"))
            return

        if self._live:
            self._live.update(Text(""))
        stamp = time.strftime("%H:%M:%S", time.localtime(cap.t))
        head = Text(f"[{stamp}] ", style="dim")
        if self.show_source and cap.source and cap.source != cap.target:
            src = Text(cap.source, style="dim")
            src.truncate(400, overflow="ellipsis")
            self.console.print(head + Text(f"{cap.src_lang or '??'} ", style="dim cyan") + src)
            self.console.print(Text("           ") + Text(cap.target, style="bold white"))
        else:
            self.console.print(head + Text(cap.target or cap.source, style="bold white"))
        if self.timing:
            rt = cap.lag_ms / 1000.0
            self.console.print(
                Text(
                    f"           asr {cap.asr_ms}ms · mt {cap.mt_ms}ms · "
                    f"lag {rt:.1f}s · audio {cap.audio_s:.1f}s",
                    style="dim yellow",
                )
            )

    def close(self) -> None:
        if self._live:
            self._live.stop()


class FileSink(Sink):
    """Rolling plain-text file for an OBS 'Text (GDI+/FreeType)' source."""

    def __init__(self, path: str, lines: int = 2, include_source: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.buf: collections.deque[str] = collections.deque(maxlen=max(1, lines))
        self.include_source = include_source

    def emit(self, cap: Caption) -> None:
        if not cap.final:
            return
        self.buf.append(f"{cap.source}\n{cap.target}" if self.include_source else cap.target)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text("\n".join(self.buf) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)  # atomic, so OBS never reads a half-written file

    def close(self) -> None:
        pass


class JsonlSink(Sink):
    def __init__(self, path: str, include_partials: bool = False):
        self.fh = open(path, "a", encoding="utf-8", buffering=1)
        self.include_partials = include_partials

    def emit(self, cap: Caption) -> None:
        if cap.final or self.include_partials:
            self.fh.write(json.dumps(asdict(cap), ensure_ascii=False) + "\n")

    def close(self) -> None:
        self.fh.close()


class WebSocketSink(Sink):
    """Serves the overlay page and pushes captions to it.

    Point an OBS Browser Source at http://host:port/ , or just open it.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, history: int = 25):
        self.host, self.port = host, port
        self._clients: set = set()
        self._history: collections.deque[str] = collections.deque(maxlen=history)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown: asyncio.Event | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ws-sink")
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError("websocket server failed to start")

    # -- server thread -----------------------------------------------------

    def _run(self) -> None:
        asyncio.run(self._serve())

    async def _serve(self) -> None:
        try:
            from websockets.asyncio.server import serve  # websockets >= 13
        except ImportError:  # pragma: no cover - older websockets
            from websockets.server import serve  # type: ignore

        self._loop = asyncio.get_running_loop()
        self._shutdown = asyncio.Event()
        page = (Path(__file__).parent / "web" / "overlay.html").read_bytes()

        async def process_request(connection, request):
            """Serve the overlay over plain HTTP; let websocket upgrades through."""
            if request.headers.get("Upgrade", "").lower() == "websocket":
                return None
            if request.path in ("/", "/index.html", "/overlay.html"):
                return _http_ok(connection, page)
            return _http_404(connection)

        async with serve(
            self._handler, self.host, self.port,
            process_request=process_request,
            ping_interval=20, ping_timeout=20,
        ):
            log.info("overlay: http://%s:%d/", self.host, self.port)
            self._ready.set()
            await self._shutdown.wait()

    async def _handler(self, ws) -> None:
        self._clients.add(ws)
        try:
            for msg in list(self._history):
                await ws.send(msg)
            await ws.wait_closed()
        finally:
            self._clients.discard(ws)

    # -- producer side -----------------------------------------------------

    def emit(self, cap: Caption) -> None:
        msg = json.dumps(asdict(cap), ensure_ascii=False)
        if cap.final:
            self._history.append(msg)
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._broadcast, msg)

    def _broadcast(self, msg: str) -> None:
        for ws in list(self._clients):
            # fire-and-forget: a stalled overlay must never stall transcription
            asyncio.create_task(_safe_send(ws, msg))

    def close(self) -> None:
        # Let serve() unwind on its own; calling loop.stop() out from under
        # asyncio.run() tears down mid-await and raises on the way out.
        if self._loop and self._shutdown:
            self._loop.call_soon_threadsafe(self._shutdown.set)
            self._thread.join(timeout=3)


async def _safe_send(ws, msg: str) -> None:
    try:
        await ws.send(msg)
    except Exception:
        pass


def _http_ok(connection, body: bytes):
    # Built by hand rather than via connection.respond(): Headers is a
    # multidict whose __setitem__ appends, so patching a respond() result
    # would emit two Content-Type headers.
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    return Response(
        200,
        "OK",
        Headers(
            {
                "Content-Type": "text/html; charset=utf-8",
                "Content-Length": str(len(body)),
                "Cache-Control": "no-store",
            }
        ),
        body,
    )


def _http_404(connection):
    from http import HTTPStatus

    return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")


class Fanout(Sink):
    def __init__(self, sinks: list[Sink]):
        self.sinks = sinks

    def emit(self, cap: Caption) -> None:
        for s in self.sinks:
            try:
                s.emit(cap)
            except Exception:
                log.exception("sink %s failed", type(s).__name__)

    def close(self) -> None:
        for s in self.sinks:
            try:
                s.close()
            except Exception:
                pass
