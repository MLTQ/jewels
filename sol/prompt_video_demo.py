"""Serve a small LAN browser demo for the frozen prompt-to-Jewel video proof."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any
from urllib.parse import unquote, urlparse

from sol.prompt_video_runtime import PromptVideoPaths, PromptVideoRuntime


def parse_byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Parse the single HTTP byte range used by browser video players."""
    if not header:
        return None
    if not header.startswith("bytes=") or "," in header or size <= 0:
        raise ValueError("unsupported byte range")
    first, last = header[6:].split("-", 1)
    if not first:
        suffix = int(last)
        if suffix <= 0:
            raise ValueError("invalid byte range")
        return max(0, size - suffix), size - 1
    start = int(first)
    end = size - 1 if not last else min(int(last), size - 1)
    if start < 0 or start >= size or end < start:
        raise ValueError("unsatisfiable byte range")
    return start, end


def demo_html(prompts: tuple[str, ...], learned_available: bool) -> str:
    """Return the self-contained, dependency-free demo interface."""
    prompts_json = json.dumps(prompts).replace("</", "<\\/")
    learned_disabled = "" if learned_available else "disabled"
    learned_note = (
        "Free-form mode maps new wording into the same three learned scene families."
        if learned_available
        else "The optional learned speaker checkpoint is not installed."
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jewel Field — prompt to video proof</title>
<style>
:root {{ color-scheme: dark; --ink:#f6f1e7; --muted:#a7a29a; --line:#35322f;
  --gold:#f0b95b; --teal:#66d8cb; --panel:rgba(24,23,22,.91); }}
* {{ box-sizing:border-box; }}
body {{ margin:0; min-height:100vh; color:var(--ink); font:15px/1.5 Inter,ui-sans-serif,system-ui;
  background:radial-gradient(circle at 15% 10%,#183a38 0,transparent 25%),
  radial-gradient(circle at 85% 20%,#4a3020 0,transparent 26%),#0b0c0c; }}
body:before {{ content:""; position:fixed; inset:0; pointer-events:none; opacity:.16;
  background-image:linear-gradient(30deg,transparent 48%,#fff 49%,transparent 50%);
  background-size:34px 34px; }}
.wrap {{ width:min(1160px,calc(100% - 36px)); margin:0 auto; padding:54px 0 70px; }}
header {{ display:flex; justify-content:space-between; gap:30px; align-items:end; margin-bottom:28px; }}
.eyebrow {{ color:var(--teal); text-transform:uppercase; letter-spacing:.18em; font-size:12px; }}
h1 {{ margin:5px 0 0; font:600 clamp(35px,6vw,68px)/.98 Georgia,serif; letter-spacing:-.04em; }}
.claim {{ max-width:430px; color:var(--muted); font-size:16px; }}
.grid {{ display:grid; grid-template-columns:minmax(300px,.82fr) minmax(420px,1.35fr); gap:18px; }}
.card {{ position:relative; border:1px solid var(--line); border-radius:18px; padding:22px;
  background:var(--panel); box-shadow:0 18px 80px #0008; backdrop-filter:blur(14px); }}
label.title {{ display:block; margin:0 0 8px; font-weight:650; }}
textarea,input {{ width:100%; color:var(--ink); background:#0d0e0e; border:1px solid #44413d;
  border-radius:10px; padding:12px; font:inherit; outline:none; }}
textarea {{ min-height:112px; resize:vertical; }}
textarea:focus,input:focus {{ border-color:var(--teal); box-shadow:0 0 0 3px #66d8cb22; }}
.examples {{ display:flex; flex-wrap:wrap; gap:7px; margin:9px 0 20px; }}
.chip {{ border:1px solid #48443f; background:#22211f; color:#ddd5c9; border-radius:999px;
  padding:6px 10px; cursor:pointer; font-size:12px; }}
.mode {{ display:grid; gap:9px; margin:10px 0 18px; }}
.mode label {{ display:grid; grid-template-columns:20px 1fr; gap:9px; align-items:start; color:var(--muted); }}
.mode input {{ width:auto; margin-top:4px; accent-color:var(--gold); }}
.mode strong {{ display:block; color:var(--ink); }}
.row {{ display:grid; grid-template-columns:1fr auto; gap:12px; align-items:end; }}
button.go {{ border:0; border-radius:11px; padding:13px 19px; color:#17110a; background:var(--gold);
  font-weight:750; cursor:pointer; min-width:150px; }}
button.go:disabled {{ opacity:.55; cursor:wait; }}
.truth {{ border-top:1px solid var(--line); margin-top:20px; padding-top:17px; color:var(--muted); font-size:13px; }}
.stage {{ min-height:430px; display:grid; place-items:center; overflow:hidden; }}
.empty {{ text-align:center; color:var(--muted); max-width:390px; }}
.orb {{ width:78px; height:78px; margin:0 auto 18px; transform:rotate(45deg); border:1px solid #777;
  background:linear-gradient(135deg,#66d8cb55,#f0b95b33); box-shadow:inset 0 0 35px #fff2; }}
#result {{ display:none; width:100%; }}
video {{ display:block; width:100%; border-radius:12px; background:#000; image-rendering:auto; }}
.resultbar {{ display:flex; justify-content:space-between; gap:16px; margin-top:13px; align-items:start; }}
.scene {{ color:var(--teal); font-weight:650; }}
.download {{ color:var(--gold); text-decoration:none; white-space:nowrap; }}
.status {{ min-height:24px; margin-top:12px; color:var(--muted); }}
.status.error {{ color:#ff9b8d; }}
details {{ margin-top:12px; color:var(--muted); }} pre {{ white-space:pre-wrap; font-size:11px; color:#cbc4b8; }}
@media (max-width:820px) {{ header {{ display:block; }} .claim {{ margin-top:18px; }}
  .grid {{ grid-template-columns:1fr; }} .stage {{ min-height:330px; }} }}
</style>
</head>
<body><main class="wrap">
<header><div><div class="eyebrow">Native Jewel grammar / proof demo</div><h1>Speak into<br>the volume.</h1></div>
<div class="claim">A prompt and seed emit a scene token, persistent trajectory tokens, and 72,000
irregular time-distorted Gaussian Jewels—then render a 49-frame video.</div></header>
<div class="grid"><section class="card">
<label class="title" for="prompt">Describe the video</label>
<textarea id="prompt"></textarea><div class="examples" id="examples"></div>
<label class="title">Speaker mode</label><div class="mode">
<label><input type="radio" name="mode" value="exact" checked><span><strong>Proven prompts</strong>
The passing controlled experiment. Choose one of the examples above.</span></label>
<label><input type="radio" name="mode" value="learned" {learned_disabled}><span><strong>Free-form wording · experimental</strong>
{learned_note}</span></label></div>
<div class="row"><div><label class="title" for="seed">Seed</label><input id="seed" type="number" value="20260914"></div>
<button class="go" id="generate">Generate video</button></div>
<div class="status" id="status"></div>
<div class="truth"><strong>What this is:</strong> a genuine prompt-only native Jewel generation path.<br>
<strong>What it is not yet:</strong> an open-vocabulary production text-to-video model. Local detail is
soft and sparkly; macro tokens are still backed by a small source vocabulary.</div>
</section><section class="card stage"><div class="empty" id="empty"><div class="orb"></div>
Choose a proven prompt or try a nearby free-form description. A full render takes a little while on the GPU.</div>
<div id="result"><video id="video" controls loop playsinline></video><div class="resultbar">
<div><div class="scene" id="scene"></div><div id="programline"></div></div>
<a class="download" id="download" download>Download MP4 ↓</a></div>
<details><summary>Show emitted program and provenance</summary><pre id="metadata"></pre></details></div>
</section></div></main>
<script>
const prompts={prompts_json}; const promptBox=document.querySelector('#prompt');
prompts.forEach((p,i)=>{{const b=document.createElement('button');b.className='chip';b.textContent=['Ballerina','Retriever','Welder'][i]||`Prompt ${{i+1}}`;b.onclick=()=>{{promptBox.value=p;document.querySelector('[value=exact]').checked=true;}};document.querySelector('#examples').appendChild(b);}});
promptBox.value=prompts[0]||'';
document.querySelector('#generate').onclick=async()=>{{
 const button=document.querySelector('#generate'),status=document.querySelector('#status');button.disabled=true;
 status.className='status';status.textContent='Speaking the Jewel program, casting the field, and rendering 49 frames…';
 try {{const response=await fetch('/api/generate',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{prompt:promptBox.value,seed:Number(document.querySelector('#seed').value),mode:document.querySelector('[name=mode]:checked').value}})}});
 const data=await response.json();if(!response.ok) throw new Error(data.error||'Generation failed');
 document.querySelector('#empty').style.display='none';document.querySelector('#result').style.display='block';
 const video=document.querySelector('#video');video.src=data.video_url;video.load();video.play().catch(()=>{{}});
 document.querySelector('#download').href=data.video_url;document.querySelector('#scene').textContent=`Emitted scene: ${{data.metadata.program_scene_label}}`;
 const p=data.metadata.program;document.querySelector('#programline').textContent=`scene ${{p.scene_token}} · foreground ${{p.foreground_token}} · background ${{p.background_token}} · 72,000 Jewels`;
 document.querySelector('#metadata').textContent=JSON.stringify(data.metadata,null,2);status.textContent='Video generated. Change the seed to cast a different program.';
 }} catch(error) {{status.className='status error';status.textContent=error.message;}} finally {{button.disabled=false;}}
}};
</script></body></html>"""


class DemoApplication:
    """Serialize GPU generation and expose demo metadata to HTTP handlers."""

    def __init__(self, runtime: PromptVideoRuntime, output_dir: Path) -> None:
        self.runtime = runtime
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        prompt = str(request.get("prompt", ""))
        mode = str(request.get("mode", "exact"))
        seed = int(request.get("seed", 20260914))
        if not -(2**63) <= seed < 2**63:
            raise ValueError("seed must be a signed 64-bit integer")
        with self.lock:
            video, metadata_path, metadata = self.runtime.generate_video(
                prompt, seed, mode=mode, output_dir=self.output_dir
            )
        return {
            "video_url": f"/videos/{video.name}",
            "metadata_url": f"/videos/{metadata_path.name}",
            "metadata": metadata,
        }


def make_handler(application: DemoApplication) -> type[BaseHTTPRequestHandler]:
    """Bind one application instance to a standard-library HTTP handler."""

    class DemoHandler(BaseHTTPRequestHandler):
        server_version = "JewelPromptDemo/1"

        def _send_bytes(
            self, payload: bytes, content_type: str, status: int = HTTPStatus.OK
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, value: Any, status: int = HTTPStatus.OK) -> None:
            self._send_bytes(
                json.dumps(value).encode(), "application/json; charset=utf-8", status
            )

        def _serve_video_artifact(self, name: str) -> None:
            if Path(name).name != name or Path(name).suffix not in {".mp4", ".json"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            path = application.output_dir / name
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            size = path.stat().st_size
            if path.suffix == ".json":
                self._send_bytes(path.read_bytes(), "application/json; charset=utf-8")
                return
            try:
                byte_range = parse_byte_range(self.headers.get("Range"), size)
            except (ValueError, TypeError):
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            start, end = byte_range or (0, size - 1)
            self.send_response(
                HTTPStatus.PARTIAL_CONTENT if byte_range else HTTPStatus.OK
            )
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if byte_range:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_bytes(
                    demo_html(
                        application.runtime.prompts,
                        application.runtime.learned_available,
                    ).encode(),
                    "text/html; charset=utf-8",
                )
            elif parsed.path == "/api/status":
                self._json({
                    "ready": True,
                    "prompts": application.runtime.prompts,
                    "learned_available": application.runtime.learned_available,
                })
            elif parsed.path.startswith("/videos/"):
                self._serve_video_artifact(unquote(parsed.path.removeprefix("/videos/")))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/generate":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("invalid request size")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("request body must be a JSON object")
                self._json(application.generate(request))
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except Exception as error:  # Keep the LAN demo responsive after GPU errors.
                self.log_error("generation failed: %r", error)
                self._json({"error": f"Generation failed: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    return DemoHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    paths = PromptVideoPaths.from_project_root(args.project_root)
    output = args.output_dir or (
        args.project_root / "sol" / "results" / "jewel_prompt_demo_v1" / "generated"
    )
    runtime = PromptVideoRuntime(paths, device=args.device)
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(DemoApplication(runtime, output))
    )
    print(f"Jewel prompt demo ready at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
