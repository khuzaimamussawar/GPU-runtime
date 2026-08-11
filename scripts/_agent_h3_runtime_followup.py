from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


# Manifests: BasicScheduler.denoise is a real per-job patch path.
for rel, needle, replacement in [
    (
        "workflows/manifests/fl2va_manifest.json",
        '    "steps": "16.inputs.steps",\n    "scheduler": "16.inputs.scheduler",',
        '    "steps": "16.inputs.steps",\n    "scheduler": "16.inputs.scheduler",\n    "denoise": "16.inputs.denoise",',
    ),
    (
        "workflows/manifests/ref2va_manifest.json",
        '    "steps": "18.inputs.steps",\n    "scheduler": "18.inputs.scheduler",',
        '    "steps": "18.inputs.steps",\n    "scheduler": "18.inputs.scheduler",\n    "denoise": "18.inputs.denoise",',
    ),
]:
    path = ROOT / rel
    text = path.read_text(encoding="utf-8")
    text = replace_once(text, needle, replacement, f"{rel} denoise path")
    path.write_text(text, encoding="utf-8")


# Base workflow patcher accepts denoise.
path = ROOT / "src/common/h3_runtime.py"
text = path.read_text(encoding="utf-8")
text = replace_once(
    text,
    '        "scheduler": required.get("scheduler"),\n        "sampler": required.get("sampler"),',
    '        "scheduler": required.get("scheduler"),\n        "denoise": required.get("denoise") or optional.get("denoise"),\n        "sampler": required.get("sampler"),',
    "runtime denoise simple path",
)
text = replace_once(
    text,
    '        "scheduler",\n        "sampler",',
    '        "scheduler",\n        "denoise",\n        "sampler",',
    "runtime denoise supported setting",
)
path.write_text(text, encoding="utf-8")


# Production adapter: exact multi-source Director audio + real Comfy sampler progress.
path = ROOT / "src/common/runtime_adapter.py"
text = path.read_text(encoding="utf-8")
text = replace_once(
    text,
    "import math\nimport shutil\nimport subprocess\n",
    "import asyncio\nimport json\nimport math\nimport shutil\nimport subprocess\nimport time\nimport urllib.parse\n",
    "runtime adapter progress imports",
)
text = replace_once(
    text,
    "_ACTIVE_PROGRESS_CALLBACK: ProgressCallback | None = None\n",
    "_ACTIVE_PROGRESS_CALLBACK: ProgressCallback | None = None\n_ACTIVE_JOB_ID: str | None = None\n",
    "active job id",
)

insert_after_audio = '''def _materialize_reference_audio(\n    ref: dict[str, Any] | None, target_dir: Path, index: int\n) -> str:\n'''
if insert_after_audio not in text:
    raise RuntimeError("Could not locate reference-audio helper")
# Insert the composite helper immediately before _materialize_inputs, after the
# existing _materialize_reference_audio function body.
marker = "\n\ndef _materialize_inputs(\n"
if text.count(marker) != 1:
    raise RuntimeError(f"materialize inputs marker: expected 1, found {text.count(marker)}")
composite_helper = r'''

def _materialize_reference_audio_segments(
    refs: list[Any], target_dir: Path
) -> str:
    """Materialize exact ordered Director source ranges into one H3 audio ref."""
    paths: list[Path] = []
    for index, value in enumerate(refs):
        ref = runtime.media_ref(value)
        if not ref:
            continue
        relative = _materialize_reference_audio(ref, target_dir, index)
        paths.append(_absolute_input_path(relative))

    if not paths:
        raise runtime.H3RuntimeError("referenceAudioSegments did not contain usable audio")
    if len(paths) == 1:
        return _relative_input_path(paths[0])

    output = target_dir / "director_audio_segments_h3.wav"
    if output.exists():
        return _relative_input_path(output)

    cmd = ["ffmpeg", "-y"]
    for source in paths:
        cmd += ["-i", str(source)]
    concat_inputs = "".join(f"[{index}:a:0]" for index in range(len(paths)))
    cmd += [
        "-filter_complex",
        f"{concat_inputs}concat=n={len(paths)}:v=0:a=1[outa]",
        "-map", "[outa]",
        "-ac", "2",
        "-ar", "48000",
        "-c:a", "pcm_s16le",
        str(output),
    ]
    subprocess.run(cmd, check=True)
    return _relative_input_path(output)
'''
text = text.replace(marker, composite_helper + marker, 1)

text = replace_once(
    text,
    '    audio_refs = _as_list(inputs.get("referenceAudios") or inputs.get("referenceAudio") or inputs.get("audio"))\n\n    if len(refs) > 9:',
    '    audio_segments = _as_list(inputs.get("referenceAudioSegments"))\n    audio_refs = _as_list(inputs.get("referenceAudios") or inputs.get("referenceAudio") or inputs.get("audio"))\n\n    if len(refs) > 9:',
    "read Director audio segments",
)
text = replace_once(
    text,
    '    if not refs and not video_refs and not audio_refs:\n        raise runtime.H3RuntimeError(\n            "h3_ref2va requires at least one image, video, or audio reference"\n        )\n',
    '    if not refs and not video_refs and not audio_refs and not audio_segments:\n        raise runtime.H3RuntimeError(\n            "h3_ref2va requires at least one image, video, or audio reference"\n        )\n',
    "Director audio counts as Ref2VA input",
)

old_audio_loop = '''    for index, ref in enumerate(audio_refs):
        loader_path = optional.get(f"referenceAudio{index}")
        if not loader_path:
            raise runtime.H3RuntimeError(f"Missing workflow path for reference audio {index}")
        runtime.set_path(
            workflow,
            loader_path,
            _materialize_reference_audio(runtime.media_ref(ref), media_dir, index),
        )

    _prune_ref2va_placeholders(
        workflow,
        optional,
        image_count=len(refs),
        video_count=len(video_refs),
        audio_count=len(audio_refs),
    )
'''
new_audio_loop = '''    audio_binding_count = 0
    if audio_segments:
        loader_path = optional.get("referenceAudio0")
        if not loader_path:
            raise runtime.H3RuntimeError("Missing workflow path for Director reference audio")
        runtime.set_path(
            workflow,
            loader_path,
            _materialize_reference_audio_segments(audio_segments, media_dir),
        )
        audio_binding_count = 1
    else:
        for index, ref in enumerate(audio_refs):
            loader_path = optional.get(f"referenceAudio{index}")
            if not loader_path:
                raise runtime.H3RuntimeError(f"Missing workflow path for reference audio {index}")
            runtime.set_path(
                workflow,
                loader_path,
                _materialize_reference_audio(runtime.media_ref(ref), media_dir, index),
            )
        audio_binding_count = len(audio_refs)

    _prune_ref2va_placeholders(
        workflow,
        optional,
        image_count=len(refs),
        video_count=len(video_refs),
        audio_count=audio_binding_count,
    )
'''
text = replace_once(text, old_audio_loop, new_audio_loop, "composite Director audio binding")

# Replace coarse generating wait with a WebSocket progress path. ComfyUI ships
# aiohttp in its own requirements; if WebSocket setup fails, keep the existing
# history polling path and never invent a percentage.
old_wait = '''def _wait_for_output_video(prompt_id: str, started_at: float) -> Path:
    _emit_progress("generating")
    return _ORIGINAL_WAIT_FOR_OUTPUT_VIDEO(prompt_id, started_at)
'''
new_wait = r'''async def _wait_for_output_video_with_progress(
    prompt_id: str, started_at: float, client_id: str
) -> Path | None:
    try:
        import aiohttp
    except Exception:
        return None

    deadline = time.time() + int(runtime.os.environ.get("COMFY_JOB_TIMEOUT_SECONDS", "3600"))
    last_history: dict[str, Any] | None = None
    base = runtime.COMFY_URL.rstrip("/")
    ws_base = base.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    ws_url = f"{ws_base}/ws?clientId={urllib.parse.quote(client_id, safe='')}"

    timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_connect=10, sock_read=None)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                websocket = await session.ws_connect(ws_url, heartbeat=30)
            except Exception as exc:
                print(f"[SceneBuilder H3] Comfy progress websocket unavailable: {exc}")
                return None

            async with websocket:
                while time.time() < deadline:
                    try:
                        async with session.get(
                            f"{base}/history/{urllib.parse.quote(prompt_id, safe='')}",
                            timeout=aiohttp.ClientTimeout(total=20),
                        ) as response:
                            if response.ok:
                                last_history = await response.json()
                                video = runtime.find_video_in_history(last_history)
                                if video:
                                    return video
                    except Exception:
                        pass

                    try:
                        message = await websocket.receive(timeout=2.0)
                    except asyncio.TimeoutError:
                        continue
                    if message.type == aiohttp.WSMsgType.TEXT:
                        try:
                            payload = json.loads(message.data)
                        except Exception:
                            continue
                        data = payload.get("data") if isinstance(payload, dict) else None
                        if not isinstance(data, dict):
                            continue
                        event_prompt_id = str(data.get("prompt_id") or data.get("promptId") or "")
                        if event_prompt_id and event_prompt_id != prompt_id:
                            continue
                        if str(payload.get("type") or "") == "progress":
                            value = data.get("value")
                            maximum = data.get("max")
                            try:
                                value_num = float(value)
                                max_num = float(maximum)
                            except (TypeError, ValueError):
                                continue
                            if max_num > 0:
                                percent = max(0, min(100, round(value_num / max_num * 100)))
                                _emit_progress("generating", percent)
                    elif message.type in {
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    }:
                        break
    except Exception as exc:
        print(f"[SceneBuilder H3] Comfy progress listener failed: {exc}")
        return None

    scanned = runtime.newest_video_file(started_at)
    if scanned:
        return scanned
    if last_history:
        print(f"[SceneBuilder H3] Comfy progress listener ended without output: {last_history}")
    return None


def _wait_for_output_video(prompt_id: str, started_at: float) -> Path:
    _emit_progress("generating")
    client_id = _ACTIVE_JOB_ID
    if client_id:
        try:
            video = asyncio.run(_wait_for_output_video_with_progress(prompt_id, started_at, client_id))
            if video:
                return video
        except Exception as exc:
            print(f"[SceneBuilder H3] falling back to history-only Comfy polling: {exc}")
    return _ORIGINAL_WAIT_FOR_OUTPUT_VIDEO(prompt_id, started_at)
'''
text = replace_once(text, old_wait, new_wait, "real Comfy progress listener")

text = replace_once(
    text,
    "    global _ACTIVE_PROGRESS_CALLBACK\n",
    "    global _ACTIVE_PROGRESS_CALLBACK, _ACTIVE_JOB_ID\n",
    "runtime active globals",
)
text = replace_once(
    text,
    "    previous_callback = _ACTIVE_PROGRESS_CALLBACK\n    _ACTIVE_PROGRESS_CALLBACK = progress_callback\n",
    "    previous_callback = _ACTIVE_PROGRESS_CALLBACK\n    previous_job_id = _ACTIVE_JOB_ID\n    _ACTIVE_PROGRESS_CALLBACK = progress_callback\n    _ACTIVE_JOB_ID = job_id\n",
    "set runtime active job id",
)
text = replace_once(
    text,
    "        _ACTIVE_PROGRESS_CALLBACK = previous_callback\n        # Always remove downloaded inputs.",
    "        _ACTIVE_PROGRESS_CALLBACK = previous_callback\n        _ACTIVE_JOB_ID = previous_job_id\n        # Always remove downloaded inputs.",
    "restore runtime active job id",
)
path.write_text(text, encoding="utf-8")
