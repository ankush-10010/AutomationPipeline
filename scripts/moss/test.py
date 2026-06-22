import json

# Helper to make a code cell
def code_cell(source_lines):
    source = [l + '\n' for l in source_lines[:-1]] + [source_lines[-1]]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": ""},
        "outputs": [],
        "source": source
    }

def md_cell(lines):
    return {
        "cell_type": "markdown",
        "metadata": {"id": ""},
        "source": [l + '\n' for l in lines[:-1]] + [lines[-1]]
    }

cells = []

# ── HEADER ────────────────────────────────────────────────────────────────────
cells.append(md_cell([
    "# MOSS-TTS Pipeline — Colab Setup",
    "",
    "**Run cells top to bottom.** Each cell is safe to re-run individually.",
    "",
    "| Step | Cell | Notes |",
    "|------|------|-------|",
    "| 1 | GPU check | Detects CUDA, picks the right torch backend |",
    "| 2 | Clone repo | Clones MOSS-TTS + audio-tokenizer submodule |",
    "| 3 | Create venv | Isolated `/content/moss_venv` — zero Colab conflicts |",
    "| 4 | Install PyTorch | Installs torch/torchaudio for your CUDA version |",
    "| 5 | Install MOSS-TTS | Core package + transformers, gradio, etc. |",
    "| 6 | Write pipeline | Writes `moss_tts_pipeline.py` to disk |",
    "| 7 | Launch | Starts Gradio — opens a public share URL |",
    "",
    "> **Tip:** If you restart the runtime, skip to **Cell 7** — the venv persists in `/content/`."
]))

# ── CELL 1: GPU / CUDA detection ──────────────────────────────────────────────
cells.append(code_cell([
    "# @title Cell 1 — GPU check + pick torch backend { display-mode: \"form\" }",
    "import subprocess, re, os, sys",
    "",
    "def detect_cuda():",
    "    try:",
    "        out = subprocess.run(['nvcc', '--version'], capture_output=True, text=True).stdout",
    "        m = re.search(r'release (\\d+)\\.(\\d+)', out)",
    "        if m:",
    "            major, minor = int(m.group(1)), int(m.group(2))",
    "            code = major * 10 + minor   # e.g. 124 for CUDA 12.4",
    "            if code >= 128: return 'cu128', f'{major}.{minor}'",
    "            if code >= 124: return 'cu124', f'{major}.{minor}'",
    "            if code >= 121: return 'cu121', f'{major}.{minor}'",
    "            if code >= 118: return 'cu118', f'{major}.{minor}'",
    "    except Exception:",
    "        pass",
    "    return 'cpu', 'N/A'",
    "",
    "TORCH_IDX, cuda_ver = detect_cuda()",
    "TORCH_URL = f'https://download.pytorch.org/whl/{TORCH_IDX}'",
    "VENV      = '/content/moss_venv'",
    "VENV_PY   = f'{VENV}/bin/python'",
    "VENV_PIP  = f'{VENV}/bin/pip'",
    "",
    "# Print GPU info",
    "gpu = subprocess.run(",
    "    ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],",
    "    capture_output=True, text=True",
    ").stdout.strip()",
    "",
    "print(f'GPU:          {gpu or \"None\"}') ",
    "print(f'CUDA version: {cuda_ver}')",
    "print(f'Torch index:  {TORCH_IDX}')",
    "print()",
    "if TORCH_IDX == 'cpu':",
    "    print('⚠️  No GPU found. Inference will work but will be slow.')",
    "else:",
    "    print('✅  GPU ready.')"
]))

# ── CELL 2: Clone repo ─────────────────────────────────────────────────────────
cells.append(code_cell([
    "# @title Cell 2 — Clone MOSS-TTS { display-mode: \"form\" }",
    "os.chdir('/content')",
    "",
    "if not os.path.exists('/content/MOSS-TTS/.git'):",
    "    subprocess.run(['git', 'clone', 'https://github.com/OpenMOSS/MOSS-TTS.git'], check=True)",
    "    os.chdir('/content/MOSS-TTS')",
    "    subprocess.run(['git', 'submodule', 'update', '--init', '--recursive'], check=True)",
    "    print('✅  Cloned.')",
    "else:",
    "    os.chdir('/content/MOSS-TTS')",
    "    print('Already cloned — skipping.')"
]))

# ── CELL 3: Create venv ────────────────────────────────────────────────────────
cells.append(code_cell([
    "# @title Cell 3 — Create isolated virtual environment { display-mode: \"form\" }",
    "# Creates /content/moss_venv — isolated from Colab's system packages.",
    "# Re-running this cell is safe (skips if already exists).",
    "",
    "import subprocess, sys, os",
    "if not os.path.exists(VENV_PIP):",
    "    print('Installing virtualenv...')",
    "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'virtualenv'], check=True)",
    "    print('Creating venv...')",
    "    subprocess.run([sys.executable, '-m', 'virtualenv', VENV], check=True)",
    "    print(f'✅  venv created at {VENV}')",
    "else:",
    "    print(f'venv already exists with pip at {VENV} — skipping.')"
]))

# ── CELL 4: Install PyTorch ────────────────────────────────────────────────────
cells.append(code_cell([
    "# @title Cell 4 — Install PyTorch (CUDA-matched) into venv { display-mode: \"form\" }",
    "# This installs the latest torch/torchaudio compatible with your CUDA.",
    "# Skips the version pinned in MOSS-TTS's pyproject.toml to avoid conflicts.",
    "",
    "import subprocess",
    "subprocess.run([VENV_PIP, 'uninstall', '-y', 'torch', 'torchaudio'], capture_output=True)",
    "",
    "print(f'Installing torch + torchaudio from {TORCH_URL} ...')",
    "subprocess.run([",
    "    VENV_PIP, 'install', '-q',",
    "    '--index-url', TORCH_URL,",
    "    '--extra-index-url', 'https://pypi.org/simple',",
    "    'torch', 'torchaudio',",
    "], check=True)",
    "",
    "# Verify",
    "result = subprocess.run([VENV_PY, '-c',",
    "    'import torch; print(f\"torch {torch.__version__}  CUDA={torch.version.cuda}\")'],",
    "    capture_output=True, text=True)",
    "print(result.stdout.strip())",
    "print('✅  PyTorch ready.')"
]))

# ── CELL 5: Install MOSS-TTS + deps ───────────────────────────────────────────
cells.append(code_cell([
    "# @title Cell 5 — Install MOSS-TTS base package + all other deps { display-mode: \"form\" }",
    "# Installs the MOSS-TTS Python package WITHOUT the [torch-runtime] extra",
    "# (that extra pins torch to cu128 and would override what we just installed).",
    "# torch is already in the venv — pip won't reinstall it.",
    "",
    "os.chdir('/content/MOSS-TTS')",
    "",
    "# 5a. MOSS-TTS core package (no torch extras)",
    "print('Installing MOSS-TTS package ...')",
    "r = subprocess.run([",
    "    VENV_PIP, 'install', '-q', '-e', '.',",
    "    '--index-url', TORCH_URL,",
    "    '--extra-index-url', 'https://pypi.org/simple',",
    "], capture_output=True, text=True)",
    "if r.returncode != 0:",
    "    print('[WARN] Base install output:')",
    "    print(r.stderr[-2000:])",
    "else:",
    "    print('  MOSS-TTS package ✅')",
    "",
    "# 5b. transformers >= 5.0 (MOSS-TTS requires it; Colab ships 4.x)",
    "print('Installing transformers >= 5.0 ...')",
    "subprocess.run([VENV_PIP, 'install', '-q', 'transformers>=5.0.0', 'accelerate'], check=True)",
    "print('  transformers ✅')",
    "",
    "# 5c. Gradio + pipeline extras",
    "print('Installing gradio + extras ...')",
    "subprocess.run([VENV_PIP, 'install', '-q', 'gradio', 'orjson'], check=True)",
    "print('  gradio ✅')",
    "",
    "print()",
    "print('✅  All dependencies installed.')"
]))

# ── CELL 6: Write pipeline script ─────────────────────────────────────────────
PIPELINE_SRC = r'''
"""
MOSS-TTS Pipeline — Colab Edition
No voice cloning needed.
  Tab A: MOSS-TTS direct (random timbre, 20+ languages)
  Tab B: MOSS-VoiceGenerator preset voices (text-described voices)
"""
from __future__ import annotations
import argparse, importlib.util, time
from functools import lru_cache
import gradio as gr
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

MOSS_TTS_MODEL     = "OpenMOSS-Team/MOSS-TTS"
MOSS_VG_MODEL      = "OpenMOSS-Team/MOSS-VoiceGenerator"

PRESET_VOICES = {
    "News Anchor [EN]": "Clear, authoritative news anchor voice with a neutral American English accent, steady pacing, and professional broadcast delivery.",
    "Calm Narrator [EN]": "A calm, measured narrator voice for audiobooks and documentaries. Warm, unhurried tone with natural pauses.",
    "Cheerful Host [EN]": "Energetic, friendly podcast host voice — enthusiastic and conversational, upbeat without being frantic.",
    "Wise Elder [EN]": "A distinguished elderly male voice, slow and deliberate with warmth and gravitas. Slightly gravelly, resonant.",
    "Soft & Clear Female [EN]": "A soft, gentle female voice — pleasant and clear with a warm, soothing quality and natural rhythm.",
    "Precise & Neutral [EN]": "Crisp, precise neutral voice for technical narration. Consistent tempo, clean articulation, no accent.",
    "播音主持 [ZH]": "标准普通话播音主持风格，吐字清晰，语调平稳，声音浑厚，具有专业广播感。",
    "温情叙述 [ZH]": "温暖亲切的女性叙述声音，语速适中，情感细腻，如同讲故事般娓娓道来。",
    "活力青春 [ZH]": "年轻活泼的男性声音，充满朝气和活力，语调轻快自然，感染力强。",
    "Custom — write your own": "",
}
PRESET_NAMES = list(PRESET_VOICES.keys())

LANGUAGE_TAGS = [
    "Auto (omit)","Chinese","English","Arabic","Czech","Danish","Dutch","Finnish",
    "French","German","Greek","Hebrew","Hindi","Hungarian","Italian","Japanese",
    "Korean","Macedonian","Malay","Persian (Farsi)","Polish","Portuguese",
    "Romanian","Russian","Spanish","Swahili","Swedish","Tagalog","Thai",
    "Turkish","Vietnamese",
]

def resolve_attn(device: torch.device, dtype: torch.dtype) -> str:
    if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
        if importlib.util.find_spec("flash_attn") is not None:
            major, _ = torch.cuda.get_device_capability(device)
            if major >= 8: return "flash_attention_2"
        return "sdpa"
    return "eager"

@lru_cache(maxsize=4)
def load_model(model_path: str, device_str: str):
    device = torch.device(device_str)
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    attn   = resolve_attn(device, dtype)
    print(f"[load] {model_path}  device={device}  attn={attn}", flush=True)
    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, normalize_inputs=True)
    if hasattr(proc, "audio_tokenizer"):
        proc.audio_tokenizer = proc.audio_tokenizer.to(device)
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True,
        attn_implementation=attn, torch_dtype=dtype,
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    sr = int(getattr(proc.model_config, "sampling_rate", 24000))
    return model, proc, device, sr

def _infer(model, proc, device, sr, conversations, mode, temp, top_p, top_k, rep_pen, max_tok):
    batch = proc(conversations, mode=mode)
    ids  = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    with torch.no_grad():
        outs = model.generate(
            input_ids=ids, attention_mask=mask,
            max_new_tokens=int(max_tok),
            audio_temperature=float(temp), audio_top_p=float(top_p),
            audio_top_k=int(top_k), audio_repetition_penalty=float(rep_pen),
        )
    msgs = proc.decode(outs)
    if not msgs or msgs[0] is None:
        raise RuntimeError("Model returned no audio.")
    audio = msgs[0].audio_codes_list[0]
    arr = audio.detach().float().cpu().numpy() if isinstance(audio, torch.Tensor) else np.asarray(audio, np.float32)
    return sr, arr.reshape(-1).astype(np.float32)

def gen_direct(text, lang, temp, top_p, top_k, rep_pen, max_tok, dev):
    text = (text or "").strip()
    if not text: raise gr.Error("Enter some text first.")
    t0 = time.monotonic()
    model, proc, device, sr = load_model(MOSS_TTS_MODEL, dev)
    kw = {"text": text}
    if lang and lang != "Auto (omit)": kw["language"] = lang
    sr, arr = _infer(model, proc, device, sr, [[proc.build_user_message(**kw)]],
                     "generation", temp, top_p, top_k, rep_pen, max_tok)
    status = f"Done in {time.monotonic()-t0:.1f}s | Direct | lang={lang} | temp={temp} top_p={top_p} top_k={top_k}"
    return (sr, arr), status

def gen_voice(text, preset, custom_instr, temp, top_p, top_k, rep_pen, max_tok, dev):
    text = (text or "").strip()
    if not text: raise gr.Error("Enter some text first.")
    instr = (custom_instr or "").strip() or PRESET_VOICES.get(preset, "")
    if not instr: raise gr.Error("Select a preset or fill in a custom instruction.")
    t0 = time.monotonic()
    model, proc, device, sr = load_model(MOSS_VG_MODEL, dev)
    sr, arr = _infer(model, proc, device, sr,
                     [[proc.build_user_message(text=text, instruction=instr)]],
                     "generation", temp, top_p, top_k, rep_pen, max_tok)
    status = (f"Done in {time.monotonic()-t0:.1f}s | VoiceGen | preset={preset}\n"
              f"instruction: {instr[:100]}{'...' if len(instr)>100 else ''}")
    return (sr, arr), status

def on_preset(name):
    instr = PRESET_VOICES.get(name, "")
    return gr.update(value=instr, interactive=(instr == ""),
                     placeholder="Describe the voice: gender, age, tone, emotion, speed…")

def build_ui(device_str):
    with gr.Blocks(title="MOSS-TTS Pipeline") as demo:
        gr.Markdown("# MOSS-TTS Pipeline\n*No voice cloning required — two generation modes below.*")
        dev_state = gr.State(device_str)
        with gr.Tabs():
            with gr.TabItem("Direct Generation  (MOSS-TTS 8B)"):
                gr.Markdown("*Generates speech with a neutral/random timbre. Supports 20+ languages.*")
                with gr.Row():
                    with gr.Column(scale=3):
                        d_text = gr.Textbox(label="Text", lines=6, placeholder="Enter text…")
                        d_lang = gr.Dropdown(LANGUAGE_TAGS, value="Auto (omit)", label="Language tag (optional)")
                        with gr.Accordion("Sampling parameters", open=False):
                            d_temp = gr.Slider(0.1, 3.0, 1.7, step=0.05, label="Temperature")
                            d_topp = gr.Slider(0.1, 1.0, 0.8, step=0.01, label="Top-p")
                            d_topk = gr.Slider(1, 200, 25, step=1,        label="Top-k")
                            d_rpen = gr.Slider(0.8, 2.0, 1.0, step=0.05,  label="Repetition penalty")
                            d_mnt  = gr.Slider(256, 8192, 4096, step=128, label="Max new tokens")
                        gr.Button("Generate Speech", variant="primary").click(
                            gen_direct,
                            inputs=[d_text, d_lang, d_temp, d_topp, d_topk, d_rpen, d_mnt, dev_state],
                            outputs=[gr.Audio(label="Output", type="numpy"),
                                     gr.Textbox(label="Status", lines=3, interactive=False)])
                    with gr.Column(scale=2):
                        pass
            with gr.TabItem("Preset Voices  (MOSS-VoiceGenerator 1.7B)"):
                gr.Markdown("*Voice style set by text description — no reference audio needed.*")
                with gr.Row():
                    with gr.Column(scale=3):
                        v_text   = gr.Textbox(label="Text", lines=5, placeholder="Enter text…")
                        v_preset = gr.Dropdown(PRESET_NAMES, value=PRESET_NAMES[0], label="Voice preset")
                        v_instr  = gr.Textbox(label="Voice instruction (auto-filled)", lines=3,
                                              value=PRESET_VOICES[PRESET_NAMES[0]], interactive=False)
                        v_preset.change(on_preset, v_preset, v_instr)
                        with gr.Accordion("Sampling parameters (VoiceGen defaults)", open=False):
                            v_temp = gr.Slider(0.1, 3.0, 1.5, step=0.05, label="Temperature")
                            v_topp = gr.Slider(0.1, 1.0, 0.6, step=0.01, label="Top-p")
                            v_topk = gr.Slider(1, 200, 50,  step=1,       label="Top-k")
                            v_rpen = gr.Slider(0.8, 2.0, 1.1, step=0.05,  label="Repetition penalty")
                            v_mnt  = gr.Slider(256, 8192, 4096, step=128, label="Max new tokens")
                        gr.Button("Generate Speech", variant="primary").click(
                            gen_voice,
                            inputs=[v_text, v_preset, v_instr, v_temp, v_topp, v_topk, v_rpen, v_mnt, dev_state],
                            outputs=[gr.Audio(label="Output", type="numpy"),
                                     gr.Textbox(label="Status", lines=3, interactive=False)])
                    with gr.Column(scale=2):
                        pass
        gr.Markdown("---\nMOSS-TTS Family · Apache-2.0")
    return demo

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--share", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print("[WARN] No CUDA — falling back to CPU"); args.device = "cpu"
    build_ui(args.device).queue(max_size=4).launch(
        server_name="0.0.0.0", server_port=args.port, share=args.share, theme=gr.themes.Soft())

if __name__ == "__main__":
    main()
'''

cells.append(code_cell([
    "# @title Cell 6 — Write pipeline script to disk { display-mode: \"form\" }",
    "SCRIPT = r'''" + PIPELINE_SRC + "'''",
    "",
    "with open('/content/MOSS-TTS/moss_tts_pipeline.py', 'w') as f:",
    "    f.write(SCRIPT)",
    "print('✅  /content/MOSS-TTS/moss_tts_pipeline.py written.')"
]))

# ── CELL 7: Launch ─────────────────────────────────────────────────────────────
cells.append(code_cell([
    "# @title Cell 7 — Launch Gradio app { display-mode: \"form\" }",
    "# Runs the pipeline in the venv as a background subprocess.",
    "# Streams output until the public share URL appears, then stops blocking.",
    "",
    "import threading, re",
    "",
    "# Auto-detect device",
    "is_cuda = subprocess.run([VENV_PY, '-c', 'import torch; print(torch.cuda.is_available())'],",
    "                          capture_output=True, text=True).stdout.strip() == 'True'",
    "device  = 'cuda:0' if is_cuda else 'cpu'",
    "print(f'Launching on {device} ...')",
    "",
    "proc = subprocess.Popen(",
    "    [VENV_PY, '/content/MOSS-TTS/moss_tts_pipeline.py',",
    "     '--device', device, '--share'],",
    "    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,",
    "    cwd='/content/MOSS-TTS'",
    ")",
    "",
    "print('Waiting for app to start (model downloads happen here on first run)...')",
    "print('This may take several minutes the first time.\\n')",
    "",
    "url_found = threading.Event()",
    "",
    "def stream_output():",
    "    for line in proc.stdout:",
    "        print(line, end='', flush=True)",
    "        if 'gradio.live' in line or 'Running on public URL' in line:",
    "            url_found.set()",
    "",
    "t = threading.Thread(target=stream_output, daemon=True)",
    "t.start()",
    "",
    "# Wait until URL found, timeout, or process dies",
    "import time",
    "start_wait = time.time()",
    "while not url_found.is_set():",
    "    if proc.poll() is not None:",
    "        print(f'\\n❌ App crashed with exit code {proc.returncode} (If code is -9, it ran out of system RAM!)')",
    "        break",
    "    if time.time() - start_wait > 900:",
    "        print('\\n❌ Timed out waiting for app to start.')",
    "        break",
    "    time.sleep(1)",
    "",
    "if url_found.is_set():",
    "    print('\\n✅  App is live — use the public URL above to open it.')"
]))

# ── OPTIONAL CELL: Kill app ────────────────────────────────────────────────────
cells.append(md_cell([
    "### Optional — stop the app",
    "Run the cell below to kill the Gradio server."
]))

cells.append(code_cell([
    "# @title Stop app (optional) { display-mode: \"form\" }",
    "try:",
    "    proc.terminate()",
    "    print('App stopped.')",
    "except NameError:",
    "    print('No running app found.')"
]))

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"}
    },
    "cells": cells
}

with open('MOSS_TTS_Colab.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Done")