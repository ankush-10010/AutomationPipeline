"""
MOSS-TTS Pipeline
=================
A clean text-to-speech pipeline built on the MOSS-TTS Family.

Two generation modes — no voice cloning required:

  MODE A  |  MOSS-TTS Direct
           |  Generates speech directly from text with a random/neutral timbre.
           |  Supports 20+ languages. Great for quick synthesis.

  MODE B  |  MOSS-VoiceGenerator  (Preset Voices)
           |  Generates speech guided by a text-based voice style instruction.
           |  Pick a built-in preset or write your own voice description.
           |  No reference audio needed.

Setup (one-time):
    conda create -n moss-tts python=3.12 -y && conda activate moss-tts
    git clone https://github.com/OpenMOSS/MOSS-TTS.git
    cd MOSS-TTS
    pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime]"
    pip install gradio orjson

Usage:
    python moss_tts_pipeline.py
    python moss_tts_pipeline.py --device cpu          # CPU fallback
    python moss_tts_pipeline.py --share               # public Gradio link
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor

# ──────────────────────────────────────────────────────────────────────────────
#  PyTorch backend tweaks (matches official MOSS-TTS guidance)
# ──────────────────────────────────────────────────────────────────────────────
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# ──────────────────────────────────────────────────────────────────────────────
#  Model identifiers
# ──────────────────────────────────────────────────────────────────────────────
MOSS_TTS_MODEL = "OpenMOSS-Team/MOSS-TTS"
MOSS_VOICE_GEN_MODEL = "OpenMOSS-Team/MOSS-VoiceGenerator"

# ──────────────────────────────────────────────────────────────────────────────
#  Preset voices  (MOSS-VoiceGenerator — instruction field)
# ──────────────────────────────────────────────────────────────────────────────
PRESET_VOICES: dict[str, dict] = {
    "🎙️  News Anchor  [EN]": {
        "instruction": (
            "Clear, authoritative news anchor voice with a neutral American English accent, "
            "steady pacing, measured tone, and professional broadcast delivery."
        ),
    },
    "📚  Calm Narrator  [EN]": {
        "instruction": (
            "A calm, measured narrator voice for audiobooks and documentaries. Warm, unhurried "
            "tone with natural pauses and thoughtful delivery."
        ),
    },
    "🎧  Cheerful Host  [EN]": {
        "instruction": (
            "Energetic, friendly podcast host voice — enthusiastic and conversational, "
            "upbeat without being frantic, radiating approachability and warmth."
        ),
    },
    "🧙  Wise Elder  [EN]": {
        "instruction": (
            "A distinguished elderly male voice, slow and deliberate with warmth and gravitas. "
            "Slightly gravelly, resonant, carrying quiet authority."
        ),
    },
    "🌸  Soft & Clear Female  [EN]": {
        "instruction": (
            "A soft, gentle female voice — pleasant and clear with a warm, soothing quality, "
            "natural rhythm, and a hint of a smile in the tone."
        ),
    },
    "🔬  Precise & Neutral  [EN]": {
        "instruction": (
            "Crisp, precise neutral voice ideal for technical narration and e-learning. "
            "Consistent tempo, clean articulation, no regional accent, zero flamboyance."
        ),
    },
    "📰  播音主持  [ZH]": {
        "instruction": (
            "标准普通话播音主持风格，吐字清晰，语调平稳，声音浑厚，具有专业广播感，"
            "适合新闻播报和正式场合。"
        ),
    },
    "📖  温情叙述  [ZH]": {
        "instruction": (
            "温暖亲切的女性叙述声音，语速适中，情感细腻，如同讲故事般娓娓道来，"
            "充满感情，适合有声书和情感类内容。"
        ),
    },
    "🔥  活力青春  [ZH]": {
        "instruction": (
            "年轻活泼的男性声音，充满朝气和活力，语调轻快自然，感染力强，"
            "适合综艺节目和年轻化品牌内容。"
        ),
    },
    "✍️  Custom — write your own  ": {
        "instruction": "",  # user fills the Custom Instruction box
    },
}

PRESET_NAMES = list(PRESET_VOICES.keys())

# ──────────────────────────────────────────────────────────────────────────────
#  Language tags (MOSS-TTS direct mode)
# ──────────────────────────────────────────────────────────────────────────────
LANGUAGE_TAGS = [
    "Auto (omit)",
    "Chinese", "English", "Arabic", "Czech", "Danish", "Dutch",
    "Finnish", "French", "German", "Greek", "Hebrew", "Hindi",
    "Hungarian", "Italian", "Japanese", "Korean", "Macedonian",
    "Malay", "Persian (Farsi)", "Polish", "Portuguese", "Romanian",
    "Russian", "Spanish", "Swahili", "Swedish", "Tagalog", "Thai",
    "Turkish", "Vietnamese",
]

# ──────────────────────────────────────────────────────────────────────────────
#  Attention implementation helper
# ──────────────────────────────────────────────────────────────────────────────
def resolve_attn(device: torch.device, dtype: torch.dtype) -> str:
    if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
        if importlib.util.find_spec("flash_attn") is not None:
            major, _ = torch.cuda.get_device_capability(device)
            if major >= 8:
                return "flash_attention_2"
        return "sdpa"
    return "eager"


# ──────────────────────────────────────────────────────────────────────────────
#  Cached model loaders  (only one instance per model path)
# ──────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=4)
def load_model(model_path: str, device_str: str):
    """Load and cache a model + processor pair."""
    device = torch.device(device_str)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    attn = resolve_attn(device, dtype)

    print(f"[load] {model_path}  device={device}  attn={attn}", flush=True)

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        normalize_inputs=True,
    )
    if hasattr(processor, "audio_tokenizer"):
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation=attn,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    sample_rate = int(getattr(processor.model_config, "sampling_rate", 24000))
    return model, processor, device, sample_rate


# ──────────────────────────────────────────────────────────────────────────────
#  Core inference
# ──────────────────────────────────────────────────────────────────────────────
def _run_inference(
    *,
    model,
    processor,
    device: torch.device,
    sample_rate: int,
    conversations: list,
    mode: str,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_new_tokens: int,
) -> tuple[int, np.ndarray]:
    batch = processor(conversations, mode=mode)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            audio_temperature=float(temperature),
            audio_top_p=float(top_p),
            audio_top_k=int(top_k),
            audio_repetition_penalty=float(repetition_penalty),
        )

    messages = processor.decode(outputs)
    if not messages or messages[0] is None:
        raise RuntimeError("Model returned no decodable audio.")

    audio = messages[0].audio_codes_list[0]
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().float().cpu().numpy()
    else:
        audio_np = np.asarray(audio, dtype=np.float32)

    if audio_np.ndim > 1:
        audio_np = audio_np.reshape(-1)

    return sample_rate, audio_np.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
#  Gradio callback — MOSS-TTS Direct
# ──────────────────────────────────────────────────────────────────────────────
def generate_direct(
    text: str,
    language_tag: str,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_new_tokens: int,
    device_str: str,
) -> tuple[tuple, str]:
    t0 = time.monotonic()
    text = (text or "").strip()
    if not text:
        raise gr.Error("Please enter some text to synthesize.")

    model, processor, device, sr = load_model(MOSS_TTS_MODEL, device_str)

    user_kwargs: dict = {"text": text}
    if language_tag and language_tag != "Auto (omit)":
        user_kwargs["language"] = language_tag

    conversations = [[processor.build_user_message(**user_kwargs)]]
    sr, audio = _run_inference(
        model=model, processor=processor, device=device, sample_rate=sr,
        conversations=conversations, mode="generation",
        temperature=temperature, top_p=top_p, top_k=top_k,
        repetition_penalty=repetition_penalty, max_new_tokens=max_new_tokens,
    )

    elapsed = time.monotonic() - t0
    status = (
        f"✅  Done in {elapsed:.1f}s  |  mode: MOSS-TTS Direct  |  "
        f"lang={language_tag}  |  temp={temperature}  top_p={top_p}  "
        f"top_k={top_k}  rep_pen={repetition_penalty}"
    )
    return (sr, audio), status


# ──────────────────────────────────────────────────────────────────────────────
#  Gradio callback — MOSS-VoiceGenerator
# ──────────────────────────────────────────────────────────────────────────────
def generate_voice_gen(
    text: str,
    preset_name: str,
    custom_instruction: str,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_new_tokens: int,
    device_str: str,
) -> tuple[tuple, str]:
    t0 = time.monotonic()
    text = (text or "").strip()
    if not text:
        raise gr.Error("Please enter some text to synthesize.")

    # Resolve instruction
    preset = PRESET_VOICES.get(preset_name, {})
    instruction = (custom_instruction or "").strip() or preset.get("instruction", "")
    if not instruction:
        raise gr.Error(
            "Please select a preset or enter a custom voice description in the instruction box."
        )

    model, processor, device, sr = load_model(MOSS_VOICE_GEN_MODEL, device_str)

    conversations = [[processor.build_user_message(text=text, instruction=instruction)]]
    sr, audio = _run_inference(
        model=model, processor=processor, device=device, sample_rate=sr,
        conversations=conversations, mode="generation",
        temperature=temperature, top_p=top_p, top_k=top_k,
        repetition_penalty=repetition_penalty, max_new_tokens=max_new_tokens,
    )

    elapsed = time.monotonic() - t0
    status = (
        f"✅  Done in {elapsed:.1f}s  |  mode: VoiceGenerator  |  "
        f"preset={preset_name}  |  temp={temperature}  top_p={top_p}  "
        f"top_k={top_k}  rep_pen={repetition_penalty}\n"
        f"instruction: {instruction[:120]}{'…' if len(instruction) > 120 else ''}"
    )
    return (sr, audio), status


# ──────────────────────────────────────────────────────────────────────────────
#  Helper: fill instruction box when a preset is chosen
# ──────────────────────────────────────────────────────────────────────────────
def on_preset_change(preset_name: str):
    preset = PRESET_VOICES.get(preset_name, {})
    instruction = preset.get("instruction", "")
    is_custom = (instruction == "")
    return (
        gr.update(value=instruction, interactive=is_custom,
                  placeholder="Describe the voice: gender, age, tone, emotion, accent, speed…"),
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Gradio UI
# ──────────────────────────────────────────────────────────────────────────────
CSS = """
body { font-family: 'Inter', system-ui, sans-serif; }

.tab-header { font-size: 15px; font-weight: 600; }

#gen-btn-direct, #gen-btn-voice {
    background: #0ea5e9;
    color: white;
    font-weight: 600;
    border-radius: 8px;
    padding: 10px 0;
    font-size: 15px;
}
#gen-btn-direct:hover, #gen-btn-voice:hover { background: #0284c7; }

.status-box textarea {
    font-size: 12px;
    color: #64748b;
    font-family: monospace;
}

.voice-card {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px;
    background: #f8fafc;
}

.section-label {
    font-size: 12px;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}
"""

def build_ui(device_str: str) -> gr.Blocks:
    with gr.Blocks(title="MOSS-TTS Pipeline", css=CSS, theme=gr.themes.Soft()) as demo:

        gr.Markdown(
            """
            # 🔊 MOSS-TTS Pipeline
            **Text-to-speech generation — no voice cloning required.**
            Choose *Direct Generation* for quick synthesis or *Preset Voices* for styled output.
            """
        )

        # ── shared device state (hidden) ────────────────────────────────────
        device_state = gr.State(device_str)

        with gr.Tabs():

            # ════════════════════════════════════════════════════════════════
            #  TAB A — MOSS-TTS Direct
            # ════════════════════════════════════════════════════════════════
            with gr.TabItem("🎙️ Direct Generation  (MOSS-TTS)", elem_classes="tab-header"):
                gr.Markdown(
                    "_No voice style required. The model generates speech with a neutral/random timbre. "
                    "Supports 20+ languages via the optional language tag._"
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        d_text = gr.Textbox(
                            label="Text to synthesize",
                            lines=7,
                            placeholder="Enter your text here…\n\nYou can also use Pinyin: nin2 hao3\nor IPA: /həloʊ, wɜːld/",
                        )
                        d_lang = gr.Dropdown(
                            choices=LANGUAGE_TAGS,
                            value="Auto (omit)",
                            label="Language tag  (optional — helps accuracy for non-EN/ZH text)",
                        )

                        with gr.Accordion("⚙️ Sampling parameters", open=False):
                            d_temp = gr.Slider(0.1, 3.0, value=1.7, step=0.05,
                                               label="Temperature  — higher = more varied")
                            d_topp = gr.Slider(0.1, 1.0, value=0.8, step=0.01,
                                               label="Top-p  (nucleus sampling)")
                            d_topk = gr.Slider(1, 200, value=25, step=1,
                                               label="Top-k")
                            d_rpen = gr.Slider(0.8, 2.0, value=1.0, step=0.05,
                                               label="Repetition penalty  (>1 discourages loops)")
                            d_mnt  = gr.Slider(256, 8192, value=4096, step=128,
                                               label="Max new tokens  (controls max audio length)")

                        d_btn = gr.Button("Generate Speech", variant="primary",
                                          elem_id="gen-btn-direct")

                    with gr.Column(scale=2):
                        d_audio  = gr.Audio(label="Output", type="numpy")
                        d_status = gr.Textbox(label="Status", lines=4, interactive=False,
                                              elem_classes="status-box")

                d_btn.click(
                    fn=generate_direct,
                    inputs=[d_text, d_lang, d_temp, d_topp, d_topk, d_rpen, d_mnt, device_state],
                    outputs=[d_audio, d_status],
                )

            # ════════════════════════════════════════════════════════════════
            #  TAB B — MOSS-VoiceGenerator  (Preset Voices)
            # ════════════════════════════════════════════════════════════════
            with gr.TabItem("✨ Preset Voices  (MOSS-VoiceGenerator)", elem_classes="tab-header"):
                gr.Markdown(
                    "_Voice style is controlled by a text description — no reference audio needed. "
                    "Pick a built-in preset or select **Custom** and write your own._"
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        v_text = gr.Textbox(
                            label="Text to synthesize",
                            lines=5,
                            placeholder="Enter the text you want spoken…",
                        )

                        with gr.Group(elem_classes="voice-card"):
                            v_preset = gr.Dropdown(
                                choices=PRESET_NAMES,
                                value=PRESET_NAMES[0],
                                label="Voice preset",
                            )
                            v_instruction = gr.Textbox(
                                label="Voice instruction  (auto-filled from preset, or write your own)",
                                lines=3,
                                value=PRESET_VOICES[PRESET_NAMES[0]]["instruction"],
                                interactive=False,
                                placeholder="Describe the voice: gender, age, tone, emotion, accent, speed…",
                            )

                        with gr.Accordion("⚙️ Sampling parameters", open=False):
                            gr.Markdown(
                                "_Recommended defaults for VoiceGenerator: "
                                "temp=1.5, top-p=0.6, top-k=50, rep-pen=1.1_"
                            )
                            v_temp = gr.Slider(0.1, 3.0, value=1.5, step=0.05,
                                               label="Temperature")
                            v_topp = gr.Slider(0.1, 1.0, value=0.6, step=0.01,
                                               label="Top-p")
                            v_topk = gr.Slider(1, 200, value=50, step=1,
                                               label="Top-k")
                            v_rpen = gr.Slider(0.8, 2.0, value=1.1, step=0.05,
                                               label="Repetition penalty")
                            v_mnt  = gr.Slider(256, 8192, value=4096, step=128,
                                               label="Max new tokens")

                        v_btn = gr.Button("Generate Speech", variant="primary",
                                          elem_id="gen-btn-voice")

                    with gr.Column(scale=2):
                        v_audio  = gr.Audio(label="Output", type="numpy")
                        v_status = gr.Textbox(label="Status", lines=4, interactive=False,
                                              elem_classes="status-box")

                # Wire preset → instruction box
                v_preset.change(
                    fn=on_preset_change,
                    inputs=[v_preset],
                    outputs=[v_instruction],
                )

                v_btn.click(
                    fn=generate_voice_gen,
                    inputs=[v_text, v_preset, v_instruction,
                            v_temp, v_topp, v_topk, v_rpen, v_mnt, device_state],
                    outputs=[v_audio, v_status],
                )

        gr.Markdown(
            """
            ---
            **Models:** [MOSS-TTS](https://huggingface.co/OpenMOSS-Team/MOSS-TTS) · 
            [MOSS-VoiceGenerator](https://huggingface.co/OpenMOSS-Team/MOSS-VoiceGenerator) · 
            [GitHub](https://github.com/OpenMOSS/MOSS-TTS) · Apache-2.0
            """
        )

    return demo


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MOSS-TTS Pipeline")
    parser.add_argument("--device", default="cuda:0",
                        help="Torch device string (e.g. cuda:0, cpu)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    parser.add_argument("--preload", action="store_true",
                        help="Download and cache both models before the UI starts")
    args = parser.parse_args()

    # Fall back gracefully when no GPU is available
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print("[WARN] CUDA not available — falling back to CPU (will be slow)")
        args.device = "cpu"

    if args.preload:
        print("[Startup] Pre-loading MOSS-TTS …", flush=True)
        load_model(MOSS_TTS_MODEL, args.device)
        print("[Startup] Pre-loading MOSS-VoiceGenerator …", flush=True)
        load_model(MOSS_VOICE_GEN_MODEL, args.device)
        print("[Startup] Both models ready.", flush=True)

    demo = build_ui(args.device)
    demo.queue(max_size=8, default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
