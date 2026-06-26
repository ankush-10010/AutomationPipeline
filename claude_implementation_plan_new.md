# Comprehensive Implementation Plan for Exponential View Growth on @rickify-i4y

Based on a thorough analysis of top-performing short-form theory channels—specifically **@ThinkAlto** (Rick and Morty) and **@NyleTrix** (Ben 10)—and benchmarking them against our automated outputs on **@rickify-i4y**, this implementation plan details the exact architectural, prompt-level, and code-level modifications required in our 8-phase AI pipeline to dramatically increase views, retention, and engagement.

This document serves as the absolute blueprint for our pipeline upgrade.

---

## 📊 1. Competitive Research & Benchmarking

Our current automated outputs on `@rickify-i4y` produce factually grounded, well-edited videos. However, they are overly "documentary-style" and fail to trigger the psychological triggers necessary to retain the hyper-stimulated YouTube Shorts audience.

**Deep Dive into @ThinkAlto (Rick & Morty):**
- **Hooks:** Never starts with a question. Always starts with a pattern interrupt. Example: *"Rick actually LIED about the Central Finite Curve."* They leverage the "Curiosity Gap"—giving a shocking conclusion first and making the viewer watch to see *how* they arrived there.
- **Pacing:** Rapid escalation. The visual on screen changes every 1.5 to 2.5 seconds.
- **Visual Engagement:** Bouncy, high-contrast captions synced to word level. Frequent micro-zooms on character faces to simulate intensity.

**Deep Dive into @NyleTrix (Ben 10):**
- **Truth Sandwiching:** They ground wild alien theories with a highly specific, canonical fact within the first 5 seconds to build immense credibility. Example: *"The Omnitrix has NEVER scanned this alien, despite it being in the database."*
- **Lore Subversion:** They take established lore that fans accept and subvert it.
- **Continuous Audio Loop:** The end of their Shorts perfectly flows into the beginning of the Short, tricking the user into watching the first 3 seconds again, which spikes the retention graph above 100%.

**Benchmarking `@rickify-i4y` (Our Current Output):**
- **Weaknesses:** Our `topic_prompt.txt` generates questions ("Why did Rick..."). Questions give the viewer a chance to say "I don't care" and scroll. Our FFmpeg Ken Burns effect in `assembler.py` is too slow and smooth, lacking the snappy CapCut-style edits.
- **Strengths:** Our Verifier-Corrector loop (Phase 1b) gives us 10x more factual accuracy than human creators. Our YOLOv8 bounding boxes give us the capability for perfect facial zooms.

---

## 🛠️ Phase 1: Overhaul Topic Mining

We must force the Topic Miner (`scripts/topic_miner.py`) to generate definitive, shocking claims instead of questions.

### 1.1 Update `prompts/topic_prompt.txt`

Replace the entire contents of `prompts/topic_prompt.txt` with the following rigid template:

```text
You are a viral YouTube Shorts content strategist for a channel about {show_name}.

Given the following episode summaries and character information:
---
{show_context}
---

Character details:
{character_details}

Key themes of the show: {themes}

Generate {num_topics} scroll-stopping short-form video topics. 
DO NOT GENERATE QUESTIONS. Every topic must be a definitive, shocking, or counter-intuitive claim about the lore or characters.

Topic styles to use:
- The "Liar" Format: "Why [Character] lied about [Major Event]"
- The "Secret Flaw" Format: "The hidden flaw in [Technology/Character] that nobody noticed"
- The "Dark Truth" Format: "The dark truth behind [Innocent Event/Scene]"
- The "Canon Breaking" Format: "The one time [Character] completely broke canon"

Rules:
- Each topic must be a definitive statement. No question marks.
- Topics must exploit a "Curiosity Gap" (a shocking conclusion that requires the viewer to watch to understand how you got there).
- The "hook" must be a 1-sentence pattern interrupt.
- Avoid topics that require extensive plot setup to understand.

Previously covered topics (DO NOT repeat or rephrase these):
{completed_topics}

Output ONLY a valid JSON array. Each element should be an object with these fields:
- "topic": the full statement (e.g., "Rick's portal fluid is actually a weapon.")
- "hook": a 1-sentence scroll-stopping hook to open the video.
- "answer_angle": a brief note on what the answer/insight is (for your reference).
- "difficulty": "easy", "medium", "hard".

Example output format:
[
  {{
    "topic": "Rick didn't destroy the Citadel to save Morty.",
    "hook": "Rick didn't destroy the Citadel to save Morty — he did it to save himself.",
    "answer_angle": "The Citadel represented conformity, and Evil Morty proved Rick's philosophy was weaponized.",
    "difficulty": "medium"
  }}
]
```

### 1.2 Update `scripts/topic_miner.py` Config

Ensure that `scripts/topic_miner.py` parses the new `topic` structure correctly and passes it to the `pipeline_state.json`.

```python
# In scripts/topic_miner.py - Adjust parsing logic if necessary
import json

def parse_topics(llm_output):
    try:
        topics_json = json.loads(llm_output)
        # Ensure no questions were generated as a safety check
        for t in topics_json:
            if "?" in t["topic"]:
                t["topic"] = t["topic"].replace("?", ".") # Basic fallback
        return topics_json
    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM output: {e}")
        return []
```

---

## 📝 Phase 2: Restructure Script Generation

The current `script_prompt.txt` has a good 4-act structure, but we need to optimize it for the "Continuous Audio Loop" and stricter pacing.

### 2.1 Update `prompts/script_prompt.txt`

Replace the entire contents of `prompts/script_prompt.txt` with the following:

```text
SYSTEM INSTRUCTIONS / PROMPT TEMPLATE:

You are a cynical, hyper-intelligent archivist of the Citadel of Ricks. You write relentless, viral YouTube Shorts scripts uncovering the darkest lore and hidden truths of {show_name}.

Your exact narration persona is: {narrator_style}
Your target topic: {topic}
Your retrieved multiversal database (Subtitles, Wikis, Theories):
{context}

CRITICAL DIRECTIVES:

STRICT WORD COUNT: The total narration text across all sections must be exactly 130 to 160 words. No exceptions.

MANDATORY 4-ACT SCRIPT STRUCTURE:
You MUST structure your script into exactly four psychological retention beats using these exact headers:

[HOOK]
(0-3s, max 12 words): State a ruthless, definitive lore bomb immediately. Do not ask questions. Do not greet the audience. State a mind-bending claim as an absolute truth.

[PROOF]
(3-10s, max 25 words): Present one specific visual detail or factual quote from the {context} that proves your hook is grounded in canon. Build trust immediately.

[ESCALATION]
(10-30s, max 80 words): Build the cynical theory step by step. Keep sentences brutally short (max 10 words per sentence). Punchy. Fast. Inject a rhetorical question every 3 sentences to force viewer engagement.

[PAYOFF]
(30-45s, max 20 words): Deliver the definitive conclusion. 
CRITICAL LOOPING RULE: The final sentence MUST perfectly set up the first sentence of the [HOOK] grammatically. If the Hook is "Rick's portal fluid isn't green for aesthetics," the Payoff must end with "...which is exactly why."

NEGATIVES: Never say "To construct this script", "I will first identify", "Let's dive in". Speak purely in-universe.
BANNED VOCABULARY: {avoid_phrases}, "delve", "unpack", "realm", "tapestry", "mind-blowing".

EXAMPLE OF PERFECT LOOPING SCRIPT (DO NOT COPY THIS TOPIC):
[HOOK]
Rick's portal fluid isn't green for aesthetics.

[PROOF]
In the Citadel archives, a hidden diagnostic report confirms standard portal fluid emits a lethal subspace frequency. Now look at Morty's bedroom.

[ESCALATION]
In season two, Rick installed a lead-lined vent directly above Morty's bed, leaving the rest of the house exposed. Why? Because Rick isn't protecting his grandson out of love. Morty's delta-brainwaves act as a natural cloaking device against the Federation. If Morty absorbs too much radiation, his brain frequencies degrade, rendering Rick's shield useless.

[PAYOFF]
He keeps him alive because a broken shield gets you killed. And that is exactly why...

FINAL OUTPUT:
Generate ONLY your assigned topic script using the exact bracket headers.
```

---

## 🎬 Phase 3: High-Retention Video Assembly (CapCut-ification)

This is the most critical engineering change. We need to move away from slow Ken Burns effects to high-impact, TikTok-style edits.

### 3.1 Implement Word-Level Bouncy Captions (`scripts/captioner.py`)

We need to generate ASS (Advanced SubStation Alpha) subtitles instead of standard SRT to allow for dynamic word highlighting.

Create a new function in `scripts/captioner.py` to generate `.ass` files from `faster-whisper` word-level timestamps.

```python
# Add to scripts/captioner.py
import datetime

def generate_dynamic_ass_subtitles(word_segments, output_ass_path):
    """
    Generates an .ass subtitle file with a 'karaoke' style effect
    where the currently spoken word pops out in bright yellow.
    """
    ass_header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Impact,110,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,3,5,10,10,250,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    with open(output_ass_path, 'w', encoding='utf-8') as f:
        f.write(ass_header)
        
        # Group words into short phrases (max 3-4 words per line for Shorts)
        lines = []
        current_line = []
        for word in word_segments:
            current_line.append(word)
            if len(current_line) >= 4 or word['word'].strip().endswith(('.', ',', '?', '!')):
                lines.append(current_line)
                current_line = []
        if current_line:
            lines.append(current_line)
            
        for line in lines:
            if not line: continue
            start_time = str(datetime.timedelta(seconds=line[0]['start']))[:11]
            end_time = str(datetime.timedelta(seconds=line[-1]['end']))[:11]
            
            # Create the ASS text string with karaoke highlighting
            # {\c&H00FFFF&} changes color to yellow for the active word
            text = ""
            for i, word in enumerate(line):
                duration_ms = int((word['end'] - word['start']) * 100)
                # K tag for duration, color override for active
                text += f"{{\\k{duration_ms}}}{{\\c&H00FFFF&}}{word['word'].strip()}{{\\c&HFFFFFF&}} "
            
            f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text.strip()}\n")
```

### 3.2 Implement Rapid Cuts and Micro-Zooms (`scripts/assembler.py`)

We will use FFmpeg filters to simulate a human editor adding rapid zooms on key beats. We will leverage the YOLO bounding boxes generated in Phase 4.

```python
# Add to scripts/assembler.py

def generate_zoom_filter(clip_duration, box_center_x, box_center_y, video_width=1080, video_height=1920):
    """
    Generates an FFmpeg zoompan filter string that snaps to 120% scale 
    exactly 0.5 seconds into the clip, centered on the YOLO bounding box.
    """
    zoom_factor = 1.2
    # Calculate crop coordinates based on YOLO center to keep character in frame
    x_val = f"({box_center_x} - (iw/zoom)/2)"
    y_val = f"({box_center_y} - (ih/zoom)/2)"
    
    # Snap zoom: if time > 0.5s, zoom to 1.2, else stay at 1.0
    zoom_expr = f"if(gte(time,0.5), {zoom_factor}, 1.0)"
    
    filter_string = f"zoompan=z='{zoom_expr}':x='{x_val}':y='{y_val}':d=1:s={video_width}x{video_height}:fps=30"
    return filter_string

def build_ffmpeg_command(clips_manifest, audio_path, subtitle_ass_path, output_path):
    """
    Constructs the complex FFmpeg command to concatenate clips, 
    apply dynamic zooms, add the looping audio, and burn the ASS subtitles.
    """
    # 1. Base command
    cmd = ["ffmpeg", "-y"]
    
    # 2. Add video inputs
    for clip in clips_manifest:
        cmd.extend(["-i", clip['path']])
        
    # 3. Add audio input
    cmd.extend(["-i", audio_path])
    
    # 4. Build filter complex
    filter_complex = ""
    for i, clip in enumerate(clips_manifest):
        # Assuming clips_manifest includes YOLO bounding box center (x,y)
        bx = clip.get('box_x', 540)
        by = clip.get('box_y', 960)
        
        zoom_filter = generate_zoom_filter(clip['duration'], bx, by)
        filter_complex += f"[{i}:v]scale=1080:1920,setsar=1/1,{zoom_filter}[v{i}];"
        
    # Concatenate all processed clips
    concat_inputs = "".join([f"[v{i}]" for i in range(len(clips_manifest))])
    filter_complex += f"{concat_inputs}concat=n={len(clips_manifest)}:v=1:a=0[concat_v];"
    
    # Apply ASS subtitles
    filter_complex += f"[concat_v]ass='{subtitle_ass_path}'[final_v]"
    
    cmd.extend(["-filter_complex", filter_complex])
    cmd.extend(["-map", "[final_v]", "-map", f"{len(clips_manifest)}:a"])
    
    # 5. Output settings for YouTube Shorts (H.264, AAC)
    cmd.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-c:a", "aac", "-b:a", "192k", "-shortest"])
    cmd.extend([output_path])
    
    return cmd
```

**Rule for Assembly:** Max clip length is 3 seconds. If `assembler.py` receives a clip requirement of 6 seconds, it must chop it into two 3-second segments and apply a horizontal flip (`hflip`) or scale change to the second half to force a visual reset.

---

## 🎨 Phase 4: Thumbnail and Metadata Optimization

### 4.1 Update `prompts/thumbnail_prompt.txt`

The thumbnail generation prompt must direct the generation of high-contrast, click-baity imagery.

```text
You are an expert YouTube thumbnail designer.
Based on the topic: {topic}
And the script hook: {hook}

Describe the most click-worthy, high-contrast, shocking frame we should extract or generate for the thumbnail.
It MUST feature a recognizable character from {show_name} looking shocked, angry, or secretive.
Include instructions for adding a glowing RED circle or RED arrow pointing to a specific background detail.
Keep the description under 40 words.
```

### 4.2 YouTube Metadata SEO Optimization (`scripts/publisher.py`)

YouTube Shorts rely heavily on initial seed audience matching.

Modify the upload metadata payload in `scripts/publisher.py`:
- **Title Constraint**: The YouTube title MUST be under 50 characters and NOT contain the name of the show. The name of the show goes in the hashtags.
- **Example Title**: `Rick lied to us all. 😳 #rickandmorty #theory`
- **Description**: Add parasitic SEO tags to the bottom of the description targeting high-volume searches.

```python
# Add to scripts/publisher.py

def generate_optimized_metadata(topic, script):
    # Extract hook for title
    hook = script.split("[HOOK]")[1].split("[PROOF]")[0].strip()
    
    # Keep title under 50 chars for optimal Shorts display
    title = hook[:45] + "..." if len(hook) > 45 else hook
    title = title + " #rickandmorty #theory"
    
    description = f"""
{script}

Dive deep into the darkest Rick and Morty theories, lore, and hidden details.
#rickandmorty #rickandmortytheory #evilmorty #adultswim #cartoonnetwork
    """
    
    return {
        "title": title,
        "description": description,
        "tags": ["rick and morty", "theory", "lore", "evil morty", "explained", "hidden details"]
    }
```

---

## 🚀 5. Execution Roadmap

1. **Immediate Execution (Day 1):** Replace the contents of `prompts/topic_prompt.txt` and `prompts/script_prompt.txt`. Run the orchestrator in `--dry-run` mode to evaluate the newly generated Hooks and Scripts for the required 4-act structure and looping finish.
2. **Core Development (Days 2-3):** Implement the `generate_dynamic_ass_subtitles` function in `scripts/captioner.py`. This is the single biggest ROI change for retention. Test against an existing audio file.
3. **Visual Engineering (Days 4-5):** Integrate the FFmpeg `zoompan` filters in `scripts/assembler.py`. Ensure YOLO bounding boxes are successfully passed from the matching phase to the assembler.
4. **Testing (Day 6):** Run a full pipeline test. Manually verify that the audio loop works perfectly (no silence at the end of the video) and that the visual cuts occur rapidly.
5. **Deployment (Day 7):** Deploy to `@rickify-i4y` and monitor the 10-second retention curve in YouTube Studio. If the curve drops below 80% at 10 seconds, decrease the average clip length in `assembler.py` to 2 seconds.

This plan moves the pipeline from an automated documentary generator to a high-retention, algorithmic view-farming engine matching the exact structural DNA of ThinkAlto and NyleTrix.
