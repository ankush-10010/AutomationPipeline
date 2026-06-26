---
name: implementation-plan
description: Create a deep, thorough implementation plan for a feature or growth goal. Performs real competitor research, reads the full codebase context, benchmarks against specific channels/products, and produces a detailed actionable document with code-level specifics.
---

# /implementation-plan — Deep Research & Implementation Planning Skill

## Purpose

This skill produces implementation plans at Claude Sonnet depth: competitor video-by-video analysis, codebase-specific code changes with actual file names and function names read from the repo, priority matrices, and a phased roadmap with success metrics. Short plans are a failure mode. If the output is under 400 lines for a complex goal, it is incomplete.

## Mandatory Research Steps — Do NOT skip any

Do all of the following before writing a single word of the plan. Research first, write second.

### Step 1: Read the full project context

Read `CLAUDE.md` in the project root. Then read these files if they exist:
- `pipeline_state.json` — understand current pipeline phases and state
- `requirements.txt` — understand what libraries are available
- `README.md` — understand the architecture overview
- Every file in `prompts/` — read every prompt template the pipeline uses
- Every file in `scripts/` or `src/` — understand the actual implementation
- `config/` directory if it exists — read all config files
- `graphify-out/` directory — read the knowledge graph output to understand entities, relationships, and discovered patterns in the project

Do not summarize — read fully so you can reference specific function names, line numbers, and actual prompt text in the plan.

### Step 2: Competitor research — channel by channel

For each competitor channel mentioned by the user, research it thoroughly using web search. Do not rely on general knowledge. For each channel:

- Search for the channel name + "youtube strategy" and "youtube shorts analysis"
- Find their top 10 most-viewed videos (by view count, not recency)
- For each of the top 5 videos, analyze:
  - The exact title format (what emotional trigger it uses)
  - The hook structure (first 3 seconds — what claim or question is made)
  - Estimated retention pattern (does the description or comments mention it?)
  - Visual style (what the thumbnail looks like, color scheme, text overlay style)
  - Script density (how many words per second based on transcript if findable)
  - What makes it different from generic content in this niche
- Identify patterns across their top videos: what topic formats, title structures, and hook types appear repeatedly
- Note what they do that our current pipeline does NOT do

### Step 3: Analyze our current output quality

Read files in `output/` or `clips/` or equivalent output directories to understand what we currently produce. If there are completed videos, note the file names.

Search for the user's channel (mentioned in the request) to understand:
- Current view counts on recent uploads
- Title and thumbnail style currently being used
- What the comment section says (audience signals)
- How our output compares to the competitor channels you just researched

### Step 4: Identify the specific gaps

Now produce a gap analysis table before writing the plan. For each dimension (hook quality, visual pacing, script structure, SEO, thumbnail, metadata), rate our current output vs. each competitor channel. This is the foundation of the plan.

---

## Plan Structure — Required Sections

Write the plan in this exact structure. Do not skip sections. Each section must be substantive — if a section is two paragraphs, it is not complete enough.

### Section 1: Executive Summary (1 page)
What is the core thesis of this plan? What is the single most important thing we need to change, and why will it matter? Write this as a narrative paragraph, not a bullet list.

### Section 2: Competitive Intelligence Report
For each competitor channel, write a dedicated subsection with:
- Channel overview (niche, posting frequency, subscriber range if findable)
- Their top 5 videos with titles, approximate view counts, and what made them perform
- Identified formula: the repeating patterns in their best content
- Specific techniques we should steal (name them precisely)
- Specific things they do that would NOT work for our automated pipeline (be honest)

### Section 3: Current Pipeline Audit
For each phase of our pipeline (reference actual phase names from `pipeline_state.json`), state:
- What the phase currently does
- Where it is strong
- Where it fails to match competitor quality
- Which specific file or prompt needs to change

### Section 4: Prioritized Change List
Order changes by impact × effort. For each change:
- Which file exactly (full relative path)
- What currently exists there (quote the relevant section)
- What needs to change (write the actual new prompt text or code, not a description of it)
- What output difference this will produce
- How to verify it worked (what to check in the output)

### Section 5: Phased Roadmap
Four phases. Each phase must specify:
- What gets done (specific tasks with file names)
- Success metric (how do we know this phase worked — view counts, retention percentages, specific observable outputs)
- How long it should take (realistic estimate based on complexity)
- What the blocker risks are

### Section 6: Measurement & Iteration Plan
How do we know the plan is working? What YouTube Studio metrics do we watch? At what thresholds do we iterate? What do we do if a change hurts performance?

### Section 7: Quick Wins (Do These First)
List 3-5 changes that take less than 2 hours to implement but will have immediate impact. These are the first things to do before anything else. For each, write the exact change needed.

---

## Output Standards

- Write in prose paragraphs for analysis and context. Use structured sections (headers) for the plan itself.
- Reference actual file names, function names, and prompt text from the codebase — not generic placeholders.
- When proposing new prompt text, write the full prompt, not a description of what the prompt should say.
- When proposing code changes, write the actual code diff or new function, not "update function X to do Y."
- Minimum output: 400 lines. If you are under that, you have not done the research step properly.
- Do not end the plan with "next steps" — the entire document IS the next steps. End with the measurement plan.

## Failure modes to avoid

- Writing a plan from general knowledge without actually reading the codebase — every file-specific recommendation must be grounded in what you read.
- Writing competitor analysis without actually searching for the competitor channels.
- Proposing changes to files that don't exist in the project.
- Generic advice that applies to any YouTube channel rather than specific changes to this specific automated pipeline.
- Treating the plan as a brainstorm — every item must be actionable with a specific file and a specific change.
