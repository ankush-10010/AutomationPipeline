---
name: explain
description: Provide a deep, thorough explanation of a file, function, system, or architecture. Uses the Anthropic "Explanatory" output style — educational insights before and after each implementation choice.
---

# /explain — Deep Explanation Skill

Inspired by Anthropic's built-in `Explanatory` output style (`src/constants/outputStyles.ts`) and the "ant-internal" verbose communication mode.

## Goal

Produce a thorough, educational explanation of the target code, system, or concept. This is not a quick summary — the goal is that after reading your explanation, the user could modify, debug, or extend the target with confidence.

## What to explain

If the user specified a file, function, class, or concept — focus there. If they said "explain this codebase" or similar — start with the top-level architecture.

## Steps

### 1. Orient the reader

Begin with a one-paragraph "what is this and why does it exist?" — written for someone who has never seen this code before. Include:
- What problem it solves
- Where it fits in the larger system
- What it is NOT responsible for (the boundary)

### 2. Map the structure

For files/modules: list the main exports (functions, classes, constants) and what each one does in one sentence.

For systems/architectures: draw the component diagram in text or Mermaid. Show what calls what, what owns what, and what the data flow is.

### 3. Walk through the key logic

For the most important or non-obvious function/path:
- State the preconditions (what must be true when this is called)
- Walk through the logic step by step, explaining the WHY behind each significant decision
- State the postconditions (what is guaranteed when it returns)
- Note the error states and what triggers them

### 4. Highlight non-obvious things

For each thing that would surprise a competent developer reading this for the first time:
- Name it explicitly (don't let the reader discover it by accident)
- Explain why it works that way (historical reason, performance constraint, external dependency requirement, etc.)
- Note whether it's a known quirk, a deliberate design choice, or a debt that could be improved

### 5. Educational insight blocks

Before and after significant implementation sections, add insight blocks in this format:

```
★ Insight ─────────────────────────────────────
[2-3 key educational points about this implementation choice,
 specific to this codebase — not generic programming concepts]
─────────────────────────────────────────────────
```

These insights should focus on:
- Why this approach was chosen over the alternatives
- What invariant or constraint it relies on
- How it connects to a broader pattern in the codebase

### 6. How to extend or modify it

End with a practical section: if a developer needed to add a feature, fix a bug, or change behavior in this code, what are the right entry points? What would they need to change, and what are the risks of getting it wrong?

## Style rules

- Write in flowing prose. Not bullet fragments.
- Expand technical terms on first use.
- Assume the reader is a competent developer but has no prior context on this specific codebase.
- Include file paths and line numbers (format: `path/to/file.ts:42`) for every claim about where something happens.
- This skill intentionally produces long output. Do not truncate for brevity.
