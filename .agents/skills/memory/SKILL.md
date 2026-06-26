---
name: memory
description: Review the project's CLAUDE.md and session memory. Propose what to persist, what to clean up, and what is outdated or conflicting.
---

# /memory — Memory Review & Cleanup Skill

Adapted from Anthropic's internal `/remember` skill (`src/skills/bundled/remember.ts`).

## Goal

Review all memory layers for this project and produce a clear structured report of proposed changes, grouped by action type. Do NOT apply any changes — present proposals for the user to approve or reject individually.

## Steps

### 1. Gather all memory layers

Read the following files if they exist:
- `CLAUDE.md` in the project root — project-wide instructions for the agent
- `CLAUDE.local.md` in the project root — personal/user-specific agent instructions (not committed)
- `.agents/AGENTS.md` — Antigravity workspace rules

Note the contents and which layers are present. If any file is missing, note that too.

### 2. Review the current session context

Think about what you have learned during this session:
- Project-specific conventions you discovered (naming, structure, test patterns)
- Commands that are useful for this project (how to run tests, how to build, etc.)
- Non-obvious facts about the codebase architecture
- Preferences the user expressed during this session

### 3. Classify each piece of knowledge

For each piece of knowledge or instruction, determine the best destination:

| Destination | What belongs there | Examples |
|---|---|---|
| **CLAUDE.md** | Project conventions that all contributors using an AI agent should follow | "use bun not npm", "API routes use kebab-case", "test command is bun test", "prefer functional style" |
| **CLAUDE.local.md** | Personal instructions specific to this user, not applicable to other contributors | "I prefer concise responses", "always explain trade-offs", "don't auto-commit" |
| **.agents/AGENTS.md** | Workspace-level behavioral rules for the Antigravity agent | Tool use preferences, output style overrides, project-specific safety rules |
| **Discard** | Temporary context, session-specific observations, or things that are already captured | "we just fixed bug X" — already visible in git history |

**Important distinctions:**
- CLAUDE.md and CLAUDE.local.md contain **instructions for the AI**, not the user's preferences for external tools (editor theme, IDE keybindings don't belong here).
- Workflow practices (PR conventions, merge strategies, branch naming) are ambiguous — ask the user whether they're personal or team-wide before deciding.
- When unsure of the right destination, **ask rather than guess**.

### 4. Identify cleanup opportunities

Scan across all files for:
- **Duplicates**: Knowledge captured in multiple places → propose removing the lower-priority copy
- **Outdated**: Instructions contradicted by newer information → propose updating the older one
- **Conflicts**: Direct contradictions between files → propose resolution and note which is more recent
- **Redundant with defaults**: Instructions telling the agent to do what it already does by default → propose removing

### 5. Present the report

Structure your output as:

**New additions to propose:**
For each: destination file, the exact text to add, and a brief rationale.

**Cleanup to propose:**
For each: which file, what to remove/change, and why.

**Ambiguous — need your input:**
For each: the knowledge, the two candidate destinations, and a question for the user to resolve.

**No action needed:**
Brief note on things that are already well-captured or that should stay as working session context.

## Rules

- Present ALL proposals **before** making any changes.
- Do NOT modify any files without the user explicitly approving a specific proposal.
- Do NOT create new files unless the target file doesn't exist yet and a proposal is approved.
- Keep CLAUDE.md entries as short, imperative instructions. Not prose explanations.
