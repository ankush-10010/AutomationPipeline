---
name: review
description: Perform a thorough, adversarial code review of recent changes or a specified file/PR. Checks correctness, security, performance, style, and test coverage.
---

# /review — Deep Code Review Skill

Extracted from Claude Code's internal verification agent pattern (`VERIFICATION_AGENT_TYPE`) and Anthropic's code quality philosophy.

## Goal

Perform an independent, adversarial review of the specified code. Do NOT soften findings. Your job is to find problems, not to validate the author's choices.

## Steps

### 1. Understand the scope

If the user specified a file, PR, commit range, or function — focus there. If nothing was specified:
- Run `git diff HEAD~1 HEAD` to see the most recent commit.
- Run `git status` to see uncommitted changes.

### 2. Read every changed file in full

Do not skim. Read the complete file, not just the diff, to understand context. Pay attention to:
- What the code is trying to do
- What the existing patterns and conventions are
- Whether the new code is consistent with the existing patterns

### 3. Check for correctness

For each changed function or module, ask:
- Are there off-by-one errors?
- Are there edge cases that would cause incorrect behavior (empty inputs, null/undefined, concurrent access, network failures)?
- Does the logic match the stated intent of the code?
- Are return values and error states handled correctly everywhere they're used?

### 4. Check for security

Scan for OWASP Top 10 and common issues:
- **Injection**: SQL injection, command injection, XSS, template injection
- **Authentication/Authorization**: Are protected routes actually protected? Is user input used in permission checks?
- **Sensitive data exposure**: Are secrets, tokens, or PII logged or returned to clients?
- **Input validation**: Is external input (user input, API responses, file content) validated at the boundary?
- **Dependency risk**: Were new packages added? Are they well-maintained and necessary?

### 5. Check for performance

- Are there N+1 database queries (a query inside a loop)?
- Are expensive operations (file I/O, network calls) parallelized where possible?
- Is there unnecessary re-computation that could be memoized or cached?
- Are there memory leaks (event listeners not removed, large objects held in closures)?

### 6. Check code style & maintainability

Against the project's existing patterns:
- Does naming follow the project's conventions?
- Is the abstraction level appropriate? (Not too generic, not too specific)
- Are there comments explaining WHY where the logic is non-obvious?
- Are there unnecessary comments explaining WHAT (which the code already expresses)?
- Is the code testable? Are dependencies injected rather than hard-coded?

### 7. Check test coverage

- Do new functions have corresponding tests?
- Do the tests cover the happy path AND meaningful edge cases?
- Are the tests actually asserting the right things, or do they pass trivially?

### 8. Run checks

If the project has a lint or type-check command, run it:
```
npm run lint
npm run typecheck  
bun run check
```

Run the test suite:
```
npm test
bun test
pytest
```

Report the actual output — do NOT claim tests pass without running them.

### 9. Write the review report

Structure your report as:

**Summary**: One paragraph on the overall quality and whether you recommend merging/accepting as-is, with changes, or rejecting.

**Critical issues** (must fix before merging):
- Each issue with: location (`file:line`), description of the problem, and a concrete fix suggestion.

**Important issues** (should fix):
- Same format.

**Minor issues / suggestions** (optional but worth knowing):
- Same format.

**Positive observations** (what was done well — brief, not praise-heavy):
- Maximum 3 bullet points.

## Rules

- Do NOT soften findings with phrases like "this might be okay" or "just a suggestion" for Critical and Important issues. Be direct.
- Do NOT approve code that has failing tests. State clearly what failed.
- If you cannot run the tests (wrong environment, missing dependencies), say so explicitly — do not imply the code is correct.
- If the scope is large (>500 lines changed), prioritize the highest-risk areas first and note that you did so.
