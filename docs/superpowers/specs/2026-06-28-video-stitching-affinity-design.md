# Design Specification: Video Stitching Affinity & Persistence

**Date:** 2026-06-28
**Context:** The video assembly pipeline suffers from visual discontinuity. While recent fixes added cooldown windows to prevent literal loops of the same clip, the video still randomly jumps between seasons and empty rooms because segments are evaluated in total isolation. 
**Goal:** Make the final video feel like a human-edited, cohesive narrative by keeping clips anchored to a central episode (when applicable) and maintaining visual focus on subjects across sentences.

## 1. Architecture Overview

This design enhances Phase 4 (`clip_matcher.py`) with two new contextual memory systems:
1. **Global Episode Affinity**: A pre-computation step that reads the entire script to determine if there is a dominant episode. If found, clips from that episode receive a scoring advantage.
2. **Subject Persistence (Memory)**: A sliding state within the segment matching loop that carries over character and location context when the current sentence lacks explicit subjects (e.g., when using pronouns).

---

## 2. Global Episode Affinity (Implementation)

### Component: `calculate_dominant_episode`
Before iterating through the segments in `build_manifest`, we will analyze the script holistically.

**Data Flow:**
1. Concatenate all segment text into a single `full_script` string.
2. Run `extract_keywords(full_script)` to get the core themes/actions. *Note: `extract_keywords` already strips out common English stop-words ("i", "am", "from", "the", etc.) ensuring we only match on meaningful nouns and verbs.*
3. Iterate over all eligible clips in the `clip_index`. Group them by episode prefix (e.g., `s1e1`, `s9e1`).
4. For each episode prefix, sum the keyword overlap between the `full_script` keywords and the clips' tags/actions.
5. Identify the highest-scoring episode prefix. If its score is significantly higher than the runner-up (or simply > 0 if only a few episodes exist), assign it as `dominant_episode_key`.

**Scoring Impact:**
Inside `match_keyword` and `match_semantic`, if `dominant_episode_key` is provided and the candidate clip's filename matches the prefix, add a `+2.0` score bonus.

---

## 3. Subject Persistence (Implementation)

### Component: `Context Memory Variables`
Inside the `build_manifest` segment loop, we will track the narrative focus.

**Data Flow:**
1. Initialize two variables outside the segment loop: `active_characters = set()` and `active_locations = set()`.
2. For Segment N, extract characters and locations from the text.
3. **Inheritance Rule:**
   - If Segment N mentions explicitly new characters, `active_characters` is overwritten with the new characters.
   - If Segment N mentions NO characters, it inherits the characters from `active_characters`.
   - The exact same logic applies to `active_locations`.
4. **Scoring:** The matching strategies (`match_keyword`, `match_semantic`) will now score the segment using the *inherited* characters/locations instead of strictly the ones found in the text.
5. **State Update:** After a clip is successfully selected, the system updates `active_characters` to reflect who is *actually* in the clip (to ensure the visual truth anchors the memory). If the clip has no metadata, the memory retains the inherited state.

---

## 4. Error Handling & Edge Cases

*   **Vague Scripts (No Dominant Episode):** If the script is entirely generic and no episode scores higher than others, `dominant_episode_key` remains `None` and the pipeline functions identically to how it does today (pulling the best clip from any season).
*   **Topic Drifts:** If the script changes characters (e.g., Rick to Morty), the persistence memory instantly overwrites, preventing Rick from being forced into a scene about Morty.
*   **Empty Metadata:** Since most clips currently lack character metadata, the Persistence system relies on the few tagged clips. When a character is inherited, it will heavily bias the system to pick one of the few tagged character clips, keeping them on screen.

## 5. Testing & Verification

1.  **Affinity Test:** Run a script heavily summarizing Season 1 Episode 1. Verify via logs that `dominant_episode_key` is detected as `s1e1` and that clips with this prefix receive the `+2.0` bonus.
2.  **Persistence Test:** Run a script with two segments: "Rick enters the room." followed by "He looks around." The second segment should log that it inherited "rick" and selected a clip containing Rick, rather than cutting to a random generic shot.
