# Task Plan: Public Repository Consistency Cleanup

## Goal

Prepare the public research repository by keeping documentation and scripts English-only, aligning documented interfaces with the current source tree, removing misleading configuration-owned parameter defaults, and preserving unrelated working-tree changes.

## Current Phase

Complete

## Phases

### Phase 1: Scope and Inventory

- [x] Snapshot the dirty working tree before edits.
- [x] Identify existing script/source files and all Chinese-language occurrences.
- [x] Classify occurrences as comments/docstrings versus executable strings or data.
- [x] Record the exact edit scope in `findings.md`.
- **Status:** complete

### Phase 2: Translation Strategy

- [x] Establish terminology mappings and style conventions.
- [x] Decide how to handle mixed Chinese-English comments and docstrings.
- [x] Confirm no translation will alter identifiers, literals, or program behavior.
- **Status:** complete

### Phase 3: Translation

- [x] Translate in-scope comments and docstrings file by file.
- [x] Preserve indentation, code structure, and technical meaning.
- [x] Review focused diffs incrementally.
- **Status:** complete

### Phase 4: Verification

- [x] Re-scan existing scripts for Chinese characters.
- [x] Run syntax/compile checks for modified script files.
- [x] Run the smallest relevant tests if available and runnable.
- [x] Compare pre-edit and post-edit working-tree state.
- **Status:** complete

### Phase 5: Delivery

- [x] Summarize modified files and translation boundaries.
- [x] Report verification results and any remaining Chinese text outside scope.
- [x] Confirm unrelated user changes were preserved.
- **Status:** complete

### Phase 6: Parameter-Exposure Context Review

- [x] Inspect `m1_v2_dvr.py`, `m1_dvr_con.py`, their call sites, and current configuration files.
- [x] Inventory duplicated/default parameter values across the existing scripts.
- [x] Separate model-architecture defaults from training/loss settings sourced from configuration.
- [x] Check recent commits and document constraints in `findings.md`.
- **Status:** complete

### Phase 7: Clarification and Design Approval

- [x] Ask one focused clarification about the intended public-code abstraction boundary.
- [x] Propose two or three approaches with trade-offs and a recommendation.
- [x] Present the scoped design and obtain user approval.
- **Status:** complete

### Phase 8: Written Specification and Review

- [x] Write the approved design under `docs/superpowers/specs/` and commit it separately.
- [x] Run the required independent specification review loop.
- [x] Ask the user to review the approved written specification.
- **Status:** complete

### Phase 9: Implementation Planning

- [x] Invoke the required `writing-plans` skill after written-spec approval.
- [x] Record the implementation and verification sequence.
- **Status:** complete

### Phase 10: Implementation and Verification

- [x] Replace duplicated public-script parameter defaults with configuration-derived settings.
- [x] Review all existing scripts for the same misleading pattern within the approved scope.
- [x] Verify structural consistency and document any non-executable illustrative conventions.
- [x] Deliver the focused diff and verification evidence.
- **Status:** complete

### Phase 11: Repository-Wide Consistency Audit

- [x] Snapshot the current dirty worktree before the new audit.
- [x] Inventory every existing documentation and source/script file.
- [x] Scan for non-English text and configuration-owned numeric defaults.
- [x] Cross-check README paths, commands, modules, and workflow descriptions against existing files.
- [x] Record findings without changing implementation files.
- **Status:** complete

### Phase 12: Audit Design Approval

- [x] Clarify the parameter-disclosure boundary and treatment of explanatory scientific constants.
- [x] Define the treatment of historical planning documents and examples in the proposed design.
- [x] Present two or three remediation approaches and a recommendation.
- [x] Obtain approval for the exact edit boundary.
- **Status:** complete

### Phase 13: Written Audit Specification and Review

- [x] Write the approved design under `docs/superpowers/specs/`.
- [x] Complete the required independent specification review loop.
- [x] Ask the user to review the approved specification.
- **Status:** complete

### Phase 14: Implementation Planning

- [x] Produce a file-by-file implementation and verification plan.
- [x] Review the plan before editing repository content.
- **Status:** complete

### Phase 15: Consistency Cleanup and README Update

- [x] Apply approved documentation and script corrections.
- [x] Update README last from the verified repository state.
- [x] Run English, parameter-exposure, path/link, AST, and scope checks.
- [x] Deliver the focused diff and limitations without disturbing unrelated changes.
- **Status:** complete

## Key Questions

1. Which file extensions and directories constitute scripts/source in the current working tree?
2. Which Chinese occurrences are comments or docstrings rather than runtime strings or data?
3. Are there existing user changes overlapping any target file?
4. What verification is possible given the repository's current deleted files?
5. Which numeric defaults in model/loss signatures duplicate configuration values?
6. Should public supplementary scripts remain structurally executable, or prioritize concise reading-only pseudocode?
7. Should model architecture dataclass defaults also be removed, or only training/loss defaults?
8. Should internal planning/specification documents be part of the public English and parameter-exposure audit, or treated as development records?
9. Which numeric examples are explanatory process constants versus misleading experiment configuration values?
10. Which README commands and paths remain truthful in the intentionally incomplete reading-only tree?

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Treat Python docstrings as developer-facing comments unless they are clearly runtime/user-facing content | The request concerns code documentation, and docstrings commonly serve that role |
| Do not translate identifiers, dictionary keys, filenames, data values, or user-facing runtime messages without separate justification | Changing executable strings could alter behavior or interfaces |
| Work only on files that currently exist; do not restore files already deleted in the dirty worktree | Existing deletions belong to the user and are outside this task |
| Translate full comment/docstring sentences while preserving embedded English identifiers and tensor shapes | Produces natural English without changing the code contract |
| Use concise imperative/descriptive English consistent with surrounding Python documentation | Keeps style uniform and avoids adding new scientific claims |
| Use full-repository AST parsing instead of pytest for this comments-only change | The tracked test files and key modules are already deleted in the user's worktree, while AST parsing validates syntax without imports or generated caches |
| For the parameter-exposure task, remove defaults only for configuration-owned learned-model hyperparameters | The user explicitly chose this boundary while retaining structural dimensions and M0 process constants |
| Use one injected configuration object for constrained loss settings | This removes numeric defaults and long scalar signatures while keeping parameter provenance explicit |
| Remove or abstract only concrete configuration-managed experiment values | The user confirmed that parameter names, M0 process constants, regional periods, geographic thresholds, and unit conversions remain legitimate scientific/contextual information |
| Use a reading-oriented consistency cleanup without labeling README as supplementary reading code | The user selected the curated cleanup but explicitly requested a neutral project README with no non-executable install, CLI, or test commands |
| Replace the detailed runner with an injected-backend four-model workflow | This preserves the scientific order and training-fold isolation without restoring deleted operational modules or retaining exploratory-model branches |

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| `git diff --check` reports six trailing-whitespace lines in the pre-existing `runner_dvr.py` diff | 1 | Verified the affected whitespace was present in the pre-task file view; preserve it as unrelated user work and run focused checks on files/lines changed by this task |
| Combined verification command was rejected because its temporary-cache cleanup used `rm -rf` | 1 | Replace bytecode compilation with read-only `ast.parse` over every existing Python file; no cache or cleanup required |
| `configs/*.yaml` inspection failed with zsh `no matches found` | 1 | Treat the empty current `configs/` tree as a finding; inspect tracked `HEAD` paths and call sites without relying on an unresolved glob |
| Attempted to remove an apparent duplicate plan line that was produced by overlapping `sed` display ranges, not the file | 1 | Verified with targeted `rg`; the plan contains only one line and requires no edit |
| First repository-consistency specification review found eight scope and verification gaps | 1 | Revised the regional interfaces, rollout contract, seed ownership, worktree manifest, objective fields, data signatures, README claims, and zero-Han rule; second review approved the specification |
| First bulk implementation-plan revision patch did not match the current plan text | 1 | Split the revision into smaller targeted patches after locating exact headings and snippets |
| First repository-consistency implementation-plan review found three execution gaps | 1 | Added a complete path-union manifest, reusable validator, explicit runner event/identity contract, and read-only legacy objective migration; second review approved the plan |

## Notes

- Re-read this plan before changing scope or starting bulk edits.
- Update `findings.md` after every two repository search/view operations.
- Use `apply_patch` for source and planning-file edits.
