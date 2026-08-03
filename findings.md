# Findings and Decisions

## Requirements

- Inspect all existing repository scripts/source files for Chinese comments.
- Translate Chinese comments into accurate, natural English.
- Include developer-facing docstrings in the audit.
- Preserve code logic, interfaces, literals, and formatting.
- Preserve all unrelated pre-existing working-tree changes and deletions.
- Review the public supplementary scripts, especially `m1_v2_dvr.py` and `m1_dvr_con.py`.
- Do not expose duplicated training/loss parameter values in function signatures when the actual experiment obtains them from configuration.
- Refactor the supplementary scripts so relevant parameters are obtained from configuration.
- Treat these files primarily as reading supplements; runtime executability is not a hard requirement.
- Re-audit the entire current repository for documentation/script consistency.
- Keep public-facing documentation, comments, and docstrings in English.
- Ensure configuration-managed experiment and learned-model parameters are not duplicated as concrete defaults or presented as independent sources of truth.
- Update README only after the repository audit establishes the current public surface.

## Research Findings

- Audit baseline Git-status snapshot: `/tmp/rice-public-status.before-project-audit.Ob9Ymu`, SHA-256 `1472121a0601036643cc219a820e9176b6d2908f59adef7128f36ea5980762d4`.
- The current filesystem contains 39 visible files from `rg --files`: `README.md`, three planning records, `requirements.txt`, six helper scripts, 26 package Python files, and placeholder `.gitkeep` files. There are no existing `configs/`, `tests/`, CLI, figure/table builder, modifier-interpretability, reviving-offset-sensitivity, or dataset/config-loader files.
- README is materially inconsistent with the existing tree: it documents `configs/`, `tests/`, editable-package installation through the deleted `pyproject.toml`, the deleted `rice_phenology_hypernet.cli`, deleted data/config modules, deleted analyses, and deleted figure/table builders as if currently available.
- README's opening scientific background, objectives, process-preserving DVR method, and bounded regional-plausibility framing remain aligned with the intended project narrative and should be preserved unless source inspection reveals a contradiction.
- Recent commits are documentation-only design commits on top of `origin/main`; the corresponding spec files are currently deleted in the worktree. This audit must not restore them implicitly.
- The repository-wide Han-character scan found only two Chinese phrases, both retained as historical translation examples in `findings.md`; all current README and Python source/script content is already English-only.
- AST inventory found no remaining configuration-owned learned-model defaults in `m1_v2_dvr.py` or `m1_dvr_con.py`. Their only numeric dataclass defaults are the approved structural dimensions (`input_dim=5`, `state_dim=2`), and `stage_index=None` remains a runtime input default.
- Other Python defaults fall into several categories that require source tracing before any edit: M0/physics process constants; regional projection operational defaults such as period, seed, chunk size, and device; helper-script geographic/aggregation defaults; runtime path/context defaults; and optional API controls. They must not be removed merely because they are numeric.
- `runner_dvr.py` still passes configuration fields explicitly to multiple common DVR-loss calls. These references expose parameter names and provenance but no concrete values; whether that violates the user's intended disclosure boundary depends on whether “parameter leakage” means concrete values only or also verbose parameter interfaces.
- Internal planning records contain historical descriptions of parameter names and prior values/cleanup decisions. Treating those records as public documentation would require redaction or removal; treating them as development-only records would keep the audit focused on README and distributed source.
- `settings.py` still defines paths for `configs/`, generated figures, and tables and creates artifact directories at import time. The absence of those source/config directories therefore reflects an intentionally incomplete public reading tree, not a self-contained runnable package.
- `runtime.py` snapshots any available YAML files but does not supply experiment parameters itself. Its `SETTINGS` and optional path/run arguments are dependency-injection or runtime-control defaults, not disclosed learned-model hyperparameters.
- `regional_grid_projection.py` is not independently runnable because it imports the deleted `data.dataset_dvr` module. It also exposes a concrete deployment run identifier and seed in `run_regional_grid_projection`; unlike period definitions and process constants, these are experiment-selection values and are plausible parameter-disclosure issues.
- `regional_grid_analysis.py` imports the deleted `figures.builder` module. README therefore must not advertise regional analysis/figure commands as currently executable.
- Regional period ranges, the weather-sequence limit, file names, and the reviving-date construction rule define the documented analysis/data contract. They should remain unless the user defines “parameter leakage” to include all study-specific design choices, which would make the supplementary code scientifically opaque.
- Static import resolution found 13 imports of currently missing local modules. The largest cluster is `runner_dvr.py`, which still imports the deleted configuration loader, DVR dataset, feature builder, split helpers, and six deleted learned-model variants; `data/io.py`, `threshold_utils.py`, `regional_grid_projection.py`, and `regional_grid_analysis.py` also reference deleted modules.
- The current runner is broader than the four-model public narrative: alongside the included M0, M1-V2, and M1-DVR-CON paths, it retains branches for deleted M1, M1-V3, M1-V4, radiation-constrained, M3-direct, and M3-sequential models. This is a major script-consistency issue even for reading-only code because readers cannot distinguish public methods from removed exploratory paths.
- `runner_dvr.py` consistently obtains learned-model and loss settings through `get_project_config()`. Its repeated scalar forwarding exposes names but no numeric values; a single configuration-object interface would improve readability, but common loss/model modules needed for a complete refactor are deleted.
- Helper-script defaults mostly encode dataset contracts, geographic subsets, unit conversions, request batching, or CLI ergonomics. They are not learned-model hyperparameters and should remain unless a concrete external configuration source exists.
- Complete absolute-and-relative import resolution found 21 references to missing local modules. Additional inconsistencies occur in `data/__init__.py` and `experiments/__init__.py`, whose lazy public exports still advertise deleted datasets, diagnostics, modifier analysis, and reviving-sensitivity modules.
- `m1_dvr_con.py` retains a function-local import of the deleted `models.dvr_loss` helper. Its configuration interface is now clean, but the displayed loss implementation is not self-contained.
- `models/__init__.py` is aligned with the three included model modules (two M0 classes plus M1-V2 and M1-DVR-CON), while the experiment/data package surfaces are not aligned with existing files.
- README's input-path references point to intentionally absent private data, which can remain as examples only if clearly labeled. Its `configs/data.yaml` reference is invalid even as an example because the configuration directory is not part of the public tree.
- `requirements.txt` exists, but `pip install -e .` is unsupported because the packaging file is deleted. Installing requirements alone would still not make workflows runnable because 21 local imports are unresolved.
- The deleted typed configuration schema confirms that split seeds, DVR batch seeds, learning schedules, learned-model settings, loss weights, sequence limits, and model-selection settings are experiment configuration. Concrete copies of those values must not be reintroduced into current source or README.
- The deleted CLI shows that fixed deployment seeds/run identifiers were previously hard-coded at the command layer rather than loaded through the typed project configuration. Under the user-approved disclosure rule, these are still concrete experiment selections and should be removed from current readable function signatures and README examples.
- The current README edit is layered on the original minimal public README. Its new Background/Objectives/Method/Significance sections are valuable and can be retained; the inherited layout, installation, CLI, workflow, and test sections require replacement to reflect the reduced tree.
- Internal `task_plan.md`, `findings.md`, and `progress.md` are development records rather than the scientific public interface. They should remain out of README navigation, but because the user requested an entire-project English audit, their two remaining Chinese example phrases should still be translated or generalized.
- The user selected the curated reading-oriented cleanup with one explicit README adjustment: do not describe the repository as “supplementary reading code.” README should present the scientific project neutrally while omitting unsupported installation, CLI, and test instructions.
- Runner symbol tracing shows that exploratory-model cleanup is not a small import edit. Deleted M1/M1-V3/M1-V4/radiation/M3 symbols are referenced across training, evaluation, prediction, task dispatch, materialization, and checkpoint helpers; several referenced training/prediction helpers are themselves absent. Removing these paths would reshape a large pre-existing uncommitted runner file.
- M1-V2 training still shares the deleted `compute_dvr_loss` implementation formerly imported through M1. Eliminating all deleted-model dependencies while retaining the detailed training path would require either relocating/recreating that loss implementation or reducing the runner to a higher-level paper-model orchestration sketch.
- Because `runner_dvr.py` contains extensive user-owned uncommitted work, a broad deletion is materially riskier than the previous two-call cleanup and requires an explicit scope decision before design approval.
- The user explicitly chose the broad runner trim despite that risk: the replacement should retain only `m0_t`, `m0_dvr`, `m1_v2_dvr`, and `m1_dvr_con` and remove the detailed exploratory-model branches.
- The three included model modules provide clean construction/forward interfaces. M0 is self-contained apart from included daylength/physics modules; M1-V2 is self-contained; M1-DVR-CON is self-contained except for its deleted common DVR objective.
- The current 2,760-line runner has roughly 70 top-level classes/functions. A readable four-model replacement should preserve the scientifically important sequence—configuration injection, training-fold-only requirement estimation, learned daily-DVR correction, threshold accumulation, sequential rollout, and evaluation—without retaining absent data pipelines, exploratory models, multiprocessing, checkpoint discovery, or CLI-oriented batch machinery.
- A small shared DVR-objective module with a required configuration protocol would resolve the remaining learned-loss dependency without reintroducing numeric defaults. This should be a new narrowly named module rather than restoring the deleted legacy file unchanged.
- The user approved the complete design: a four-model simplified runner, a shared config-injected objective, stale package-export cleanup, required regional experiment identifiers, full English audit, and a neutral README with no unsupported commands.
- `data/io.py` can be made consistent without restoring the deleted configuration loader: raw weather/phenology paths and processed output paths can be explicit required inputs or a caller-supplied path object. This also removes misleading path defaults.
- `threshold_utils.py` only needs threshold column names and rounding precision from the deleted feature module. The column names already exist in `models/m0.py`; the utility can depend on that included definition and retain rounding precision as an output-format constant.
- `data/__init__.py` should export only the explicit I/O/daylength helpers that remain, and `experiments/__init__.py` should export only the new simplified four-model runner surface. Lazy exports for deleted datasets, diagnostics, modifier analysis, sensitivity analysis, and batch/deployment entry points should be removed.
- The approved design is now recorded in `docs/superpowers/specs/2026-08-03-repository-consistency-readme-design.md` and committed alone as `785812f`; no staged user deletion or source modification was included in that commit.
- The first independent specification review found eight issues: unresolved regional dependencies after runner replacement; insufficiently explicit daily-DVR/threshold/rollout sequencing; duplicate seed ownership; weak dirty-index/worktree preservation; incomplete objective-field ownership; ambiguous data-I/O signatures; overly broad README claims; and a contradiction between the English exemption and zero-Han acceptance check.
- The required correction direction is concrete: add a small shared DVR-core contract, inject regional model providers, remove regional figure-builder claims, make runner execute correction/accumulation/crossing order directly, keep seed only in the experiment specification, capture persistent worktree/index hash manifests, enumerate every objective field, define exact path dataclasses/signatures, narrow README claims, and require true zero-Han content.
- The revised specification resolves all eight first-review findings and was committed alone as `598323e` (`docs: refine repository consistency design`).
- The second independent specification review approved the revision with all prior issues resolved. Implementation planning remains gated on the user's review of the written specification.
- The user approved direct execution plus final add/commit/push. The implementation plan may therefore include a final repository commit, but it must still distinguish the approved current deletion set from accidental new changes before staging.
- The legacy common DVR objective contains exactly the event, terminal, shrinkage, smoothness, mean-anchor, stage multiplier, epsilon, first-crossing, and MAE calculations named by the revised spec. Its equations can be transferred to the new objective module while replacing all scalar arguments with one required config object.
- Regional projection uses only three dataset constants from the deleted dataset module, so those can move cleanly into `dvr_core.py`. Its old runner dependencies are concentrated in device resolution, prepared-model representation/loading, prediction typing, and model-order recovery; these should be replaced by a local device helper plus the approved provider/spec contracts.
- Regional analysis figure code occupies most of the file after its core climatology/metrics functions. The approved simplification can remove the figure result, builder entry point, plot helpers, and figure metadata while retaining analysis/path/metric helpers through the numeric summary section.
- The shared DVR-core constants recover cleanly from the deleted dataset module: five stage names, the two photoperiod-sensitive stages, and the five weather feature names. These are scientific/interface definitions rather than tunable experiment values.
- The repository is on `main`, tracks `origin/main`, and is six commits ahead before implementation. The HTTPS origin is `https://github.com/SmartAG-NWAFU/rice_phenology_hypernet.git`; final push should target `origin main` after verified staging and commit.
- Porcelain-v2 status confirms a mixed index/worktree: numerous staged deletions, staged-plus-unstaged planning/model/runner changes, and unstaged README/M1-V2 changes. The audit manifest must capture all three layers before any new source edit, and the final commit will intentionally include the approved accumulated public-cleanup state because the user explicitly requested add/commit/push.
- The implementation plan is saved at `docs/superpowers/plans/2026-08-03-repository-consistency-readme.md`. It uses nine tasks, static red/green contracts in place of the deleted pytest suite, a persistent audit manifest, and one verified final stage/commit/push sequence.
- The first independent plan review found three execution gaps: the manifest path universe omitted the explicit HEAD/index/untracked union; runner verification did not expose separate base-DVR and internal rollout events or prove requirement-object identity at every stage; and the red/green contract was descriptive rather than one reusable executable artifact.
- The plan must add `scripts/validate_repository_consistency.py`, expand the audit union and post-stage comparison, separate stage-input/base-DVR/modifier backend calls, trace correction/accumulation/crossing/advancement, and read the deleted legacy objective only through `git show HEAD:...`.
- The revised plan now addresses all three issues with an explicit HEAD/index/untracked union, a reusable validator script, distinct backend methods, stage-start input, rollout tracing, requirement-object identity assertions, and an explicit read-only `git show` migration source.
- The second independent implementation-plan review approved the revised plan. The user explicitly authorized implementation on `main`, final staging, commit, and push, so the normal worktree isolation requirement is intentionally overridden.
- The persistent pre-implementation manifest captures every path in the HEAD/index/untracked union and records a SHA-256 of `b803ee900d47062181dc6b6cc1bb7cb0aa6ad6c5e7770e414c2f718331b684d6`.
- The reusable repository validator establishes a complete failing baseline across language, import, runner, parameter, export, regional, README, and orchestration-contract categories. The final runner should remain dependency-light so the synthetic backend check does not require PyTorch.
- A new repository-wide audit began after the model-parameter cleanup. The working tree remains intentionally dirty and now also shows the previously committed design/specification files as deleted in the worktree; these deletions must be treated as user state unless explicitly brought into scope.
- The audit must distinguish existing public documentation/source from deleted tracked paths. README should describe the current reading-only tree, not silently assume deleted CLI, configuration, test, figure-builder, or data-pipeline modules are present.
- The repository is already dirty, with many deleted tracked files and a modified `runner_dvr.py`; a before/after status comparison is required.
- The previous README task was superseded before README implementation began; two design-only commits already exist from that task.
- The current working tree contains 26 existing Python files and no other script-language files among the scanned extensions.
- Chinese characters occur in exactly three existing Python files: `models/m0.py`, `models/m1_dvr_con.py`, and `experiments/runner_dvr.py`.
- All occurrences found in the first scan are in comments or developer-facing docstrings; no identifiers, dictionary keys, filenames, or runtime messages need translation.
- `runner_dvr.py` already had user modifications before this task, so edits there must be limited to the identified comment/docstring lines and reviewed as a focused diff against the pre-task version where possible.
- The comments in `m0.py` consistently describe a temperature-only process baseline that omits photoperiod scaling; translations should use `temperature-only` and retain code identifiers such as `thermal`, `photo`, and `factor = 1.0` verbatim.
- The comments in `m1_dvr_con.py` describe stage-decaying background-information injection, sigmoid gates, gate-prior and monotonic regularization, and stage-specific heads. Translation must preserve these established code concepts without revising their scientific claims.
- The final inventory contains 85 Chinese-bearing lines: 11 in `m0.py`, 50 in `m1_dvr_con.py`, and 24 in `runner_dvr.py`.
- The Chinese-bearing regions in `runner_dvr.py` are comments/docstrings for M1-DVR-CON configuration/training, gate logging, temperature-only DVR helpers, checkpoint loading, and deployment-artifact materialization.
- The pre-existing `runner_dvr.py` diff is very large (2,424 additions and 631 deletions versus `HEAD`), so whole-file Git diffs cannot isolate this task. Verification will use the Han-character inventory plus exact target-line review and must not reformat the file.
- The first post-translation Han-character scan returned no matches across any existing source/script file.
- Focused diffs for `m0.py` and `m1_dvr_con.py` contain only comment/docstring translations.
- Targeted post-edit inspection of `runner_dvr.py` confirms its 24 Chinese-bearing lines were replaced with English in place, including materialization return/raise documentation not shown in the first compact search excerpt.
- Fresh verification found zero Han characters across all existing script/source extensions.
- Read-only `ast.parse` succeeded for all 26 existing Python files under `scripts/` and `src/`.
- `git diff --check` succeeded for `m0.py` and `m1_dvr_con.py`, the two target files that were clean before this task.
- Before/after status comparison shows the user's pre-existing deletions remain, `runner_dvr.py` remains modified with only its targeted comment/docstring lines translated, and this task added new modified-status entries only for `m0.py`, `m1_dvr_con.py`, and the three planning files.
- The deleted test suite cannot provide a meaningful focused pytest run in the current worktree; for a comments/docstrings-only task, full-source AST parsing is the available non-invasive verification.
- A new task now targets misleading parameter exposure. Context and design review must precede implementation because the boundary between configuration access and reading-only pseudocode affects multiple files and APIs.
- `m1_v2_dvr.py` exposes architecture defaults in `M1V2DvrConfig` (`input_dim`, `hidden_size`, `dropout`, `modifier_cap`, `event_beta`) but imports its loss function from the currently deleted `models/dvr_loss.py`.
- `m1_dvr_con.py` exposes the same architecture defaults plus `state_dim` and `background_gate_prior`, and directly exposes eleven loss/regularization defaults in `compute_m1_dvr_con_loss`.
- `runner_dvr.py` already obtains model and loss settings from `get_project_config().experiment...` and passes them explicitly into constructors/loss functions; this establishes configuration as the intended source of truth.
- The current `configs/` tree contains no YAML files, and `src/rice_phenology_hypernet/config.py` is already deleted in the working tree. The requested files are therefore currently illustrative/incomplete, consistent with the user's statement that they are reading supplements rather than executable code.
- The tracked `HEAD` version of `config.py` defines `DvrCorrectionConfig` and `ConstrainedDvrCorrectionConfig` fields for every named loss weight, stage multiplier, model hyperparameter, gate prior, and gate penalty. This confirms that the duplicated defaults are not intended as an independent source of truth.
- The tracked `HEAD` version of `dvr_loss.py` duplicates the same loss defaults that the user cited. That file is currently deleted, but `m1_v2_dvr.py` still imports and re-exports its `compute_dvr_loss`, so the public reading path is currently conceptually incomplete.
- No `configs/` YAML file is tracked in `HEAD`; only the typed configuration loader was tracked. Any reading-only refactor should show configuration access symbolically and should not pretend a runnable public configuration bundle exists.
- The only recorded commit touching the named model/config/loss files is the initial public reproducibility release (`48a89fc`); the much larger current `runner_dvr.py` state is uncommitted user work and must remain outside the design except for reference checks.
- The broader numeric-default scan found three different categories: process-definition constants in `m0.py`/`physics.py`, operational interface defaults in regional/runner functions, and configuration-owned learned-model/loss hyperparameters in the two named model paths. Only the third category directly matches the user's concern.
- `M0Parameters` and thermal/photoperiod response defaults are not represented in `DvrCorrectionConfig`; removing them under this task would conflate fixed process-model definitions with experiment hyperparameters.
- The recommended clarification boundary is therefore whether to remove all configuration-owned learned-model defaults (architecture plus loss settings) while retaining structural dimensions (`input_dim`, `state_dim`) and fixed process-model constants.
- The user confirmed that all configuration-managed defaults should be removed from the two learned models, including architecture, gate, and loss settings, while `input_dim`, `state_dim`, and M0 process constants remain explicit.
- Direct call-site inspection found one explicit `M1V2DvrModel` construction, two constrained-loss calls, and no external no-argument construction of either named model. Requiring model configuration objects is therefore consistent with the visible runner path.
- The approved approach uses explicit configuration-object injection rather than global configuration lookup or long required-scalar signatures.
- The written design removes the stale `m1_v2_dvr.py` import/re-export of the already deleted common loss helper, while keeping loss configuration responsibility in the experiment runner.
- First independent specification review identified four gaps: `eps` provenance, removal of the gate-prior vector and hard-coded initialization epsilon, isolated protection for the already-modified runner, and stronger structural assertions.
- The revised design treats `config.eps` as the exact reading-only representation of `experiment.m1_dvr_con.eps` named by the user, while explicitly acknowledging that the deleted public loader snapshot does not contain that field.
- The revised design replaces the gate-initialization literal with `torch.finfo(prior.dtype).eps` and requires a byte-for-byte pre-edit runner snapshot so only two constrained-loss call hunks may change.
- The second independent specification review approved the revised design with all four issues resolved.
- The user approved the written specification and authorized continuation.
- The implementation plan preserves the dirty current worktree instead of creating a separate worktree because the required runner changes must be applied on top of the user's uncommitted version.
- The implementation plan intentionally avoids source commits: an implementation commit touching `runner_dvr.py` could capture unrelated user work, while an isolated snapshot/diff provides safer evidence.
- First independent implementation-plan review found four issues: preserve exact `batch["stage_index"]`, count ten legacy runner kwargs, replace short-circuit checks with one complete AST contract, and correct macOS snapshot/status handling.
- The revised plan uses terminal `XXXXXX` templates, snapshots both runner bytes and Git status, validates the exact one-config loss signature plus all eleven `config.<field>` accesses, and checks both runner calls structurally.
- The second independent implementation-plan review approved the corrected plan with no remaining issues.
- Inline execution was explicitly selected by the user. The approved no-worktree exception remains necessary because `runner_dvr.py` must be edited on top of its uncommitted user state.
- Pre-edit runner snapshot: `/tmp/runner_dvr.before-config-cleanup.Nr5tto`, SHA-256 `adb3faa627f438343f7db11e93bd0c5cadcc8a192742682d06bd7035bc7b9b6f` (identical to the source before implementation).
- Pre-edit Git-status snapshot: `/tmp/rice-phenology-status.before-config-cleanup.wu01hA`.
- The complete pre-change contract check failed as expected on all designed conditions: model defaults, optional constructors, stale M1-V2 loss exposure, gate-prior/epsilon literals, expanded constrained-loss signature, missing config accesses, and both expanded runner calls.
- Post-edit runner AST confirms exactly two constrained-loss calls, each with only `model`, `config`, and `stage_index` keywords; `config=cfg` and `batch["stage_index"]` are preserved.
- Byte-for-byte runner comparison against `/tmp/runner_dvr.before-config-cleanup.Nr5tto` contains exactly two hunks, both replacing the ten legacy config kwargs with `config=cfg`.
- The final complete structural contract passed: both learned-model constructors require configuration, all configuration-owned tunable fields are required, the constrained loss reads all eleven settings through `config`, and only `stage_index` retains its runtime default.
- Read-only `ast.parse` again succeeded for all 26 existing Python files under `scripts/` and `src/`; focused whitespace checks passed for both modified model files.
- The final status comparison added `m1_v2_dvr.py` as the only newly modified source path. Files already staged and edited during execution changed from `A ` or `M ` to `AM` or `MM`; all pre-existing deleted and unrelated modified paths retained their scope.
- No pytest, import, or end-to-end execution result is claimed because the current reading-only worktree lacks the deleted configuration loader, common DVR loss module, and tests.
- The implementation remains uncommitted on `main`, as required, so the user's existing staged and unstaged work is preserved without integration or cleanup operations.

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| Inventory first, then patch only classified comment/docstring occurrences | Prevents accidental translation of executable strings or research data |
| Use Unicode-aware searches across existing source-like files | Chinese text may occur in comments, docstrings, or mixed-language lines |
| Scope the implementation to the three existing Python files containing Han characters | The full source-extension scan found no Chinese text in the other existing script files |
| Preserve code terms and shapes verbatim while translating surrounding prose | Names such as `background_gate`, `stage_state`, `thermal`, and tensor dimensions are part of the implementation contract |
| Describe gated background context as `stage-decaying background-information injection` | This preserves the code's intended gate behavior without renaming variables |
| Use `temperature-only` and `photoperiod effects` for the corresponding process concepts | These are standard, concise technical terms |
| Keep arrow symbols, inequalities, variable names, and shape annotations unchanged | They encode model relationships rather than natural-language prose |

## Issues Encountered

| Issue | Resolution |
|-------|------------|
| `git diff --check` reports six trailing-whitespace lines in `runner_dvr.py` | These lines belong to the large pre-existing user diff and were visible before translation; leave them untouched and disclose the focused-check limitation |
| Verification shell command was rejected due to a temporary-directory cleanup command | Use read-only AST parsing instead of generating bytecode caches |
| Shell glob `configs/*.yaml` had no matches | Inspect Git tree paths explicitly and avoid unmatched globs |

## Resources

- `task_plan.md`
- Repository Git status and tracked source-file inventory
- `src/rice_phenology_hypernet/models/m0.py`
- `src/rice_phenology_hypernet/models/m1_dvr_con.py`
- `src/rice_phenology_hypernet/experiments/runner_dvr.py`

## Visual/Browser Findings

- None; this is a filesystem and source-code task.

## Repository Consistency Implementation Findings

- The common loss equations migrated without numerical changes; only parameter access changed from scalar defaults to the required `DvrLossConfig` object.
- A dependency-light runner is sufficient to express the paper workflow. Model imports can remain function-local, while backend injection keeps splitting, features, fitting, and scoring outside the scientific rollout.
- The same training-derived requirement mapping is now passed by identity to learned fitting and every stage input, base-DVR, and modifier calculation within a fold.
- Regional artifact selection can be decoupled from the deleted deployment implementation by requiring a provider that returns one validated model for each name in `PAPER_MODEL_NAMES`.
- Removing figure infrastructure from regional analysis preserves climatology, period aggregation, heading/maturity metrics, and run-metadata behavior while eliminating missing imports and unsupported README claims.
- The final README describes only retained files and interfaces and contains no runnable installation, CLI, or test instructions.
