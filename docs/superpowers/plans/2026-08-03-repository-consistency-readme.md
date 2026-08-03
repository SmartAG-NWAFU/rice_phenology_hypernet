# Repository Consistency and README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one coherent English-only four-model repository, remove concrete experiment settings from public interfaces, align all local imports and package exports with the retained tree, and rewrite README without unsupported commands or capabilities.

**Architecture:** Replace the operational 2,760-line runner with a small injected-backend orchestration module that owns daily-DVR correction, accumulation, threshold crossing, and sequential stage advancement. Move shared scientific constants and the config-injected DVR objective into focused modules, use explicit path/provider contracts for omitted infrastructure, and update README last from the verified final filesystem.

**Tech Stack:** Python 3.10+, dataclasses, typing `Protocol`, NumPy, pandas, PyTorch, Git, `rg`, Python `ast`.

---

## Working-Tree and Commit Rules

The current `main` worktree contains staged deletions and mixed staged/unstaged
changes that the user has explicitly asked to finish, add, commit, and push.
Do not create a separate worktree because the approved runner replacement must
supersede the current uncommitted runner content. Before any source edit,
capture persistent worktree/index evidence. Keep implementation edits unstaged
until final verification; then `git add -A` only after the staged diff has been
reviewed as the approved accumulated public cleanup.

Reference specification:
`docs/superpowers/specs/2026-08-03-repository-consistency-readme-design.md`.

## File Map

**Create**

- `docs/superpowers/audits/2026-08-03-pre-implementation-manifest.md` — persistent HEAD/index/worktree evidence and edit allowlist.
- `scripts/validate_repository_consistency.py` — reusable accumulating red/green/staged-tree contract.
- `src/rice_phenology_hypernet/experiments/dvr_core.py` — shared stage/model/weather contracts.
- `src/rice_phenology_hypernet/models/dvr_objective.py` — common config-injected DVR loss.

**Replace**

- `src/rice_phenology_hypernet/experiments/runner_dvr.py` — concise four-model orchestration.

**Modify**

- `README.md`
- `findings.md`
- `progress.md`
- `task_plan.md`
- `src/rice_phenology_hypernet/types.py`
- `src/rice_phenology_hypernet/data/__init__.py`
- `src/rice_phenology_hypernet/data/io.py`
- `src/rice_phenology_hypernet/experiments/__init__.py`
- `src/rice_phenology_hypernet/experiments/threshold_utils.py`
- `src/rice_phenology_hypernet/experiments/regional_grid_projection.py`
- `src/rice_phenology_hypernet/experiments/regional_grid_analysis.py`
- `src/rice_phenology_hypernet/models/__init__.py`
- `src/rice_phenology_hypernet/models/m1_dvr_con.py`

`runtime.py`, `settings.py`, `m0.py`, `m1_v2_dvr.py`, `physics.py`, helper
scripts, and all pre-existing deleted paths are verify-only unless a later
structural check proves a direct contradiction.

### Task 1: Persist the dirty-worktree contract and run red checks

**Files:**

- Create: `docs/superpowers/audits/2026-08-03-pre-implementation-manifest.md`
- Create: `scripts/validate_repository_consistency.py`
- Update: `findings.md`, `progress.md`

- [x] **Step 1: Capture complete pre-edit evidence**

Run a read-only Python script whose path universe is the union of:

```bash
git ls-tree -r --name-only HEAD
git ls-files --stage
git ls-files --others --exclude-standard
```

For every path in that union, print:

```text
path | HEAD blob or ABSENT | index blob or ABSENT | porcelain status or CLEAN | worktree SHA-256 or ABSENT
```

Also print the current branch/upstream/ahead-behind state. Paste the exact
output into the audit manifest with this allowlist:

```text
README.md
docs/superpowers/audits/2026-08-03-pre-implementation-manifest.md
docs/superpowers/plans/2026-08-03-repository-consistency-readme.md
findings.md
progress.md
scripts/validate_repository_consistency.py
task_plan.md
src/rice_phenology_hypernet/types.py
src/rice_phenology_hypernet/data/__init__.py
src/rice_phenology_hypernet/data/io.py
src/rice_phenology_hypernet/experiments/__init__.py
src/rice_phenology_hypernet/experiments/dvr_core.py
src/rice_phenology_hypernet/experiments/runner_dvr.py
src/rice_phenology_hypernet/experiments/threshold_utils.py
src/rice_phenology_hypernet/experiments/regional_grid_projection.py
src/rice_phenology_hypernet/experiments/regional_grid_analysis.py
src/rice_phenology_hypernet/models/__init__.py
src/rice_phenology_hypernet/models/dvr_objective.py
src/rice_phenology_hypernet/models/m1_dvr_con.py
```

Record the manifest SHA-256 in `progress.md`.

- [x] **Step 2: Run the complete pre-change contract and confirm failure**

Create `scripts/validate_repository_consistency.py` as one accumulating
validator. It must never stop at the first error and must report all current
violations:

- Han characters in source/text files;
- exploratory model names/imports in `runner_dvr.py`;
- any local import resolving to a missing module;
- the deleted common loss import in `m1_dvr_con.py`;
- fixed experiment seed/run-ID defaults;
- stale package exports;
- README install/CLI/pytest/missing-path/unavailable-capability claims.

The validator must also contain the synthetic recording-backend contract used
after runner replacement. Before replacement it should catch the import or
interface mismatch and record it as another failure rather than crash.

Run exactly:

```bash
python3 scripts/validate_repository_consistency.py
```

Expected: `REPOSITORY_CONSISTENCY_CONTRACT_FAILED` with failures from every
category. Save the exact output in `progress.md` as the red baseline.

### Task 2: Add shared DVR contracts and config-injected objective

**Files:**

- Create: `src/rice_phenology_hypernet/experiments/dvr_core.py`
- Create: `src/rice_phenology_hypernet/models/dvr_objective.py`
- Modify: `src/rice_phenology_hypernet/models/m1_dvr_con.py`
- Modify: `src/rice_phenology_hypernet/models/__init__.py`

- [x] **Step 1: Create the shared scientific contract**

Implement:

```python
from dataclasses import dataclass
import numpy as np

DVR_STAGE_NAMES = ("tillering", "jointing", "booting", "heading", "maturity")
PHOTO_SENSITIVE_STAGES = frozenset({"booting", "heading"})
DEFAULT_WEATHER_FEATURES = ("TemAver", "TemMin", "TemMax", "daylength", "Precipitation")
PAPER_MODEL_NAMES = ("m0_t", "m0_dvr", "m1_v2_dvr", "m1_dvr_con")


@dataclass(frozen=True)
class StageInputs:
    doy: np.ndarray
    mask: np.ndarray
    model_inputs: object | None


@dataclass(frozen=True)
class StageRolloutResult:
    completion_doy: float
    next_start_doy: float
    corrected_dvr: np.ndarray
    cumulative_progress: np.ndarray
```

No seed, run ID, loss weight, or learned-model value belongs in this file.

- [x] **Step 2: Create the common objective with a required config**

Inspect the deleted source only with:

```bash
git show HEAD:src/rice_phenology_hypernet/models/dvr_loss.py
```

Do not restore or unstage `models/dvr_loss.py`. Move its equations into
`dvr_objective.py` and define:

```python
class DvrLossConfig(Protocol):
    event_loss_weight: float
    terminal_loss_weight: float
    shrink_loss_weight: float
    smooth_loss_weight: float
    mean_anchor_loss_weight: float
    stage_anchor_multipliers: tuple[float, ...]
    stage_terminal_weights: tuple[float, ...]
    stage_shrink_multipliers: tuple[float, ...]
    eps: float


def compute_dvr_loss(
    outputs: dict[str, torch.Tensor],
    true_duration: torch.Tensor,
    mask: torch.Tensor,
    *,
    config: DvrLossConfig,
    stage_index: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    ...
```

Every legacy scalar access becomes `config.<field>`. Preserve the existing
event, weighted terminal, shrinkage, smoothness, mean-anchor, first-crossing,
and duration-MAE equations.

- [x] **Step 3: Point the constrained objective at the shared module**

Change `ConstrainedDvrLossConfig` to extend `DvrLossConfig` plus required gate
weights. Replace the function-local deleted import and expanded scalar call
with:

```python
base_loss, base_stats = compute_dvr_loss(
    outputs,
    true_duration,
    mask,
    config=config,
    stage_index=stage_index,
)
```

Export the common objective types/functions from `models/__init__.py` only if
the final local-import check remains clean; do not re-export them from
`m1_v2_dvr.py`.

- [x] **Step 4: Run focused green checks**

Use AST assertions to verify the full field set, the exact objective keyword
interface, no numeric defaults, no `.dvr_loss` import, and successful parsing.
Expected: `DVR_OBJECTIVE_CONTRACT_OK`.

### Task 3: Make data, threshold, and package surfaces internally consistent

**Files:**

- Modify: `src/rice_phenology_hypernet/types.py`
- Modify: `src/rice_phenology_hypernet/data/io.py`
- Modify: `src/rice_phenology_hypernet/data/__init__.py`
- Modify: `src/rice_phenology_hypernet/experiments/threshold_utils.py`
- Modify: `src/rice_phenology_hypernet/experiments/__init__.py`

- [x] **Step 1: Add explicit raw-path ownership**

Add to `types.py`:

```python
@dataclass(frozen=True)
class RawDataPaths:
    weather: Path
    phenology: Path
```

- [x] **Step 2: Remove global configuration from data I/O**

Use these exact signatures:

```python
def load_raw_weather(path: Path) -> pd.DataFrame: ...
def load_raw_phenology(path: Path) -> pd.DataFrame: ...
def prepare_data_assets(
    raw_paths: RawDataPaths,
    prepared_paths: PreparedDataPaths,
) -> PreparedDataPaths: ...
def load_clean_data(prepared_paths: PreparedDataPaths) -> tuple[pd.DataFrame, pd.DataFrame]: ...
```

Create only the parent directories of explicit output paths. Do not call
`prepare_data_assets` from `load_clean_data`; raise `FileNotFoundError` with
both expected paths when prepared inputs are absent.

- [x] **Step 3: Clean package exports**

`data/__init__.py` exports only existing I/O helpers, path contracts, and
`DayLengthCalculator`. Remove every deleted dataset lazy export.

`experiments/__init__.py` exports only `PAPER_MODEL_NAMES`,
`ExperimentSpec`, `DvrExperimentBundle`, and `run_dvr_experiment` from the
new runner/core. Remove every deleted diagnostic, modifier, sensitivity,
deployment, and batch export.

- [x] **Step 4: Repair threshold utility provenance**

Import `THRESHOLD_COLUMNS` from `models.m0`, define
`THRESHOLD_DECIMALS = 2` as an output-format constant, and keep the merge and
prior-map calculations unchanged.

- [x] **Step 5: Run focused checks**

AST-assert the four exact data signatures, absence of
`rice_phenology_hypernet.config`, no deleted lazy exports, and threshold import
resolution. Expected: `PACKAGE_SURFACE_CONTRACT_OK`.

### Task 4: Replace the runner with the four-model scientific workflow

**Files:**

- Replace: `src/rice_phenology_hypernet/experiments/runner_dvr.py`

- [x] **Step 1: Define compact public contracts**

The replacement contains:

```python
@dataclass(frozen=True)
class ExperimentSpec:
    task: str
    model_name: str
    seed: int


@dataclass(frozen=True)
class FoldRecords:
    fold: int
    train_records: pd.DataFrame
    test_records: pd.DataFrame


@dataclass(frozen=True)
class DvrExperimentBundle:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    audit: dict[str, object]
```

Add protocols for the learned configuration sections and a
`DvrWorkflowBackend` with these separate responsibilities:

```python
def split_records(self, records: pd.DataFrame, *, task: str, seed: int) -> tuple[FoldRecords, ...]: ...
def estimate_stage_requirements(self, train_records: pd.DataFrame) -> Mapping[str, float]: ...
def fit_learned_model(self, model_name: str, model: object, train_records: pd.DataFrame, requirements: Mapping[str, float], config: object, *, seed: int) -> object: ...
def build_stage_inputs(self, record: pd.Series, stage_name: str, stage_start_doy: float, requirements: Mapping[str, float]) -> StageInputs: ...
def calculate_base_dvr(self, model_name: str, stage_name: str, inputs: StageInputs, requirements: Mapping[str, float]) -> np.ndarray: ...
def calculate_modifier(self, model_name: str, model: object, stage_name: str, inputs: StageInputs, requirements: Mapping[str, float]) -> np.ndarray | None: ...
def build_prediction_record(self, record: pd.Series, stage_predictions: Mapping[str, float], *, fold: int, model_name: str) -> dict[str, object]: ...
def score(self, predictions: pd.DataFrame) -> pd.DataFrame: ...
```

No backend method may perform correction, accumulation, threshold crossing, or
stage advancement. The same requirement object must be passed to fitting,
stage-input construction, base-DVR calculation, and modifier calculation.

- [x] **Step 2: Implement configuration-derived model construction**

`build_paper_model(model_name, config)` constructs exactly the four specified
models. M1 configuration dataclasses receive all tunable values from injected
config sections; only structural dimensions remain local.

- [x] **Step 3: Implement runner-owned daily correction and crossing**

Implement:

```python
def rollout_stage(
    inputs: StageInputs,
    base_dvr: np.ndarray,
    requirement: float,
    modifier: np.ndarray | None,
    stage_start_doy: float,
    *,
    trace: list[str] | None = None,
) -> StageRolloutResult:
    effective_modifier = np.ones_like(base_dvr) if modifier is None else modifier
    if trace is not None:
        trace.append("correct")
    corrected = np.where(inputs.mask, base_dvr * effective_modifier, 0.0)
    if trace is not None:
        trace.append("accumulate")
    cumulative = np.cumsum(corrected)
    if trace is not None:
        trace.append("cross")
    crossings = np.flatnonzero(inputs.mask & (cumulative >= requirement))
    completion = float(inputs.doy[crossings[0]]) if len(crossings) else float("nan")
    if trace is not None:
        trace.append("advance")
    next_start = completion + 1.0 if np.isfinite(completion) else float("nan")
    return StageRolloutResult(completion, next_start, corrected, cumulative)
```

Reject shape mismatches and non-positive requirements with clear English
errors.

- [x] **Step 4: Implement sequential fold execution**

`run_dvr_experiment(spec, config, backend)` must:

1. validate `spec.model_name` against `PAPER_MODEL_NAMES`;
2. obtain folds with `spec.seed`;
3. estimate one requirement object from each fold's training records and retain
   its identity;
4. pass that identical object into learned fitting;
5. loop through test records and the five stages in order;
6. pass the identical requirement object into separate stage-input, base-DVR,
   and modifier backend calls, selecting the active stage scalar only in the
   runner immediately before rollout;
7. call `rollout_stage(inputs, base_dvr, stage_requirement, modifier,
   stage_start_doy, trace=...)` directly;
8. use `next_start_doy` for the next stage;
9. build predictions and score them; and
10. record requirement provenance and call-order metadata.

- [x] **Step 5: Run a recording-backend check**

Construct a synthetic single-fold backend whose event log and object IDs prove:

```text
split -> estimate -> fit -> stage_inputs -> base_dvr -> modifier -> correct ->
accumulate -> cross -> advance (repeat five stages) -> score
```

Assert that modifier application changes the crossing only through
`base_dvr * modifier` before `np.cumsum`; `stage_start_doy` is supplied to each
rollout and equals the prior result's `next_start_doy`; and `id(requirements)`
is identical in fitting, stage-input, base-DVR, and modifier calls across all
five stages. Expected: `FOUR_MODEL_RUNNER_CONTRACT_OK`.

### Task 5: Decouple regional projection from deleted deployment code

**Files:**

- Modify: `src/rice_phenology_hypernet/experiments/regional_grid_projection.py`

- [x] **Step 1: Replace deleted imports and types**

Import stage/weather/model constants from `dvr_core.py`. Replace
`MaterializedProcessModel` typing with a local `RegionalPredictionModel`
protocol or `Any` inside the provider-owned wrapper. Add a local device
resolver rather than importing the old runner helper.

- [x] **Step 2: Add required regional experiment selection**

Define:

```python
@dataclass(frozen=True)
class RegionalProjectionSpec:
    deployment_run_id: str
    seed: int
    period: str


class RegionalModelProvider(Protocol):
    def prepare_models(
        self,
        *,
        spec: RegionalProjectionSpec,
        device: torch.device,
    ) -> list[PreparedDeploymentModel]: ...
```

Change `run_regional_grid_projection` to require `spec` and `model_provider`.
Remove `deployment_run_id`, `seed`, and the duplicated period argument from its
signature; all three come from `spec` with no defaults.

- [x] **Step 3: Thread provider/spec through projection**

Replace `_load_prepared_models` with provider validation. Ensure returned names
are unique, drawn from `PAPER_MODEL_NAMES`, and ordered by that tuple. Use
`spec.deployment_run_id`, `spec.seed`, and `spec.period` only for metadata and
provider calls. Preserve regional preparation, weather batching, prediction,
chunking, multiprocessing structure, and scientific constants.

- [x] **Step 4: Run focused checks**

AST-assert no imports from `data.dataset_dvr` or old runner deployment helpers,
required regional spec/provider arguments, no concrete seed/run ID defaults,
and four-model ordering. Expected: `REGIONAL_PROJECTION_CONTRACT_OK`.

### Task 6: Reduce regional analysis to climatology and metrics

**Files:**

- Modify: `src/rice_phenology_hypernet/experiments/regional_grid_analysis.py`

- [x] **Step 1: Remove unavailable figure infrastructure**

Delete the figure-builder and matplotlib normalization imports, figure result
dataclass, figure constants, `build_regional_grid_figures`, every plot helper,
the `build_figures`/`figures_dir` arguments, and figure paths from result
dataclasses/metadata.

- [x] **Step 2: Preserve numeric analysis**

Keep period resolution, path resolution, climatology aggregation,
heading/maturity metrics, period metrics, `_r2_score`, `_display_path`, and
JSON writing. Import `PAPER_MODEL_NAMES` from `dvr_core.py` and use it for
stable model ordering.

- [x] **Step 3: Run focused checks**

Assert no import/reference to deleted figures or `DEPLOYMENT_MODEL_NAMES`, no
figure API in public signatures/results, and preserved climatology/metrics
functions. Expected: `REGIONAL_ANALYSIS_CONTRACT_OK`.

### Task 7: Complete the English audit and rewrite README last

**Files:**

- Modify: `findings.md`
- Modify: `README.md`
- Update: `task_plan.md`, `progress.md`

- [x] **Step 1: Remove the last Han-character examples**

Replace the two historical Chinese phrase examples in `findings.md` with
English-only descriptions. Re-scan all existing text/source extensions and
require zero matches.

- [x] **Step 2: Re-inventory the final filesystem**

Run `rg --files` after source cleanup. Record the actual module tree and verify
all README paths against that inventory before writing prose.

- [x] **Step 3: Rewrite README**

Retain the scientific topics but correct repository claims. Use exactly these
main sections:

```text
Background
Objectives
Methodological Approach
Significance
Model Framework
Workflow
Repository Layout
Configuration and Parameter Provenance
Expected Data Interfaces
Regional Analysis Scope
Repository Scope
```

Do not include installation, CLI, pytest, seed, run-ID, YAML, interpretability,
diagnostic, sensitivity, general figure/table builder, supplementary-label, or
executable-reproducibility claims. The layout block must list only final paths.

- [x] **Step 4: Run README and language checks**

Assert all layout paths exist and banned headings/phrases/commands are absent.
Expected: `README_CONTRACT_OK` and `ENGLISH_ONLY_CONTRACT_OK`.

### Task 8: Run complete verification and close planning records

**Files:**

- Verify: all current Markdown/Python/shell/TOML/YAML/text files
- Update: `task_plan.md`, `findings.md`, `progress.md`

- [x] **Step 1: Run all-Python AST parsing**

Parse every existing `.py` under `scripts/` and `src/` with `ast.parse`.
Expected: `PYTHON_AST_OK: <count> files parsed`.

- [x] **Step 2: Run repository-wide local-import resolution**

Resolve absolute and relative `rice_phenology_hypernet` imports against actual
`.py` or package `__init__.py` paths. Expected:
`LOCAL_IMPORT_CONTRACT_OK: 0 missing imports`.

- [x] **Step 3: Run the complete green contract**

Run the exact reusable validator:

```bash
python3 scripts/validate_repository_consistency.py
```

It must cover the focused runner, objective, package, regional, README,
English, local-import, and recording-backend checks. Expected:
`REPOSITORY_CONSISTENCY_CONTRACT_OK` with no failure lines.

- [x] **Step 4: Run whitespace and diff review**

Run `git diff --check`. Inspect each allowlisted source/doc diff, with special
attention to the complete runner replacement and regional deletions. Record
line counts and source responsibilities in `progress.md`.

- [x] **Step 5: Verify the persistent audit manifest**

Regenerate the complete HEAD/index/untracked union table without editing the
baseline manifest. Assert:

- every non-allowlisted existing worktree hash matches;
- every pre-existing index blob/status is unchanged before final staging;
- every pre-existing deletion remains absent;
- every new/changed worktree path is allowlisted; and
- the runner is the sole approved whole-file replacement.

Expected: `DIRTY_WORKTREE_SCOPE_OK`.

- [x] **Step 6: Close planning records**

Mark Phases 13-15 complete and record exact verification evidence and the
pytest/end-to-end limitation. Do not claim runtime tests.

### Task 9: Stage, commit, and push the approved repository state

**Files:**

- Stage: the complete verified current repository state
- Commit: one final cleanup commit after the two existing design commits
- Push: `origin main`

- [x] **Step 1: Invoke the publication skill**

Read and follow `github:yeet` before staging. Confirm branch, upstream, remote,
and that no credential or generated data file has appeared.

- [x] **Step 2: Stage and inspect**

Run:

```bash
git add -A
git status --short
git diff --cached --check
git diff --cached --stat
git diff --cached --name-status
```

Regenerate the union of HEAD, staged index, and remaining untracked paths.
Compare the complete staged path set—not only deletions—against the manifest
and allowlist. Abort if any unapproved path appears, any pre-existing deletion
was restored, or any path expected in the final cleanup is silently absent.

- [x] **Step 3: Re-run verification on the staged tree**

Run `python3 scripts/validate_repository_consistency.py` plus the all-Python AST
and local-import checks again after staging. Expected: identical successful
results.

- [x] **Step 4: Commit**

Use:

```bash
git commit -m "refactor: align four-model public workflow"
```

Verify the commit contains the approved accumulated cleanup and no unrelated
path.

- [x] **Step 5: Push**

Run:

```bash
git push origin main
```

Verify `main` is synchronized with `origin/main` and report the final commit
hash. If authentication or remote policy blocks the push, preserve the local
commit and report the exact error without retrying destructively.
