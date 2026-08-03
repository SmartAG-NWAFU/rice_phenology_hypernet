# Repository Consistency and README Design

**Status:** Approved for implementation design on 2026-08-03

## Objective

Align the current public repository's English documentation, model scope,
configuration provenance, and source interfaces. Replace the oversized DVR
runner with a concise four-model workflow, remove concrete experiment settings
from public signatures and documentation, and rewrite README from the verified
current tree.

The four paper models are:

- `m0_t`
- `m0_dvr`
- `m1_v2_dvr`
- `m1_dvr_con`

## Repository Constraints

The working tree is intentionally dirty. Many tracked modules, configuration
files, tests, builders, and packaging files are already deleted. Those
deletions belong to the user and must not be restored implicitly. Existing
changes in `README.md`, `runner_dvr.py`, `m0.py`, `m1_v2_dvr.py`, and
`m1_dvr_con.py` must be preserved unless this specification explicitly
replaces the affected content.

Before implementation, write a persistent audit manifest under
`docs/superpowers/audits/`. For every tracked path, record the HEAD blob ID,
index blob ID and status, worktree SHA-256 when the path exists, and whether the
path is absent. The manifest must also record the exact implementation
allowlist. Source changes must remain unstaged so every pre-existing index blob
remains unchanged. At completion, all non-allowlisted worktree hashes must
match, every pre-existing deletion must remain absent, and the runner's changed
hash must be identified as the approved whole-file replacement.

The code is intended to communicate the scientific workflow and does not need
to provide a complete executable release. Nevertheless, every edited Python
file must remain syntactically valid, interfaces shown together must agree,
and the simplified runner must not import deleted exploratory models.

README must not describe the repository as "supplementary reading code." It
must use neutral project framing and must not include installation, CLI, or
test commands that the current tree cannot support.

## Disclosure Boundary

Remove or abstract concrete values that select or tune an experiment:

- learned-model architecture and regularization values supplied by project
  configuration;
- loss weights and stage multipliers;
- training schedules, optimizer settings, model-selection settings, and batch
  seeds;
- fixed deployment seed values and deployment run identifiers; and
- concrete experiment identifiers in README examples.

Retain information needed to understand the scientific and data contracts:

- parameter names and types;
- `input_dim` and `state_dim` structural dimensions;
- M0 temperature and photoperiod process constants;
- stage names and threshold definitions;
- regional periods and their year ranges;
- geographic filters, data aggregation rules, unit conversions, and output
  formatting constants; and
- operational controls such as device selection, worker counts, request
  batching, and chunk sizes when they are not experiment settings.

No learned-model or loss parameter may regain a concrete default.

## Simplified Runner Architecture

Replace `src/rice_phenology_hypernet/experiments/runner_dvr.py` with a concise
four-model orchestration module. The replacement should be understandable as a
single scientific workflow rather than a complete operational training system.

### Retained responsibilities

The runner must show these steps in order:

1. receive an externally supplied configuration and experiment specification;
2. split parent site-year records according to the requested evaluation task;
3. estimate stage requirements from the current training fold only;
4. construct one of the four paper models;
5. fit a learned daily-DVR modifier when the selected model is learned;
6. calculate the process-derived daily base DVR for the active stage;
7. apply the learned positive modifier to that daily DVR before accumulation;
8. accumulate the corrected DVR and identify the first threshold crossing;
9. advance the predicted completion date to the next stage's start state and
   repeat the same daily sequence; and
10. return prediction and evaluation records.

The runner, not one opaque backend method, must own steps 6-9. The estimated
training-fold requirement object must be passed unchanged to model fitting and
every stage rollout for all four models. A learned backend may produce trained
models or daily modifier sequences, but it may not perform hidden accumulation,
threshold crossing, or stage advancement.

### Public types and interfaces

Use small dataclasses and `Protocol` contracts rather than imports from deleted
data, split, CLI, or configuration modules. At minimum, the runner should
define:

- the exact `PAPER_MODEL_NAMES` tuple;
- an experiment specification containing task, model name, and the sole
  required seed value;
- a configuration protocol exposing the two learned-model configuration
  sections without concrete values;
- a workflow-backend protocol for split construction, training-fold
  requirement estimation, learned-model fitting, stage-input construction,
  process base-DVR calculation, learned modifier calculation, and scoring; and
- a compact experiment result containing predictions, metrics, and audit
  metadata.

The public entry point must be
`run_dvr_experiment(spec, config, backend)`. It must not accept a second seed
argument. It must validate the four-model name set and preserve the
training-fold-only requirement-estimation boundary.

Define a pure runner-owned threshold-crossing function whose inputs include the
base-DVR sequence, optional learned modifier sequence, valid-day mask, stage
requirement, and stage start state. It must multiply before cumulative summation
and return the first crossing date plus the next-stage start state. Sequential
rollout must call this function once per stage in order.

### Model construction

- `m0_t` constructs `M0TPhenologyModel`.
- `m0_dvr` constructs `M0PhenologyModel`.
- `m1_v2_dvr` constructs `M1V2DvrModel` from a required
  `M1V2DvrConfig` populated from the injected configuration.
- `m1_dvr_con` constructs `M1ConDvrModel` from a required
  `M1ConDvrConfig` populated from the injected configuration.

The runner may retain structural input/state dimensions. It must not copy any
learned-model or loss value.

### Shared DVR core

Add `src/rice_phenology_hypernet/experiments/dvr_core.py` for definitions used
by both the simplified runner and regional projection:

- `DVR_STAGE_NAMES`;
- `PHOTO_SENSITIVE_STAGES`;
- `DEFAULT_WEATHER_FEATURES`;
- the exact four-name `PAPER_MODEL_NAMES` tuple; and
- small stage-input/result dataclasses used by runner-owned rollout.

This module contains scientific structure, names, and tensor/data contracts;
it must not contain learned-model values, loss weights, experiment seeds, or
run identifiers.

### Removed responsibilities

Delete imports, branches, helpers, and identifiers for:

- legacy `m1_dvr`;
- M1-V3 and M1-V4 variants;
- radiation-constrained variants;
- M3 direct and sequential variants;
- deleted modifier/diagnostic workflows;
- CLI-oriented batch dispatch;
- multiprocessing fold execution;
- checkpoint discovery across removed models; and
- detailed deployment materialization tied to concrete run identifiers.

## Shared DVR Objective

Add a narrowly scoped shared objective module under
`src/rice_phenology_hypernet/models/`. Do not restore the deleted legacy
module unchanged.

The module must:

- define `DvrLossConfig` containing, without defaults,
  `event_loss_weight`, `terminal_loss_weight`, `shrink_loss_weight`,
  `smooth_loss_weight`, `mean_anchor_loss_weight`,
  `stage_anchor_multipliers`, `stage_terminal_weights`,
  `stage_shrink_multipliers`, and `eps`;
- implement the common event, terminal, shrinkage, smoothness, and optional
  mean-anchor/stage-aware terms;
- accept one required configuration object rather than a long set of scalar
  hyperparameters;
- allow only `stage_index` to retain a runtime `None` default; and
- expose no concrete learned-model, stage-weight, or epsilon values.

`m1_dvr_con.py` must use this shared objective and extend it with gate-prior
and gate-monotonic regularization. Its `ConstrainedDvrLossConfig` must extend
the common protocol with required `gate_prior_weight` and
`gate_monotonic_weight` fields. The required model configuration continues to
own `background_gate_prior` without a default.
`m1_v2_dvr.py` remains a model-definition module and does not re-export the
objective.

## Package and Utility Consistency

### Data package

Remove lazy exports for deleted dataset classes from `data/__init__.py`.
Add a required `RawDataPaths` dataclass with `weather` and `phenology` paths.
Use the existing `PreparedDataPaths` output contract. The exact interfaces are:

- `load_raw_weather(path: Path)`;
- `load_raw_phenology(path: Path)`;
- `prepare_data_assets(raw_paths: RawDataPaths, prepared_paths: PreparedDataPaths)`;
  and
- `load_clean_data(prepared_paths: PreparedDataPaths)`.

`load_clean_data` must raise `FileNotFoundError` when prepared weather or
phenology data are absent; it must not trigger implicit preparation. Remove the
deleted global project-configuration import and export only these existing,
internally consistent helpers.

### Experiment package

Remove lazy exports for deleted diagnostics, modifier analysis, regional
sensitivity, batch deployment, and other absent entry points. Export only the
simplified four-model runner surface that exists after the rewrite.

### Threshold utilities

Remove the import from the deleted feature-engineering module. Reuse the
included threshold-column definition and retain rounding precision only as an
output-format constant.

### Regional projection

Keep regional periods, year ranges, geographic/data rules, sequence limits,
and operational controls. Remove concrete deployment run identifiers and seed
defaults from public projection interfaces; require them explicitly wherever
they select an experiment artifact.

Replace imports from deleted `data.dataset_dvr` with the constants and
dataclasses in `experiments/dvr_core.py`. Remove imports of deployment artifact
classes/functions from the old runner. Define a `RegionalModelProvider`
protocol that receives a required `RegionalProjectionSpec` containing the
deployment run identifier, seed, and period, and returns prepared prediction
models for the exact four paper model names. Regional projection may retain
chunking and device controls, but artifact selection must flow only through the
required specification/provider pair.

### Regional analysis

Do not advertise or export deleted general figure-builder infrastructure.
Remove the deleted figure-builder import, figure-result dataclass,
`build_regional_grid_figures`, figure-generation helpers, `build_figures`
control, and figure paths from analysis results/metadata. Keep the regional
climatology and metrics algorithms with English documentation. Replace
`DEPLOYMENT_MODEL_NAMES` with `PAPER_MODEL_NAMES` from `dvr_core.py`.

### Runtime and settings

Preserve path construction, run metadata, and artifact-directory behavior.
`ProjectSettings` and ordinary runtime-control defaults are not experiment
parameter disclosures.

## English Audit

Scan every existing Markdown, Python, shell, TOML, YAML, and text file for Han
characters. Translate or generalize the two historical Chinese examples in
`findings.md`. The current repository has no required Han-bearing identifier,
column name, filename, or data value, so no allowlist applies: the final scan
must return zero matches.

All new source comments, docstrings, specification text, and README prose must
be English.

## README Design

Retain the scientific topics, but revise the existing prose rather than
preserving its repository claims:

- Background;
- Objectives;
- Methodological Approach; and
- Significance.

In particular, replace "public supplementary code" with neutral wording;
remove claims that the current repository includes modifier diagnostics,
interpretability analysis, reviving-offset sensitivity, or general
figure/table builders; and narrow "supports reproducible analysis" to a claim
about transparent method structure rather than executable reproduction.

Replace the remaining README with sections derived from the final filesystem:

1. **Model Framework** — explain the two process baselines and two learned DVR
   modifiers.
2. **Workflow** — describe requirement estimation, daily DVR calculation,
   learned correction, accumulation, threshold crossing, and sequential
   rollout without command examples.
3. **Repository Layout** — list only files and directories that exist after
   implementation.
4. **Configuration and Parameter Provenance** — state that experiment and
   learned-model values are supplied externally and are not duplicated in the
   public model definitions.
5. **Expected Data Interfaces** — describe required conceptual weather,
   phenology, and regional inputs without referring to a missing YAML file.
6. **Regional Analysis Scope** — present regional output as a bounded
   plausibility analysis, not grid-cell validation.
7. **Repository Scope** — list omitted raw data, trained weights, generated
   outputs, configuration values, and manuscript assets.

Remove:

- editable-install and environment setup instructions;
- all `python -m rice_phenology_hypernet.cli` commands;
- the public CLI command list;
- pytest commands and test-suite claims;
- references to missing YAML files, builders, diagnostics, or sensitivity
  modules;
- fixed seeds, deployment run identifiers, and concrete learned parameters;
  and
- any statement that labels the repository as supplementary reading code.
- all uses of "supplementary" as a repository label;
- unavailable interpretability, diagnostic, sensitivity, general figure/table
  builder, or full reproducibility claims.

## Verification Contract

The implementation is complete only when all of the following checks pass:

1. A repository-wide Han-character scan returns no matches in any existing
   Markdown, Python, shell, TOML, YAML, or text file; no allowlist is permitted.
2. Every existing Python file under `scripts/` and `src/` parses with
   `ast.parse`.
3. The simplified runner contains exactly the four paper model identifiers and
   no exploratory-model imports, branches, or string literals.
4. A repository-wide absolute-and-relative local-import audit reports no import
   of a missing `rice_phenology_hypernet` module. Regional modules are included
   in this check.
5. A recording backend run over a synthetic fold proves this event order:
   split, estimate requirements from training records, fit with the identical
   requirement object, calculate stage inputs/base DVR, apply modifier before
   accumulation, cross threshold, advance stage, score. The same requirement
   object must reach every stage rollout.
6. Learned-model/objective fields, function signatures, module constants, and
   documentation contain no configuration-owned concrete default. The check
   must cover all common and constrained loss fields enumerated above.
7. `ExperimentSpec.seed` is the only seed source in the simplified runner, and
   public regional artifact selection contains no fixed seed or deployment run
   identifier.
8. README contains no installation, CLI, or pytest command and no reference to
   missing configuration, test, builder, diagnostic, or sensitivity paths.
   It also contains no `supplementary` repository label, unavailable capability
   claim, or executable-reproducibility claim.
9. README's repository tree entries resolve to paths that exist after the
   edits.
10. Focused `git diff --check` succeeds for files edited by this task, excluding
   previously documented runner whitespace that disappears with the approved
   replacement.
11. The persistent pre-edit audit manifest proves: every non-allowlisted
    worktree hash is unchanged; every pre-existing index blob/status is
    unchanged; every pre-existing deletion remains absent; and only allowlisted
    paths have new worktree hashes. The runner is explicitly recorded as the
    approved whole-file replacement.

Pytest, package importability, training, and end-to-end execution must not be
claimed unless the missing runtime dependencies and tests are independently
restored and executed, which is outside this design.

## Expected Impact

The repository will present one coherent four-model scientific workflow,
without exposing concrete experiment settings or advertising unavailable
commands. The simplified runner will intentionally trade operational breadth
for a clear representation of configuration flow, training-fold isolation,
daily-DVR correction, threshold accumulation, and sequential phenology
prediction.
