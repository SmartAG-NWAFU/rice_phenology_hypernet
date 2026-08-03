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
6. apply the corrected daily DVR before accumulation;
7. identify stage completion by threshold crossing;
8. roll predicted stages forward sequentially; and
9. return prediction and evaluation records.

### Public types and interfaces

Use small dataclasses and `Protocol` contracts rather than imports from deleted
data, split, CLI, or configuration modules. At minimum, the runner should
define:

- the exact `PAPER_MODEL_NAMES` tuple;
- an experiment specification containing task, model name, and required seed;
- a configuration protocol exposing the two learned-model configuration
  sections without concrete values;
- a workflow-backend protocol for split construction, training-fold
  requirement estimation, learned-model fitting, sequential rollout, and
  scoring; and
- a compact experiment result containing predictions, metrics, and audit
  metadata.

The public experiment function must require the configuration, backend, and
seed explicitly. It must validate the four-model name set and preserve the
training-fold-only requirement-estimation boundary.

### Model construction

- `m0_t` constructs `M0TPhenologyModel`.
- `m0_dvr` constructs `M0PhenologyModel`.
- `m1_v2_dvr` constructs `M1V2DvrModel` from a required
  `M1V2DvrConfig` populated from the injected configuration.
- `m1_dvr_con` constructs `M1ConDvrModel` from a required
  `M1ConDvrConfig` populated from the injected configuration.

The runner may retain structural input/state dimensions. It must not copy any
learned-model or loss value.

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

- define a protocol containing the common DVR loss settings with no defaults;
- implement the common event, terminal, shrinkage, smoothness, and optional
  stage-aware terms;
- accept one required configuration object rather than a long set of scalar
  hyperparameters;
- allow only `stage_index` to retain a runtime `None` default; and
- expose no concrete learned-model, stage-weight, or epsilon values.

`m1_dvr_con.py` must use this shared objective and extend it with gate-prior
and gate-monotonic regularization from its required configuration object.
`m1_v2_dvr.py` remains a model-definition module and does not re-export the
objective.

## Package and Utility Consistency

### Data package

Remove lazy exports for deleted dataset classes from `data/__init__.py`.
Refactor `data/io.py` so raw input paths are explicit required inputs or come
from a caller-supplied path object; do not import the deleted global project
configuration loader. Export only existing, internally consistent helpers.

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

### Regional analysis

Do not advertise or export deleted general figure-builder infrastructure.
Keep the regional analysis algorithms and their English documentation. If
figure-building code cannot be separated safely from the deleted builder, the
analysis interface should focus on climatology and metrics rather than claim a
currently available general figure workflow.

### Runtime and settings

Preserve path construction, run metadata, and artifact-directory behavior.
`ProjectSettings` and ordinary runtime-control defaults are not experiment
parameter disclosures.

## English Audit

Scan every existing Markdown, Python, shell, TOML, YAML, and text file for Han
characters. Translate or generalize the two historical Chinese examples in
`findings.md`. Do not translate identifiers, column names, file names, or data
values unless they are natural-language documentation.

All new source comments, docstrings, specification text, and README prose must
be English.

## README Design

Preserve and lightly edit the existing:

- Background;
- Objectives;
- Methodological Approach; and
- Significance.

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

## Verification Contract

The implementation is complete only when all of the following checks pass:

1. A repository-wide Han-character scan returns no matches in existing text
   and source files.
2. Every existing Python file under `scripts/` and `src/` parses with
   `ast.parse`.
3. The simplified runner contains exactly the four paper model identifiers and
   no exploratory-model imports, branches, or string literals.
4. The runner imports only existing local modules.
5. Learned-model and objective dataclasses/protocols contain no
   configuration-owned concrete defaults.
6. Public experiment-selection interfaces contain no fixed seed or deployment
   run identifier.
7. README contains no installation, CLI, or pytest command and no reference to
   missing configuration, test, builder, diagnostic, or sensitivity paths.
8. README's repository tree entries resolve to paths that exist after the
   edits.
9. Focused `git diff --check` succeeds for files edited by this task, excluding
   previously documented runner whitespace that disappears with the approved
   replacement.
10. Final Git status compared with
    `/tmp/rice-public-status.before-project-audit.Ob9Ymu` shows only approved
    source/documentation changes and preserves all pre-existing deletions.

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
