# Config-Derived Learned-Model Parameters Design

## Goal

Make the public supplementary model scripts unambiguous about parameter
provenance: learned-model and loss hyperparameters come from the experiment
configuration, while structural dimensions and fixed process-model constants
remain visible in code.

The files are reading supplements. They should be internally coherent and
syntactically valid, but the current public working tree is not expected to be
runnable because its configuration loader, common loss module, tests, and YAML
configuration files are absent or deleted.

## Scope

Primary files:

- `src/rice_phenology_hypernet/models/m1_v2_dvr.py`
- `src/rice_phenology_hypernet/models/m1_dvr_con.py`
- the directly affected configuration factories and constrained-loss call sites
  in `src/rice_phenology_hypernet/experiments/runner_dvr.py`

Out of scope:

- fixed process-model constants in `m0.py` and `physics.py`;
- structural constants such as `input_dim`, `state_dim`, the five ordered
  stages, and the single GRU layer;
- operational defaults for seeds, worker counts, regional chunking, or figure
  presentation;
- restoration of deleted configuration, loss, or test files; and
- changes to model equations, loss formulas, training flow, or experiment
  behavior.

## Model Configuration Objects

Retain `M1V2DvrConfig` and `M1ConDvrConfig` as compact model-construction
objects because deployment artifacts and the runner already use them.

For `M1V2DvrConfig`:

- `hidden_size`, `dropout`, `modifier_cap`, and `event_beta` become required
  fields without numeric defaults;
- `input_dim = 5` remains an explicit structural default and is placed after
  all required fields to satisfy dataclass ordering.

For `M1ConDvrConfig`:

- `hidden_size`, `dropout`, `modifier_cap`, `event_beta`, and
  `background_gate_prior` become required fields without defaults;
- `input_dim = 5` and `state_dim = 2` remain structural defaults and are placed
  after all required fields.

Both model constructors will require a configuration object. They will no
longer accept `None` or instantiate a default configuration internally. The
existing runner factories remain the visible provenance link: they read the
project experiment configuration and construct the model configuration objects.

## Loss Configuration

`compute_m1_dvr_con_loss` will replace its individual configuration-owned
keyword arguments with one `config` object. A small structural typing contract
may document the required fields without assigning values. The function will
read:

- event, terminal, shrinkage, smoothness, and mean-anchor weights;
- stage anchor, terminal, and shrinkage multipliers;
- gate-prior and gate-monotonic weights; and
- the numerical-stability epsilon

through `config.<field>` references. `stage_index` remains a separate argument
because it is batch data rather than a hyperparameter. The loss equations and
statistics remain unchanged.

The two constrained-loss call sites in `runner_dvr.py` will pass the existing
experiment configuration object once instead of expanding it into individual
weight arguments.

`m1_v2_dvr.py` does not define its own loss function. Its stale import and
re-export of the currently deleted common `dvr_loss.py` helper will be removed;
the experiment runner remains responsible for applying configuration-derived
loss settings.

## Reading Clarity

Short comments or docstrings will state that tunable values are populated from
the experiment configuration. No concrete learned-model or loss hyperparameter
values will appear in the two model scripts. Parameter names remain visible so
readers can understand the model and loss components without mistaking example
numbers for independent defaults.

## Verification

Because this task targets reading-only supplementary code in an already
incomplete working tree, verification will be structural:

- scan both model scripts for forbidden numeric defaults attached to
  configuration-owned fields;
- inspect the focused diffs for model/loss equation changes;
- parse all existing Python files with `ast.parse`;
- run `git diff --check` where pre-existing whitespace permits; and
- compare Git status before and after to preserve unrelated deletions and the
  user's large pre-existing `runner_dvr.py` changes.

No full execution or pytest claim will be made while the configuration loader,
common loss module, and tests remain deleted.
