# Rice Phenology Hypernet

## Background

Accurate prediction of crop phenological stages supports crop management,
cultivar evaluation, and assessment of climate-related production risks.
Process-based phenology models represent development as daily weather-driven
increments that accumulate until stage-specific requirements are reached.
Fixed response functions and calibrated requirements, however, may not fully
capture development across sites and years. Small daily development-rate (DVR)
errors can consequently shift threshold-crossing dates and propagate through
later stages.

## Objectives

This project examines whether machine learning can improve cross-environment
crop phenology prediction by correcting daily DVR within the accumulated-
development framework. It compares process baselines and hybrid models under
random site-year, unseen-site, and later-year evaluation tasks while
preserving a common definition of stage completion.

## Methodological Approach

The method first calculates a process-derived daily DVR from temperature and,
where relevant, photoperiod. The hybrid models learn positive daily modifiers
from weather sequences and stage context. Each modifier is applied before
daily rates are accumulated. Stage completion is then determined by the first
requirement crossing, and the following stage begins on the next day. Thus,
learning changes the daily development input without replacing the process
model's state update, threshold crossing, or sequential rollout.

## Repository Layout

```text
src/rice_phenology_hypernet/      process models, hybrid models, objectives,
                                  experiment contracts, and regional analysis
scripts/china_rice_calendar/      regional rice-calendar preparation helpers
scripts/meteo_download/           regional weather download and standardization
data/                             placeholders for private data products
artifacts/                        placeholders for generated outputs
```

## Package Architecture

| Area             | Responsibility                                                                                          |
| ---------------- | ------------------------------------------------------------------------------------------------------- |
| `models/`      | Process models, learned DVR modifiers, physical response functions, and configuration-driven objectives |
| `experiments/` | Shared contracts, four-model orchestration, summaries, threshold utilities, and regional workflows      |
| `data/`        | Explicit raw/prepared data loading and daylength calculation                                            |
| `evaluation/`  | Stage-level and aggregated prediction metrics                                                           |
| `types.py`     | Shared path and result dataclasses                                                                      |
| `settings.py`  | Repository-relative directory definitions                                                               |
| `runtime.py`   | Run directories, manifests, metadata, and output discovery                                              |

## Core Interfaces

- `RawDataPaths` and `PreparedDataPaths` make input and output ownership
  explicit.
- `DvrExperimentConfig` supplies records and model settings without embedding
  experiment values in model definitions.
- `DvrWorkflowBackend` separates data splitting, feature construction,
  training, prediction records, and scoring from the common rollout logic.
- `StageInputs` and `StageRolloutResult` define the boundary around daily DVR
  correction and sequential stage advancement.
- `RegionalProjectionSpec` and `RegionalModelProvider` separate regional
  projection from model-artifact loading.

## Data and Execution Flow

1. Callers provide explicit data paths and experiment configuration.
2. Data modules standardize station tables and load prepared assets.
3. The experiment backend creates folds, estimates training-only requirements,
   and prepares model inputs.
4. The runner constructs one of the four models and applies the shared stage
   rollout in a fixed order.
5. Predictions, metrics, audit metadata, and optional regional summaries are
   returned through typed result objects or written to run-specific paths.

## Supporting Scripts

| Directory                        | Purpose                                                             |
| -------------------------------- | ------------------------------------------------------------------- |
| `scripts/china_rice_calendar/` | Download, extract, coarsen, and inspect regional rice-calendar data |
| `scripts/meteo_download/`      | Download and standardize regional gridded weather data              |

These scripts prepare external inputs. Model training and experiment settings
remain outside the helper scripts.

## Extension Points

- Implement `DvrWorkflowBackend` to connect a different split, feature, or
  training pipeline to the shared runner.
- Extend model construction explicitly when adding a model; the public model
  registry remains limited to the four retained model names.
- Implement `RegionalModelProvider` to load trained artifacts without coupling
  projection code to one storage layout.
- Add data adapters around the explicit path contracts instead of introducing
  hidden global paths.
