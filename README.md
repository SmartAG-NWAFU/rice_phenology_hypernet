# Rice Phenology Hypernet

## Background

Accurate prediction of rice phenological stages supports crop management,
cultivar evaluation, and assessment of climate-related production risks.
Process-based phenology models represent development as daily weather-driven
increments that accumulate until stage-specific requirements are reached.
Fixed response functions and calibrated requirements, however, may not fully
capture development across sites and years. Small daily development-rate (DVR)
errors can consequently shift threshold-crossing dates and propagate through
later stages.

## Objectives

This project examines whether machine learning can improve cross-environment
rice phenology prediction by correcting daily DVR within the accumulated-
development framework. It compares process baselines and hybrid models under
sample-level, site-extrapolation, and year-extrapolation evaluation tasks while
preserving a common definition of stage completion.

## Methodological Approach

The method first calculates a process-derived daily DVR from temperature and,
where relevant, photoperiod. The hybrid models learn positive daily modifiers
from weather sequences and stage context. Each modifier is applied before
daily rates are accumulated. Stage completion is then determined by the first
requirement crossing, and the following stage begins on the next day. Thus,
learning changes the daily development input without replacing the process
model's state update, threshold crossing, or sequential rollout.

## Significance

Correcting DVR at the daily process entry point provides a transparent bridge
between process knowledge and sequence learning. The shared rollout makes it
possible to attribute differences among models to their daily development
formulation rather than to different post-processing rules for predicted stage
dates. This structure also supports analysis of how prediction skill transfers
across environmental settings.

## Model Framework

The retained workflow covers the four models reported in the study:

- `m0_t`: a temperature-only process baseline;
- `m0_dvr`: a photothermal process baseline;
- `m1_v2_dvr`: a recurrent model that learns positive daily DVR modifiers;
- `m1_dvr_con`: a constrained recurrent modifier model with stage-dependent
  background-information gates.

All four models share five sequential prediction stages: tillering, jointing,
booting, heading, and maturity. The M0 process-response constants remain part
of the scientific model definition. Learned-model architecture settings,
regularization settings, loss weights, seeds, and artifact identifiers are
supplied by external experiment configuration rather than duplicated as
public defaults.

## Workflow

The core experiment structure is:

1. split records according to the selected evaluation task;
2. estimate stage requirements from training records only;
3. fit the selected learned model when applicable;
4. construct daily inputs for the active stage;
5. calculate the process-derived base DVR and optional learned modifier;
6. correct and accumulate daily DVR until the stage requirement is crossed;
7. advance sequentially through all five stages; and
8. summarize stage predictions with MAE, RMSE, bias, and R-squared metrics.

The experiment runner owns correction, accumulation, threshold crossing, and
stage advancement. Project-specific splitting, feature construction, fitting,
and scoring are expressed through an injected backend so that the scientific
order remains explicit.

## Repository Layout

```text
src/rice_phenology_hypernet/      process models, hybrid models, objectives,
                                  experiment contracts, and regional analysis
scripts/china_rice_calendar/      regional rice-calendar preparation helpers
scripts/meteo_download/           regional weather download and standardization
data/                             placeholders for private data products
artifacts/                        placeholders for generated outputs
docs/superpowers/                 design, implementation, and audit records
```

## Configuration and Parameter Provenance

Experiment-controlled values are represented by required configuration
protocols and dataclasses. The learned models require configuration objects,
and the common DVR objective requires a loss-configuration object. The runner
passes the same training-derived requirement object through model fitting and
every stage calculation within a fold. Regional artifact selection likewise
requires a `RegionalProjectionSpec` and a `RegionalModelProvider`; no seed or
deployment identifier is embedded in the public projection interface.

Structural dimensions used to explain tensor interfaces remain visible in the
model definitions. M0 thermal and photoperiod response constants also remain
visible because they define the process model rather than an experiment run.

## Expected Data Interfaces

Station weather is supplied through an explicit path and contains `SID`,
`Date`, and `TemAver`; the retained preparation logic also recognizes `year`,
`TemMin`, `TemMax`, `Precipitation`, and `Radiation`. Dates must be parseable,
and numeric weather fields are normalized before site-year matching.

Station phenology is supplied through a separate explicit path. Required
record metadata include station identifier, year, latitude, longitude, and a
reviving date. Stage dates from reviving through maturity must follow a valid
chronological order. `RawDataPaths` identifies the two private inputs, while
`PreparedDataPaths` identifies all prepared outputs. Loading prepared data does
not silently trigger preprocessing.

Regional projection uses point-year inputs linked to daily weather shards.
Each point includes location, transplanting and reviving DOY, and remote-
sensing heading and maturity references. Preparation helpers retain the
documented regional periods and their associated year ranges.

## Regional Analysis Scope

Regional projection applies the same ordered set of four paper models and
produces yearly stage predictions. The analysis module aggregates these values
to period-specific multi-year climatology and calculates heading and maturity
MAE, RMSE, bias, R-squared, and sample counts against remote-sensing reference
dates. This regional comparison is a bounded plausibility analysis; it does
not by itself establish independent grid-cell deployment validation.

## Repository Scope

The repository contains the scientific model definitions, configuration-
derived interfaces, four-model orchestration, data contracts, regional input
helpers, and numeric analysis logic. Private station data, experiment-specific
configuration values, and trained artifacts are intentionally outside the
repository. The documentation therefore focuses on method structure and data
provenance.
