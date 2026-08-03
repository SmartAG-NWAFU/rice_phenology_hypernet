# README Project Overview Design

## Goal

Add a concise English introduction near the beginning of the root `README.md`
so that readers can understand the project's research background, objectives,
methodological approach, and significance before encountering the workflow and
reproduction instructions.

## Placement and Structure

Insert four short sections immediately after the repository title and before
the existing workflow list:

1. `Background`
2. `Objectives`
3. `Methodological Approach`
4. `Significance`

Keep each section to one compact paragraph. Retain the existing technical
workflow, installation, input, command, and testing documentation. Convert the
current lowercase `workflow` label into a Markdown heading and remove its empty
trailing list item.

## Content

The introduction will:

- explain why accurate rice phenology prediction matters and why fixed process
  responses can be difficult to transfer across sites and years;
- state that the project evaluates process-preserving correction of daily
  development rate (DVR) before accumulated development and threshold crossing;
- summarize the temperature and photothermal process baselines, learned daily
  DVR modifiers, shared accumulated-development rules, sequential rollout, and
  sample/site/year extrapolation protocols;
- describe the regional grid workflow as a plausibility check rather than an
  independent validation exercise; and
- explain the value of integrating machine learning at a process-relevant entry
  point while preserving interpretable phenology-model structure.

## Claim Boundaries

The new text will not introduce performance statistics, physiological claims,
or assertions of operational regional validation that are not established by
the public repository. Internal model identifiers may remain in the workflow
list, while the introductory prose will emphasize the scientific model classes
and shared DVR mechanism.

## Verification

After editing `README.md`:

- inspect the focused diff for `README.md`;
- run `git diff --check -- README.md`; and
- confirm that no unrelated working-tree changes were modified.
