# Data Directory

This directory is intentionally empty in the public repository.

Expected private inputs:

- `raw/daily_temperature.csv`
- `raw/obser_pheno_catalog_origin.xlsx`
- `boundary/china.json`
- `boundary/provinces.json`

Generated processed files are written under `processed/` and are ignored by
git. Regional helper scripts may also write intermediate feature products under
`data/artifacts/`; those files are ignored.
