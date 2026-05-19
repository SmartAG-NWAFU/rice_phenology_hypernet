# Meteo Download Scripts

This directory contains historical-weather helpers that call a local
`meteo-data-download` service backed by Google Earth Engine / ERA5-Land.

These scripts are optional convenience tools for rebuilding private inputs.
They do not download or process scenario weather data.

## Station Historical Weather

Use `download_site_wether.py` to build the default station weather input:

```bash
python3 scripts/meteo_download/download_site_wether.py \
  --catalog-path data/raw/obser_pheno_catalog_origin.xlsx \
  --output-path data/raw/daily_temperature.csv
```

Important defaults:

```text
catalog path:          data/raw/obser_pheno_catalog_origin.xlsx
output path:           data/raw/daily_temperature.csv
scratch output root:   data/processed/site_weather_gee_era5_1981_2020_raw
API base URL:          http://127.0.0.1:8000
```

## Regional Historical Weather

Use `download_regional_grid_weather_gee.py` for daily regional grid weather:

```bash
python3 scripts/meteo_download/download_regional_grid_weather_gee.py
```

Standardize the sharded output:

```bash
python3 scripts/meteo_download/standardize_regional_grid_weather_gee.py
```

The standardizer writes daily clean shards and point-year summaries under
`data/processed/regional_grid_weather_gee_era5_2003_2022_clean/`.

## Requirements

- Docker container `meteo-data-download` must be running.
- Google Earth Engine authentication must be configured for the local service.
- The container output mount should expose host `data/processed` as
  `/app/outputs`.

Health check:

```bash
curl -s http://127.0.0.1:8000/health
```

Expected shape:

```json
{"ok": true, "service": "meteo_data_download"}
```
