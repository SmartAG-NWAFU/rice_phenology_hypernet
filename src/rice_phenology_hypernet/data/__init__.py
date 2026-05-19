from .io import (
    PHENOLOGY_STAGE_COLUMNS,
    load_clean_data,
    load_raw_phenology,
    load_raw_weather,
    prepare_data_assets,
)


def __getattr__(name: str):
    if name == "RiceSampleDataset":
        from .dataset import RiceSampleDataset

        return RiceSampleDataset
    if name == "RiceDvrStageDataset":
        from .dataset_dvr import RiceDvrStageDataset

        return RiceDvrStageDataset
    if name == "RiceDvrSeasonDataset":
        from .dataset_dvr import RiceDvrSeasonDataset

        return RiceDvrSeasonDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "PHENOLOGY_STAGE_COLUMNS",
    "RiceDvrSeasonDataset",
    "RiceDvrStageDataset",
    "RiceSampleDataset",
    "load_clean_data",
    "load_raw_phenology",
    "load_raw_weather",
    "prepare_data_assets",
]
