# Data and frozen basin split

The study uses CAMELS-US Daymet time series, version 1.2, and catchment
attributes, version 2.0. Obtain the files from the official distribution and
accept its data-use terms:

- [Time-series description and download](https://ncar.github.io/hydrology/datasets/CAMELS_timeseries)
- [Catchment attributes description and download](https://ncar.github.io/hydrology/datasets/CAMELS_attributes)
- [Time-series DOI: 10.5065/D6MW2F4D](https://doi.org/10.5065/D6MW2F4D)
- [Attributes DOI: 10.5065/D6G73C3Q](https://doi.org/10.5065/D6G73C3Q)

The code's MIT license does not replace the data providers' terms. The small
attribute tables in `splits/` are CAMELS-derived research inputs; attribute
definitions and underlying data remain attributed to CAMELS-US. No raw daily
meteorological or streamflow observations are included.

Pass the following directories to `scripts/prepare_data.py`:

```text
basin_mean_forcing/daymet/<HUC2>/<basin_id>_lump_cida_forcing_leap.txt
usgs_streamflow/<HUC2>/<basin_id>_streamflow_qc.txt
camels_attributes_v2.0/camels_topo.txt
camels_attributes_v2.0/camels_soil.txt
camels_attributes_v2.0/camels_vege.txt
camels_attributes_v2.0/camels_geol.txt
```

The first two root arguments are searched recursively, so an unzipped archive
may contain intermediate directories. The attribute root must directly contain
the four attribute tables. Duplicate files for the same basin are rejected.
No user-specific Windows path is required.

## Frozen membership

`splits/formal_pub_527/train.txt`, `val.txt`, and `test.txt` contain the exact
370/50/107 basin assignment. Preserve leading zeros. The split is an input to
reproduction, not a new random draw. Its manifest contains file checksums.
`reference_531.txt`, the four exclusions, and the period-restricted basin list
document how the 527-basin pool was retained. Overlapping catchments assigned
together in the study remain together in these lists.

## Preprocessing

Five daily inputs are precipitation, shortwave radiation, maximum and minimum
temperature, and vapour pressure. Static inputs contain four climate summaries,
16 land-surface attributes, and one missingness indicator. Climate summaries
are recomputed for 1 October 1980–30 September 1989. The original misspelling
`geol_porostiy` is mapped to `geol_porosity`.

Scaling and imputation use the 370 training basins only. Dynamic scaling uses
their training-period forcing. Streamflow is converted from ft³/s to mm/day
using the area in the forcing-file header. Area as a model attribute receives
`log1p`; geological permeability is already logarithmic and is not transformed
again. Forest cover is a fraction, whereas sand, silt, and clay are percentages.

The builder reads training and validation streamflow in their specified label
periods. The training cache contains no test streamflow. The original basin
quality screening did examine streamflow availability; this is distinct from
using test labels to fit model parameters or select checkpoints.

`data_contract/source_hashes.json` records the study's raw-file fingerprints.
The builder checks these and verifies recomputed attributes against the frozen
table. A mismatch identifies a different data version or modified input and
requires investigation rather than silently changing the experiment.

## Data citations

Newman, A.J., et al. (2015). Development of a large-sample watershed-scale
hydrometeorological data set for the contiguous USA: data set characteristics
and assessment of regional variability in hydrologic model performance.
*Hydrology and Earth System Sciences*, 19, 209–223.
https://doi.org/10.5194/hess-19-209-2015

Addor, N., Newman, A.J., Mizukami, N., Clark, M.P. (2017). The CAMELS data set:
catchment attributes and meteorology for large-sample studies.
*Hydrology and Earth System Sciences*, 21, 5293–5313.
https://doi.org/10.5194/hess-21-5293-2017

