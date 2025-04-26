# Compound Flooding Integration Pipeline

This **README** documents the **full, clean‑room workflow** that takes you
from only raw **GESLA 3.0** station files + `metadata.csv` → fully merged
CSV files

```
{station_code}_ERA5_with_sea_level.csv
```
which contain **hourly ERA5 meteorology, GESLA sea‑level values, and
Meteostat ground‑station precipitation**.

The process has **three executable stages** (four files total).  Two more
modules are imported as helpers but never run directly.  All other Python
files can be deleted without affecting reproduction of the merged CSVs.

---

## 0. Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python ≥ 3.9** | tested on 3.10 |
| `pandas`, `numpy`, `xarray` | core data libraries |
| `meteostat` | ground precipitation |
| `earthengine‑api`, `geemap` | ERA5 download via Google Earth Engine (GEE) |
| Google Earth Engine account & billing‑enabled project ID | needed by Stage 1 |

```bash
pip install pandas numpy xarray meteostat earthengine-api geemap

# one‑time GEE auth
earthengine authenticate
```

Directory layout expected **before** running anything:
```
compound_flooding/
└── data/
    └── GESLA/
        ├── GESLA3.0_ALL/           # raw *.txt or *.csv sea‑level files
        └── metadata.csv            # from GESLA project
```

---

## 1 ▶ Download ERA5 → `{code}_ERA5.csv`

**Script to run:** `ERA5_data_retrieval.py`

This contacts GEE, samples the ERA5‑Land hourly image collection at each
GESLA station coordinate, **merges** those values with local‑time GESLA
sea‑level observations, and writes

```
compound_flooding/GESLA_ERA5/
  240A_ERA5.csv
  876B_ERA5.csv
  ...
```

Under the hood the script imports `EfficientCompoundFloodingProcessor`
(inside *compound_flooding_processor.py*) only to obtain the list of
CONUS station locations; you do **not** run that helper separately.

Example command:
```bash
python ERA5_data_retrieval.py \
  --base-path /path/to/compound_flooding \
  --min-years 5          # skip short records (optional)
```

**Output columns** include ERA5 variables and a `sea_level` column, but
*not* ground precipitation yet.

---

## 2 ▶ Attach GESLA Metadata & Resample → `{code}_ERA5_with_sea_level.csv`

**Script to run:** `gesla_era5_integrator.py`

Reads the Stage‑1 `*_ERA5.csv` files, uses `GeslaDataset` from
`gesla.py` to ensure sea‑level data are hourly and aligned, and writes
into a new directory:

```
compound_flooding/GESLA_ERA5_with_sea_level/
  240A_ERA5_with_sea_level.csv   # ERA5 + hourly sea_level
  ...
```

Run with:
```bash
python gesla_era5_integrator.py \
  --base-path /path/to/compound_flooding
```

You may inspect a sample row – it should now have **ERA5 mets +
sea_level**, but precipitation is still missing.

---

## 3 ▶ Add Meteostat Precipitation (final step)

**Script to run:** `ground_precipitation_integrator.py`

This script:
1. Scans every `*_ERA5_with_sea_level.csv` produced in Stage 2.
2. Finds the nearest Meteostat weather stations (adaptive search box).
3. Downloads hourly `prcp` using `meteostat.Hourly(..).fetch()`.
4. Joins the `prcp` series to each CSV (writing in place).

```bash
python ground_precipitation_integrator.py \
  --base-path /path/to/compound_flooding \
  --n-processes 24     # optional: parallel workers
```

After the script completes **each file is final**:
```
station_code,datetime,t2m,sp,sea_level,prcp
240A,1993-01-01T00:00Z,276.4,101325,0.42,0.0
...
```

---

## 4. File/Module inventory

| Keep? | Python file | Reason |
|-------|-------------|--------|
| ✔ | `gesla.py` | core loader used by Stages 1&2 |
| ✔ | `ERA5_data_retrieval.py` | Stage 1 driver |
| ✔ | `gesla_era5_integrator.py` | Stage 2 driver |
| ✔ | `ground_precipitation_integrator.py` | Stage 3 driver |
| ➟ helper | `compound_flooding_processor.py` | imported by Stage 1, not executed directly |
| ✖ | `bayesian_network_analysis.py` | post‑analysis only |
| ✖ | `compound_visualizer.py` | plotting only |
| ✖ | `test.py` | demo utility |

---

## Troubleshooting

| Issue | Likely cause | Suggested fix |
|-------|--------------|---------------|
| `earthengine.Initialize` error | Not authenticated | `earthengine authenticate` |
| No `sea_level` after Stage 2 | Station code mismatch | Confirm `site_code` in `metadata.csv` |
| `prcp` all NaN | No Meteostat station nearby | Increase `MAX_SEARCH_DEG` constant |

---

# Compound Flooding Toolkit

A modular Python codebase for ingesting, preprocessing, and analyzing compound flood data from GESLA tide-gauge and ERA5 variables.

## Features so far

- **`data_io.py`**: Validate paths, load station metadata (CSV), list station codes, and load individual station CSVs (pandas/Dask).
- **`preprocess.py`**: Clean and prepare time-series data:
  - Replace sentinel `-99.9999` with `NaN`
  - Convert units:
    - `total_precipitation` from meters → mm/h
    - `surface_pressure` from Pa → hPa
    - `temperature_2m` from Kelvin → °C
  - Normalize timestamps to UTC
  - Drop duplicates
  - Optional linear detrend of `sea_level`
  - Interpolate small gaps (configurable hours)
  - Clip extreme spikes (configurable threshold)
- **`cli.py`**: Command-line interface with parallel ingest:
  - `ingest` sub-command: ingest & preprocess **all** stations in parallel using `ProcessPoolExecutor`
  - Flags for detrending, gap interpolation, spike clipping, and worker count

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Saurav-JSU/compound_flooding.git
   cd compound_flooding
   ```
2. Create and activate a Python environment (Python ≥ 3.10):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install pandas xarray
   # Optional: Dask for large CSVs
   pip install "dask[dataframe]"
   ```

## Project Structure

```
compound_flooding/             # Project root
├─ data/                       # Raw data directory
│  └─ GESLA/usa_metadata.csv   # Station metadata
├─ compound_flooding/          # Source code
│  ├─ data_io.py
│  ├─ preprocess.py
│  └─ cli.py
├─ outputs/                    # Processed outputs (generated)
│  └─ cleaned/                 # NetCDF files per station
└─ README.md
```

## Usage

### 1. Smoke-test `data_io.py`

Validate paths, list station codes, and preview a station:
```bash
python src/compound_flooding/data_io.py \
  --metadata data/GESLA/usa_metadata.csv \
  --station-dir compound_flooding/GESLA_ERA5_with_sea_level
```

Expected output:
```
Verifying paths...
  OK: paths are valid.
Listing station codes...
  Found N stations; sample: ["551A", ...]
Loading data for station: 551A...
  Loaded 5 rows; columns: ["datetime", ...]
Smoke test completed successfully.
```

### 2. Smoke-test `preprocess.py`

Clean a single station CSV:
```bash
python src/compound_flooding/preprocess.py \
  --station-csv compound_flooding/GESLA_ERA5_with_sea_level/551A_ERA5_with_sea_level.csv \
  --detrend \
  --max-gap 3 \
  --spike 5.0
```

This prints the cleaned `xarray.Dataset` and a few sample values.

### 3. Parallel ingest with CLI

Process all stations in parallel and save per-station NetCDFs:
```bash
python -m src.compound_flooding.cli ingest \
  --metadata compound_flooding/data/GESLA/usa_metadata.csv \
  --station-dir compound_flooding/GESLA_ERA5_with_sea_level \
  --output-dir outputs/cleaned \
  --detrend \
  --max-gap 3 \
  --spike 5.0 \
  --workers 96
```

This will spawn up to 96 workers to speed up ingest and preprocessing of ~2145 stations.

## Next Steps

- **Tier 1 statistics**: Implement univariate POT/GPD fits, empirical joint exceedance, CPR calculations.
- **Tier 2 copulas**: Bivariate copula modeling (Gumbel, t, etc.) and diagnostics.
- **Tier 3 advanced**: Vine copulas, Bayesian networks, and ML approaches.

Feel free to explore and extend! Contributions are welcome via pull requests.

