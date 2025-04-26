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

# Summary of Preprocessed Data for Tier 1 Analysis

## 1. Dataset Overview

After quality control and correction processes, our final dataset consists of:

| Quality Tier | Count | Description | Usage |
|--------------|-------|-------------|-------|
| Tier A | 199 | Excellent quality stations | Use original data |
| Tier B (improved) | ~17 | Successfully corrected stations | Use corrected data |
| Total Usable | ~206 | Combined high-quality stations | Primary dataset for Tier 1 |
| Tier C/D | ~116 | Low quality or uncorrectable | Exclude from analysis |

The usable dataset provides comprehensive coverage across U.S. coastal regions with approximately 206 stations. The full list of usable station codes is available in `data/processed/usable_stations_for_tier1.csv`.

## 2. Key Characteristics

### 2.1 Temporal Coverage
- **Typical Period**: 1981-2021 (~40 years)
- **Temporal Resolution**: Hourly
- **Average Record Length**: 34.93 years per station

### 2.2 Sea Level Data
- **Missing Data**: <5% for Tier A stations, 5-10% for corrected Tier B stations
- **Typical 99th Percentile**: ~2.92 meters (varies by region)
- **Null Value Indicator**: -99.9999

### 2.3 Precipitation Data
- **Units**: Millimeters (mm), standardized from multiple sources
- **Typical 99th Percentile**: ~21.47 mm
- **Source Priority**: Ground data where available, ERA5 as backup

### 2.4 Compound Events
- **Joint Exceedances**: Average of ~1500 per station using 95th percentile thresholds
- **Conditional Probability Ratio**: Average of 2.32 (indicating significant dependence)

## 3. Data Corrections Applied

| Correction Type | Description | Affected Stations |
|-----------------|-------------|-------------------|
| Sea Level Outliers | Unit conversion or capping of physically implausible values | ~10 stations |
| Missing Data Fill | Linear interpolation of short gaps (1-2 hours) | ~12 stations |
| Precipitation Standardization | Consolidated ground and ERA5 data with unit standardization | All stations |

## 4. File Structure for Tier 1 Analysis

### 4.1 Input Files
- **Original Data**: `compound_flooding/GESLA_ERA5_with_sea_level/{SITE_CODE}_ERA5_with_sea_level.csv`
- **Corrected Data**: `data/processed/corrected/{SITE_CODE}_corrected.csv`
- **Station List**: `data/processed/usable_stations_for_tier1.csv`
- **Metadata**: `compound_flooding/GESLA/usa_metadata.csv`

### 4.2 Key Columns
- `datetime`: Hourly timestamps (UTC)
- `sea_level`: Water level measurements in meters
- `precipitation_mm`: Consolidated precipitation in millimeters
- `total_precipitation_mm`: ERA5 precipitation in millimeters
- `ground_precipitation`: Ground-based precipitation (varies in units)
- Additional meteorological variables: wind components, pressure, temperature

## 5. Recommendations for Tier 1 Analysis

1. **Data Loading**:
   - Check the quality tier of each station before loading
   - Load from corrected files for improved Tier B stations
   - Verify null values are properly handled (-99.9999)

2. **Threshold Selection**:
   - Use the 95th or 99th percentiles as starting points for EVA thresholds
   - Consider regional differences in threshold selection
   - Validate thresholds with mean residual life plots as specified in methodology

3. **Declustering**:
   - Implement 48-hour separation between extreme events as specified in methodology
   - Handle autocorrelation in hourly data to ensure independent peak events

4. **Joint Exceedance Analysis**:
   - Use the consolidated precipitation_mm column for joint analysis
   - Verify CPR calculations with appropriate null hypothesis testing
   - Explore temporal offsets between peaks (±6h, ±12h, ±24h)

By using this preprocessed dataset, the Tier 1 extreme value analysis and joint exceedance analysis can proceed with high-quality, standardized data, ensuring robust results for the compound flooding study.

© 2025 Saurav Bhattarai – Jackson State University Water Lab

