# Compound Flooding Analysis Framework

A modular, research-grade codebase for analyzing compound coastal flood events, following the methodology outlined in the Compound_Flooding.pdf document. This project implements a three-tier approach for analyzing sea level and precipitation extremes to identify and characterize compound flooding events.

## Project Structure

```
compound_flooding/
├── pyproject.toml           # Project metadata and dependencies
├── requirements.txt         # Pinned dependencies
├── README.md                # This file
├── src/
│   └── compound_flooding/   # Main package
│       ├── __init__.py      
│       ├── data_io.py       # Data I/O utilities
│       ├── preprocess.py    # Preprocessing and QC
│       ├── tier1_stats.py   # Tier 1 statistical analysis
│       ├── tier2_copula.py  # Tier 2 copula modeling
│       ├── tier3_vine_bn.py # Tier 3 advanced modeling
│       ├── viz.py           # Visualization utilities
│       └── cli.py           # Command-line interface
└── tests/                   # Test suite
    ├── __init__.py
    ├── test_data_io.py
    ├── test_preprocess.py
    └── test_cli.py
```

## Data Directory Structure

The codebase expects the following data directory structure:

```
compound_flooding/GESLA_ERA5_with_sea_level/
└── {STATION}_ERA5_with_sea_level.csv   # One file per station

data/GESLA/
└── usa_metadata.csv                    # Station metadata
```

## Installation

### Development Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/compound_flooding.git
cd compound_flooding

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

### With GPU Support

```bash
pip install -e ".[gpu]"
```

## Usage

### Command-line Interface

```bash
# Process all stations and produce cleaned NetCDF files
python cli.py ingest --detrend-sea-level

# Process specific stations
python cli.py ingest --station-codes 240A 241A --detrend-sea-level

# Run with GPU acceleration
python cli.py ingest --gpu --use-dask --max-workers 8

# Run Tier 1 analysis (planned)
# python cli.py tier1

# Generate plots (planned)
# python cli.py plot
```

## Methodology

The compound flooding analysis follows a three-tier approach:

1. **Tier 1: Baseline Statistical Analysis**
   - Univariate extreme value analysis
   - Empirical joint exceedance analysis
   - Threshold selection
   - Lead/lag analysis

2. **Tier 2: Dependence Modeling**
   - Bivariate copula modeling
   - Conditional exceedance analysis
   - Joint return period estimation

3. **Tier 3: Advanced Multivariate & Causal Models**
   - Bayesian Networks
   - Vine copulas
   - Machine learning approaches
   - Non-stationarity and climate change

## Testing

Run the test suite using pytest:

```bash
pytest
```

## License

MIT License