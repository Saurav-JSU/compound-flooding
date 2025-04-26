# src/compound_flooding/data_io.py
"""
Module: data_io.py
Responsibilities:
- Validate input paths
- Load station metadata and individual station time-series CSVs
- Offer switchable backend (pandas/Dask) for scalability
- Smoke-test CLI to verify functionality against real data
"""
from pathlib import Path
import pandas as pd

# Try Dask for large-data support
try:
    import dask.dataframe as dd
    _USE_DASK = True
except ImportError:
    _USE_DASK = False


def validate_paths(metadata_path: str, station_dir: str) -> None:
    """
    Ensure metadata CSV and station directory exist.
    Raises FileNotFoundError or NotADirectoryError on issues.
    """
    meta = Path(metadata_path)
    st_dir = Path(station_dir)
    if not meta.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not st_dir.is_dir():
        raise NotADirectoryError(f"Station directory not found: {station_dir}")


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Read station master metadata into a pandas DataFrame.
    Expects a 'SITE CODE' column.
    """
    df = pd.read_csv(metadata_path)
    if 'SITE CODE' not in df.columns:
        raise KeyError("Metadata CSV must include 'SITE CODE' column")
    return df


def list_station_codes(metadata_path: str) -> list[str]:
    """
    Return list of station codes from metadata.
    """
    df = load_metadata(metadata_path)
    return df['SITE CODE'].astype(str).tolist()


def load_station_data(station_code: str, station_dir: str):
    """
    Load merged GESLA+ERA5 CSV for a given station code.
    Uses Dask for large files if available.

    Returns
    -------
    pandas.DataFrame or dask.DataFrame
    """
    filename = f"{station_code}_ERA5_with_sea_level.csv"
    filepath = Path(station_dir) / filename
    if not filepath.is_file():
        raise FileNotFoundError(f"Station file not found: {filepath}")
    parse_dates = ['datetime']
    if _USE_DASK:
        return dd.read_csv(str(filepath), parse_dates=parse_dates)
    return pd.read_csv(filepath, parse_dates=parse_dates)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Smoke-test data_io module with real metadata and station files"
    )
    parser.add_argument(
        '--metadata', required=True,
        help='Path to data/GESLA/usa_metadata.csv'
    )
    parser.add_argument(
        '--station-dir', required=True,
        help='Path to GESLA_ERA5_with_sea_level directory'
    )
    args = parser.parse_args()

    print("Verifying paths...")
    try:
        validate_paths(args.metadata, args.station_dir)
        print("  OK: paths are valid.")
    except Exception as e:
        print(f"  ERROR: {e}")
        exit(1)

    print("Listing station codes...")
    codes = list_station_codes(args.metadata)
    print(f"  Found {len(codes)} station codes; sample: {codes[:5]}")

    if codes:
        sample = codes[0]
        print(f"Loading data for station: {sample}...")
        try:
            df = load_station_data(sample, args.station_dir)
            # If Dask, compute a small sample
            if _USE_DASK:
                df = df.head(5)
            else:
                df = df.iloc[:5]
            print(f"  Loaded {len(df)} rows; columns: {list(df.columns)}")
        except Exception as e:
            print(f"  ERROR loading station {sample}: {e}")
            exit(1)

    print("data_io smoke test completed successfully.")
