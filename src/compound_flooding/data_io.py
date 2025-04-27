# src/compound_flooding/data_io.py
"""
Module: data_io.py
Responsibilities:
- Validate metadata path and station directory
- Load station metadata CSV into pandas DataFrame
- List available station codes
- Load station CSV (flexible pattern matching)
"""
import os
import glob
import pandas as pd


def validate_paths(metadata_path: str, station_dir: str) -> None:
    """
    Ensure metadata CSV and station directory both exist.
    """
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not os.path.isdir(station_dir):
        raise NotADirectoryError(f"Station directory not found: {station_dir}")
    print("Verifying paths...")
    print(f"  Metadata: {metadata_path}")
    print(f"  Station dir: {station_dir}")
    print("  OK: paths are valid.")


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Load station metadata CSV. Returns DataFrame with 'SITE CODE' as str.
    """
    df = pd.read_csv(metadata_path)
    if 'SITE CODE' not in df.columns:
        raise KeyError("Metadata CSV missing 'SITE CODE' column")
    df['SITE CODE'] = df['SITE CODE'].astype(str)
    print(f"Loaded metadata: {len(df)} stations")
    return df


def list_station_codes(meta: pd.DataFrame) -> list[str]:
    """
    Return list of station codes from metadata DataFrame.
    """
    codes = meta['SITE CODE'].tolist()
    print(f"Listing station codes... Found {len(codes)} codes; sample: {codes[:5]}")
    return codes


def load_station_data(station_code: str, station_dir: str) -> pd.DataFrame:
    """
    Load station CSV by code, matching any filename containing the code.

    Example: code='240A' matches '240A_ERA5_with_sea_level.csv'.
    """
    pattern = os.path.join(station_dir, f"*{station_code}*_ERA5_with_sea_level.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No station CSV found for code '{station_code}' in {station_dir}")
    filepath = matches[0]
    df = pd.read_csv(filepath, parse_dates=['datetime'])
    print(f"Loading data for station: {station_code} → {os.path.basename(filepath)}")
    print(f"  Loaded {len(df)} rows; columns: {list(df.columns)}")
    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Smoke-test data_io module')
    parser.add_argument('--metadata', required=True, help='Path to metadata CSV')
    parser.add_argument('--station-dir', required=True, help='Directory of station CSVs')
    parser.add_argument('--sample-code', default=None, help='One station code to test')
    args = parser.parse_args()

    # Validate
    validate_paths(args.metadata, args.station_dir)
    # Load metadata and list codes
    meta = load_metadata(args.metadata)
    codes = list_station_codes(meta)
    # Optionally load one station
    code = args.sample_code or codes[0]
    df = load_station_data(code, args.station_dir)
    print("data_io smoke test completed successfully.")
