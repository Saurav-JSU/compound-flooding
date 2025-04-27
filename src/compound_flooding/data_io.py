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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_paths(metadata_path: str, station_dir: str) -> bool:
    """
    Ensure metadata CSV and station directory both exist.
    
    Parameters
    ----------
    metadata_path : str
        Path to the metadata CSV file
    station_dir : str
        Directory containing station CSV files
        
    Returns
    -------
    bool
        True if both paths are valid, raises appropriate exception otherwise
    
    Raises
    ------
    FileNotFoundError
        If metadata file doesn't exist
    NotADirectoryError
        If station directory doesn't exist
    PermissionError
        If paths exist but aren't accessible
    """
    # Check if metadata file exists
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Check if metadata file is readable
    if not os.access(metadata_path, os.R_OK):
        raise PermissionError(f"Metadata file is not readable: {metadata_path}")
        
    # Check if station directory exists
    if not os.path.exists(station_dir):
        raise NotADirectoryError(f"Station directory not found: {station_dir}")
    
    # Check if station directory is readable
    if not os.access(station_dir, os.R_OK):
        raise PermissionError(f"Station directory is not accessible: {station_dir}")
        
    # Check if station directory is a directory
    if not os.path.isdir(station_dir):
        raise NotADirectoryError(f"Path is not a directory: {station_dir}")
    
    logger.info("Verifying paths...")
    logger.info(f"  Metadata: {metadata_path}")
    logger.info(f"  Station dir: {station_dir}")
    logger.info("  OK: paths are valid.")
    
    return True


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Load station metadata CSV.
    
    Parameters
    ----------
    metadata_path : str
        Path to the metadata CSV file
        
    Returns
    -------
    pd.DataFrame
        DataFrame with 'SITE CODE' as str
        
    Raises
    ------
    KeyError
        If 'SITE CODE' column is missing from metadata
    ValueError
        If the metadata file is empty
    pd.errors.EmptyDataError
        If file is empty or has no data
    pd.errors.ParserError
        If there's an error parsing the CSV
    """
    try:
        df = pd.read_csv(metadata_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Metadata file is empty: {metadata_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing metadata file: {e}")
    
    # Check if dataframe is empty
    if df.empty:
        raise ValueError(f"Metadata file contains no data: {metadata_path}")
        
    # Check if required column exists
    if 'SITE CODE' not in df.columns:
        raise KeyError("Metadata CSV missing 'SITE CODE' column")
    
    # Convert SITE CODE to string
    df['SITE CODE'] = df['SITE CODE'].astype(str)
    
    # Check for required fields
    required_fields = ['SITE CODE', 'LATITUDE', 'LONGITUDE']
    missing_fields = [field for field in required_fields if field not in df.columns]
    if missing_fields:
        logger.warning(f"Metadata missing recommended fields: {', '.join(missing_fields)}")
    
    logger.info(f"Loaded metadata: {len(df)} stations")
    return df


def list_station_codes(meta: pd.DataFrame) -> list[str]:
    """
    Return list of station codes from metadata DataFrame.
    
    Parameters
    ----------
    meta : pd.DataFrame
        Metadata DataFrame with 'SITE CODE' column
        
    Returns
    -------
    list[str]
        List of station codes
        
    Raises
    ------
    KeyError
        If 'SITE CODE' column is missing
    """
    if not isinstance(meta, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
        
    if 'SITE CODE' not in meta.columns:
        raise KeyError("Metadata DataFrame missing 'SITE CODE' column")
    
    codes = meta['SITE CODE'].tolist()
    logger.info(f"Listing station codes... Found {len(codes)} codes; sample: {codes[:5] if len(codes) >= 5 else codes}")
    return codes


def load_station_data(station_code: str, station_dir: str) -> pd.DataFrame:
    """
    Load station CSV by code, matching any filename containing the code.
    
    Parameters
    ----------
    station_code : str
        Station code to search for
    station_dir : str
        Directory containing station CSV files
        
    Returns
    -------
    pd.DataFrame
        DataFrame with station data including 'datetime' column
        
    Raises
    ------
    FileNotFoundError
        If no matching station file is found
    ValueError
        If station data is empty or missing required columns
    """
    if not station_code or not isinstance(station_code, str):
        raise ValueError(f"Invalid station code: {station_code}")
    
    if not os.path.isdir(station_dir):
        raise NotADirectoryError(f"Station directory not found: {station_dir}")
    
    # Find all files matching the pattern
    pattern = os.path.join(station_dir, f"*{station_code}*_ERA5_with_sea_level.csv")
    matches = glob.glob(pattern)
    
    if not matches:
        raise FileNotFoundError(f"No station CSV found for code '{station_code}' in {station_dir}")
    
    filepath = matches[0]
    logger.info(f"Loading data for station: {station_code} → {os.path.basename(filepath)}")
    
    try:
        df = pd.read_csv(filepath, parse_dates=['datetime'])
    except KeyError:
        # Try loading without parsing dates first, to check if 'datetime' column exists
        try:
            df = pd.read_csv(filepath)
            if 'datetime' not in df.columns:
                raise ValueError(f"Station CSV missing 'datetime' column: {filepath}")
            # Now parse dates
            df['datetime'] = pd.to_datetime(df['datetime'])
        except Exception as e:
            raise ValueError(f"Error loading station data: {e}")
    except Exception as e:
        raise ValueError(f"Error loading station data: {e}")
    
    # Check if dataframe is empty
    if df.empty:
        raise ValueError(f"Station CSV contains no data: {filepath}")
    
    # Check for required columns
    required_columns = ['datetime', 'sea_level']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Station CSV missing required columns: {', '.join(missing_columns)}")
    
    logger.info(f"  Loaded {len(df)} rows; columns: {list(df.columns)}")
    return df


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Smoke-test data_io module')
    parser.add_argument('--metadata', required=True, help='Path to metadata CSV')
    parser.add_argument('--station-dir', required=True, help='Directory of station CSVs')
    parser.add_argument('--sample-code', default=None, help='One station code to test')
    args = parser.parse_args()

    try:
        # Validate
        validate_paths(args.metadata, args.station_dir)
        print("✓ Path validation successful")
        
        # Load metadata and list codes
        meta = load_metadata(args.metadata)
        print(f"✓ Metadata loaded successfully: {len(meta)} stations")
        
        codes = list_station_codes(meta)
        print(f"✓ Station codes listed successfully: {len(codes)} codes")
        
        # Optionally load one station
        code = args.sample_code or codes[0] if codes else None
        if code:
            df = load_station_data(code, args.station_dir)
            print(f"✓ Station data loaded successfully: {len(df)} rows for {code}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        else:
            print("No station codes available for testing")
            
        print("data_io smoke test completed successfully.")
    except Exception as e:
        print(f"Error during test: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)