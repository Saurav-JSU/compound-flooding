"""
Data I/O utilities for compound flooding analysis.

This module handles loading and validation of GESLA tide-gauge data
and associated ERA5 variables for compound flood analysis.
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import pandas as pd
import numpy as np
import xarray as xr
import dask.dataframe as dd

# Type aliases
PathLike = Union[str, Path]


class DataLoader:
    """
    Data loader for GESLA tide-gauge and ERA5 data.
    
    This class handles validation, lazy loading, and preprocessing of
    GESLA tide-gauge data and associated ERA5 variables.
    
    Parameters
    ----------
    data_dir : PathLike, optional
        Directory containing station CSV files.
        Default is 'compound_flooding/GESLA_ERA5_with_sea_level'.
    metadata_path : PathLike, optional
        Path to the metadata CSV file.
        Default is 'data/GESLA/usa_metadata.csv'.
    use_dask : bool, optional
        Whether to use Dask for lazy loading. Default is False.
    """
    
    def __init__(
        self,
        data_dir: PathLike = Path("compound_flooding/GESLA_ERA5_with_sea_level"),
        metadata_path: PathLike = Path("compound_flooding/data/GESLA/usa_metadata.csv"),
        use_dask: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.metadata_path = Path(metadata_path)
        self.use_dask = use_dask
        
        # Validate paths
        self._validate_paths()
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Get station codes
        self.station_codes = self._get_station_codes()
    
    def _validate_paths(self) -> None:
        """
        Validate that data directory and metadata file exist.
        
        Raises
        ------
        FileNotFoundError
            If data directory or metadata file does not exist.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
    
    def _load_metadata(self) -> pd.DataFrame:
        """
        Load station metadata from CSV file.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing station metadata with appropriate types.
        """
        # Define dtypes for the main columns
        dtypes = {
            "FILE NAME": str,
            "SITE NAME": str,
            "SITE CODE": str,
            "COUNTRY": "category",
            "LATITUDE": float,
            "LONGITUDE": float,
            "NUMBER OF YEARS": float,
            "TIME ZONE HOURS": float,
            "DATUM INFORMATION": str,
            "INSTRUMENT": "category",
            "RECORD QUALITY": "category"
        }
        
        # Load metadata
        try:
            metadata = pd.read_csv(self.metadata_path)
            
            # Apply dtypes where columns exist
            for col, dtype in dtypes.items():
                if col in metadata.columns:
                    metadata[col] = metadata[col].astype(dtype)
            
            # Convert date columns
            date_cols = ["START DATE/TIME", "END DATE/TIME"]
            for col in date_cols:
                if col in metadata.columns:
                    metadata[col] = pd.to_datetime(metadata[col], utc=True, errors='coerce')
            
            return metadata
        except Exception as e:
            raise ValueError(f"Error loading metadata file: {e}")
    
    def _get_station_codes(self) -> List[str]:
        """
        Get list of station codes from metadata or file directory.
        
        Returns
        -------
        List[str]
            List of station codes.
        """
        if "SITE CODE" in self.metadata.columns:
            return self.metadata["SITE CODE"].unique().tolist()
        else:
            # Fallback to extracting from filenames
            pattern = str(self.data_dir / "*_ERA5_with_sea_level.csv")
            files = glob.glob(pattern)
            
            # Extract station codes from filenames
            codes = [Path(f).stem.split('_')[0] for f in files]
            return sorted(set(codes))  # Return unique sorted codes
    
    def get_station_path(self, station_code: str) -> Path:
        """
        Get path to station data CSV file.
        
        Parameters
        ----------
        station_code : str
            Station code.
        
        Returns
        -------
        Path
            Path to station data CSV file.
        
        Raises
        ------
        FileNotFoundError
            If station data file does not exist.
        """
        path = self.data_dir / f"{station_code}_ERA5_with_sea_level.csv"
        
        if not path.exists():
            raise FileNotFoundError(f"Station data file not found: {path}")
        
        return path
    
    def load_station_data(
        self, 
        station_code: str, 
        columns: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        """
        Load data for a specific station.
        
        Parameters
        ----------
        station_code : str
            Station code.
        columns : List[str], optional
            List of columns to load. If None, all columns are loaded.
        start_date : str, optional
            Start date in ISO format (YYYY-MM-DD).
        end_date : str, optional
            End date in ISO format (YYYY-MM-DD).
        
        Returns
        -------
        Union[pd.DataFrame, dd.DataFrame]
            DataFrame containing station data.
            Returns a dask DataFrame if `use_dask` is True.
        """
        path = self.get_station_path(station_code)
        
        # Determine read function and parameters based on dask usage
        if self.use_dask:
            read_func = dd.read_csv
            kwargs = {
                'assume_missing': True,
                'blocksize': '64MB'  # Adjust based on typical file size
            }
        else:
            read_func = pd.read_csv
            kwargs = {}
        
        # Load data
        df = read_func(
            path,
            parse_dates=['datetime'],
            **kwargs
        )
        
        # Filter columns if specified
        if columns is not None:
            # Ensure datetime is included
            if 'datetime' not in columns:
                columns = ['datetime'] + columns
            df = df[columns]
        
        # Filter by date range if specified
        if start_date is not None:
            if self.use_dask:
                df = df[df.datetime >= start_date]
            else:
                df = df[df.datetime >= pd.Timestamp(start_date)]
        
        if end_date is not None:
            if self.use_dask:
                df = df[df.datetime <= end_date]
            else:
                df = df[df.datetime <= pd.Timestamp(end_date)]
        
        # If using pandas, set datetime as index
        if not self.use_dask:
            df.set_index('datetime', inplace=True)
        
        return df
    
    def load_station_data_as_xarray(
        self, 
        station_code: str, 
        columns: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> xr.Dataset:
        """
        Load data for a specific station as an xarray Dataset.
        
        Parameters
        ----------
        station_code : str
            Station code.
        columns : List[str], optional
            List of columns to load. If None, all columns are loaded.
        start_date : str, optional
            Start date in ISO format (YYYY-MM-DD).
        end_date : str, optional
            End date in ISO format (YYYY-MM-DD).
        
        Returns
        -------
        xr.Dataset
            Dataset containing station data.
        """
        df = self.load_station_data(station_code, columns, start_date, end_date)
        
        # If using dask, compute before conversion
        if self.use_dask:
            df = df.compute()
            df.set_index('datetime', inplace=True)
        
        # Convert to xarray
        ds = df.to_xarray()
        
        # Rename time dimension if it's not already called 'time'
        if 'datetime' in ds.dims and 'time' not in ds.dims:
            ds = ds.rename({'datetime': 'time'})
        
        # Add station metadata as attributes
        station_meta = self.get_station_metadata(station_code)
        if station_meta is not None:
            ds.attrs.update({
                'station_code': station_code,
                'station_name': station_meta.get('SITE NAME', ''),
                'latitude': station_meta.get('LATITUDE', np.nan),
                'longitude': station_meta.get('LONGITUDE', np.nan),
                'datum_information': station_meta.get('DATUM INFORMATION', ''),
                'source': 'GESLA_ERA5'
            })
        
        return ds
    
    def get_station_metadata(self, station_code: str) -> Optional[Dict]:
        """
        Get metadata for a specific station.
        
        Parameters
        ----------
        station_code : str
            Station code.
        
        Returns
        -------
        Optional[Dict]
            Dictionary containing station metadata or None if not found.
        """
        if "SITE CODE" in self.metadata.columns:
            station_meta = self.metadata[self.metadata["SITE CODE"] == station_code]
            
            if not station_meta.empty:
                return station_meta.iloc[0].to_dict()
        
        # Try matching by file name if site code not found
        if "FILE NAME" in self.metadata.columns:
            file_pattern = f"{station_code}_*"
            station_meta = self.metadata[self.metadata["FILE NAME"].str.startswith(file_pattern)]
            
            if not station_meta.empty:
                return station_meta.iloc[0].to_dict()
        
        return None
    
    def get_all_stations_data(
        self, 
        columns: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Union[pd.DataFrame, dd.DataFrame]]:
        """
        Load data for all stations.
        
        Parameters
        ----------
        columns : List[str], optional
            List of columns to load. If None, all columns are loaded.
        start_date : str, optional
            Start date in ISO format (YYYY-MM-DD).
        end_date : str, optional
            End date in ISO format (YYYY-MM-DD).
        
        Returns
        -------
        Dict[str, Union[pd.DataFrame, dd.DataFrame]]
            Dictionary mapping station codes to DataFrames.
        """
        result = {}
        
        for station_code in self.station_codes:
            try:
                df = self.load_station_data(station_code, columns, start_date, end_date)
                result[station_code] = df
            except Exception as e:
                print(f"Error loading data for station {station_code}: {e}")
        
        return result


# Example Usage
if __name__ == "__main__":
    import tempfile
    import shutil
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir_path = Path(tempdir)
        
        # Create fake metadata file
        metadata_path = tempdir_path / "metadata.csv"
        pd.DataFrame({
            "SITE CODE": ["TEST1", "TEST2"],
            "SITE NAME": ["Test Station 1", "Test Station 2"],
            "LATITUDE": [40.0, 41.0],
            "LONGITUDE": [-74.0, -75.0]
        }).to_csv(metadata_path, index=False)
        
        # Create fake data directory and files
        data_dir = tempdir_path / "data"
        data_dir.mkdir()
        
        # Create a fake station data file
        test_data = pd.DataFrame({
            "datetime": pd.date_range(start="2000-01-01", periods=100, freq="H"),
            "sea_level": np.random.normal(0, 0.1, 100),
            "total_precipitation": np.random.exponential(0.1, 100)
        })
        test_data.to_csv(data_dir / "TEST1_ERA5_with_sea_level.csv", index=False)
        
        # Initialize data loader
        loader = DataLoader(data_dir=data_dir, metadata_path=metadata_path)
        
        # Test loading data
        print(f"Found station codes: {loader.station_codes}")
        
        # Load data for TEST1
        df = loader.load_station_data("TEST1")
        print("\nLoaded data for TEST1:")
        print(df.head())
        
        # Convert to xarray
        ds = loader.load_station_data_as_xarray("TEST1")
        print("\nConverted to xarray Dataset:")
        print(ds)