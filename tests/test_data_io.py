"""
Unit tests for data_io module.
"""

import os
import pytest
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

# Adjust path to import the module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from compound_flooding.data_io import DataLoader
except ImportError:
    # Fallback approach if package is not installed
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from data_io import DataLoader


@pytest.fixture
def sample_data_dir():
    """Create a temporary directory with sample data files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create metadata file
        metadata_df = pd.DataFrame({
            "FILE NAME": ["STATION1_FILE", "STATION2_FILE"],
            "SITE NAME": ["Test Station 1", "Test Station 2"],
            "SITE CODE": ["STATION1", "STATION2"],
            "COUNTRY": ["USA", "USA"],
            "LATITUDE": [40.7, 41.5],
            "LONGITUDE": [-74.0, -75.2],
            "START DATE/TIME": ["2000-01-01 00:00:00", "2000-01-01 00:00:00"],
            "END DATE/TIME": ["2020-12-31 23:00:00", "2020-12-31 23:00:00"],
            "NUMBER OF YEARS": [21.0, 21.0],
            "TIME ZONE HOURS": [0.0, 0.0],
            "DATUM INFORMATION": ["NAVD88", "NAVD88"],
            "INSTRUMENT": ["Gauge", "Gauge"],
            "RECORD QUALITY": ["Good", "Good"]
        })
        metadata_path = tmp_path / "metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        
        # Create data directory
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create sample station data
        for station_code in ["STATION1", "STATION2"]:
            # Create a time series
            dates = pd.date_range(start="2020-01-01", periods=48, freq="H")
            
            # Create sample data
            df = pd.DataFrame({
                "datetime": dates,
                "sea_level": np.random.normal(0, 0.1, len(dates)),
                "total_precipitation": np.random.exponential(0.001, len(dates)),
                "u_component_of_wind_10m": np.random.normal(0, 2, len(dates)),
                "v_component_of_wind_10m": np.random.normal(0, 2, len(dates)),
                "surface_pressure": np.random.normal(101325, 500, len(dates)),
                "temperature_2m": np.random.normal(288, 5, len(dates)),
                "ground_precipitation": np.random.exponential(0.001, len(dates))
            })
            
            # Save to CSV
            file_path = data_dir / f"{station_code}_ERA5_with_sea_level.csv"
            df.to_csv(file_path, index=False)
        
        yield {
            "data_dir": data_dir,
            "metadata_path": metadata_path,
            "station_codes": ["STATION1", "STATION2"]
        }


def test_data_loader_initialization(sample_data_dir):
    """Test DataLoader initialization."""
    # Initialize DataLoader
    loader = DataLoader(
        data_dir=sample_data_dir["data_dir"],
        metadata_path=sample_data_dir["metadata_path"]
    )
    
    # Check that DataLoader is initialized correctly
    assert loader is not None
    assert loader.data_dir == sample_data_dir["data_dir"]
    assert loader.metadata_path == sample_data_dir["metadata_path"]
    assert not loader.use_dask
    
    # Check that metadata is loaded
    assert loader.metadata is not None
    assert len(loader.metadata) == 2
    
    # Check that station codes are extracted
    assert set(loader.station_codes) == set(sample_data_dir["station_codes"])


def test_load_station_data(sample_data_dir):
    """Test loading station data."""
    # Initialize DataLoader
    loader = DataLoader(
        data_dir=sample_data_dir["data_dir"],
        metadata_path=sample_data_dir["metadata_path"]
    )
    
    # Load data for one station
    df = loader.load_station_data("STATION1")
    
    # Check that data is loaded correctly
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 48  # 48 hours as per our sample data
    assert "sea_level" in df.columns
    assert "total_precipitation" in df.columns


def test_load_station_data_with_filters(sample_data_dir):
    """Test loading station data with filters."""
    # Initialize DataLoader
    loader = DataLoader(
        data_dir=sample_data_dir["data_dir"],
        metadata_path=sample_data_dir["metadata_path"]
    )
    
    # Load data with column filter
    columns = ["sea_level", "total_precipitation"]
    df = loader.load_station_data("STATION1", columns=columns)
    
    # Check that only requested columns are present
    assert set(df.columns) == set(columns)
    
    # Load data with date filter
    start_date = "2020-01-01 12:00:00"
    end_date = "2020-01-02 12:00:00"
    df = loader.load_station_data(
        "STATION1", 
        start_date=start_date,
        end_date=end_date
    )
    
    # Check that dates are filtered correctly
    assert len(df) == 25  # From hour 12 to hour 36 = 25 hours


def test_load_station_data_as_xarray(sample_data_dir):
    """Test loading station data as xarray Dataset."""
    # Initialize DataLoader
    loader = DataLoader(
        data_dir=sample_data_dir["data_dir"],
        metadata_path=sample_data_dir["metadata_path"]
    )
    
    # Load data as xarray
    ds = loader.load_station_data_as_xarray("STATION1")
    
    # Check that data is loaded correctly
    assert ds is not None
    assert isinstance(ds, xr.Dataset)
    assert "sea_level" in ds.data_vars
    assert "total_precipitation" in ds.data_vars
    assert "time" in ds.dims
    assert len(ds.time) == 48
    
    # Check that station metadata is added as attributes
    assert "station_code" in ds.attrs
    assert ds.attrs["station_code"] == "STATION1"
    assert "latitude" in ds.attrs
    assert "longitude" in ds.attrs


def test_get_station_metadata(sample_data_dir):
    """Test getting station metadata."""
    # Initialize DataLoader
    loader = DataLoader(
        data_dir=sample_data_dir["data_dir"],
        metadata_path=sample_data_dir["metadata_path"]
    )
    
    # Get metadata for one station
    meta = loader.get_station_metadata("STATION1")
    
    # Check that metadata is retrieved correctly
    assert meta is not None
    assert meta["SITE CODE"] == "STATION1"
    assert meta["SITE NAME"] == "Test Station 1"
    assert meta["LATITUDE"] == 40.7
    assert meta["LONGITUDE"] == -74.0


def test_invalid_station_code(sample_data_dir):
    """Test handling of invalid station code."""
    # Initialize DataLoader
    loader = DataLoader(
        data_dir=sample_data_dir["data_dir"],
        metadata_path=sample_data_dir["metadata_path"]
    )
    
    # Try to load data for non-existent station
    with pytest.raises(FileNotFoundError):
        loader.load_station_data("NONEXISTENT")


def test_get_all_stations_data(sample_data_dir):
    """Test loading data for all stations."""
    # Initialize DataLoader
    loader = DataLoader(
        data_dir=sample_data_dir["data_dir"],
        metadata_path=sample_data_dir["metadata_path"]
    )
    
    # Load data for all stations
    stations_data = loader.get_all_stations_data(columns=["sea_level"])
    
    # Check that data is loaded correctly for all stations
    assert len(stations_data) == 2
    assert "STATION1" in stations_data
    assert "STATION2" in stations_data
    assert "sea_level" in stations_data["STATION1"].columns
    assert len(stations_data["STATION1"]) == 48
    assert len(stations_data["STATION2"]) == 48


if __name__ == "__main__":
    # Run tests manually
    pytest.main(["-xvs", __file__])