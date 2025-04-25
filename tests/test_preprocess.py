"""
Unit tests for preprocess module.
"""

import pytest
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import tempfile
from datetime import datetime

# Adjust path to import the module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from compound_flooding.preprocess import DataPreprocessor
except ImportError:
    # Fallback approach if package is not installed
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from preprocess import DataPreprocessor


@pytest.fixture
def sample_dataset():
    """Create a sample xarray dataset for testing."""
    # Create time coordinate
    time = pd.date_range(start="2020-01-01", periods=100, freq="H")
    
    # Create sample data with trend, spikes, and gaps
    # Sea level with linear trend (0.5m over 100 hours), noise, spikes, and gaps
    trend = np.linspace(0, 0.5, 100)  # Linear trend
    noise = np.random.normal(0, 0.1, 100)  # Random noise
    sea_level = trend + noise
    
    # Add spikes
    sea_level[10] = 5.0  # Large spike
    sea_level[50] = -3.0  # Negative spike
    
    # Add gaps (NaN values)
    sea_level[30:32] = np.nan  # Small gap (2 hours)
    sea_level[70:75] = np.nan  # Larger gap (5 hours)
    
    # Create precipitation data (in meters, small values)
    total_precip = np.random.exponential(0.001, 100)  # in meters
    ground_precip = np.random.exponential(0.001, 100)  # in meters
    
    # Wind components
    u_wind = np.random.normal(0, 5, 100)
    v_wind = np.random.normal(0, 5, 100)
    
    # Pressure in Pa
    pressure = np.random.normal(101325, 500, 100)
    
    # Temperature in K
    temperature = np.random.normal(288, 5, 100)
    
    # Create dataset
    ds = xr.Dataset(
        data_vars={
            'sea_level': ('time', sea_level),
            'total_precipitation': ('time', total_precip),
            'ground_precipitation': ('time', ground_precip),
            'u_component_of_wind_10m': ('time', u_wind),
            'v_component_of_wind_10m': ('time', v_wind),
            'surface_pressure': ('time', pressure),
            'temperature_2m': ('time', temperature)
        },
        coords={
            'time': time
        }
    )
    
    # Add random quality flags for testing
    # 1 = good, 2 = interpolated, 3 = doubtful, 4 = spike, 5 = missing
    sea_level_flags = np.ones(100, dtype=int)  # Initialize as all good
    sea_level_flags[10] = 4  # Mark the spike
    sea_level_flags[50] = 4  # Mark the other spike
    sea_level_flags[30:32] = 5  # Mark the small gap as missing
    sea_level_flags[70:75] = 5  # Mark the larger gap as missing
    
    ds['sea_level_flag'] = ('time', sea_level_flags)
    
    return ds


def test_preprocessor_initialization():
    """Test DataPreprocessor initialization."""
    preprocessor = DataPreprocessor()
    assert preprocessor is not None


def test_check_timezone(sample_dataset):
    """Test timezone checking functionality."""
    preprocessor = DataPreprocessor()
    
    # Original dataset should have no timezone info
    assert sample_dataset.time.dt.tz is None
    
    # After preprocessing, time should have UTC timezone
    processed_ds = preprocessor._check_timezone(sample_dataset)
    
    # Get timezone from the converted dataset
    try:
        # Different xarray versions handle this differently
        tz = processed_ds.time.dt.tz
    except AttributeError:
        # For older xarray versions, convert to pandas first
        tz = pd.DatetimeIndex(processed_ds.time.values).tz
    
    assert str(tz) == 'UTC'


def test_check_units(sample_dataset):
    """Test unit checking and conversion."""
    preprocessor = DataPreprocessor()
    
    # Set units attribute for testing
    sample_dataset.total_precipitation.attrs['units'] = 'm'
    
    # Run unit check
    processed_ds = preprocessor._check_units(sample_dataset)
    
    # Check that precipitation units were converted
    assert processed_ds.total_precipitation.attrs['units'] == 'mm h⁻¹'
    
    # Check that values were scaled appropriately (m to mm)
    assert (processed_ds.total_precipitation.values > sample_dataset.total_precipitation.values).all()
    assert np.allclose(processed_ds.total_precipitation.values, 
                      sample_dataset.total_precipitation.values * 1000)


def test_handle_gaps(sample_dataset):
    """Test gap handling functionality."""
    preprocessor = DataPreprocessor()
    
    # Run gap handling
    processed_ds = preprocessor._handle_gaps(sample_dataset)
    
    # Check that small gaps (2 hours) are filled
    assert not np.isnan(processed_ds.sea_level.values[30:32]).any()
    
    # Large gaps should still be NaN (default max_gap_size is 2)
    assert np.isnan(processed_ds.sea_level.values[70:75]).any()


def test_handle_spikes(sample_dataset):
    """Test spike handling functionality."""
    preprocessor = DataPreprocessor()
    
    # Original dataset has spikes
    assert sample_dataset.sea_level.values[10] == 5.0
    assert sample_dataset.sea_level.values[50] == -3.0
    
    # Run spike handling
    processed_ds = preprocessor._handle_spikes(sample_dataset)
    
    # Check that spikes are handled - either removed or reduced
    assert processed_ds.sea_level.values[10] != 5.0
    assert processed_ds.sea_level.values[50] != -3.0
    
    # Spike handling should generate a spike flag
    assert 'sea_level_spike_flag' in processed_ds.data_vars


def test_detrend_sea_level(sample_dataset):
    """Test sea level detrending functionality."""
    preprocessor = DataPreprocessor()
    
    # Original dataset has a trend
    trend_original = np.polyfit(np.arange(len(sample_dataset.time)), 
                               sample_dataset.sea_level.values, 1)[0]
    assert abs(trend_original) > 0.001  # Should have a non-zero trend
    
    # Run detrending
    processed_ds = preprocessor._detrend_sea_level(sample_dataset)
    
    # Check that original sea level is preserved
    assert 'sea_level_original' in processed_ds
    
    # Detrended sea level should have trend info in attributes
    assert 'sea_level_trend_mm_per_year' in processed_ds.attrs
    
    # Check that the trend is significantly reduced in the detrended data
    # Skip NaN values in calculation
    mask = ~np.isnan(processed_ds.sea_level.values)
    trend_detrended = np.polyfit(np.arange(len(processed_ds.time))[mask], 
                                processed_ds.sea_level.values[mask], 1)[0]
    assert abs(trend_detrended) < abs(trend_original) * 0.1  # Trend should be reduced by 90%+


def test_full_preprocessing(sample_dataset):
    """Test the full preprocessing workflow."""
    preprocessor = DataPreprocessor()
    
    # Run full preprocessing
    processed_ds = preprocessor.preprocess_station(
        ds=sample_dataset,
        detrend_sea_level=True,
        handle_gaps=True,
        handle_spikes=True
    )
    
    # Check that preprocessing steps are recorded in attributes
    assert 'preprocessed' in processed_ds.attrs
    assert processed_ds.attrs['preprocessed'] == 'True'
    
    # Check preprocessing steps
    steps = processed_ds.attrs['preprocessing_steps'].split(',')
    assert 'timezone_check' in steps
    assert 'unit_check' in steps
    assert 'gap_handling' in steps
    assert 'spike_handling' in steps
    assert 'sea_level_detrending' in steps
    
    # Check that spikes are removed and small gaps are filled
    assert processed_ds.sea_level.values[10] != 5.0
    assert processed_ds.sea_level.values[50] != -3.0
    assert not np.isnan(processed_ds.sea_level.values[30])
    assert not np.isnan(processed_ds.sea_level.values[31])


def test_save_preprocessed_data(sample_dataset):
    """Test saving preprocessed data to NetCDF."""
    preprocessor = DataPreprocessor()
    
    # Run preprocessing
    processed_ds = preprocessor.preprocess_station(
        ds=sample_dataset,
        detrend_sea_level=True,
        handle_gaps=True,
        handle_spikes=True
    )
    
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.nc') as tmp:
        tmp_path = Path(tmp.name)
        
        # Save data
        preprocessor.save_preprocessed_data(
            ds=processed_ds,
            output_path=tmp_path,
            compress=True
        )
        
        # Check that file exists
        assert tmp_path.exists()
        
        # Try to load it back
        reloaded = xr.open_dataset(tmp_path)
        
        # Check that key data and attributes are preserved
        assert 'sea_level' in reloaded.data_vars
        assert 'preprocessed' in reloaded.attrs
        assert reloaded.attrs['preprocessed'] == 'True'
        assert 'sea_level_trend_mm_per_year' in reloaded.attrs


if __name__ == "__main__":
    # Run tests manually
    pytest.main(["-xvs", __file__])