"""
Unit tests for cli module.
"""

import pytest
import sys
from pathlib import Path
import tempfile
import pandas as pd
import numpy as np
import xarray as xr
import argparse
import os
import shutil

# Add parent path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from compound_flooding.cli import setup_parser, ingest_command
except ImportError:
    # Fallback for direct import
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from cli import setup_parser, ingest_command


@pytest.fixture
def sample_data_environment():
    """Create a temporary directory with sample data files for CLI testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create the necessary directory structure
        data_dir = tmp_path / "compound_flooding" / "GESLA_ERA5_with_sea_level"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_dir = tmp_path / "data" / "GESLA"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        output_dir = tmp_path / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file
        metadata_df = pd.DataFrame({
            "FILE NAME": ["TEST1_FILE", "TEST2_FILE"],
            "SITE NAME": ["Test Station 1", "Test Station 2"],
            "SITE CODE": ["TEST1", "TEST2"],
            "COUNTRY": ["USA", "USA"],
            "LATITUDE": [40.7, 41.5],
            "LONGITUDE": [-74.0, -75.2],
            "START DATE/TIME": ["2020-01-01 00:00:00", "2020-01-01 00:00:00"],
            "END DATE/TIME": ["2020-01-31 23:00:00", "2020-01-31 23:00:00"],
            "NUMBER OF YEARS": [0.08, 0.08],  # ~1 month
            "TIME ZONE HOURS": [0.0, 0.0],
            "DATUM INFORMATION": ["NAVD88", "NAVD88"],
            "INSTRUMENT": ["Gauge", "Gauge"],
            "RECORD QUALITY": ["Good", "Good"]
        })
        metadata_path = metadata_dir / "usa_metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        
        # Create sample station data files
        for station_code in ["TEST1", "TEST2"]:
            # Create sample time series
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
            
            # Add spike and gap for testing preprocessing
            if station_code == "TEST1":
                df.loc[10, "sea_level"] = 5.0  # Spike
                df.loc[20:22, "sea_level"] = np.nan  # Gap
            
            # Save to CSV
            file_path = data_dir / f"{station_code}_ERA5_with_sea_level.csv"
            df.to_csv(file_path, index=False)
        
        # Create an empty pytest import module
        init_file = Path(__file__).parent / "__init__.py"
        if not init_file.exists():
            init_file.touch()
        
        # Test module file paths
        data_io_path = Path(__file__).parent.parent / "compound_flooding" / "data_io.py"
        preprocess_path = Path(__file__).parent.parent / "compound_flooding" / "preprocess.py"
        
        # Create compound_flooding directory in tempdir as well to ensure imports work
        compound_dir = tmp_path / "compound_flooding"
        compound_dir.mkdir(exist_ok=True)
        
        # Create __init__.py
        (compound_dir / "__init__.py").touch()
        
        # Copy relevant modules to tmp_path (if they exist)
        if data_io_path.exists() and preprocess_path.exists():
            shutil.copy(data_io_path, compound_dir / "data_io.py")
            shutil.copy(preprocess_path, compound_dir / "preprocess.py")
        else:
            # Fallback to parent directory
            data_io_parent = Path(__file__).parent.parent.parent / "data_io.py"
            preprocess_parent = Path(__file__).parent.parent.parent / "preprocess.py"
            
            if data_io_parent.exists() and preprocess_parent.exists():
                shutil.copy(data_io_parent, compound_dir / "data_io.py")
                shutil.copy(preprocess_parent, compound_dir / "preprocess.py")
        
        yield {
            "root_dir": tmp_path,
            "data_dir": data_dir,
            "metadata_path": metadata_path,
            "output_dir": output_dir,
            "station_codes": ["TEST1", "TEST2"]
        }


def test_parser_setup():
    """Test command-line argument parser setup."""
    parser = setup_parser()
    
    # Test basic parser functionality
    assert parser is not None
    assert isinstance(parser, argparse.ArgumentParser)
    
    # Parse some test arguments
    args = parser.parse_args(['--data-dir', '/test/data', 'ingest'])
    
    # Check that arguments are parsed correctly
    assert args.data_dir == '/test/data'
    assert args.command == 'ingest'
    assert not args.gpu  # Default is False
    
    # Test ingest subcommand parser
    args = parser.parse_args([
        'ingest',
        '--station-codes', 'TEST1', 'TEST2',
        '--detrend-sea-level',
        '--overwrite'
    ])
    
    assert args.command == 'ingest'
    assert args.station_codes == ['TEST1', 'TEST2']
    assert args.detrend_sea_level is True
    assert args.overwrite is True
    assert args.no_gap_handling is False  # Default
    assert args.no_spike_handling is False  # Default


@pytest.mark.parametrize("detrend", [True, False])
@pytest.mark.parametrize("handle_gaps", [True, False])
def test_ingest_command(sample_data_environment, detrend, handle_gaps, monkeypatch):
    """Test the ingest command functionality with different preprocessing options."""
    # Setup test environment paths
    root_dir = sample_data_environment["root_dir"]
    data_dir = sample_data_environment["data_dir"]
    metadata_path = sample_data_environment["metadata_path"]
    output_dir = sample_data_environment["output_dir"]
    
    # Add test directory to path to ensure imports work
    sys.path.insert(0, str(root_dir))
    
    # Mock arguments
    class MockArgs:
        verbose = 1
        data_dir = str(data_dir)
        metadata_path = str(metadata_path)
        output_dir = str(output_dir)
        use_dask = False
        gpu = False
        station_codes = ["TEST1"]
        start_date = None
        end_date = None
        detrend_sea_level = detrend
        no_gap_handling = not handle_gaps
        no_spike_handling = False
        overwrite = True
        max_workers = 1
    
    # Run ingest command
    args = MockArgs()
    
    # Redirect stdout to capture output
    from io import StringIO
    import sys
    
    # Save original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Create string buffer for capturing output
    captured_output = StringIO()
    sys.stdout = captured_output
    sys.stderr = captured_output
    
    try:
        # Run ingest command
        ingest_command(args)
        
        # Reset stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        # Get the captured output
        output = captured_output.getvalue()
        
        # Check that processing completed successfully
        assert "Ingest completed" in output
        assert "Processed: 1/1 stations" in output
        
        # Check that output file was created
        output_file = output_dir / "netcdf" / "TEST1_preprocessed.nc"
        assert output_file.exists(), f"Output file not found: {output_file}"
        
        # Load the output file and check results
        ds = xr.open_dataset(output_file)
        
        # Check basic structure
        assert "sea_level" in ds.data_vars
        assert "time" in ds.dims
        
        # Check detrending based on parameter
        if detrend:
            assert "sea_level_original" in ds.data_vars
            assert "sea_level_trend_mm_per_year" in ds.attrs
        
        # Check preprocessing steps in attributes
        assert "preprocessed" in ds.attrs
        assert ds.attrs["preprocessed"] == "True"
        
        steps = ds.attrs["preprocessing_steps"].split(",")
        assert "timezone_check" in steps
        assert "unit_check" in steps
        assert "sea_level_detrending" in steps if detrend else "sea_level_detrending" not in steps
        assert "gap_handling" in steps if handle_gaps else "gap_handling" not in steps
        
    finally:
        # Restore stdout and stderr even if test fails
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def test_error_handling(sample_data_environment, monkeypatch):
    """Test error handling in the ingest command."""
    # Setup test environment paths
    root_dir = sample_data_environment["root_dir"]
    data_dir = sample_data_environment["data_dir"]
    metadata_path = sample_data_environment["metadata_path"]
    output_dir = sample_data_environment["output_dir"]
    
    # Add test directory to path to ensure imports work
    sys.path.insert(0, str(root_dir))
    
    # Mock arguments with non-existent station
    class MockArgs:
        verbose = 2  # More verbose for detailed error
        data_dir = str(data_dir)
        metadata_path = str(metadata_path)
        output_dir = str(output_dir)
        use_dask = False
        gpu = False
        station_codes = ["NONEXISTENT"]  # This station doesn't exist
        start_date = None
        end_date = None
        detrend_sea_level = False
        no_gap_handling = False
        no_spike_handling = False
        overwrite = True
        max_workers = 1
    
    # Run ingest command with error
    args = MockArgs()
    
    # Redirect stdout to capture output
    from io import StringIO
    import sys
    
    # Save original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Create string buffer for capturing output
    captured_output = StringIO()
    sys.stdout = captured_output
    sys.stderr = captured_output
    
    try:
        # Run ingest command
        ingest_command(args)
        
        # Reset stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        # Get the captured output
        output = captured_output.getvalue()
        
        # Check that error was handled
        assert "Error processing station NONEXISTENT" in output
        assert "Errors: 1/1 stations" in output
        
    finally:
        # Restore stdout and stderr even if test fails
        sys.stdout = original_stdout
        sys.stderr = original_stderr


if __name__ == "__main__":
    # Run tests manually
    pytest.main(["-xvs", __file__])