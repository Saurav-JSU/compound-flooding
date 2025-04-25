# How to Run the Compound Flooding Analysis Code

Based on the errors you encountered, here are step-by-step instructions to get the code running:

## Setup Instructions

1. **Copy the Fixed Preprocessor**:
   Replace the original `preprocess.py` with the fixed version I provided, which handles:
   - Sea level missing values (`-99.9999`)
   - Proper ERA5 precipitation conversion from meters to mm/h
   - Error handling for detrending when all values are identical or invalid

2. **Create the Correct Directory Structure**:
   ```bash
   # From your project root:
   mkdir -p outputs/netcdf
   ```

3. **Run with Explicit Path Arguments**:
   ```bash
   python src/compound_flooding/cli.py ingest \
     --data-dir path/to/your/station/csv/files \
     --metadata-path path/to/your/metadata.csv \
     --detrend-sea-level \
     --verbose
   ```

4. **Process a Single Station First**:
   ```bash
   python src/compound_flooding/cli.py ingest \
     --station-codes YOUR_STATION_CODE \
     --data-dir path/to/your/station/csv/files \
     --metadata-path path/to/your/metadata.csv \
     --verbose
   ```

## Important Notes About the Data

1. **Handling `-99.9999` Values**:
   The fixed preprocessor automatically detects and converts `-99.9999` sea level values to NaN, which is a common missing data code.

2. **ERA5 Precipitation Units**:
   The code now correctly detects and converts ERA5 precipitation from meters to mm/h, looking for values that are very small (typical of meter units).

3. **Gaps in Time Series**:
   The large number of gaps detected (198,349) suggests there might be significant periods with missing data. The preprocessor will attempt to interpolate small gaps (≤2 hours) but will leave larger gaps as NaN.

4. **Detrending Failures**:
   If all valid sea level values are identical or if there are numerical issues, detrending will be skipped with a warning message.

## Troubleshooting

1. **Check Your CSV Files**:
   Ensure all station CSV files follow the format:
   ```
   datetime,total_precipitation,u_component_of_wind_10m,v_component_of_wind_10m,surface_pressure,temperature_2m,sea_level,ground_precipitation
   ```

2. **Metadata File Format**:
   Your metadata CSV should have columns including:
   ```
   FILE NAME,SITE NAME,SITE CODE,COUNTRY,LATITUDE,LONGITUDE,START DATE/TIME,END DATE/TIME,NUMBER OF YEARS,TIME ZONE HOURS,DATUM INFORMATION,INSTRUMENT,RECORD QUALITY
   ```

3. **Check Error Messages**:
   Look for specific error messages related to:
   - File not found errors (wrong paths)
   - Data processing errors (malformed data)
   - Detrending errors (numerical issues)

4. **Enable Verbose Mode**:
   Always use the `--verbose` flag (or `-v`) to see detailed logging output:
   ```bash
   python src/compound_flooding/cli.py ingest --verbose --detrend-sea-level
   ```

By following these instructions, you should be able to successfully run the preprocessing stage for your compound flooding analysis.