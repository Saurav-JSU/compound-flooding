#!/bin/bash

# Default values
STATION_LIST="station_list.txt"
NETCDF_DIR="outputs/cleaned"
TIER2_DIR="outputs/tier2"
OUTPUT_DIR="outputs/tier3"
WORKERS=160
COASTLINE_ANGLE=0.0
RANDOM_STATE=""

# Help function
function show_help {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -s, --station-list FILE    Station list file (default: $STATION_LIST)"
    echo "  -n, --netcdf-dir DIR       Directory containing cleaned NetCDF files (default: $NETCDF_DIR)"
    echo "  -t, --tier2-dir DIR        Directory containing Tier 2 outputs (default: $TIER2_DIR)"
    echo "  -o, --output-dir DIR       Directory to save Tier 3 outputs (default: $OUTPUT_DIR)"
    echo "  -w, --workers N            Number of parallel workers (default: $WORKERS)"
    echo "  -c, --coastline-angle N    Coastline angle in degrees (default: $COASTLINE_ANGLE)"
    echo "  -r, --random-state N       Random state for reproducibility"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            ;;
        -s|--station-list)
            STATION_LIST="$2"
            shift 2
            ;;
        -n|--netcdf-dir)
            NETCDF_DIR="$2"
            shift 2
            ;;
        -t|--tier2-dir)
            TIER2_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -c|--coastline-angle)
            COASTLINE_ANGLE="$2"
            shift 2
            ;;
        -r|--random-state)
            RANDOM_STATE="--random-state $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR/models"
mkdir -p "$OUTPUT_DIR/summary"

# Activate conda environment and run the batch process
echo "Starting Tier 3 batch processing with the following parameters:"
echo "Station list: $STATION_LIST"
echo "NetCDF directory: $NETCDF_DIR"
echo "Tier2 directory: $TIER2_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Number of workers: $WORKERS"
echo "Coastline angle: $COASTLINE_ANGLE"
if [[ -n "$RANDOM_STATE" ]]; then
    echo "Random state: $RANDOM_STATE"
fi

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate compound_flooding

# Run the batch process
echo "Running batch process..."
start_time=$(date +%s)

python src/compound_flooding/tier3_bn.py batch \
    --station-list "$STATION_LIST" \
    --netcdf-dir "$NETCDF_DIR" \
    --tier2-dir "$TIER2_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --workers "$WORKERS" \
    --coastline-angle "$COASTLINE_ANGLE" \
    $RANDOM_STATE

end_time=$(date +%s)
elapsed=$((end_time - start_time))

# Print summary
echo "Batch processing completed in $((elapsed / 60)) minutes and $((elapsed % 60)) seconds."
echo "Results saved to $OUTPUT_DIR" 