# src/compound_flooding/cli.py
"""
CLI wrapper for the Compound Flooding toolkit.
Sub-commands:
  ingest  : Ingest and preprocess all stations (parallel)
Usage example:
  python -m src.compound_flooding.cli ingest \
    --metadata compound_flooding/data/GESLA/usa_metadata.csv \
    --station-dir compound_flooding/GESLA_ERA5_with_sea_level \
    --output-dir outputs/cleaned \
    --detrend \
    --max-gap 3 \
    --spike 5.0 \
    --workers 96
"""
import argparse
import os
import multiprocessing
from functools import partial
import concurrent.futures
import pandas as pd

from src.compound_flooding.data_io import validate_paths, load_metadata, load_station_data
from src.compound_flooding.preprocess import preprocess_dataframe


def _process_station(
    code: str,
    station_dir: str,
    output_dir: str,
    detrend: bool,
    max_gap: int,
    spike: float
) -> str:
    """
    Ingest, preprocess, and save one station to NetCDF.
    """
    try:
        # Load raw CSV
        df = load_station_data(code, station_dir)
        # Convert Dask to pandas if needed
        if hasattr(df, 'compute'):
            df = df.compute()
        # Preprocess
        ds = preprocess_dataframe(
            df,
            detrend=detrend,
            max_gap_hours=max_gap,
            spike_threshold=spike
        )
        # Save
        os.makedirs(output_dir, exist_ok=True)
        outpath = os.path.join(output_dir, f"{code}.nc")
        ds.to_netcdf(outpath)
        return f"[OK]   {code} → {outpath}"
    except Exception as e:
        return f"[ERR]  {code}: {e}"


def ingest_all(
    metadata: str,
    station_dir: str,
    output_dir: str,
    detrend: bool,
    max_gap: int,
    spike: float,
    workers: int
) -> None:
    """
    Parallel ingest & preprocess for all stations.
    """
    validate_paths(metadata, station_dir)
    meta = load_metadata(metadata)
    codes = meta['SITE CODE'].astype(str).tolist()

    # Determine worker count
    num_workers = workers or multiprocessing.cpu_count()
    print(f"Running ingest for {len(codes)} stations with {num_workers} workers...")

    func = partial(
        _process_station,
        station_dir=station_dir,
        output_dir=output_dir,
        detrend=detrend,
        max_gap=max_gap,
        spike=spike
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for msg in executor.map(func, codes):
            print(msg)


def main():
    parser = argparse.ArgumentParser(
        prog='compound_flooding',
        description='Compound Flooding Toolkit'
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # ingest sub-command
    p_ingest = sub.add_parser('ingest', help='Ingest & preprocess all stations')
    p_ingest.add_argument('--metadata', default='compound_flooding/data/GESLA/usa_metadata.csv', help='Metadata CSV')
    p_ingest.add_argument('--station-dir', default='compound_flooding/GESLA_ERA5_with_sea_level', help='Station CSV directory')
    p_ingest.add_argument('--output-dir', default='outputs/cleaned', help='Output NetCDF directory')
    p_ingest.add_argument('--detrend', action='store_true', help='Detrend sea_level')
    p_ingest.add_argument('--max-gap', type=int, default=2, help='Max gap hours to interpolate')
    p_ingest.add_argument('--spike', type=float, default=None, help='Clip sea_level spikes')
    p_ingest.add_argument('--workers', type=int, default=0, help='Number of parallel workers (0=all cores)')

    args = parser.parse_args()
    if args.command == 'ingest':
        ingest_all(
            metadata=args.metadata,
            station_dir=args.station_dir,
            output_dir=args.output_dir,
            detrend=args.detrend,
            max_gap=args.max_gap,
            spike=args.spike,
            workers=args.workers
        )

if __name__ == '__main__':
    main()