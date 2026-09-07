# process_era5.py
"""
Process ERA5 meteorological reanalysis data for Module A and B

Input: ERA5_*.nc files (monthly meteorological variables)
Output:
  - era5_daily_aggregates.nc (daily aggregated meteorological data)
  - weather_statistics.csv (daily weather statistics)
  - era5_stats.json (processing statistics)
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm

# Configuration
INPUT_DIR = Path('data/raw/era5')
OUTPUT_DIR = Path('data/processed/era5')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MATOPIBA bounds (approximate grid cells in ERA5)
MATOPIBA_BOUNDS = {
    'north': 0,
    'south': -15,
    'east': -40,
    'west': -65
}

# ERA5 key variables for fire analysis
ERA5_VARIABLES = [
    '2m_temperature',
    '2m_dewpoint_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'total_precipitation',
    'soil_moisture_0_7cm',
    'surface_solar_radiation_downwards'
]

def find_era5_files():
    """Find all ERA5 NetCDF files"""
    print("[INFO] Searching for ERA5 files...")

    nc_files = sorted(INPUT_DIR.glob('ERA5_*.nc'))

    if not nc_files:
        print("[WARNING] No ERA5 NetCDF files found")
        print(f"Expected location: {INPUT_DIR}")
        return []

    print(f"[OK] Found {len(nc_files)} ERA5 files")
    return nc_files

def load_single_era5(filepath):
    """Load single ERA5 NetCDF file"""
    try:
        # Open with xarray
        data = xr.open_dataset(filepath)

        # Extract date from filename (format: ERA5_YYYYMM.nc)
        filename = filepath.stem
        year_month = filename.split('_')[1]
        year = int(year_month[:4])
        month = int(year_month[4:6])

        data.attrs['year'] = year
        data.attrs['month'] = month

        return data

    except Exception as e:
        print(f"[WARNING] Error loading {filepath}: {e}")
        return None

def crop_to_matopiba(data):
    """
    Crop ERA5 data to MATOPIBA region
    ERA5 uses latitude/longitude as dimensions
    """
    try:
        # Slice by latitude and longitude bounds
        cropped = data.sel(
            latitude=slice(MATOPIBA_BOUNDS['north'], MATOPIBA_BOUNDS['south']),
            longitude=slice(MATOPIBA_BOUNDS['west'], MATOPIBA_BOUNDS['east']),
            drop=False
        )
        return cropped
    except Exception as e:
        print(f"[WARNING] Error cropping to MATOPIBA: {e}")
        # Return full data if crop fails
        return data

def aggregate_to_daily(hourly_data):
    """
    Aggregate hourly ERA5 data to daily statistics
    Input: hourly data with time dimension
    Output: daily min, max, mean for each variable
    """
    print("[INFO] Aggregating hourly data to daily...")

    if 'time' not in hourly_data.dims:
        print("[WARNING] No time dimension found in ERA5 data")
        return None

    # Resample to daily (mean)
    daily_mean = hourly_data.resample(time='D').mean()

    # Also compute daily min and max for temperature
    if '2m_temperature' in hourly_data.data_vars:
        daily_min_temp = hourly_data['2m_temperature'].resample(time='D').min()
        daily_max_temp = hourly_data['2m_temperature'].resample(time='D').max()

        daily_mean['2m_temperature_min'] = daily_min_temp
        daily_mean['2m_temperature_max'] = daily_max_temp

    # Accumulate precipitation (sum instead of mean)
    if 'total_precipitation' in hourly_data.data_vars:
        precip_daily = hourly_data['total_precipitation'].resample(time='D').sum()
        daily_mean['total_precipitation'] = precip_daily

    print(f"[OK] Aggregated to {len(daily_mean.time)} daily records")
    return daily_mean

def calculate_wind_components(data):
    """
    Calculate wind magnitude and direction from U and V components
    U = East-West component (positive = East)
    V = North-South component (positive = North)
    """
    print("[INFO] Calculating wind magnitude and direction...")

    u_var = '10m_u_component_of_wind'
    v_var = '10m_v_component_of_wind'

    if u_var in data.data_vars and v_var in data.data_vars:
        # Wind magnitude: sqrt(U² + V²)
        wind_magnitude = np.sqrt(
            data[u_var]**2 + data[v_var]**2
        )
        data['wind_magnitude'] = wind_magnitude

        # Wind direction: atan2(V, U) in degrees (0-360)
        wind_direction = np.arctan2(data[v_var], data[u_var]) * 180 / np.pi
        wind_direction = (wind_direction + 360) % 360  # Convert to 0-360 range
        data['wind_direction'] = wind_direction

        print("[OK] Wind components calculated")

    return data

def calculate_drying_indices(data):
    """
    Calculate Fire Weather Index (FWI) components
    Simple approximation using available variables
    """
    print("[INFO] Calculating drying and fire weather indices...")

    # Temperature and humidity stress (simple approximation)
    if '2m_temperature' in data.data_vars and '2m_dewpoint_temperature' in data.data_vars:
        temp = data['2m_temperature']
        dewpoint = data['2m_dewpoint_temperature']

        # Relative humidity approximation
        # RH ≈ 100 * (es(Td) / es(T))
        # Simplified: RH ≈ 100 * (10^(Td/500) / 10^(T/500))
        rh = 100 * (10**(dewpoint/500) / 10**(temp/500))
        rh = np.clip(rh, 0, 100)
        data['relative_humidity'] = rh

        # Drying index (inverse of humidity)
        data['drying_index'] = 100 - rh

        print("[OK] Drying indices calculated")

    return data

def process_monthly_file(filepath):
    """Process single ERA5 monthly file"""
    # Load
    data = load_single_era5(filepath)
    if data is None:
        return None

    # Crop to MATOPIBA
    data_cropped = crop_to_matopiba(data)

    # Aggregate to daily
    daily_data = aggregate_to_daily(data_cropped)
    if daily_data is None:
        return None

    # Calculate wind components
    daily_data = calculate_wind_components(daily_data)

    # Calculate drying indices
    daily_data = calculate_drying_indices(daily_data)

    return {
        'filepath': filepath,
        'year': data.attrs['year'],
        'month': data.attrs['month'],
        'data': daily_data
    }

def merge_monthly_files(processed_files):
    """Merge multiple monthly ERA5 files into single time series"""
    print("[INFO] Merging monthly data into annual/complete time series...")

    if not processed_files:
        return None

    # Concatenate along time dimension
    data_list = [f['data'] for f in processed_files]
    merged = xr.concat(data_list, dim='time')

    # Sort by time
    merged = merged.sortby('time')

    print(f"[OK] Merged {len(processed_files)} months")
    print(f"[OK] Temporal range: {merged.time.min().values} to {merged.time.max().values}")

    return merged

def calculate_weather_statistics(merged_data):
    """Calculate summary weather statistics"""
    print("[INFO] Computing weather statistics...")

    stats_records = []

    # Global statistics
    for var in merged_data.data_vars:
        var_data = merged_data[var].values

        if var_data.size > 0 and not np.all(np.isnan(var_data)):
            stats_records.append({
                'variable': var,
                'mean': float(np.nanmean(var_data)),
                'min': float(np.nanmin(var_data)),
                'max': float(np.nanmax(var_data)),
                'std': float(np.nanstd(var_data))
            })

    stats_df = pd.DataFrame(stats_records)

    # Temporal statistics (if daily data exists)
    if 'time' in merged_data.dims:
        temporal_stats = {
            'temporal_coverage': f"{merged_data.time.min().values} to {merged_data.time.max().values}",
            'days_covered': len(merged_data.time)
        }
    else:
        temporal_stats = {}

    print(f"[OK] Computed statistics for {len(stats_df)} variables")

    return stats_df, temporal_stats

def save_outputs(merged_data, stats_df, temporal_stats):
    """Save processed data and statistics"""
    print("[INFO] Saving outputs...")

    # Save as NetCDF
    nc_path = OUTPUT_DIR / 'era5_daily_aggregates.nc'
    try:
        merged_data.to_netcdf(nc_path)
        print(f"[OK] Daily aggregates saved to: {nc_path}")
    except Exception as e:
        print(f"[WARNING] Failed to save NetCDF: {e}")

    # Save statistics
    if not stats_df.empty:
        csv_path = OUTPUT_DIR / 'weather_statistics.csv'
        stats_df.to_csv(csv_path, index=False)
        print(f"[OK] Statistics saved to: {csv_path}")

    # Save summary JSON
    summary = {
        'processing_date': datetime.now().isoformat(),
        'spatial_resolution_degrees': 0.25,
        'spatial_resolution_km': 25,
        'spatial_bounds': MATOPIBA_BOUNDS,
        'temporal_stats': temporal_stats,
        'variables_processed': list(merged_data.data_vars),
        'data_source': 'ECMWF ERA5 Reanalysis'
    }

    summary_path = OUTPUT_DIR / 'era5_stats.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Summary saved to: {summary_path}")

    return {
        'data_file': str(nc_path),
        'stats_file': str(csv_path),
        'summary_file': str(summary_path)
    }

def main():
    """Main processing pipeline"""
    print("\n" + "="*60)
    print("ERA5 METEOROLOGICAL DATA PROCESSING - ETAPA 2")
    print("="*60)

    # Step 1: Find files
    nc_files = find_era5_files()
    if not nc_files:
        print("\n[ERROR] No ERA5 files found!")
        print("[INFO] Ensure files are downloaded to:", INPUT_DIR)
        print("[INFO] Expected filenames: ERA5_YYYYMM.nc")
        return

    # Step 2: Process each monthly file
    print("\n[INFO] Processing individual monthly files...")
    processed_files = []

    for filepath in tqdm(nc_files, desc="Processing ERA5 files"):
        file_data = process_monthly_file(filepath)
        if file_data is not None:
            processed_files.append(file_data)

    print(f"[OK] Successfully processed {len(processed_files)} files")

    if not processed_files:
        print("[ERROR] No files were successfully processed")
        return

    # Step 3: Merge monthly files
    merged_data = merge_monthly_files(processed_files)
    if merged_data is None:
        print("[ERROR] Failed to merge monthly data")
        return

    # Step 4: Calculate statistics
    stats_df, temporal_stats = calculate_weather_statistics(merged_data)

    # Step 5: Save outputs
    output_info = save_outputs(merged_data, stats_df, temporal_stats)

    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Monthly files processed: {len(processed_files)}")
    print(f"Data variables: {len(merged_data.data_vars)}")
    print(f"Temporal coverage: {len(merged_data.time)} days")
    if temporal_stats:
        print(f"Period: {temporal_stats.get('temporal_coverage', 'N/A')}")

    return merged_data

if __name__ == '__main__':
    merged = main()
    print("\n[INFO] All raw data processing complete!")
    print("[INFO] Next step: Feature engineering for Module A and B")
    print("[INFO] Run: python src/preprocessing/feature_engineering.py")
