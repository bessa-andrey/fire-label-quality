# process_mcd64a1.py
"""
Process MODIS MCD64A1 (Burned Area) GeoTIFF data for Module A and B

Input: MCD64A1_*.tif files (monthly burned area)
Output:
  - burned_area_composite.tif (merged monthly layers)
  - burn_date_stats.csv (pixel-level statistics)
  - mcd64a1_stats.json (processing statistics)
"""

import rasterio
import rioxarray
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm

# Configuration
INPUT_DIR = Path('data/raw/mcd64a1')
OUTPUT_DIR = Path('data/processed/burned_area')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MATOPIBA bounds
MATOPIBA_BOUNDS = {
    'north': 0,
    'south': -15,
    'east': -40,
    'west': -65
}

def find_mcd64a1_files():
    """Find all MCD64A1 GeoTIFF files"""
    print("[INFO] Searching for MCD64A1 files...")

    tif_files = sorted(INPUT_DIR.glob('MCD64A1_*.tif'))

    if not tif_files:
        print("[WARNING] No MCD64A1 GeoTIFF files found")
        print(f"Expected location: {INPUT_DIR}")
        return []

    print(f"[OK] Found {len(tif_files)} MCD64A1 files")
    return tif_files

def load_single_mcd64a1(filepath):
    """Load single MCD64A1 GeoTIFF file"""
    try:
        # Open with rioxarray for easy geospatial handling
        data = rioxarray.open_rasterio(filepath)

        # Extract date from filename (format: MCD64A1_YYYYMM.tif)
        filename = filepath.stem
        year_month = filename.split('_')[1]
        year = int(year_month[:4])
        month = int(year_month[4:6])

        data.attrs['year'] = year
        data.attrs['month'] = month
        data.attrs['date'] = f"{year}-{month:02d}-01"

        return data

    except Exception as e:
        print(f"[WARNING] Error loading {filepath}: {e}")
        return None

def extract_burn_dates(data):
    """
    Extract burn dates from BurnDate band
    0 = not burned
    1-365 = day of year when burned
    """
    # Get the first (and usually only) band
    burn_dates = data.values[0]

    # Create mask: 1 where burned, 0 where not burned
    burned_mask = (burn_dates > 0).astype(np.uint8)

    return burn_dates, burned_mask

def crop_to_matopiba(data):
    """Crop GeoTIFF to MATOPIBA bounds"""
    try:
        # Use rio to access CRS and transform
        cropped = data.rio.clip_box(
            minx=MATOPIBA_BOUNDS['west'],
            miny=MATOPIBA_BOUNDS['south'],
            maxx=MATOPIBA_BOUNDS['east'],
            maxy=MATOPIBA_BOUNDS['north']
        )
        return cropped
    except Exception as e:
        print(f"[WARNING] Error cropping to MATOPIBA: {e}")
        return data

def process_monthly_file(filepath):
    """Process single MCD64A1 monthly file"""
    # Load
    data = load_single_mcd64a1(filepath)
    if data is None:
        return None

    # Crop to MATOPIBA
    data_cropped = crop_to_matopiba(data)

    # Extract burn dates
    burn_dates, burned_mask = extract_burn_dates(data_cropped)

    return {
        'filepath': filepath,
        'year': data.attrs['year'],
        'month': data.attrs['month'],
        'date': data.attrs['date'],
        'data': data_cropped,
        'burn_dates': burn_dates,
        'burned_mask': burned_mask
    }

def merge_monthly_layers(processed_files):
    """
    Merge multiple monthly MCD64A1 files into single xarray Dataset
    Creates band stack with temporal dimension
    """
    print("[INFO] Merging monthly layers...")

    if not processed_files:
        print("[ERROR] No processed files to merge")
        return None

    # Create list of data arrays with time coordinate
    data_arrays = []
    time_coords = []

    for file_data in processed_files:
        data_arrays.append(file_data['data'])

        # Create pandas timestamp for time coordinate
        year = file_data['year']
        month = file_data['month']
        time_coords.append(pd.Timestamp(year=year, month=month, day=1))

    # Stack into DataArray with time dimension
    merged = xr.concat(data_arrays, dim='time')
    merged.coords['time'] = time_coords

    print(f"[OK] Merged {len(data_arrays)} monthly layers")
    print(f"[OK] Spatial shape: {merged.y.size} x {merged.x.size}")
    print(f"[OK] Temporal range: {time_coords[0]} to {time_coords[-1]}")

    return merged

def calculate_burned_area_stats(merged_data):
    """Calculate statistics for burned areas"""
    print("[INFO] Calculating burned area statistics...")

    # Extract burn date band
    burn_dates = merged_data.values

    # Total burned pixels (any time period)
    total_burned_pixels = (burn_dates > 0).any(axis=0).sum()

    # Burned area in km² (assuming 500m resolution = 0.25 km²)
    pixel_area_km2 = 0.25
    total_burned_area_km2 = total_burned_pixels * pixel_area_km2

    # Statistics by month
    monthly_stats = []
    for time_idx in range(len(merged_data.time)):
        month_data = burn_dates[time_idx]
        burned_pixels = (month_data > 0).sum()
        burned_area = burned_pixels * pixel_area_km2

        monthly_stats.append({
            'year_month': str(merged_data.time.values[time_idx])[:7],
            'burned_pixels': int(burned_pixels),
            'burned_area_km2': float(burned_area)
        })

    stats = {
        'processing_date': datetime.now().isoformat(),
        'total_burned_pixels': int(total_burned_pixels),
        'total_burned_area_km2': float(total_burned_area_km2),
        'spatial_resolution_m': 500,
        'pixel_area_km2': pixel_area_km2,
        'temporal_coverage': f"{merged_data.time.min().values} to {merged_data.time.max().values}",
        'spatial_bounds': MATOPIBA_BOUNDS,
        'monthly_breakdown': monthly_stats
    }

    print(f"[OK] Total burned area: {stats['total_burned_area_km2']:.0f} km²")

    return stats

def save_merged_geotiff(merged_data, output_path):
    """Save merged data back to GeoTIFF"""
    print(f"[INFO] Saving merged GeoTIFF...")

    try:
        # Write with rasterio/rio backend
        merged_data.rio.to_raster(output_path)
        print(f"[OK] Saved to: {output_path}")
        return True
    except Exception as e:
        print(f"[WARNING] Could not save as GeoTIFF: {e}")
        print("[INFO] Will save as NetCDF instead")
        return False

def save_netcdf(merged_data, output_path):
    """Save merged data as NetCDF (more flexible format)"""
    print(f"[INFO] Saving as NetCDF...")

    try:
        merged_data.to_netcdf(output_path)
        print(f"[OK] Saved to: {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save NetCDF: {e}")
        return False

def main():
    """Main processing pipeline"""
    print("\n" + "="*60)
    print("MCD64A1 BURNED AREA PROCESSING - ETAPA 2")
    print("="*60)

    # Step 1: Find files
    tif_files = find_mcd64a1_files()
    if not tif_files:
        print("\n[ERROR] No MCD64A1 files found!")
        print("[INFO] Ensure files are downloaded to:", INPUT_DIR)
        print("[INFO] Expected filenames: MCD64A1_YYYYMM.tif")
        return

    # Step 2: Process each monthly file
    print("\n[INFO] Processing individual monthly files...")
    processed_files = []

    for filepath in tqdm(tif_files, desc="Loading MCD64A1 files"):
        file_data = process_monthly_file(filepath)
        if file_data is not None:
            processed_files.append(file_data)

    print(f"[OK] Successfully processed {len(processed_files)} files")

    if not processed_files:
        print("[ERROR] No files were successfully processed")
        return

    # Step 3: Merge monthly layers
    merged_data = merge_monthly_layers(processed_files)
    if merged_data is None:
        print("[ERROR] Failed to merge data")
        return

    # Step 4: Calculate statistics
    stats = calculate_burned_area_stats(merged_data)

    # Step 5: Save outputs
    print("\n[INFO] Saving outputs...")

    # Try to save as GeoTIFF first
    geotiff_path = OUTPUT_DIR / 'mcd64a1_burned_area.tif'
    success = save_merged_geotiff(merged_data, geotiff_path)

    # Also save as NetCDF for flexibility
    netcdf_path = OUTPUT_DIR / 'mcd64a1_burned_area.nc'
    save_netcdf(merged_data, netcdf_path)

    # Save statistics
    stats_file = OUTPUT_DIR / 'mcd64a1_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[OK] Statistics saved to: {stats_file}")

    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Monthly files processed: {len(processed_files)}")
    print(f"Total burned area: {stats['total_burned_area_km2']:.0f} km²")
    print(f"Total burned pixels: {stats['total_burned_pixels']}")
    print(f"Temporal coverage: {stats['temporal_coverage']}")

    return merged_data

if __name__ == '__main__':
    merged = main()
    print("\n[INFO] Next step: Process Sentinel-2 imagery")
    print("[INFO] Run: python src/preprocessing/process_sentinel2.py")
