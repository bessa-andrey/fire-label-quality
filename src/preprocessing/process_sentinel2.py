# process_sentinel2.py
"""
Process Sentinel-2 optical imagery for Module A and B

Input: Sentinel2_*.tif files (RGB + NDVI composites)
Output:
  - sentinel2_ndvi_composite.tif (merged NDVI layers)
  - sentinel2_rgb_composite.tif (merged RGB layers)
  - vegetation_stats.csv (NDVI statistics)
  - sentinel2_stats.json (processing statistics)
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
INPUT_DIR = Path('data/raw/sentinel2')
OUTPUT_DIR = Path('data/processed/sentinel2')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MATOPIBA bounds
MATOPIBA_BOUNDS = {
    'north': 0,
    'south': -15,
    'east': -40,
    'west': -65
}

def find_sentinel2_files():
    """Find all Sentinel-2 GeoTIFF files"""
    print("[INFO] Searching for Sentinel-2 files...")

    tif_files = sorted(INPUT_DIR.glob('Sentinel2_*.tif'))

    if not tif_files:
        print("[WARNING] No Sentinel-2 GeoTIFF files found")
        print(f"Expected location: {INPUT_DIR}")
        return []

    print(f"[OK] Found {len(tif_files)} Sentinel-2 files")
    return tif_files

def load_single_sentinel2(filepath):
    """Load single Sentinel-2 GeoTIFF file"""
    try:
        # Open with rioxarray
        data = rioxarray.open_rasterio(filepath)

        # Extract metadata from filename (format: Sentinel2_YYYY_season.tif)
        filename = filepath.stem
        parts = filename.split('_')
        year = int(parts[1])
        season = parts[2] if len(parts) > 2 else 'unknown'

        data.attrs['year'] = year
        data.attrs['season'] = season
        data.attrs['date'] = f"{year}-07-01" if season == 'dry' else f"{year}-11-01"

        return data

    except Exception as e:
        print(f"[WARNING] Error loading {filepath}: {e}")
        return None

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

def extract_bands(data):
    """
    Extract bands from Sentinel-2 file
    Expected bands: B4 (Red), B3 (Green), B2 (Blue), NDVI
    Or all bands available
    """
    bands = {}

    # Data should have bands as first dimension
    if hasattr(data, 'band') and len(data.band) > 0:
        # If bands are labeled
        band_names = [str(b) for b in data.band.values]
    else:
        # Generic band numbering
        band_names = [f'band_{i+1}' for i in range(data.shape[0])]

    # Extract each band
    for idx, band_name in enumerate(band_names):
        if idx < data.shape[0]:
            bands[band_name] = data.isel(band=idx).values

    return bands

def calculate_ndvi(red_band, nir_band):
    """
    Calculate NDVI from Red (B4) and NIR (B8) bands
    NDVI = (NIR - Red) / (NIR + Red)
    Range: -1 to +1
      +1 = dense vegetation
       0 = mixed vegetation/soil
      -1 = water
    """
    # Avoid division by zero
    denominator = nir_band.astype(float) + red_band.astype(float)
    denominator[denominator == 0] = np.nan

    ndvi = (nir_band.astype(float) - red_band.astype(float)) / denominator

    # Clip to valid range
    ndvi = np.clip(ndvi, -1, 1)

    return ndvi

def process_monthly_file(filepath):
    """Process single Sentinel-2 file"""
    # Load
    data = load_single_sentinel2(filepath)
    if data is None:
        return None

    # Crop to MATOPIBA
    data_cropped = crop_to_matopiba(data)

    # Extract bands
    bands = extract_bands(data_cropped)

    return {
        'filepath': filepath,
        'year': data.attrs['year'],
        'season': data.attrs['season'],
        'date': data.attrs['date'],
        'data': data_cropped,
        'bands': bands
    }

def merge_by_season(processed_files):
    """
    Merge Sentinel-2 files by season
    Dry season: July-October (create single composite)
    Wet season: November-June (create single composite)
    """
    print("[INFO] Merging files by season...")

    # Group by season
    dry_files = [f for f in processed_files if f['season'] == 'dry']
    wet_files = [f for f in processed_files if f['season'] == 'wet']

    composites = {}

    # Merge dry season
    if dry_files:
        print(f"[OK] Found {len(dry_files)} dry season files")
        dry_data_arrays = [f['data'] for f in dry_files]
        dry_merged = xr.concat(dry_data_arrays, dim='time')
        dry_merged.coords['time'] = [pd.Timestamp(year=f['year'], month=7, day=1) for f in dry_files]
        composites['dry'] = dry_merged

    # Merge wet season
    if wet_files:
        print(f"[OK] Found {len(wet_files)} wet season files")
        wet_data_arrays = [f['data'] for f in wet_files]
        wet_merged = xr.concat(wet_data_arrays, dim='time')
        wet_merged.coords['time'] = [pd.Timestamp(year=f['year'], month=11, day=1) for f in wet_files]
        composites['wet'] = wet_merged

    return composites

def calculate_vegetation_stats(merged_composites, processed_files):
    """Calculate vegetation indices statistics"""
    print("[INFO] Calculating vegetation statistics...")

    stats_records = []

    for file_data in processed_files:
        bands = file_data['bands']

        # Try to calculate NDVI from available bands
        # Look for Red, NIR, or generic band indices
        red_band = None
        nir_band = None

        # Check if NDVI already computed (band named 'NDVI' or similar)
        for band_name in bands.keys():
            if 'ndvi' in band_name.lower():
                stats_records.append({
                    'year': file_data['year'],
                    'season': file_data['season'],
                    'date': file_data['date'],
                    'metric': 'NDVI',
                    'mean': float(np.nanmean(bands[band_name])),
                    'min': float(np.nanmin(bands[band_name])),
                    'max': float(np.nanmax(bands[band_name])),
                    'std': float(np.nanstd(bands[band_name]))
                })
                break

    if not stats_records:
        print("[WARNING] Could not calculate NDVI - bands not labeled as expected")
        print("[INFO] Expected bands: B4 (Red), B8 (NIR), or pre-computed NDVI")

    return pd.DataFrame(stats_records) if stats_records else None

def save_merged_geotiff(merged_data, output_path):
    """Save merged data as GeoTIFF"""
    print(f"[INFO] Saving GeoTIFF to {output_path}...")

    try:
        merged_data.rio.to_raster(output_path)
        print(f"[OK] Saved to: {output_path}")
        return True
    except Exception as e:
        print(f"[WARNING] GeoTIFF save failed: {e}")
        return False

def save_netcdf(merged_data, output_path):
    """Save merged data as NetCDF"""
    print(f"[INFO] Saving NetCDF to {output_path}...")

    try:
        merged_data.to_netcdf(output_path)
        print(f"[OK] Saved to: {output_path}")
        return True
    except Exception as e:
        print(f"[WARNING] NetCDF save failed: {e}")
        return False

def main():
    """Main processing pipeline"""
    print("\n" + "="*60)
    print("SENTINEL-2 IMAGERY PROCESSING - ETAPA 2")
    print("="*60)

    # Step 1: Find files
    tif_files = find_sentinel2_files()
    if not tif_files:
        print("\n[ERROR] No Sentinel-2 files found!")
        print("[INFO] Ensure files are downloaded to:", INPUT_DIR)
        print("[INFO] Expected filenames: Sentinel2_YYYY_season.tif")
        return

    # Step 2: Process each file
    print("\n[INFO] Processing individual files...")
    processed_files = []

    for filepath in tqdm(tif_files, desc="Loading Sentinel-2 files"):
        file_data = process_monthly_file(filepath)
        if file_data is not None:
            processed_files.append(file_data)

    print(f"[OK] Successfully processed {len(processed_files)} files")

    if not processed_files:
        print("[ERROR] No files were successfully processed")
        return

    # Step 3: Merge by season
    merged_composites = merge_by_season(processed_files)

    # Step 4: Calculate vegetation statistics
    veg_stats_df = calculate_vegetation_stats(merged_composites, processed_files)
    if veg_stats_df is not None:
        veg_stats_file = OUTPUT_DIR / 'vegetation_stats.csv'
        veg_stats_df.to_csv(veg_stats_file, index=False)
        print(f"[OK] Vegetation stats saved to: {veg_stats_file}")

    # Step 5: Save merged data
    print("\n[INFO] Saving outputs...")

    for season, merged_data in merged_composites.items():
        print(f"\n[INFO] Processing {season} season data...")

        # Save as GeoTIFF
        tif_path = OUTPUT_DIR / f'sentinel2_{season}_composite.tif'
        save_merged_geotiff(merged_data, tif_path)

        # Save as NetCDF
        nc_path = OUTPUT_DIR / f'sentinel2_{season}_composite.nc'
        save_netcdf(merged_data, nc_path)

    # Step 6: Generate summary statistics
    stats = {
        'processing_date': datetime.now().isoformat(),
        'files_processed': len(processed_files),
        'spatial_resolution_m': 10,
        'spatial_bounds': MATOPIBA_BOUNDS,
        'seasons': {
            'dry': len([f for f in processed_files if f['season'] == 'dry']),
            'wet': len([f for f in processed_files if f['season'] == 'wet'])
        },
        'years_covered': sorted(list(set(f['year'] for f in processed_files)))
    }

    stats_file = OUTPUT_DIR / 'sentinel2_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n[OK] Summary statistics saved to: {stats_file}")

    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {stats['files_processed']}")
    print(f"Dry season files: {stats['seasons']['dry']}")
    print(f"Wet season files: {stats['seasons']['wet']}")
    print(f"Years covered: {stats['years_covered']}")
    print(f"Spatial resolution: {stats['spatial_resolution_m']}m")

    return processed_files

if __name__ == '__main__':
    processed = main()
    print("\n[INFO] Next step: Process ERA5 meteorological data")
    print("[INFO] Run: python src/preprocessing/process_era5.py")
