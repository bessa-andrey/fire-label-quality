# download_mcd64a1.py
"""
Download MODIS MCD64A1 (Burned Area) from Google Earth Engine for MATOPIBA (2022-2024)

MCD64A1: Monthly burned area product
- Resolution: 500m
- Collection: MODIS/061/MCD64A1 (latest version, replaces 006)
- Coverage: 2000-present
- Temporal: Monthly composites

Output: GeoTIFF files with burned area classification
"""

import ee
import geemap
import pandas as pd
from pathlib import Path
from datetime import datetime
import os

# Initialize Earth Engine
ee.Initialize(project='mestrado25')

# Configuration
MATOPIBA_COORDS = [
    [-65, 0],    # NW
    [-40, 0],    # NE
    [-40, -15],  # SE
    [-65, -15],  # SW
    [-65, 0]     # close polygon
]

OUTPUT_DIR = Path('data/raw/mcd64a1')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def create_aoi(coordinates):
    """Create Area of Interest geometry"""
    return ee.Geometry.Polygon(coordinates)

def download_mcd64a1_monthly(year, month, aoi):
    """
    Download MCD64A1 for specific month

    Args:
        year: int (2022-2024)
        month: int (1-12)
        aoi: ee.Geometry

    Returns:
        ee.Image with burned area data
    """
    # Build date range
    start_date = f'{year}-{month:02d}-01'

    # Calculate end of month
    if month == 12:
        end_date = f'{year+1}-01-01'
    else:
        end_date = f'{year}-{month+1:02d}-01'

    print(f"  Downloading MCD64A1 for {start_date}...")

    # Load dataset
    dataset = ee.ImageCollection('MODIS/061/MCD64A1') \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date) \
        .first()

    if dataset is None:
        print(f"    [WARNING] No data for {start_date}")
        return None

    # Select BurnDate band (day of year when burned, 0 = not burned)
    burn_date = dataset.select('BurnDate')

    return burn_date

def download_mcd64a1_year(year, aoi):
    """Download all months for a given year"""
    print(f"\n[INFO] Year: {year}")

    images = []
    dates = []

    for month in range(1, 13):
        img = download_mcd64a1_monthly(year, month, aoi)
        if img is not None:
            images.append(img)
            dates.append(f'{year}-{month:02d}')

    if images:
        # Create image collection from monthly images
        collection = ee.ImageCollection(images)
        print(f"  [OK] Downloaded {len(images)} months")
        return collection, dates
    else:
        return None, []

def export_mcd64a1_to_drive(image, year, month, aoi):
    """
    Export MCD64A1 to Google Drive (to be downloaded later)

    Args:
        image: ee.Image
        year: int
        month: int
        aoi: ee.Geometry
    """
    filename = f'MCD64A1_{year}{month:02d}'

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=filename,
        folder='fireml_data',
        fileNamePrefix=filename,
        scale=500,  # 500m resolution
        region=aoi,
        crs='EPSG:4326',
        maxPixels=1e13
    )

    task.start()
    print(f"    [INFO] Export task started: {filename}")
    return task

def main():
    """Main download function"""
    print("\n" + "="*60)
    print("MODIS MCD64A1 DOWNLOAD - MATOPIBA REGION (2022-2024)")
    print("="*60)

    # Create AOI
    aoi = create_aoi(MATOPIBA_COORDS)
    print(f"\nAOI: MATOPIBA")
    print(f"  Bounds: (-65, 0) to (-40, -15)")
    print(f"  Resolution: 500m")

    all_tasks = []

    # Download for each year
    for year in [2024, 2023, 2022]:
        collection, dates = download_mcd64a1_year(year, aoi)

        if collection:
            # Export each month
            for idx, month in enumerate(range(1, 13)):
                img = download_mcd64a1_monthly(year, month, aoi)
                if img is not None:
                    task = export_mcd64a1_to_drive(img, year, month, aoi)
                    all_tasks.append({
                        'year': year,
                        'month': month,
                        'task': task
                    })

    print(f"\n{'='*60}")
    print(f"[COMPLETE] EXPORT TASKS SUBMITTED!")
    print(f"{'='*60}")
    print(f"Total export tasks: {len(all_tasks)}")
    print(f"Each task will export to your Google Drive folder: 'fireml_data/'")
    print(f"\n[INFO] Processing time: 5-30 minutes per month")
    print(f"[INFO] Check your Google Drive: https://drive.google.com")
    print(f"{'='*60}")

    # Create metadata file
    metadata = {
        'dataset': 'MCD64A1',
        'version': 'MODIS/061',
        'region': 'MATOPIBA',
        'bounds': {
            'west': -65,
            'east': -40,
            'north': 0,
            'south': -15
        },
        'resolution': '500m',
        'period': '2022-2024',
        'export_date': datetime.now().isoformat(),
        'total_tasks': len(all_tasks),
        'tasks': [{'year': t['year'], 'month': t['month']} for t in all_tasks]
    }

    # Save metadata
    import json
    metadata_file = OUTPUT_DIR / 'mcd64a1_export_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[INFO] Metadata saved to: {metadata_file}")

    return all_tasks

if __name__ == '__main__':
    tasks = main()
    print("\n[INFO] Next steps:")
    print("1. Wait for exports to complete on Google Drive")
    print("2. Download files from 'fireml_data/' folder")
    print("3. Place GeoTIFF files in: data/raw/mcd64a1/")
    print("4. Run: python src/preprocessing/process_mcd64a1.py")
