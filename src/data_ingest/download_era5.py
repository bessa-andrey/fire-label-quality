# download_era5.py
"""
Download ERA5 reanalysis data from Copernicus CDS for MATOPIBA (2022-2024)

ERA5: Hourly data on single levels from 1979 to present
- Resolution: 0.25° (approximately 25km at equator)
- Temporal: Hourly (0-23h)
- Variables: temperature, wind, humidity, precipitation, etc.
- Lag: 5 days (quasi real-time)

Output: NetCDF files with meteorological variables
"""

import ee
import os
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json

# Initialize GEE as fallback for ERA5
ee.Initialize()

# Load credentials (for future CDS use)
load_dotenv()
cds_key = os.getenv('CDS_KEY')

# Configuration
MATOPIBA_BOUNDS = {
    'north': 0,
    'south': -15,
    'east': -40,
    'west': -65
}

OUTPUT_DIR = Path('data/raw/era5')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Key variables for fire analysis
ERA5_VARIABLES = [
    '2m_temperature',           # Surface temperature (K)
    '2m_dewpoint_temperature',  # Dew point (K)
    '10m_u_component_of_wind',  # U wind component (m/s)
    '10m_v_component_of_wind',  # V wind component (m/s)
    'total_precipitation',      # Precipitation (mm)
    'soil_moisture_0_7cm',      # Soil moisture (m³/m³)
    'surface_solar_radiation_downwards',  # Solar radiation (J/m²)
]

def export_era5_via_gee(year, month):
    """
    Export ERA5 via Google Earth Engine to Google Drive
    (CDS API has compatibility issues with current setup)

    Args:
        year: int (2022-2024)
        month: int (1-12)

    Returns:
        Task info dict
    """
    print(f"  Submitting ERA5 export for {year}-{month:02d} via GEE...")

    # Create geometry for MATOPIBA
    geometry = ee.Geometry.BBox(
        west=MATOPIBA_BOUNDS['west'],
        south=MATOPIBA_BOUNDS['south'],
        east=MATOPIBA_BOUNDS['east'],
        north=MATOPIBA_BOUNDS['north']
    )

    # Build date range
    start_date = f'{year}-{month:02d}-01'
    if month == 12:
        end_date = f'{year+1}-01-01'
    else:
        end_date = f'{year}-{month+1:02d}-01'

    try:
        # Access ERA5 dataset from GEE
        era5 = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY') \
            .filterDate(start_date, end_date) \
            .filterBounds(geometry) \
            .first()

        if era5 is None:
            print(f"    [WARNING] No data for {year}-{month:02d}")
            return None

        # Export to Google Drive
        filename = f'ERA5_{year}{month:02d}'
        task = ee.batch.Export.image.toDrive(
            image=era5,
            description=filename,
            folder='fireml_data',
            fileNamePrefix=filename,
            scale=27830,  # ~1 degree at equator
            region=geometry,
            crs='EPSG:4326',
            maxPixels=1e13
        )

        task.start()
        print(f"    [INFO] Export task started: {filename}")
        return {
            'year': year,
            'month': month,
            'filename': filename,
            'task': task
        }

    except Exception as e:
        print(f"    [WARNING] Error submitting task: {e}")
        return None

def download_era5_year(year):
    """Export ERA5 for all months for a given year"""
    print(f"\n[INFO] Year: {year}")

    exported_tasks = []

    for month in range(1, 13):
        # Focus on dry season (July-October) initially
        if month >= 7 and month <= 10:
            task_info = export_era5_via_gee(year, month)
            if task_info:
                exported_tasks.append(task_info)

    return exported_tasks

def main():
    """Main export function"""
    print("\n" + "="*60)
    print("ERA5 EXPORT - MATOPIBA REGION (2022-2024)")
    print("="*60)

    print(f"\nConfiguration:")
    print(f"  Region: MATOPIBA ({MATOPIBA_BOUNDS})")
    print(f"  Resolution: 0.25 degrees (~25km)")
    print(f"  Temporal: Monthly")
    print(f"  Dataset: ECMWF ERA5 Land Monthly")
    print(f"  Method: Google Earth Engine Export to Google Drive")

    all_tasks = []

    # Export for each year - focus on dry season
    for year in [2024, 2023, 2022]:
        tasks = download_era5_year(year)
        all_tasks.extend(tasks)

    print(f"\n{'='*60}")
    print(f"[COMPLETE] EXPORT TASKS SUBMITTED!")
    print(f"{'='*60}")
    print(f"Total export tasks: {len(all_tasks)}")
    print(f"Each task will export to your Google Drive folder: 'fireml_data/'")
    print(f"\n[INFO] Processing time: 5-30 minutes per month")
    print(f"[INFO] Check your Google Drive: https://drive.google.com")

    # Create metadata file
    metadata = {
        'dataset': 'ERA5',
        'source': 'Google Earth Engine (ECMWF ERA5 Land)',
        'product': 'ERA5 Land Monthly',
        'variables': ERA5_VARIABLES,
        'region': 'MATOPIBA',
        'bounds': MATOPIBA_BOUNDS,
        'resolution': '0.25 degrees (~25km)',
        'temporal_resolution': 'Monthly',
        'period': '2022-2024',
        'focus_months': 'Dry season (July-October)',
        'export_date': datetime.now().isoformat(),
        'total_tasks': len(all_tasks),
        'tasks': [{'year': t['year'], 'month': t['month']} for t in all_tasks]
    }

    # Save metadata
    metadata_file = OUTPUT_DIR / 'era5_export_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Metadata saved to: {metadata_file}")

    return all_tasks

if __name__ == '__main__':
    tasks = main()
    print("\n[INFO] Next steps:")
    print("1. Wait for exports to complete on Google Drive (5-30 minutes per month)")
    print("2. Download files from 'fireml_data/' folder")
    print("3. Place NetCDF files in: data/raw/era5/")
    print("4. Run: python src/preprocessing/process_era5.py")
