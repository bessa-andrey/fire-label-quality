# download_sentinel2.py
"""
Download Sentinel-2 imagery from Google Earth Engine for MATOPIBA dry season (2022-2024)

Sentinel-2: High-resolution optical imagery
- Resolution: 10m (B4=Red, B3=Green, B2=Blue, B8=NIR)
- Coverage: 2015-present
- Revisit time: 5 days
- Cloud filter: <20% cloud cover

Output: GeoTIFF composites (RGB + NDVI)
"""

import ee
import json
from pathlib import Path
from datetime import datetime

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

OUTPUT_DIR = Path('data/raw/sentinel2')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def create_aoi(coordinates):
    """Create Area of Interest geometry"""
    return ee.Geometry.Polygon(coordinates)

def calculate_ndvi(image):
    """Calculate NDVI from Sentinel-2 image"""
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

def download_sentinel2_composite(year, season, aoi):
    """
    Download Sentinel-2 composite for dry season

    Args:
        year: int (2022-2024)
        season: 'dry' (July-October) or 'wet' (November-June)
        aoi: ee.Geometry

    Returns:
        ee.Image
    """
    if season == 'dry':
        start_date = f'{year}-07-01'
        end_date = f'{year}-10-31'
        season_name = 'Dry season (Jul-Oct)'
    elif season == 'wet':
        start_date = f'{year}-11-01'
        end_date = f'{year+1}-06-30'
        season_name = 'Wet season (Nov-Jun)'
    else:
        raise ValueError("Season must be 'dry' or 'wet'")

    print(f"  Downloading Sentinel-2 composite ({season_name})...")

    # Load Sentinel-2 Level 2A (atmospherically corrected)
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .map(calculate_ndvi) \
        .median()  # Median composite to reduce clouds/noise

    return s2

def export_sentinel2_to_drive(image, year, season, aoi):
    """
    Export Sentinel-2 composite to Google Drive

    Args:
        image: ee.Image
        year: int
        season: str ('dry' or 'wet')
        aoi: ee.Geometry
    """
    filename = f'Sentinel2_{year}_{season}'

    # Select RGB + NDVI bands
    export_image = image.select(['B4', 'B3', 'B2', 'NDVI'])

    task = ee.batch.Export.image.toDrive(
        image=export_image,
        description=filename,
        folder='fireml_data',
        fileNamePrefix=filename,
        scale=10,  # 10m resolution
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
    print("SENTINEL-2 DOWNLOAD - MATOPIBA REGION (2022-2024)")
    print("="*60)

    # Create AOI
    aoi = create_aoi(MATOPIBA_COORDS)
    print(f"\nAOI: MATOPIBA")
    print(f"  Bounds: (-65, 0) to (-40, -15)")
    print(f"  Resolution: 10m")
    print(f"  Cloud filter: <20%")

    all_tasks = []

    # Download for each year - focus on dry season
    for year in [2024, 2023, 2022]:
        print(f"\n[INFO] Year: {year}")

        # Download dry season composite
        s2_dry = download_sentinel2_composite(year, 'dry', aoi)
        if s2_dry is not None:
            task = export_sentinel2_to_drive(s2_dry, year, 'dry', aoi)
            all_tasks.append({
                'year': year,
                'season': 'dry',
                'task': task
            })

        # Optional: Also download wet season for comparison
        s2_wet = download_sentinel2_composite(year, 'wet', aoi)
        if s2_wet is not None:
            task = export_sentinel2_to_drive(s2_wet, year, 'wet', aoi)
            all_tasks.append({
                'year': year,
                'season': 'wet',
                'task': task
            })

    print(f"\n{'='*60}")
    print(f"[COMPLETE] EXPORT TASKS SUBMITTED!")
    print(f"{'='*60}")
    print(f"Total export tasks: {len(all_tasks)}")
    print(f"Each task will export to your Google Drive folder: 'fireml_data/'")
    print(f"\n[INFO] Processing time: 10-60 minutes per composite")
    print(f"[INFO] Check your Google Drive: https://drive.google.com")
    print(f"{'='*60}")

    # Create metadata file
    metadata = {
        'dataset': 'Sentinel-2',
        'level': 'Level 2A (atmospherically corrected)',
        'bands': {
            'B2': 'Blue (490nm, 10m)',
            'B3': 'Green (560nm, 10m)',
            'B4': 'Red (665nm, 10m)',
            'B8': 'NIR (842nm, 10m)'
        },
        'derived': {
            'NDVI': 'Normalized Difference Vegetation Index = (B8-B4)/(B8+B4)'
        },
        'region': 'MATOPIBA',
        'bounds': {
            'west': -65,
            'east': -40,
            'north': 0,
            'south': -15
        },
        'resolution': '10m',
        'period': '2022-2024',
        'cloud_filter': '<20%',
        'export_date': datetime.now().isoformat(),
        'total_tasks': len(all_tasks),
        'tasks': [{'year': t['year'], 'season': t['season']} for t in all_tasks]
    }

    # Save metadata
    metadata_file = OUTPUT_DIR / 'sentinel2_export_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[INFO] Metadata saved to: {metadata_file}")

    return all_tasks

if __name__ == '__main__':
    tasks = main()
    print("\n[INFO] Next steps:")
    print("1. Wait for exports to complete on Google Drive")
    print("2. Download files from 'fireml_data/' folder")
    print("3. Place GeoTIFF files in: data/raw/sentinel2/")
    print("4. Run: python src/preprocessing/process_sentinel2.py")
