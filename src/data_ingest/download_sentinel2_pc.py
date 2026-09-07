# download_sentinel2_pc.py
"""
Download Sentinel-2 NDVI data via Microsoft Planetary Computer.

Two outputs:
1. Monthly NDVI composites (GeoTIFF) - for visualization
2. NDVI values at hotspot locations (CSV) - for ML model

Planetary Computer is free and doesn't require authentication.
"""

import pystac_client
import planetary_computer
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import rasterio
from pyproj import Transformer
import warnings
warnings.filterwarnings('ignore')

# Output directories
OUTPUT_DIR = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'sentinel2'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MATOPIBA bounding box
MATOPIBA_BBOX = [-50, -15, -42, -2]  # west, south, east, north


def get_catalog():
    """Connect to Planetary Computer STAC catalog."""
    return pystac_client.Client.open(
        'https://planetarycomputer.microsoft.com/api/stac/v1',
        modifier=planetary_computer.sign_inplace
    )


def search_sentinel2(catalog, year, month, max_cloud=20):
    """
    Search for Sentinel-2 images for a specific month.

    Args:
        catalog: STAC catalog
        year: Year (2022-2024)
        month: Month (1-12)
        max_cloud: Maximum cloud cover percentage

    Returns:
        List of STAC items
    """
    # Build date range
    start_date = f"{year}-{month:02d}-01"
    if month == 12:
        end_date = f"{year+1}-01-01"
    else:
        end_date = f"{year}-{month+1:02d}-01"

    search = catalog.search(
        collections=['sentinel-2-l2a'],
        bbox=MATOPIBA_BBOX,
        datetime=f"{start_date}/{end_date}",
        query={'eo:cloud_cover': {'lt': max_cloud}}
    )

    return list(search.items())


def calculate_ndvi_from_item(item, bounds, resolution=100):
    """
    Calculate NDVI from a Sentinel-2 item for a specific area.

    Args:
        item: STAC item
        bounds: (west, south, east, north)
        resolution: Output resolution in meters

    Returns:
        ndvi array, transform
    """
    # Get signed URLs for B04 (Red) and B08 (NIR)
    b04_url = item.assets['B04'].href
    b08_url = item.assets['B08'].href

    try:
        with rasterio.open(b04_url) as src_b04:
            with rasterio.open(b08_url) as src_b08:
                # Calculate window from bounds
                window = from_bounds(*bounds, src_b04.transform)

                # Read data
                b04 = src_b04.read(1, window=window).astype(float)
                b08 = src_b08.read(1, window=window).astype(float)

                # Calculate NDVI
                ndvi = np.where(
                    (b08 + b04) > 0,
                    (b08 - b04) / (b08 + b04),
                    np.nan
                )

                return ndvi, src_b04.window_transform(window)

    except Exception as e:
        print(f"    Error reading {item.id}: {e}")
        return None, None


def extract_ndvi_at_points(item, points_df, catalog, sample_size=1000):
    """
    Extract NDVI values at hotspot locations.
    Uses point-by-point tile matching for accuracy.

    Args:
        item: STAC item (fallback, may not be used)
        points_df: DataFrame with latitude, longitude columns
        catalog: STAC catalog for searching tiles
        sample_size: Number of points to sample (for efficiency)

    Returns:
        DataFrame with NDVI values
    """
    # Sample points for efficiency
    if len(points_df) > sample_size:
        sample_df = points_df.sample(n=sample_size, random_state=42)
    else:
        sample_df = points_df.copy()

    ndvi_values = []
    date_range = f"{sample_df['acq_date'].min().strftime('%Y-%m')}-01/{sample_df['acq_date'].max().strftime('%Y-%m')}-28"

    for idx, row in sample_df.iterrows():
        lat, lon = row['latitude'], row['longitude']

        # Search for tile covering this point
        search = catalog.search(
            collections=['sentinel-2-l2a'],
            bbox=[lon-0.01, lat-0.01, lon+0.01, lat+0.01],
            datetime=date_range,
            query={'eo:cloud_cover': {'lt': 30}},
            limit=1
        )

        items = list(search.items())
        if not items:
            continue

        point_item = items[0]

        try:
            b04_url = point_item.assets['B04'].href
            b08_url = point_item.assets['B08'].href

            with rasterio.open(b04_url) as src_b04:
                with rasterio.open(b08_url) as src_b08:
                    # Transform coordinates to image CRS
                    transformer = Transformer.from_crs('EPSG:4326', src_b04.crs, always_xy=True)
                    x_utm, y_utm = transformer.transform(lon, lat)

                    # Sample at transformed coordinates
                    b04_val = list(src_b04.sample([(x_utm, y_utm)]))[0][0]
                    b08_val = list(src_b08.sample([(x_utm, y_utm)]))[0][0]

                    if (b04_val + b08_val) > 0:
                        ndvi = (b08_val - b04_val) / (b08_val + b04_val)
                        ndvi_values.append({
                            'latitude': lat,
                            'longitude': lon,
                            'acq_date': row['acq_date'],
                            'ndvi': float(ndvi),
                            's2_date': point_item.datetime.strftime('%Y-%m-%d')
                        })

        except Exception:
            continue

    return pd.DataFrame(ndvi_values)


def download_sample_images(year=2022, month=9, n_images=3):
    """
    Download sample Sentinel-2 images for visualization.

    Args:
        year: Year
        month: Month
        n_images: Number of images to download
    """
    print(f"\nDownloading sample images for {year}-{month:02d}...")

    catalog = get_catalog()
    items = search_sentinel2(catalog, year, month)

    print(f"  Found {len(items)} images")

    # Take first n images
    for i, item in enumerate(items[:n_images]):
        print(f"  [{i+1}/{n_images}] {item.id}")

        # Download preview (visual band)
        if 'visual' in item.assets:
            preview_url = item.assets['visual'].href
            output_file = OUTPUT_DIR / f"preview_{item.id}.tif"

            try:
                import urllib.request
                urllib.request.urlretrieve(preview_url, str(output_file))
                print(f"    Saved: {output_file.name}")
            except Exception as e:
                print(f"    Error: {e}")


def extract_ndvi_for_hotspots(hotspots_file, year=2022, month=9):
    """
    Extract NDVI values for hotspots from Sentinel-2.

    Args:
        hotspots_file: Path to FIRMS CSV
        year: Year
        month: Month
    """
    print(f"\nExtracting NDVI for hotspots ({year}-{month:02d})...")

    # Load hotspots
    df = pd.read_csv(hotspots_file)
    df['acq_date'] = pd.to_datetime(df['acq_date'])

    # Filter to specific month
    month_df = df[(df['acq_date'].dt.year == year) & (df['acq_date'].dt.month == month)]
    print(f"  Hotspots in {year}-{month:02d}: {len(month_df):,}")

    if len(month_df) == 0:
        print("  No hotspots found")
        return pd.DataFrame()

    # Get Sentinel-2 images
    catalog = get_catalog()
    items = search_sentinel2(catalog, year, month)
    print(f"  Sentinel-2 images: {len(items)}")

    if not items:
        print("  No images found")
        return pd.DataFrame()

    # Use first image as fallback (but function will search per-point)
    item = items[0]
    print(f"  Base tile: {item.id}")

    # Extract NDVI (passes catalog for per-point tile search)
    ndvi_df = extract_ndvi_at_points(item, month_df, catalog, sample_size=2000)

    if not ndvi_df.empty:
        output_file = OUTPUT_DIR / f"ndvi_hotspots_{year}{month:02d}.csv"
        ndvi_df.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")
        print(f"  NDVI stats: min={ndvi_df['ndvi'].min():.2f}, max={ndvi_df['ndvi'].max():.2f}, mean={ndvi_df['ndvi'].mean():.2f}")

    return ndvi_df


def main():
    """Main function."""
    print("=" * 60)
    print("SENTINEL-2 NDVI EXTRACTION - MATOPIBA")
    print("Using Microsoft Planetary Computer")
    print("=" * 60)

    # Path to FIRMS data
    firms_file = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'firms_hotspots' / 'firms_viirs_2022-2024.csv'

    if not firms_file.exists():
        print(f"FIRMS file not found: {firms_file}")
        return

    # Extract NDVI for dry season months
    all_ndvi = []

    for year in [2022, 2023, 2024]:
        for month in [7, 8, 9, 10, 11]:
            ndvi_df = extract_ndvi_for_hotspots(firms_file, year, month)
            if not ndvi_df.empty:
                all_ndvi.append(ndvi_df)

    # Combine all
    if all_ndvi:
        combined = pd.concat(all_ndvi, ignore_index=True)
        output_file = OUTPUT_DIR / 'ndvi_hotspots_all.csv'
        combined.to_csv(output_file, index=False)
        print(f"\nTotal NDVI values: {len(combined):,}")
        print(f"Saved to: {output_file}")

    print("\nDone!")


if __name__ == '__main__':
    main()
