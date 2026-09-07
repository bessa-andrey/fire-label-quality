# download_firms.py
"""
Download FIRMS hotspots from NASA FIRMS API for MATOPIBA region.

FIRMS = Fire Information for Resource Management System
API Documentation: https://firms.modaps.eosdis.nasa.gov/api/

This script downloads active fire detections (hotspots) with:
- latitude, longitude: location of fire detection
- confidence: detection confidence (0-100%)
- frp: Fire Radiative Power in MW
- acq_date, acq_time: acquisition datetime
- satellite: VIIRS-NOAA20, VIIRS-SNPP, MODIS

Output: CSV file with all hotspots for MATOPIBA region (2022-2024)
"""

import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from .env
FIRMS_MAP_KEY = os.getenv('FIRMS_MAP_KEY')
if not FIRMS_MAP_KEY:
    raise ValueError("FIRMS_MAP_KEY not found in .env file")

# MATOPIBA bounding box (approximate)
# Maranhao, Tocantins, Piaui, Bahia
MATOPIBA_BOUNDS = {
    'west': -50.0,
    'south': -15.0,
    'east': -42.0,
    'north': -2.0
}

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'firms_hotspots'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# FIRMS API base URL
FIRMS_API_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


def download_firms_area(source, days_back=10):
    """
    Download FIRMS hotspots for MATOPIBA region.

    Args:
        source: 'VIIRS_NOAA20_NRT', 'VIIRS_SNPP_NRT', or 'MODIS_NRT'
        days_back: number of days to download (max 10 for NRT)

    Returns:
        pandas DataFrame with hotspot data
    """
    # Build API URL
    # Format: /api/area/csv/{MAP_KEY}/{source}/{west},{south},{east},{north}/{days}
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{FIRMS_MAP_KEY}/{source}/"
        f"{MATOPIBA_BOUNDS['west']},{MATOPIBA_BOUNDS['south']},"
        f"{MATOPIBA_BOUNDS['east']},{MATOPIBA_BOUNDS['north']}/"
        f"{days_back}"
    )

    print(f"Downloading {source} hotspots...")
    print(f"Region: MATOPIBA")
    print(f"Period: last {days_back} days")

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        # Parse CSV response
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))

        print(f"Downloaded {len(df)} hotspots")
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {source}: {e}")
        return pd.DataFrame()


def download_firms_archive(source, start_date, end_date):
    """
    Download historical FIRMS data for a date range.

    Args:
        source: 'VIIRS_NOAA20_SP', 'VIIRS_SNPP_SP', or 'MODIS_SP'
        start_date: datetime object
        end_date: datetime object

    Returns:
        pandas DataFrame with all hotspots
    """
    from io import StringIO
    import time

    all_data = []
    current_date = start_date

    # Calculate total days for progress
    total_days = (end_date - start_date).days
    days_processed = 0

    print(f"\nDownloading {source} archive data...")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Total days: {total_days}")
    print("-" * 40)

    while current_date < end_date:
        # Download 10 days at a time
        days_to_download = min(10, (end_date - current_date).days)
        date_str = current_date.strftime("%Y-%m-%d")

        # Build archive URL
        url = (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
            f"{FIRMS_MAP_KEY}/{source}/"
            f"{MATOPIBA_BOUNDS['west']},{MATOPIBA_BOUNDS['south']},"
            f"{MATOPIBA_BOUNDS['east']},{MATOPIBA_BOUNDS['north']}/"
            f"{days_to_download}/{date_str}"
        )

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            # Parse CSV
            if response.text.strip() and 'latitude' in response.text:
                df = pd.read_csv(StringIO(response.text))
                all_data.append(df)
                count = len(df)
            else:
                count = 0

            days_processed += days_to_download
            progress = (days_processed / total_days) * 100
            print(f"  {date_str}: {count} hotspots ({progress:.0f}%)")

        except requests.exceptions.RequestException as e:
            print(f"  {date_str}: Error - {e}")

        # Move to next period
        current_date += timedelta(days=days_to_download)

        # Small delay to respect API limits
        time.sleep(0.5)

    # Combine all data
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal hotspots downloaded: {len(combined)}")
        return combined
    else:
        print("\nNo data downloaded")
        return pd.DataFrame()


def main(start_year=2019, end_year=2024):
    """
    Main function to download FIRMS data for MATOPIBA.

    Args:
        start_year: First year to download (default 2019 for expanded dataset)
        end_year: Last year to download (default 2024)
    """
    print("=" * 60)
    print(f"FIRMS HOTSPOT DOWNLOAD - MATOPIBA ({start_year}-{end_year})")
    print("=" * 60)

    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)

    all_data = []

    # VIIRS SNPP covers 2012+ (use for years before NOAA-20)
    if start_year < 2018:
        print("\n[INFO] VIIRS NOAA-20 not available before 2018, using VIIRS SNPP")
        snpp_end = datetime(min(2017, end_year), 12, 31)
        snpp_data = download_firms_archive('VIIRS_SNPP_SP', start_date, snpp_end)
        if not snpp_data.empty:
            snpp_data['source_sensor'] = 'VIIRS_SNPP'
            all_data.append(snpp_data)
        noaa20_start = datetime(2018, 1, 1)
    else:
        noaa20_start = start_date

    # VIIRS NOAA-20 (2018+)
    if end_year >= 2018:
        noaa20_data = download_firms_archive('VIIRS_NOAA20_SP', noaa20_start, end_date)
        if not noaa20_data.empty:
            noaa20_data['source_sensor'] = 'VIIRS_NOAA20'
            all_data.append(noaa20_data)

    # Combine all data
    if all_data:
        viirs_data = pd.concat(all_data, ignore_index=True)

        # Remove exact duplicates
        before_dedup = len(viirs_data)
        viirs_data = viirs_data.drop_duplicates(
            subset=['latitude', 'longitude', 'acq_date', 'acq_time'],
            keep='first'
        )
        after_dedup = len(viirs_data)
        if before_dedup > after_dedup:
            print(f"[INFO] Removed {before_dedup - after_dedup} duplicate detections")

        output_file = OUTPUT_DIR / f'firms_viirs_{start_year}-{end_year}.csv'
        viirs_data.to_csv(output_file, index=False)
        print(f"\nSaved to: {output_file}")

        print("\n" + "=" * 60)
        print("DOWNLOAD COMPLETE")
        print("=" * 60)
        print(f"Total hotspots: {len(viirs_data)}")
        print(f"Columns: {viirs_data.columns.tolist()}")
        print(f"Date range: {viirs_data['acq_date'].min()} to {viirs_data['acq_date'].max()}")
        print(f"Output: {output_file}")

        return viirs_data
    else:
        print("\nNo data downloaded!")
        return pd.DataFrame()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download FIRMS data for MATOPIBA')
    parser.add_argument('--start-year', type=int, default=2019, help='Start year (default: 2019)')
    parser.add_argument('--end-year', type=int, default=2024, help='End year (default: 2024)')
    args = parser.parse_args()
    main(start_year=args.start_year, end_year=args.end_year)
