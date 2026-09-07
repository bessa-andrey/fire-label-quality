# download_era5_cds.py
"""
Download ERA5 reanalysis data from Copernicus CDS for MATOPIBA (2022-2024).

CDS = Climate Data Store (Copernicus)
- Direct download using CDS API
- No Google Earth Engine required
- Uses credentials from .env file

ERA5 variables for fire analysis:
- 2m_temperature: Surface temperature
- 2m_dewpoint_temperature: Used to calculate humidity
- 10m_u/v_component_of_wind: Wind speed and direction
- total_precipitation: Precipitation

Output: NetCDF files with meteorological data
"""

import cdsapi
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get CDS credentials
CDS_URL = os.getenv('CDS_URL')
CDS_KEY = os.getenv('CDS_KEY')

if not CDS_KEY:
    raise ValueError("CDS_KEY not found in .env file")

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'era5'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MATOPIBA bounding box [North, West, South, East]
MATOPIBA_AREA = [-2, -50, -15, -42]  # N, W, S, E

# Variables to download
VARIABLES = [
    '2m_temperature',
    '2m_dewpoint_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'total_precipitation',
]


def setup_cds_credentials():
    """
    Create CDS credentials file if needed.
    """
    cds_rc = Path.home() / '.cdsapirc'

    if not cds_rc.exists():
        content = f"""url: {CDS_URL}
key: {CDS_KEY}
"""
        with open(cds_rc, 'w') as f:
            f.write(content)
        print(f"Created CDS credentials file: {cds_rc}")
    else:
        print(f"CDS credentials file exists: {cds_rc}")


def download_era5_monthly(year, month):
    """
    Download ERA5 data for a specific month.

    Args:
        year: Year (2022-2024)
        month: Month (1-12)

    Returns:
        Path to downloaded file
    """
    # Calculate days in month
    if month in [1, 3, 5, 7, 8, 10, 12]:
        days = list(range(1, 32))
    elif month in [4, 6, 9, 11]:
        days = list(range(1, 31))
    else:  # February
        if year % 4 == 0:
            days = list(range(1, 30))
        else:
            days = list(range(1, 29))

    # Convert to strings with zero padding
    days_str = [f"{d:02d}" for d in days]

    output_file = OUTPUT_DIR / f'era5_{year}{month:02d}.nc'

    if output_file.exists():
        print(f"  {year}-{month:02d}: Already exists, skipping")
        return output_file

    print(f"  Downloading ERA5 for {year}-{month:02d}...")

    try:
        client = cdsapi.Client()

        client.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': VARIABLES,
                'year': str(year),
                'month': f'{month:02d}',
                'day': days_str,
                'time': [
                    '00:00', '06:00', '12:00', '18:00'  # 4 times per day
                ],
                'area': MATOPIBA_AREA,
            },
            str(output_file)
        )

        print(f"    Saved: {output_file}")
        return output_file

    except Exception as e:
        print(f"    Error: {e}")
        return None


def main():
    """
    Main function to download ERA5 for MATOPIBA (2022-2024).
    """
    print("=" * 60)
    print("ERA5 DOWNLOAD - MATOPIBA (2022-2024)")
    print("Using Copernicus CDS API")
    print("=" * 60)

    # Setup credentials
    setup_cds_credentials()

    print(f"\nConfiguration:")
    print(f"  Region: MATOPIBA")
    print(f"  Area: N={MATOPIBA_AREA[0]}, W={MATOPIBA_AREA[1]}, S={MATOPIBA_AREA[2]}, E={MATOPIBA_AREA[3]}")
    print(f"  Variables: {len(VARIABLES)}")
    for v in VARIABLES:
        print(f"    - {v}")
    print(f"  Output: {OUTPUT_DIR}")

    # Download for each year/month
    # Focus on dry season first (July-November) - fire season
    downloaded = []

    for year in [2022, 2023, 2024]:
        print(f"\n[Year {year}]")

        for month in range(1, 13):
            result = download_era5_monthly(year, month)
            if result:
                downloaded.append(result)

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Downloaded {len(downloaded)} files")
    print(f"Output directory: {OUTPUT_DIR}")

    return downloaded


if __name__ == '__main__':
    main()
