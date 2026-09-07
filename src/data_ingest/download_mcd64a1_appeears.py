# download_mcd64a1_appeears.py
"""
Download MODIS MCD64A1 (Burned Area) via NASA AppEEARS API.

AppEEARS = Application for Extracting and Exploring Analysis Ready Samples
- Uses Earthdata credentials (same as in .env)
- Downloads directly to local disk
- No Google Earth Engine required

MCD64A1 product:
- Monthly burned area from MODIS Terra+Aqua
- Resolution: 500m
- Band: BurnDate (day of year, 0 = not burned)
"""

import requests
import time
import os
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Earthdata credentials
EARTHDATA_USER = os.getenv('EARTHDATA_USERNAME')
EARTHDATA_PASS = os.getenv('EARTHDATA_PASSWORD')

if not EARTHDATA_USER or not EARTHDATA_PASS:
    raise ValueError("EARTHDATA_USERNAME and EARTHDATA_PASSWORD required in .env")

# AppEEARS API endpoints
API_BASE = "https://appeears.earthdatacloud.nasa.gov/api"

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'mcd64a1'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MATOPIBA bounding box
MATOPIBA_BOUNDS = {
    'west': -50.0,
    'south': -15.0,
    'east': -42.0,
    'north': -2.0
}


def get_token():
    """
    Authenticate with AppEEARS and get bearer token.
    """
    print("Authenticating with NASA AppEEARS...")

    url = f"{API_BASE}/login"
    response = requests.post(url, auth=(EARTHDATA_USER, EARTHDATA_PASS))

    if response.status_code == 200:
        token = response.json()['token']
        print("  Authentication successful!")
        return token
    else:
        print(f"  Authentication failed: {response.status_code}")
        print(f"  Response: {response.text}")
        return None


def submit_task(token, start_date, end_date, task_name):
    """
    Submit an area request to AppEEARS.

    Args:
        token: Bearer token
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        task_name: Name for this task

    Returns:
        task_id if successful
    """
    print(f"\nSubmitting task: {task_name}")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Region: MATOPIBA")

    # Build task request
    task = {
        "task_type": "area",
        "task_name": task_name,
        "params": {
            "dates": [
                {
                    "startDate": start_date,
                    "endDate": end_date
                }
            ],
            "layers": [
                {
                    "product": "MCD64A1.061",
                    "layer": "Burn_Date"
                }
            ],
            "output": {
                "format": {
                    "type": "geotiff"
                },
                "projection": "geographic"
            },
            "geo": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [MATOPIBA_BOUNDS['west'], MATOPIBA_BOUNDS['north']],
                                [MATOPIBA_BOUNDS['east'], MATOPIBA_BOUNDS['north']],
                                [MATOPIBA_BOUNDS['east'], MATOPIBA_BOUNDS['south']],
                                [MATOPIBA_BOUNDS['west'], MATOPIBA_BOUNDS['south']],
                                [MATOPIBA_BOUNDS['west'], MATOPIBA_BOUNDS['north']]
                            ]]
                        }
                    }
                ]
            }
        }
    }

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    response = requests.post(f"{API_BASE}/task", headers=headers, json=task)

    if response.status_code == 202:
        task_id = response.json()['task_id']
        print(f"  Task submitted! ID: {task_id}")
        return task_id
    else:
        print(f"  Failed to submit task: {response.status_code}")
        print(f"  Response: {response.text}")
        return None


def check_task_status(token, task_id):
    """
    Check the status of a submitted task.
    """
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_BASE}/task/{task_id}", headers=headers)

    if response.status_code == 200:
        return response.json()
    return None


def wait_for_task(token, task_id, check_interval=30):
    """
    Wait for task to complete.

    Args:
        token: Bearer token
        task_id: Task ID
        check_interval: Seconds between status checks
    """
    print(f"\nWaiting for task {task_id} to complete...")
    print("  (This may take 10-30 minutes for large requests)")

    while True:
        status = check_task_status(token, task_id)

        if status is None:
            print("  Error checking status")
            return False

        task_status = status.get('status', 'unknown')
        progress = status.get('progress', 0)

        print(f"  Status: {task_status} ({progress}%)")

        if task_status == 'done':
            print("  Task completed!")
            return True
        elif task_status == 'error':
            print(f"  Task failed: {status.get('error', 'Unknown error')}")
            return False

        time.sleep(check_interval)


def get_task_files(token, task_id):
    """
    Get list of files from completed task.
    """
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_BASE}/bundle/{task_id}", headers=headers)

    if response.status_code == 200:
        return response.json()['files']
    return []


def download_files(token, task_id, output_dir):
    """
    Download all files from a completed task.
    """
    files = get_task_files(token, task_id)

    if not files:
        print("No files to download")
        return []

    print(f"\nDownloading {len(files)} files...")

    headers = {"Authorization": f"Bearer {token}"}
    downloaded = []

    for file_info in files:
        file_id = file_info['file_id']
        file_name = file_info['file_name']

        # Skip non-tif files
        if not file_name.endswith('.tif'):
            continue

        print(f"  Downloading: {file_name}")

        response = requests.get(
            f"{API_BASE}/bundle/{task_id}/{file_id}",
            headers=headers,
            stream=True
        )

        if response.status_code == 200:
            output_path = output_dir / file_name
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            downloaded.append(output_path)
            print(f"    Saved: {output_path}")
        else:
            print(f"    Failed: {response.status_code}")

    return downloaded


def main():
    """
    Main function to download MCD64A1 for MATOPIBA (2022-2024).
    """
    print("=" * 60)
    print("MCD64A1 DOWNLOAD - MATOPIBA (2022-2024)")
    print("Using NASA AppEEARS API")
    print("=" * 60)

    # Authenticate
    token = get_token()
    if not token:
        print("Failed to authenticate. Check your Earthdata credentials.")
        return

    # Submit task for entire period
    # Note: AppEEARS can handle multi-year requests
    task_id = submit_task(
        token,
        start_date="01-01-2022",
        end_date="12-31-2024",
        task_name="MCD64A1_MATOPIBA_2022-2024"
    )

    if not task_id:
        print("Failed to submit task")
        return

    # Save task info for later (in case we need to resume)
    task_info = {
        'task_id': task_id,
        'submitted': datetime.now().isoformat(),
        'product': 'MCD64A1.061',
        'region': 'MATOPIBA',
        'period': '2022-2024'
    }

    task_file = OUTPUT_DIR / 'appeears_task.json'
    with open(task_file, 'w') as f:
        json.dump(task_info, f, indent=2)
    print(f"\nTask info saved to: {task_file}")

    # Wait for completion
    print("\n" + "-" * 60)
    print("IMPORTANT: The task is now processing on NASA servers.")
    print("This can take 10-30 minutes for 3 years of data.")
    print("-" * 60)

    user_input = input("\nWait for completion now? (y/n): ").strip().lower()

    if user_input == 'y':
        if wait_for_task(token, task_id):
            # Download files
            downloaded = download_files(token, task_id, OUTPUT_DIR)

            print("\n" + "=" * 60)
            print("DOWNLOAD COMPLETE")
            print("=" * 60)
            print(f"Downloaded {len(downloaded)} files to: {OUTPUT_DIR}")
        else:
            print("\nTask did not complete successfully")
    else:
        print("\nTask submitted. Run this script again to check status and download.")
        print(f"Task ID: {task_id}")


def check_and_download(task_id=None):
    """
    Check status of existing task and download if ready.
    """
    # Load task info if not provided
    if task_id is None:
        task_file = OUTPUT_DIR / 'appeears_task.json'
        if task_file.exists():
            with open(task_file) as f:
                task_info = json.load(f)
            task_id = task_info['task_id']
        else:
            print("No pending task found. Run main() first.")
            return

    print(f"Checking task: {task_id}")

    # Authenticate
    token = get_token()
    if not token:
        return

    # Check status
    status = check_task_status(token, task_id)
    if status:
        print(f"Status: {status.get('status')} ({status.get('progress', 0)}%)")

        if status['status'] == 'done':
            downloaded = download_files(token, task_id, OUTPUT_DIR)
            print(f"\nDownloaded {len(downloaded)} files")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        # Check existing task
        check_and_download()
    else:
        # Submit new task
        main()
