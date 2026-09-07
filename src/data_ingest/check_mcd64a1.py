# check_mcd64a1.py
"""
Check MCD64A1 AppEEARS task status and download when ready.

Run periodically to check if NASA finished processing.
When done, automatically downloads the GeoTIFF files.
"""

import requests
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Credentials
EARTHDATA_USER = os.getenv('EARTHDATA_USERNAME')
EARTHDATA_PASS = os.getenv('EARTHDATA_PASSWORD')

# API
API_BASE = "https://appeears.earthdatacloud.nasa.gov/api"

# Output
OUTPUT_DIR = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'mcd64a1'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Task file
TASK_FILE = OUTPUT_DIR / 'appeears_task.json'


def get_token():
    """Authenticate with AppEEARS."""
    response = requests.post(f"{API_BASE}/login", auth=(EARTHDATA_USER, EARTHDATA_PASS))
    if response.status_code == 200:
        return response.json()['token']
    return None


def check_status(token, task_id):
    """Check task status."""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_BASE}/task/{task_id}", headers=headers)
    if response.status_code == 200:
        return response.json()
    return None


def get_files(token, task_id):
    """Get list of files from completed task."""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_BASE}/bundle/{task_id}", headers=headers)
    if response.status_code == 200:
        return response.json().get('files', [])
    return []


def download_files(token, task_id):
    """Download all GeoTIFF files from task."""
    files = get_files(token, task_id)

    if not files:
        print("No files to download")
        return []

    tif_files = [f for f in files if f['file_name'].endswith('.tif')]
    print(f"\nDownloading {len(tif_files)} GeoTIFF files...")

    headers = {"Authorization": f"Bearer {token}"}
    downloaded = []

    for i, file_info in enumerate(tif_files, 1):
        file_id = file_info['file_id']
        file_name = file_info['file_name']

        print(f"  [{i}/{len(tif_files)}] {file_name}")

        response = requests.get(
            f"{API_BASE}/bundle/{task_id}/{file_id}",
            headers=headers,
            stream=True
        )

        if response.status_code == 200:
            output_path = OUTPUT_DIR / file_name
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            downloaded.append(output_path)
        else:
            print(f"    Failed: {response.status_code}")

    return downloaded


def main():
    """Check status and download if ready."""
    print("=" * 50)
    print("MCD64A1 STATUS CHECK")
    print("=" * 50)

    # Load task ID
    if not TASK_FILE.exists():
        print("No task file found. Run download_mcd64a1_appeears.py first.")
        return

    with open(TASK_FILE) as f:
        task_info = json.load(f)

    task_id = task_info['task_id']
    print(f"Task ID: {task_id}")

    # Authenticate
    token = get_token()
    if not token:
        print("Authentication failed")
        return

    # Check status
    status = check_status(token, task_id)
    if not status:
        print("Failed to get status")
        return

    task_status = status.get('status', 'unknown')
    progress = status.get('progress', 0)

    print(f"Status: {task_status}")
    print(f"Progress: {progress}%")

    if task_status == 'done':
        print("\nTask completed! Downloading files...")
        downloaded = download_files(token, task_id)
        print(f"\nDownloaded {len(downloaded)} files to {OUTPUT_DIR}")
    elif task_status == 'error':
        print(f"\nTask failed: {status.get('error', 'Unknown error')}")
    else:
        print(f"\nStill processing. Check again later.")
        print("Run: python src/data_ingest/check_mcd64a1.py")


if __name__ == '__main__':
    main()
