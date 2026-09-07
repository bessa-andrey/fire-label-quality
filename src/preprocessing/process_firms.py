# process_firms.py
"""
Process FIRMS hotspot CSV data for Module A (Spurious Detection)

Input: firms_combined_2022-2024.csv (raw hotspots from NASA)
Output:
  - firms_processed.csv (cleaned hotspots with temporal aggregation)
  - firms_stats.json (processing statistics)
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# Configuration
INPUT_DIR = Path('data/raw/firms_hotspots')
OUTPUT_DIR = Path('data/processed/firms')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MATOPIBA bounds
MATOPIBA_BOUNDS = {
    'north': 0,
    'south': -15,
    'east': -40,
    'west': -65
}

def load_firms_data(filename='firms_viirs_2022-2024.csv'):
    """Load and validate FIRMS CSV data"""
    print(f"[INFO] Loading FIRMS data: {filename}")

    try:
        df = pd.read_csv(INPUT_DIR / filename)
        print(f"[OK] Loaded {len(df)} records")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filename}")
        return None

def validate_geometry(df):
    """Remove hotspots outside MATOPIBA bounds"""
    print(f"[INFO] Validating geometry...")

    initial_count = len(df)

    # Filter by MATOPIBA bounds
    df_filtered = df[
        (df['latitude'] >= MATOPIBA_BOUNDS['south']) &
        (df['latitude'] <= MATOPIBA_BOUNDS['north']) &
        (df['longitude'] >= MATOPIBA_BOUNDS['west']) &
        (df['longitude'] <= MATOPIBA_BOUNDS['east'])
    ].copy()

    removed = initial_count - len(df_filtered)
    print(f"[OK] Removed {removed} hotspots outside MATOPIBA bounds")

    return df_filtered

def parse_dates(df):
    """Parse acquisition dates to datetime"""
    print(f"[INFO] Parsing dates...")

    # Convert acq_date to datetime
    df['acq_datetime'] = pd.to_datetime(
        df['acq_date'].astype(str),
        format='%Y-%m-%d',
        errors='coerce'
    )

    # Add year and month columns
    df['year'] = df['acq_datetime'].dt.year
    df['month'] = df['acq_datetime'].dt.month
    df['dayofyear'] = df['acq_datetime'].dt.dayofyear

    # Check for invalid dates
    invalid_dates = df['acq_datetime'].isna().sum()
    if invalid_dates > 0:
        print(f"[WARNING] {invalid_dates} records with invalid dates")
        df = df.dropna(subset=['acq_datetime'])

    print(f"[OK] Date parsing complete. Temporal range: {df['year'].min()}-{df['year'].max()}")
    return df

def calculate_confidence_score(df):
    """Standardize confidence score (0-100 scale)"""
    print(f"[INFO] Standardizing confidence scores...")

    # VIIRS FIRMS has categorical confidence: 'l'=low, 'n'=nominal, 'h'=high
    # Convert to numeric scale (0-100)
    confidence_map = {
        'l': 30,   # low
        'n': 70,   # nominal
        'h': 90    # high
    }

    # Check if confidence is categorical or numeric
    if df['confidence'].dtype == 'object':
        df['confidence_numeric'] = df['confidence'].map(confidence_map)
        # Fill unmapped with nominal
        df['confidence_numeric'] = df['confidence_numeric'].fillna(70)
        df['confidence'] = df['confidence_numeric']
        df = df.drop(columns=['confidence_numeric'])
        print(f"[OK] Converted categorical confidence to numeric scale")
    else:
        # Already numeric
        df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
        median_conf = df['confidence'].median()
        df['confidence'] = df['confidence'].fillna(median_conf)

    print(f"[OK] Confidence: min={df['confidence'].min()}, max={df['confidence'].max()}, mean={df['confidence'].mean():.1f}")
    return df

def aggregate_temporal(df, time_window_days=3):
    """
    Aggregate hotspots by temporal/spatial proximity
    Groups nearby hotspots within same date window
    """
    print(f"[INFO] Aggregating hotspots within {time_window_days}-day windows...")

    # Sort by date and location
    df = df.sort_values(['acq_datetime', 'latitude', 'longitude'])

    # Create grid cells (0.1 degree = ~11km)
    grid_size = 0.1
    df['grid_lat'] = (df['latitude'] / grid_size).round(0) * grid_size
    df['grid_lon'] = (df['longitude'] / grid_size).round(0) * grid_size

    # Create temporal groups
    df['date_group'] = df['acq_datetime'].dt.floor('D')

    # Aggregate: count hotspots per grid cell per date
    # Note: VIIRS uses 'bright_ti4' (thermal infrared band 4) instead of 'brightness'
    brightness_col = 'bright_ti4' if 'bright_ti4' in df.columns else 'brightness'

    aggregated = df.groupby(['grid_lat', 'grid_lon', 'date_group']).agg({
        brightness_col: 'mean',
        'confidence': 'mean',
        'latitude': 'mean',
        'longitude': 'mean',
        'satellite': 'first',
        'frp': 'sum',  # Fire Radiative Power total
        'acq_datetime': 'first'
    }).reset_index()

    # Rename to standard column name
    if brightness_col != 'brightness':
        aggregated = aggregated.rename(columns={brightness_col: 'brightness'})

    # Add count
    aggregated['hotspot_count'] = df.groupby(['grid_lat', 'grid_lon', 'date_group']).size().values

    # Add year/month from aggregated datetime
    aggregated['year'] = aggregated['acq_datetime'].dt.year
    aggregated['month'] = aggregated['acq_datetime'].dt.month

    print(f"[OK] Aggregated {len(df)} records into {len(aggregated)} grid cells")

    return aggregated[['latitude', 'longitude', 'acq_datetime', 'brightness', 'confidence',
                       'satellite', 'frp', 'hotspot_count', 'year', 'month']]

def filter_confidence(df, min_confidence=30):
    """Remove low-confidence detections (likely false positives)"""
    print(f"[INFO] Filtering by confidence threshold: {min_confidence}%")

    initial_count = len(df)
    df_filtered = df[df['confidence'] >= min_confidence].copy()
    removed = initial_count - len(df_filtered)

    print(f"[OK] Removed {removed} low-confidence detections")
    return df_filtered

def calculate_persistence(df, window_days=7):
    """
    Calculate persistence metric using efficient grid-based approach.
    Counts hotspots in same grid cell across time window.
    """
    print(f"[INFO] Calculating persistence metric ({window_days}-day window)...")

    # Use grid cells for efficient spatial grouping (0.2 degrees = ~22km)
    grid_size = 0.2
    df['persist_grid_lat'] = (df['latitude'] / grid_size).round(0)
    df['persist_grid_lon'] = (df['longitude'] / grid_size).round(0)

    # Create week number for temporal grouping
    df['week'] = df['acq_datetime'].dt.isocalendar().week

    # Count hotspots per grid cell per week
    grid_counts = df.groupby(['persist_grid_lat', 'persist_grid_lon', 'week']).size().reset_index(name='week_count')

    # Merge back to get counts
    df = df.merge(grid_counts, on=['persist_grid_lat', 'persist_grid_lon', 'week'], how='left')

    # Also count total detections in grid cell across all time
    total_grid = df.groupby(['persist_grid_lat', 'persist_grid_lon']).size().reset_index(name='total_grid_count')
    df = df.merge(total_grid, on=['persist_grid_lat', 'persist_grid_lon'], how='left')

    # Persistence = sqrt(week_count * total_grid_count) - geometric mean
    df['persistence_count'] = np.sqrt(df['week_count'] * df['total_grid_count'])

    # Normalize to 0-1 scale
    max_persistence = df['persistence_count'].max()
    df['persistence_score'] = df['persistence_count'] / max(max_persistence, 1)

    # Clean up temporary columns
    df = df.drop(columns=['persist_grid_lat', 'persist_grid_lon', 'week', 'week_count', 'total_grid_count'])

    print(f"[OK] Persistence calculated. Mean score: {df['persistence_score'].mean():.3f}")

    return df

def generate_statistics(df_raw, df_processed):
    """Generate processing statistics"""
    stats = {
        'processing_date': datetime.now().isoformat(),
        'input_records': len(df_raw),
        'output_records': len(df_processed),
        'records_removed': len(df_raw) - len(df_processed),
        'temporal_range': {
            'start': str(df_processed['acq_datetime'].min()),
            'end': str(df_processed['acq_datetime'].max())
        },
        'spatial_coverage': {
            'north_lat': float(df_processed['latitude'].max()),
            'south_lat': float(df_processed['latitude'].min()),
            'east_lon': float(df_processed['longitude'].max()),
            'west_lon': float(df_processed['longitude'].min())
        },
        'confidence_stats': {
            'mean': float(df_processed['confidence'].mean()),
            'min': float(df_processed['confidence'].min()),
            'max': float(df_processed['confidence'].max())
        },
        'persistence_stats': {
            'mean_count': float(df_processed['persistence_count'].mean()),
            'max_count': int(df_processed['persistence_count'].max())
        }
    }

    return stats

def main():
    """Main processing pipeline"""
    print("\n" + "="*60)
    print("FIRMS DATA PROCESSING - ETAPA 2")
    print("="*60)

    # Step 1: Load raw data
    df_raw = load_firms_data()
    if df_raw is None:
        return

    # Step 2: Validate geometry
    df = validate_geometry(df_raw)

    # Step 3: Parse dates
    df = parse_dates(df)

    # Step 4: Standardize confidence
    df = calculate_confidence_score(df)

    # Step 5: Filter by confidence
    df = filter_confidence(df, min_confidence=30)

    # Step 6: Aggregate temporal proximity
    df = aggregate_temporal(df, time_window_days=3)

    # Step 7: Calculate persistence
    df = calculate_persistence(df, window_days=7)

    # Step 8: Generate statistics
    stats = generate_statistics(df_raw, df)

    # Step 9: Save outputs
    output_file = OUTPUT_DIR / 'firms_processed.csv'
    df.to_csv(output_file, index=False)
    print(f"\n[OK] Processed data saved to: {output_file}")

    stats_file = OUTPUT_DIR / 'firms_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[OK] Statistics saved to: {stats_file}")

    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Input records: {stats['input_records']}")
    print(f"Output records: {stats['output_records']}")
    print(f"Removed: {stats['records_removed']} ({100*stats['records_removed']/stats['input_records']:.1f}%)")
    print(f"Temporal range: {stats['temporal_range']['start'][:10]} to {stats['temporal_range']['end'][:10]}")
    print(f"Confidence (mean): {stats['confidence_stats']['mean']:.1f}%")
    print(f"Persistence (mean detections): {stats['persistence_stats']['mean_count']:.1f}")

    return df

if __name__ == '__main__':
    df = main()
    print("\n[INFO] Next step: Process MCD64A1 burned area data")
    print("[INFO] Run: python src/preprocessing/process_mcd64a1.py")
