"""
Extract Sentinel-2 spectral features and MapBiomas land use for all samples.

For each sample, computes pre/post-fire NDVI, NBR, dNBR, and MapBiomas
land use class via Google Earth Engine.

Usage:
    python src/preprocessing/extract_sentinel2_features.py
    python src/preprocessing/extract_sentinel2_features.py --resume 500
    python src/preprocessing/extract_sentinel2_features.py --input data/processed/validation/validation_dnbr_merged.csv
"""

import argparse
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PROJECT_DIR / 'outputs' / 'ground_truth_validation' / 'painel_labeled_dataset.csv'
OUTPUT_CSV = PROJECT_DIR / 'data' / 'processed' / 'sentinel2_features.csv'
MERGED_CSV = PROJECT_DIR / 'outputs' / 'ground_truth_validation' / 'painel_labeled_expanded.csv'

# Initialize Earth Engine
ee.Initialize()

# MapBiomas Collection 8
MAPBIOMAS = ee.Image(
    'projects/mapbiomas-public/assets/brazil/lulc/collection8/mapbiomas_collection80_integration_v1'
)


def extract_spectral_features(lat, lon, date_str, buffer_m=500):
    """
    Extract spectral features for one sample via GEE.

    Returns dict with 9 features, or None if no imagery available.
    """
    point = ee.Geometry.Point([lon, lat])
    geometry = point.buffer(buffer_m)
    date = ee.Date(date_str)

    # Pre-fire: 30-60 days before
    pre_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(geometry)
        .filterDate(date.advance(-60, 'day'), date.advance(-30, 'day'))
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))

    # Post-fire: 0-30 days after
    post_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(geometry)
        .filterDate(date, date.advance(30, 'day'))
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))

    # Check collection sizes (server-side)
    pre_size = pre_col.size().getInfo()
    post_size = post_col.size().getInfo()

    if pre_size == 0 and post_size == 0:
        return None

    pre_fire = pre_col.median()
    post_fire = post_col.median()

    # Build results dict
    feat = {}

    # If both collections have images, compute all indices
    if pre_size > 0 and post_size > 0:
        pre_ndvi = pre_fire.normalizedDifference(['B8', 'B4']).rename('ndvi_pre')
        post_ndvi = post_fire.normalizedDifference(['B8', 'B4']).rename('ndvi_post')
        pre_nbr = pre_fire.normalizedDifference(['B8', 'B12']).rename('nbr_pre')
        post_nbr = post_fire.normalizedDifference(['B8', 'B12']).rename('nbr_post')
        dnbr = pre_nbr.subtract(post_nbr).rename('dnbr')

        combined = ee.Image.cat([pre_ndvi, post_ndvi, pre_nbr, post_nbr, dnbr])
        stats = combined.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geometry, scale=10, maxPixels=1e8
        ).getInfo()

        dnbr_max_val = dnbr.reduceRegion(
            reducer=ee.Reducer.max(), geometry=geometry, scale=10, maxPixels=1e8
        ).getInfo().get('dnbr')

        pct_val = dnbr.gt(0.1).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geometry, scale=10, maxPixels=1e8
        ).getInfo().get('dnbr', 0)
        if pct_val is None:
            pct_val = 0

        feat = {
            'mean_ndvi_pre': stats.get('ndvi_pre'),
            'mean_ndvi_post': stats.get('ndvi_post'),
            'delta_ndvi': _safe_sub(stats.get('ndvi_pre'), stats.get('ndvi_post')),
            'mean_nbr_pre': stats.get('nbr_pre'),
            'mean_nbr_post': stats.get('nbr_post'),
            'dnbr_mean': stats.get('dnbr'),
            'dnbr_max': dnbr_max_val,
            'pct_burned': pct_val * 100,
        }
    elif pre_size > 0:
        # Only pre-fire available
        pre_ndvi = pre_fire.normalizedDifference(['B8', 'B4']).rename('ndvi_pre')
        pre_nbr = pre_fire.normalizedDifference(['B8', 'B12']).rename('nbr_pre')
        combined = ee.Image.cat([pre_ndvi, pre_nbr])
        stats = combined.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geometry, scale=10, maxPixels=1e8
        ).getInfo()
        feat = {
            'mean_ndvi_pre': stats.get('ndvi_pre'),
            'mean_nbr_pre': stats.get('nbr_pre'),
        }
    elif post_size > 0:
        # Only post-fire available
        post_ndvi = post_fire.normalizedDifference(['B8', 'B4']).rename('ndvi_post')
        post_nbr = post_fire.normalizedDifference(['B8', 'B12']).rename('nbr_post')
        combined = ee.Image.cat([post_ndvi, post_nbr])
        stats = combined.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geometry, scale=10, maxPixels=1e8
        ).getInfo()
        feat = {
            'mean_ndvi_post': stats.get('ndvi_post'),
            'mean_nbr_post': stats.get('nbr_post'),
        }

    # MapBiomas land use
    year = int(date_str[:4])
    band_name = f'classification_{year}'
    try:
        land_val = MAPBIOMAS.select(band_name).reduceRegion(
            reducer=ee.Reducer.mode(), geometry=geometry, scale=30, maxPixels=1e8
        ).getInfo().get(band_name)
        feat['land_use_class'] = land_val
    except Exception:
        feat['land_use_class'] = None

    # Return None if no spectral data at all
    if feat.get('mean_ndvi_pre') is None and feat.get('mean_nbr_pre') is None \
       and feat.get('mean_ndvi_post') is None and feat.get('mean_nbr_post') is None:
        return None

    return feat


def _safe_sub(a, b):
    if a is not None and b is not None:
        return a - b
    return None


def main():
    parser = argparse.ArgumentParser(description='Extract Sentinel-2 features via GEE')
    parser.add_argument('--input', type=str, default=None, help='Input CSV path')
    parser.add_argument('--output', type=str, default=None, help='Output features CSV path')
    parser.add_argument('--merged', type=str, default=None, help='Output merged CSV path')
    parser.add_argument('--resume', type=int, default=0, help='Resume from sample index')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of samples (0=all)')
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else DEFAULT_INPUT
    output_csv = Path(args.output) if args.output else OUTPUT_CSV
    merged_csv = Path(args.merged) if args.merged else MERGED_CSV

    print("=" * 60)
    print("EXTRACT SENTINEL-2 SPECTRAL FEATURES")
    print("=" * 60)

    df = pd.read_csv(input_path)
    df['acq_datetime'] = pd.to_datetime(df['acq_datetime'], format='mixed')
    print(f"Input: {input_path.name} ({len(df)} samples)")

    if args.limit > 0:
        df = df.head(args.limit)
        print(f"Limited to {args.limit} samples")

    total = len(df)

    # Feature columns
    feat_cols = [
        'mean_ndvi_pre', 'mean_ndvi_post', 'delta_ndvi',
        'mean_nbr_pre', 'mean_nbr_post',
        'dnbr_mean', 'dnbr_max', 'pct_burned',
        'land_use_class',
    ]

    # Resume from partial results
    if args.resume > 0 and output_csv.exists():
        df_partial = pd.read_csv(output_csv)
        print(f"Resuming from index {args.resume} ({len(df_partial)} rows in partial)")
        # Copy partial results into df
        for col in feat_cols:
            if col in df_partial.columns:
                df[col] = df_partial[col]
    else:
        for col in feat_cols:
            if col not in df.columns:
                df[col] = None

    success = 0
    errors = 0
    no_image = 0
    start_idx = args.resume

    print(f"\nProcessing {total - start_idx} samples via GEE...")
    print(f"Estimated time: {(total - start_idx) * 3 // 60} - {(total - start_idx) * 5 // 60} minutes\n")

    for i in range(start_idx, total):
        row = df.iloc[i]
        lat = row['latitude']
        lon = row['longitude']
        date_str = str(row['acq_datetime'])[:10]

        print(f"[{i+1:4d}/{total}] lat={lat:.4f}, lon={lon:.4f}, date={date_str} ... ", end='', flush=True)

        try:
            features = extract_spectral_features(lat, lon, date_str)

            if features is None:
                print("SEM IMAGEM")
                no_image += 1
            else:
                for col in feat_cols:
                    val = features.get(col)
                    if val is not None:
                        df.at[df.index[i], col] = round(val, 4) if isinstance(val, float) else val
                ndvi = features.get('delta_ndvi')
                dnbr = features.get('dnbr_mean')
                luc = features.get('land_use_class')
                print(f"OK (dNDVI={ndvi:.3f}, dNBR={dnbr:.3f}, LUC={luc})" if ndvi else "OK (partial)")
                success += 1

        except Exception as e:
            error_msg = str(e)[:80]
            print(f"ERRO: {error_msg}")
            errors += 1

        # Save progress every 25 samples
        if (i + 1) % 25 == 0:
            df[feat_cols + ['latitude', 'longitude', 'acq_datetime']].to_csv(output_csv, index=False)
            print(f"  --- Saved progress ({i+1}/{total}) ---")

        # Rate limiting
        if (i + 1) % 10 == 0:
            time.sleep(1)

    # Final save (features only)
    df[feat_cols + ['latitude', 'longitude', 'acq_datetime']].to_csv(output_csv, index=False)

    print()
    print("=" * 60)
    print("EXTRACTION RESULTS")
    print("=" * 60)
    print(f"Processed: {success + no_image + errors}/{total - start_idx}")
    print(f"  Success: {success}")
    print(f"  No image: {no_image}")
    print(f"  Errors: {errors}")

    # Merge with original dataset
    print(f"\n--- MERGING ---")
    df_orig = pd.read_csv(input_path)
    df_orig['acq_datetime'] = pd.to_datetime(df_orig['acq_datetime'], format='mixed')

    # Add features to original
    for col in feat_cols:
        df_orig[col] = df[col]

    # Stats
    n_with_features = df_orig['dnbr_mean'].notna().sum()
    print(f"Samples with spectral features: {n_with_features}/{len(df_orig)}")

    # Save merged
    merged_csv.parent.mkdir(parents=True, exist_ok=True)
    df_orig.to_csv(merged_csv, index=False)
    print(f"\nSaved features: {output_csv}")
    print(f"Saved merged:   {merged_csv}")
    print(f"\n[DONE] {n_with_features} samples with spectral features")


if __name__ == '__main__':
    main()
