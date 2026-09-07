"""
Extract MapBiomas Fogo labels for FIRMS detections.

MapBiomas Fogo Collection 3 encodes:
- Annual: pixel value = land cover class that burned (0 = not burned)
- Monthly: pixel value = month*100 + land_cover_class (0 = not burned)

For each FIRMS detection, checks if MapBiomas Fogo confirms burning
at that location in the same year/month.

Usage:
    python src/preprocessing/extract_mapbiomas_fire.py
    python src/preprocessing/extract_mapbiomas_fire.py --limit 10
    python src/preprocessing/extract_mapbiomas_fire.py --resume 500
"""

import argparse
import time
from pathlib import Path

import ee
import pandas as pd

ee.Initialize()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_CSV = PROJECT_ROOT / 'outputs' / 'ground_truth_validation' / 'painel_labeled_expanded.csv'
OUTPUT_CSV = PROJECT_ROOT / 'data' / 'processed' / 'mapbiomas_fire_labels.csv'
MERGED_CSV = PROJECT_ROOT / 'outputs' / 'ground_truth_validation' / 'painel_labeled_5strategies.csv'

MAPBIOMAS_FIRE_ANNUAL = 'projects/mapbiomas-public/assets/brazil/fire/collection3/mapbiomas_fire_collection3_annual_burned_coverage_v1'
MAPBIOMAS_FIRE_MONTHLY = 'projects/mapbiomas-public/assets/brazil/fire/collection3/mapbiomas_fire_collection3_monthly_burned_coverage_v1'


def extract_mapbiomas_fire(lat, lon, acq_date):
    """Check if MapBiomas Fogo indicates burning at this location and time.

    Returns dict with:
        mb_fire_annual: land cover class that burned in that year (0 or None = no burn)
        mb_fire_monthly: encoded month*100 + class (0 or None = no burn)
        mb_fire_month: extracted month of burn (1-12, or None)
        gt_mapbiomas_fire: 1 if burned (real fire), 0 if not burned (spurious)
    """
    point = ee.Geometry.Point([lon, lat])
    year = acq_date.year
    month = acq_date.month

    band_name = f'burned_coverage_{year}'

    try:
        # Annual check
        annual = ee.Image(MAPBIOMAS_FIRE_ANNUAL)
        if band_name not in [b['id'] for b in annual.getInfo()['bands']]:
            return None

        annual_val = annual.select(band_name).reduceRegion(
            reducer=ee.Reducer.first(), geometry=point, scale=30
        ).getInfo().get(band_name)

        # Monthly check
        monthly = ee.Image(MAPBIOMAS_FIRE_MONTHLY)
        monthly_val = monthly.select(band_name).reduceRegion(
            reducer=ee.Reducer.first(), geometry=point, scale=30
        ).getInfo().get(band_name)

        # Decode monthly value
        mb_month = None
        if monthly_val is not None and monthly_val > 0:
            mb_month = monthly_val // 100  # month = value // 100

        # Determine label
        # Strict match: same year AND same month (or adjacent month +-1)
        burned_annual = annual_val is not None and annual_val > 0
        burned_month_match = False
        if mb_month is not None:
            burned_month_match = abs(mb_month - month) <= 1  # allow +-1 month tolerance

        # Label: 1 = real fire (burned), 0 = spurious (not burned)
        if burned_annual and burned_month_match:
            gt_label = 1  # Strong evidence: same year AND month
        elif burned_annual:
            gt_label = 1  # Burned in same year (month may differ)
        else:
            gt_label = 0  # No burn detected

        return {
            'mb_fire_annual': annual_val,
            'mb_fire_monthly': monthly_val,
            'mb_fire_month': mb_month,
            'mb_fire_month_match': burned_month_match,
            'gt_mapbiomas_fire': gt_label,
        }

    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser(description='Extract MapBiomas Fogo labels')
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--merged', type=str, default=None)
    args = parser.parse_args()

    input_csv = Path(args.input) if args.input else INPUT_CSV
    output_csv = Path(args.output) if args.output else OUTPUT_CSV
    merged_csv = Path(args.merged) if args.merged else MERGED_CSV

    print("=" * 60)
    print("EXTRACT MAPBIOMAS FOGO LABELS")
    print("=" * 60)

    df = pd.read_csv(input_csv)
    df['acq_datetime'] = pd.to_datetime(df['acq_datetime'], format='mixed')
    print(f"Input: {input_csv.name} ({len(df)} samples)")

    # Add columns
    for col in ['mb_fire_annual', 'mb_fire_monthly', 'mb_fire_month',
                'mb_fire_month_match', 'gt_mapbiomas_fire']:
        if col not in df.columns:
            df[col] = None

    n_total = len(df)
    if args.limit:
        n_total = min(args.limit, n_total)

    start_idx = args.resume
    n_success = 0
    n_no_data = 0
    n_errors = 0

    print(f"Processing {n_total - start_idx} samples (from {start_idx})...")
    print(f"Estimated time: {(n_total - start_idx) * 2 // 60} - {(n_total - start_idx) * 4 // 60} minutes")
    print()

    for i in range(start_idx, n_total):
        row = df.iloc[i]
        lat = row['latitude']
        lon = row['longitude']
        acq_date = row['acq_datetime']

        try:
            result = extract_mapbiomas_fire(lat, lon, acq_date)

            if result is None:
                n_no_data += 1
                status = "NO DATA"
            else:
                for k, v in result.items():
                    df.at[i, k] = v
                gt = result['gt_mapbiomas_fire']
                n_success += 1
                month_str = f"m={result['mb_fire_month']}" if result['mb_fire_month'] else "no month"
                status = f"OK (gt={gt}, {month_str})"

        except Exception as e:
            n_errors += 1
            status = f"ERROR ({str(e)[:50]})"

        print(f"[{i+1:>4}/{n_total}] lat={lat:.4f}, lon={lon:.4f}, "
              f"date={str(acq_date)[:10]} ... {status}")

        # Rate limiting
        if (i + 1) % 10 == 0:
            time.sleep(1)

        # Save progress
        if (i + 1) % 25 == 0:
            feat_cols = ['mb_fire_annual', 'mb_fire_monthly', 'mb_fire_month',
                         'mb_fire_month_match', 'gt_mapbiomas_fire',
                         'latitude', 'longitude', 'acq_datetime']
            df[feat_cols].to_csv(output_csv, index=False)
            print(f"  --- Saved progress ({i+1}/{n_total}) ---")

    # Final save
    feat_cols = ['mb_fire_annual', 'mb_fire_monthly', 'mb_fire_month',
                 'mb_fire_month_match', 'gt_mapbiomas_fire',
                 'latitude', 'longitude', 'acq_datetime']
    df[feat_cols].to_csv(output_csv, index=False)

    print()
    print("=" * 60)
    print("EXTRACTION RESULTS")
    print("=" * 60)
    processed = n_success + n_no_data + n_errors
    print(f"Processed: {processed}/{n_total - start_idx}")
    print(f"  Success: {n_success}")
    print(f"  No data: {n_no_data}")
    print(f"  Errors: {n_errors}")

    # Stats
    gt_col = df['gt_mapbiomas_fire'].dropna()
    if len(gt_col) > 0:
        n_real = (gt_col == 1).sum()
        n_spur = (gt_col == 0).sum()
        print(f"\nLabel distribution:")
        print(f"  Real fire (1): {n_real} ({100*n_real/len(gt_col):.1f}%)")
        print(f"  Not burned (0): {n_spur} ({100*n_spur/len(gt_col):.1f}%)")

    # Merge with original
    print("\n--- MERGING ---")
    merge_cols = ['latitude', 'longitude', 'acq_datetime']
    fire_cols = ['mb_fire_annual', 'mb_fire_monthly', 'mb_fire_month',
                 'mb_fire_month_match', 'gt_mapbiomas_fire']

    df_orig = pd.read_csv(input_csv)
    df_orig['acq_datetime'] = pd.to_datetime(df_orig['acq_datetime'], format='mixed')

    for col in fire_cols:
        df_orig[col] = df[col].values[:len(df_orig)]

    n_with_label = df_orig['gt_mapbiomas_fire'].notna().sum()
    print(f"Samples with MapBiomas Fire label: {n_with_label}/{len(df_orig)}")

    df_orig.to_csv(merged_csv, index=False)
    print(f"\nSaved features: {output_csv}")
    print(f"Saved merged:   {merged_csv}")
    print(f"\n[DONE] {n_with_label} samples with MapBiomas Fire labels")


if __name__ == '__main__':
    main()
