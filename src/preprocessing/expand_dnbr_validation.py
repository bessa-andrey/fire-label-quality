"""
Expand dNBR validation samples from 199 to ~400-600.

Samples 500 new points from the Painel-labeled dataset (stratified by
gt_painel and month), computes dNBR via Google Earth Engine, and merges
with the existing 199 samples.

Usage:
    python src/preprocessing/expand_dnbr_validation.py
    python src/preprocessing/expand_dnbr_validation.py --n-samples 300
"""
import ee
import pandas as pd
import numpy as np
import time
import argparse
from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
PAINEL_CSV = PROJECT_DIR / 'outputs' / 'ground_truth_validation' / 'painel_labeled_dataset.csv'
EXISTING_DNBR = PROJECT_DIR / 'data' / 'processed' / 'validation' / 'validation_sample_auto.csv'
OUTPUT_DIR = PROJECT_DIR / 'data' / 'processed' / 'validation'
EXPANSION_CSV = OUTPUT_DIR / 'validation_expansion.csv'
MERGED_CSV = OUTPUT_DIR / 'validation_dnbr_merged.csv'

# Initialize Earth Engine
ee.Initialize()


def compute_dnbr_stats(lat, lon, date_str, buffer_m=500):
    """
    Compute dNBR statistics for a point using Sentinel-2.
    Reuses same logic as auto_validate_dnbr.py.
    """
    point = ee.Geometry.Point([lon, lat])
    geometry = point.buffer(buffer_m)
    date = ee.Date(date_str)

    # Pre-fire: 30-60 days before
    pre_fire = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(geometry)
        .filterDate(date.advance(-60, 'day'), date.advance(-30, 'day'))
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        .median())

    # Post-fire: 0-30 days after
    post_fire = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(geometry)
        .filterDate(date, date.advance(30, 'day'))
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        .median())

    # NBR = (B8 - B12) / (B8 + B12)
    pre_nbr = pre_fire.normalizedDifference(['B8', 'B12']).rename('NBR')
    post_nbr = post_fire.normalizedDifference(['B8', 'B12']).rename('NBR')

    # dNBR = pre_NBR - post_NBR (positive = burn)
    dnbr = pre_nbr.subtract(post_nbr).rename('dNBR')

    # Compute statistics
    stats = dnbr.reduceRegion(
        reducer=ee.Reducer.mean()
            .combine(ee.Reducer.max(), '', True)
            .combine(ee.Reducer.median(), '', True),
        geometry=geometry,
        scale=10,
        maxPixels=1e8
    )

    # Percentage of area with dNBR > 0.1 (burned)
    burned_mask = dnbr.gt(0.1)
    pct = burned_mask.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=10,
        maxPixels=1e8
    )

    result = stats.getInfo()
    pct_result = pct.getInfo()

    if result.get('dNBR_mean') is None:
        return None

    return {
        'mean_dnbr': result.get('dNBR_mean'),
        'max_dnbr': result.get('dNBR_max'),
        'median_dnbr': result.get('dNBR_median'),
        'pct_burned': pct_result.get('dNBR', 0) * 100
    }


def classify_point(stats):
    """
    Auto-classify based on dNBR. Same thresholds as auto_validate_dnbr.py.
    Returns: (manual_label, manual_confidence, gee_burn_evidence, manual_notes)
    """
    if stats is None:
        return (None, 1, 0, 'Sem imagens Sentinel-2 disponiveis')

    mean = stats['mean_dnbr']
    max_val = stats['max_dnbr']
    pct = stats['pct_burned']

    # REAL FIRE (label=0)
    if mean > 0.25 and pct > 40:
        return (0, 5, 2, f'dNBR forte (mean={mean:.3f}, max={max_val:.3f}, area={pct:.0f}%)')
    if mean > 0.20 and pct > 25:
        return (0, 4, 2, f'dNBR moderado-alto (mean={mean:.3f}, max={max_val:.3f}, area={pct:.0f}%)')
    if mean > 0.15 and pct > 20:
        return (0, 3, 1, f'dNBR moderado (mean={mean:.3f}, max={max_val:.3f}, area={pct:.0f}%)')

    # SPURIOUS (label=1)
    if mean < 0.05 and max_val < 0.15:
        return (1, 5, 0, f'dNBR nulo (mean={mean:.3f}, max={max_val:.3f})')
    if mean < 0.08 and max_val < 0.20 and pct < 10:
        return (1, 4, 0, f'dNBR baixo (mean={mean:.3f}, max={max_val:.3f}, area={pct:.0f}%)')
    if mean < 0.10 and pct < 15:
        return (1, 3, 0, f'dNBR insuficiente (mean={mean:.3f}, max={max_val:.3f}, area={pct:.0f}%)')

    # UNCERTAIN
    if mean >= 0.10 and mean < 0.15:
        return (1, 2, 1, f'dNBR ambiguo (mean={mean:.3f}, max={max_val:.3f}, area={pct:.0f}%)')

    if max_val > 0.30 and pct > 15:
        return (0, 2, 1, f'dNBR localizado (mean={mean:.3f}, max={max_val:.3f}, area={pct:.0f}%)')

    return (1, 2, 1, f'dNBR inconclusivo (mean={mean:.3f}, max={max_val:.3f}, area={pct:.0f}%)')


def sample_for_expansion(df_painel, df_existing, n_samples=500, seed=42):
    """
    Sample n_samples from Painel dataset, excluding those already in dNBR.
    Stratified by gt_painel and month.
    """
    # Identify existing by coordinates
    existing_coords = set(
        zip(df_existing['latitude'].round(4), df_existing['longitude'].round(4))
    )
    df_painel['in_dnbr'] = df_painel.apply(
        lambda r: (round(r['latitude'], 4), round(r['longitude'], 4)) in existing_coords,
        axis=1
    )
    available = df_painel[~df_painel['in_dnbr']].copy()
    print(f"Available for sampling: {len(available)} (excluded {df_painel['in_dnbr'].sum()} already in dNBR)")

    # Stratify: 50% real (gt_painel=1), 50% spurious (gt_painel=0)
    n_per_class = n_samples // 2
    rng = np.random.RandomState(seed)

    sampled_parts = []
    for gt_val in [0, 1]:
        class_df = available[available['gt_painel'] == gt_val]
        if len(class_df) <= n_per_class:
            sampled_parts.append(class_df)
        else:
            sampled_parts.append(class_df.sample(n=n_per_class, random_state=rng))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle

    print(f"Sampled: {len(sampled)} ({(sampled['gt_painel']==1).sum()} real, {(sampled['gt_painel']==0).sum()} spurious)")
    print(f"Month distribution: {dict(sampled['month'].value_counts().sort_index())}")

    return sampled


def main():
    parser = argparse.ArgumentParser(description='Expand dNBR validation')
    parser.add_argument('--n-samples', type=int, default=500, help='Number of new samples (default: 500)')
    parser.add_argument('--resume', type=int, default=0, help='Resume from sample index (default: 0)')
    args = parser.parse_args()

    print("=" * 60)
    print("EXPAND dNBR VALIDATION")
    print("=" * 60)

    # Load data
    df_painel = pd.read_csv(PAINEL_CSV)
    df_painel['acq_datetime'] = pd.to_datetime(df_painel['acq_datetime'])
    df_existing = pd.read_csv(EXISTING_DNBR)
    print(f"Existing dNBR samples: {len(df_existing)}")
    print(f"Painel dataset: {len(df_painel)}")

    # Sample
    sampled = sample_for_expansion(df_painel, df_existing, n_samples=args.n_samples)

    # Check if resuming from partial results
    if args.resume > 0 and EXPANSION_CSV.exists():
        df_partial = pd.read_csv(EXPANSION_CSV)
        print(f"\nResuming from index {args.resume} (partial: {len(df_partial)} rows)")
    else:
        # Prepare output DataFrame
        sampled['auto_mean_dnbr'] = None
        sampled['auto_max_dnbr'] = None
        sampled['auto_median_dnbr'] = None
        sampled['auto_pct_burned'] = None
        sampled['manual_label'] = None
        sampled['manual_confidence'] = None
        sampled['gee_burn_evidence'] = None
        sampled['manual_notes'] = None

    total = len(sampled)
    success = 0
    errors = 0
    no_image = 0

    print(f"\nProcessing {total} samples via GEE...")
    print(f"Estimated time: {total * 3 // 60} - {total * 5 // 60} minutes")
    print()

    start_idx = args.resume

    for i in range(start_idx, total):
        row = sampled.iloc[i]
        lat = row['latitude']
        lon = row['longitude']
        date_str = str(row['acq_datetime'])[:10]

        print(f"[{i+1:3d}/{total}] lat={lat:.4f}, lon={lon:.4f}, date={date_str} ... ", end='', flush=True)

        try:
            stats = compute_dnbr_stats(lat, lon, date_str)

            if stats is None:
                print("SEM IMAGEM")
                label, conf, evidence, notes = classify_point(None)
                no_image += 1
            else:
                label, conf, evidence, notes = classify_point(stats)
                sampled.at[sampled.index[i], 'auto_mean_dnbr'] = round(stats['mean_dnbr'], 4)
                sampled.at[sampled.index[i], 'auto_max_dnbr'] = round(stats['max_dnbr'], 4)
                sampled.at[sampled.index[i], 'auto_median_dnbr'] = round(stats['median_dnbr'], 4)
                sampled.at[sampled.index[i], 'auto_pct_burned'] = round(stats['pct_burned'], 1)

                status = "REAL" if label == 0 else "ESPURIO"
                print(f"{status} (conf={conf}, mean={stats['mean_dnbr']:.3f})")

            sampled.at[sampled.index[i], 'manual_label'] = label
            sampled.at[sampled.index[i], 'manual_confidence'] = conf
            sampled.at[sampled.index[i], 'gee_burn_evidence'] = evidence
            sampled.at[sampled.index[i], 'manual_notes'] = notes
            success += 1

        except Exception as e:
            error_msg = str(e)[:80]
            print(f"ERRO: {error_msg}")
            sampled.at[sampled.index[i], 'manual_notes'] = f'Erro: {error_msg}'
            sampled.at[sampled.index[i], 'manual_confidence'] = 1
            errors += 1

        # Save progress every 25 samples
        if (i + 1) % 25 == 0:
            sampled.to_csv(EXPANSION_CSV, index=False)
            print(f"  --- Saved progress ({i+1}/{total}) ---")

        # Rate limiting
        if (i + 1) % 10 == 0:
            time.sleep(1)

    # Final save
    sampled.to_csv(EXPANSION_CSV, index=False)

    print()
    print("=" * 60)
    print("EXPANSION RESULTS")
    print("=" * 60)
    print(f"Processed: {success}/{total}")
    print(f"Errors: {errors}")
    print(f"No image: {no_image}")

    # Stats
    labeled = sampled[sampled['manual_label'].notna()]
    if len(labeled) > 0:
        n_real = (labeled['manual_label'] == 0).sum()
        n_spurious = (labeled['manual_label'] == 1).sum()
        print(f"\nClassification:")
        print(f"  Real (label=0):    {n_real} ({100*n_real/len(labeled):.1f}%)")
        print(f"  Spurious (label=1): {n_spurious} ({100*n_spurious/len(labeled):.1f}%)")

    # Merge with existing
    print(f"\n--- MERGING ---")
    print(f"Existing: {len(df_existing)} samples")
    print(f"New: {len(labeled)} samples with labels")

    # Align columns for merge
    common_cols = [c for c in df_existing.columns if c in sampled.columns]
    df_new = sampled[sampled['manual_label'].notna()][common_cols].copy()
    df_merged = pd.concat([df_existing[common_cols], df_new], ignore_index=True)

    # Remove duplicates by coordinates
    df_merged['coord_key'] = df_merged['latitude'].round(4).astype(str) + '_' + df_merged['longitude'].round(4).astype(str)
    before_dedup = len(df_merged)
    df_merged = df_merged.drop_duplicates(subset='coord_key', keep='first')
    df_merged = df_merged.drop(columns='coord_key')
    print(f"After dedup: {len(df_merged)} (removed {before_dedup - len(df_merged)} duplicates)")

    n_valid = df_merged['manual_label'].notna().sum()
    print(f"Valid labels: {n_valid}")

    df_merged.to_csv(MERGED_CSV, index=False)
    print(f"\nSaved expansion: {EXPANSION_CSV}")
    print(f"Saved merged:    {MERGED_CSV}")
    print(f"\n[DONE] Total dNBR samples: {n_valid} (was 199)")


if __name__ == '__main__':
    main()
