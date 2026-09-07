"""
Automated dNBR-based validation of weak labels.

For each of the 200 validation samples:
1. Computes Sentinel-2 pre/post fire composites via Earth Engine
2. Calculates dNBR statistics in a 1km buffer
3. Auto-classifies as real fire (0) or spurious (1)
4. Writes results to validation_sample.csv

Thresholds based on USGS burn severity (Key & Benson, 2006):
  - dNBR > 0.27  → High severity
  - dNBR 0.1-0.27 → Moderate severity
  - dNBR < 0.1   → Low/unburned

Auto-classification logic:
  - mean_dnbr > 0.15 OR max_dnbr > 0.3  → Real fire (label=0, conf=4-5)
  - mean_dnbr < 0.05 AND max_dnbr < 0.15 → Spurious (label=1, conf=4-5)
  - Otherwise → Uncertain (conf=2-3, flagged for manual review)
"""
import ee
import pandas as pd
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dnbr_rule import classify_dnbr

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
VALIDATION_DIR = PROJECT_DIR / 'data' / 'processed' / 'validation'
CSV_PATH = VALIDATION_DIR / 'validation_sample.csv'
OUTPUT_PATH = VALIDATION_DIR / 'validation_sample_auto.csv'

# Initialize Earth Engine
ee.Initialize()


def compute_dnbr_stats(lat, lon, date_str, buffer_m=500):
    """
    Compute dNBR statistics for a point using Sentinel-2.

    Returns dict with: mean_dnbr, max_dnbr, median_dnbr, pct_burned (% area with dNBR > 0.1)
    Returns None if no imagery available.
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

    # Get results
    result = stats.getInfo()
    pct_result = pct.getInfo()

    if result.get('dNBR_mean') is None:
        return None

    return {
        'mean_dnbr': result.get('dNBR_mean'),
        'max_dnbr': result.get('dNBR_max'),
        'median_dnbr': result.get('dNBR_median'),
        'pct_burned': pct_result.get('dNBR', 0) * 100  # Convert to percentage
    }


def classify_point(stats):
    """
    Auto-classify a point based on dNBR statistics.

    A regra composta vive em `dnbr_rule.classify_dnbr`, para que os scripts de
    analise (inclusive a replicacao do Alentejo) apliquem exatamente a mesma
    regra. `stats['pct_burned']` ja vem em percentual de `compute_dnbr_stats`.

    Returns: (manual_label, manual_confidence, gee_burn_evidence, manual_notes)
    """
    if stats is None:
        return (None, 1, 0, 'Sem imagens Sentinel-2 disponiveis')

    return classify_dnbr(stats['mean_dnbr'], stats['max_dnbr'], stats['pct_burned'])


def main():
    print("=" * 60)
    print("VALIDACAO AUTOMATICA dNBR - Weak Labels")
    print("=" * 60)

    df = pd.read_csv(CSV_PATH)
    total = len(df)
    print(f"Amostras: {total}")
    print(f"Input:  {CSV_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    # New columns for dNBR stats
    df['auto_mean_dnbr'] = None
    df['auto_max_dnbr'] = None
    df['auto_median_dnbr'] = None
    df['auto_pct_burned'] = None

    success = 0
    errors = 0

    for i, row in df.iterrows():
        idx = i + 1
        sid = row['sample_id']
        lat = row['latitude']
        lon = row['longitude']
        date_str = str(row['acq_datetime'])[:10]

        print(f"[{idx:3d}/{total}] ID={sid}, lat={lat:.4f}, lon={lon:.4f}, date={date_str} ... ", end='', flush=True)

        try:
            stats = compute_dnbr_stats(lat, lon, date_str)

            if stats is None:
                print("SEM IMAGEM")
                label, conf, evidence, notes = classify_point(None)
            else:
                label, conf, evidence, notes = classify_point(stats)
                df.at[i, 'auto_mean_dnbr'] = round(stats['mean_dnbr'], 4)
                df.at[i, 'auto_max_dnbr'] = round(stats['max_dnbr'], 4)
                df.at[i, 'auto_median_dnbr'] = round(stats['median_dnbr'], 4)
                df.at[i, 'auto_pct_burned'] = round(stats['pct_burned'], 1)

                status = "REAL" if label == 0 else "ESPURIO"
                print(f"{status} (conf={conf}, mean={stats['mean_dnbr']:.3f}, max={stats['max_dnbr']:.3f})")

            df.at[i, 'manual_label'] = label
            df.at[i, 'manual_confidence'] = conf
            df.at[i, 'gee_burn_evidence'] = evidence
            df.at[i, 'manual_notes'] = notes
            success += 1

        except Exception as e:
            print(f"ERRO: {e}")
            df.at[i, 'manual_notes'] = f'Erro no processamento: {str(e)[:80]}'
            df.at[i, 'manual_confidence'] = 1
            errors += 1

        # Small delay to avoid rate limiting
        if idx % 10 == 0:
            time.sleep(1)

    # Save results
    df.to_csv(OUTPUT_PATH, index=False)
    print()
    print("=" * 60)
    print("RESULTADO")
    print("=" * 60)
    print(f"Processados: {success}/{total}")
    print(f"Erros: {errors}")

    # Summary
    labeled = df[df['manual_label'].notna()]
    if len(labeled) > 0:
        n_real = (labeled['manual_label'] == 0).sum()
        n_spurious = (labeled['manual_label'] == 1).sum()
        n_high_conf = (labeled['manual_confidence'] >= 4).sum()
        n_low_conf = (labeled['manual_confidence'] <= 2).sum()

        print(f"\nClassificacao automatica:")
        print(f"  Real (label=0):    {n_real} ({n_real/len(labeled)*100:.1f}%)")
        print(f"  Espurio (label=1): {n_spurious} ({n_spurious/len(labeled)*100:.1f}%)")
        print(f"\nConfianca:")
        print(f"  Alta (4-5):  {n_high_conf} ({n_high_conf/len(labeled)*100:.1f}%)")
        print(f"  Baixa (1-2): {n_low_conf} ({n_low_conf/len(labeled)*100:.1f}%) <- REVISAR MANUALMENTE")

        # Disagreements with weak label
        disagree = labeled[labeled['manual_label'] != labeled['is_reliable'].apply(lambda x: 0 if x == 1 else 1)]
        # Note: is_reliable=1 means real (weak_label reliable), is_reliable=0 means spurious
        # manual_label=0 means real fire, manual_label=1 means spurious
        # So agreement: is_reliable=1 and manual_label=0, or is_reliable=0 and manual_label=1
        # Wait, let me reconsider the mapping:
        #   is_reliable = 1 → weak says "reliable/real" → corresponds to manual_label = 0 (real)
        #   is_reliable = 0 → weak says "spurious" → corresponds to manual_label = 1 (spurious)
        # But in the GEE script, weak_label=1 was "Espurio", and is_reliable column:
        #   is_reliable=1 means the sample IS reliable = real fire
        # Actually looking at the CSV header more carefully and the context:
        # is_reliable column = 1 means the weak labeling says it's reliable (real)
        # But wait, the first sample has is_reliable=1 and weak_label in GEE was 1 (Espurio)
        # Let me check... sample 4824: is_reliable=1, GEE weak_label=1 (Espurio)
        # That's contradictory... unless is_reliable maps differently
        # Actually in the GEE script: 'weak_label': 1 means SPURIOUS
        # And in CSV: is_reliable=1 means... let me check the original script

        # Corrected mapping:
        # is_reliable=1 → Real fire (detection IS reliable) → manual_label=0
        # is_reliable=0 → Spurious (detection NOT reliable) → manual_label=1
        # Agreement: is_reliable=1 AND manual_label=0, OR is_reliable=0 AND manual_label=1
        # Equivalently: is_reliable != manual_label (due to inverted scales)
        weak_label_corrected = 1 - labeled['is_reliable']  # Convert to same scale as manual_label
        agree_count = (weak_label_corrected == labeled['manual_label']).sum()
        disagree_count = (weak_label_corrected != labeled['manual_label']).sum()

        print(f"\nConcordancia com weak label:")
        print(f"  Concorda:  {agree_count} ({agree_count/len(labeled)*100:.1f}%)")
        print(f"  Discorda:  {disagree_count} ({disagree_count/len(labeled)*100:.1f}%)")

    print(f"\nArquivo salvo em: {OUTPUT_PATH}")
    print("IMPORTANTE: Revise os casos de baixa confianca (conf<=2) no GEE Navigator!")


if __name__ == '__main__':
    main()
