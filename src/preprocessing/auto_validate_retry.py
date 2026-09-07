"""
Retry failed dNBR validations with wider temporal windows.
Only processes samples that had errors in the first run.
"""
import ee
import pandas as pd
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
VALIDATION_DIR = PROJECT_DIR / 'data' / 'processed' / 'validation'
CSV_PATH = VALIDATION_DIR / 'validation_sample_auto.csv'

ee.Initialize()


def compute_dnbr_wide(lat, lon, date_str):
    """Try progressively wider windows and higher cloud tolerance."""
    point = ee.Geometry.Point([lon, lat])
    geometry = point.buffer(500)
    date = ee.Date(date_str)

    configs = [
        # (pre_start, pre_end, post_start, post_end, cloud_pct)
        (-90, -15, -5, 45, 50),   # wider window, more cloud tolerance
        (-120, -15, -5, 60, 70),  # even wider
    ]

    for pre_s, pre_e, post_s, post_e, cloud_pct in configs:
        try:
            pre_fire = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(geometry)
                .filterDate(date.advance(pre_s, 'day'), date.advance(pre_e, 'day'))
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_pct))
                .median())

            post_fire = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(geometry)
                .filterDate(date.advance(post_s, 'day'), date.advance(post_e, 'day'))
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_pct))
                .median())

            pre_nbr = pre_fire.normalizedDifference(['B8', 'B12']).rename('NBR')
            post_nbr = post_fire.normalizedDifference(['B8', 'B12']).rename('NBR')
            dnbr = pre_nbr.subtract(post_nbr).rename('dNBR')

            stats = dnbr.reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.max(), '', True),
                geometry=geometry, scale=10, maxPixels=1e8
            )

            pct = dnbr.gt(0.1).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry, scale=10, maxPixels=1e8
            )

            result = stats.getInfo()
            pct_result = pct.getInfo()

            if result.get('dNBR_mean') is not None:
                return {
                    'mean_dnbr': result['dNBR_mean'],
                    'max_dnbr': result['dNBR_max'],
                    'pct_burned': pct_result.get('dNBR', 0) * 100
                }
        except Exception:
            continue

    return None


def classify(stats):
    if stats is None:
        return (None, 1, 0, 'Sem imagens Sentinel-2 disponiveis')

    mean = stats['mean_dnbr']
    max_val = stats['max_dnbr']
    pct = stats['pct_burned']

    # Stricter thresholds matching auto_validate_dnbr.py (500m buffer)
    if mean > 0.25 and pct > 40:
        return (0, 5, 2, f'[retry] dNBR vermelho intenso (mean={mean:.3f}, max={max_val:.3f}, area={pct:.0f}%)')
    if mean > 0.20 and pct > 25:
        return (0, 4, 2, f'[retry] dNBR laranja/vermelho (mean={mean:.3f}, max={max_val:.3f}, area={pct:.0f}%)')
    if mean > 0.15 and pct > 20:
        return (0, 3, 1, f'[retry] dNBR laranja, queimada provavel (mean={mean:.3f}, max={max_val:.3f}, area={pct:.0f}%)')
    if mean < 0.05 and max_val < 0.15:
        return (1, 5, 0, f'[retry] dNBR verde (mean={mean:.3f}, max={max_val:.3f})')
    if mean < 0.08 and max_val < 0.20 and pct < 10:
        return (1, 4, 0, f'[retry] dNBR verde (mean={mean:.3f}, max={max_val:.3f}, area={pct:.0f}%)')
    if mean < 0.10 and pct < 15:
        return (1, 3, 0, f'[retry] dNBR verde/amarelo (mean={mean:.3f}, max={max_val:.3f}, area={pct:.0f}%)')
    if mean >= 0.10 and mean < 0.15:
        return (1, 2, 1, f'[retry] dNBR amarelo, ambiguo (mean={mean:.3f}, max={max_val:.3f}, area={pct:.0f}%)')
    if max_val > 0.30 and pct > 15:
        return (0, 2, 1, f'[retry] dNBR sinal localizado (mean={mean:.3f}, max={max_val:.3f}, area={pct:.0f}%)')
    return (1, 2, 1, f'[retry] dNBR inconclusivo (mean={mean:.3f}, max={max_val:.3f}, area={pct:.0f}%)')


def main():
    df = pd.read_csv(CSV_PATH)
    error_mask = df['manual_notes'].str.contains('Erro', na=False)
    error_indices = df[error_mask].index
    total = len(error_indices)

    print(f"Retentando {total} amostras com erro...")
    print()

    recovered = 0
    still_failed = 0

    for count, i in enumerate(error_indices, 1):
        row = df.iloc[i]
        sid = row['sample_id']
        lat = row['latitude']
        lon = row['longitude']
        date_str = str(row['acq_datetime'])[:10]

        print(f"[{count:2d}/{total}] ID={sid}, date={date_str} ... ", end='', flush=True)

        stats = compute_dnbr_wide(lat, lon, date_str)
        label, conf, evidence, notes = classify(stats)

        if stats is not None:
            df.at[i, 'auto_mean_dnbr'] = round(stats['mean_dnbr'], 4)
            df.at[i, 'auto_max_dnbr'] = round(stats['max_dnbr'], 4)
            df.at[i, 'auto_pct_burned'] = round(stats['pct_burned'], 1)
            status = "REAL" if label == 0 else "ESPURIO"
            print(f"RECUPERADO -> {status} (mean={stats['mean_dnbr']:.3f})")
            recovered += 1
        else:
            print("AINDA SEM IMAGEM")
            still_failed += 1

        df.at[i, 'manual_label'] = label
        df.at[i, 'manual_confidence'] = conf
        df.at[i, 'gee_burn_evidence'] = evidence
        df.at[i, 'manual_notes'] = notes

        if count % 5 == 0:
            time.sleep(1)

    df.to_csv(CSV_PATH, index=False)

    print(f"\nRecuperados: {recovered}/{total}")
    print(f"Sem imagem:  {still_failed}/{total}")

    # Final summary
    labeled = df[df['manual_label'].notna()]
    n_real = (labeled['manual_label'] == 0).sum()
    n_spur = (labeled['manual_label'] == 1).sum()
    n_null = df['manual_label'].isna().sum()
    print(f"\nResumo final (200 amostras):")
    print(f"  Real:     {n_real}")
    print(f"  Espurio:  {n_spur}")
    print(f"  Sem dado: {n_null}")
    print(f"\nSalvo em: {CSV_PATH}")


if __name__ == '__main__':
    main()
