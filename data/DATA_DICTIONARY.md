# Data Dictionary

This document describes every column of the CSV files in this Zenodo bundle.

## `data/matopiba_fire_human_gt_v1.csv` (604 rows)

### Identifier and geolocation

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | string | Unique anonymized identifier for the FIRMS detection. |
| `latitude` | float | Decimal latitude (WGS84). |
| `longitude` | float | Decimal longitude (WGS84). |
| `acq_datetime` | datetime | FIRMS acquisition timestamp (UTC). |
| `year` | int | Year of acquisition. |
| `month` | int | Month of acquisition (1-12). |

### Label sources (raw, see README for unified mapping)

| Column | Type | Convention | Description |
|--------|------|------------|-------------|
| `is_reliable` | int | 1=Real, 0=Spurious | Weak label from the FIRMS × MCD64A1 cross-check. A detection is labelled Real when at least one MCD64A1 pixel within 500 m carries a `BurnDate` (Julian day of burning) inside the ±15-day window around the FIRMS acquisition date. Product: `MODIS/061/MCD64A1`, reduction scale 500 m. Recomputed on 2026-09-04 by `src/preprocessing/recompute_weak_labels_mcd64a1.py`. |
| `is_reliable_legacy_median` | int | 1=Real, 0=Spurious | **Superseded column, kept for traceability only. Do not use as a label source.** This is what earlier drafts of this bundle distributed under the name `is_reliable`. It is not a FIRMS × MCD64A1 cross-check: it is the median split of `reliability_score`, an in-house score built exclusively from FIRMS variables (persistence, reported confidence, detection count, FRP). Over the same 3,698 panel rows the superseded column and the strict cross-check agree on 63.39% of the detections, with a Kappa of 0.1589 between them, so the two columns record different measurements. |
| `manual_label` | int | 0=Real, 1=Spurious | dNBR-automated label from Sentinel-2 (already in unified convention). |
| `gt_mapbiomas_fire` | int (nullable) | 1=Burned, 0=Not burned | MapBiomas Fogo Collection 8 monthly burn scar label. Treat 1=Burned as "Real fire" before inverting to unified convention. |
| `gt_painel` | int (nullable) | 1=Match, 0=No match | Spatial-temporal match with CENSIPAM Painel do Fogo events. Treat 1=Match as "Real fire" before inverting to unified convention. |
| `human_label` | int | -1=Uncertain, 0=Real, 1=Spurious | Author's blind annotation based on Sentinel-2 thumbnails only. **Primary ground truth.** |
| `human_confidence` | int | 1-5 Likert | Self-rated annotator confidence (1=low, 5=high). |
| `human_notes` | string | Free text | Annotator's free-text comments per sample (in Portuguese). |

### Base features (13: FIRMS + ERA5 + temporal)

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `brightness` | float | Kelvin | FIRMS brightness temperature at detection. |
| `confidence` | int | 0-100 | FIRMS reported detection confidence. |
| `frp` | float | MW | Fire Radiative Power (FIRMS). |
| `hotspot_count` | int | count | Number of FIRMS detections within spatial neighborhood. |
| `persistence_score` | float | 0-1 | Temporal persistence score (in-house engineered). |
| `temperature` | float | °C | Surface air temperature at detection time (ERA5). |
| `dewpoint` | float | °C | Dew point temperature (ERA5). |
| `wind_speed` | float | m/s | 10-m wind speed (ERA5). |
| `precipitation` | float | mm | Total precipitation in preceding 24h (ERA5). |
| `rh` | float | % | Relative humidity (ERA5). |
| `drying_index` | float | unitless | Drying index derived from RH and temperature. |
| `month` | int | 1-12 | Month of year. |
| (`is_dry_season`) | int | 0/1 | 1 if month in [5, 6, 7, 8, 9, 10]; derived at modeling time. |

### Spectral features (9: Sentinel-2 + MapBiomas)

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `mean_ndvi_pre` | float | unitless | Mean NDVI in 500 m buffer, pre-fire window. |
| `mean_ndvi_post` | float | unitless | Mean NDVI in 500 m buffer, post-fire window. |
| `delta_ndvi` | float | unitless | NDVI pre - post. |
| `mean_nbr_pre` | float | unitless | Mean NBR (Normalized Burn Ratio), pre-fire. |
| `mean_nbr_post` | float | unitless | Mean NBR, post-fire. |
| `dnbr_mean` | float | unitless | Mean dNBR (differenced NBR) in buffer. |
| `dnbr_max` | float | unitless | Max dNBR in buffer. |
| `pct_burned` | float | 0-1 | Fraction of buffer pixels with dNBR > 0.1. |
| `land_use_class` | int | MapBiomas code | MapBiomas Collection 8 land use class (4 = Savanna Formation; full legend at mapbiomas.org). For 2023-2024 samples, imputed as 4 (mode of 2022) due to MapBiomas Collection 8 temporal coverage. |

**Coverage of the spectral block, measured on this file.** Of the 604 rows, 573 carry all
eight scene-derived columns (every column above except `land_use_class`), 597 carry at
least one, and 601 carry `land_use_class`. By year: 471 of the 484 rows of 2022, 50 of the
54 of 2023 and 52 of the 66 of 2024 are complete on the eight. A row is incomplete where
no cloud-free Sentinel-2 scene was available in the pre-fire or the post-fire window.

Until the September 2026 revision the 120 rows of 2023 and 2024 carried no spectral value
at all, because this file was enriched from the 2022 panel. It is now enriched from the
extended panel, which covers the three years. The change is strictly additive: it fills
empty cells and overwrites nothing, and the 2022 rows are identical series to series.

---

## `data/matopiba_fire_full_panel_v1.csv` (3,698 rows, 2022 only)

Same schema as above except:
- `sample_id` not present (less curated)
- `human_label`, `human_confidence`, `human_notes` not present
- `manual_label` not present (the dNBR label was only materialised for the annotated samples)
- All 3,698 rows have `gt_painel` and `gt_mapbiomas_fire` (full coverage)

Proportion labelled as real fire in this file: `is_reliable` 12.63% (467/3,698),
`is_reliable_legacy_median` 41.35% (1,529/3,698).

---

## Revision note on the weak label (2026-09-04)

Until this revision, the column named `is_reliable` in this bundle was the median split of
`reliability_score`, while the documentation described it as a FIRMS × MCD64A1 cross-check.
The two are different things and the mismatch was a documentation-and-provenance defect.

From this revision on:

- `is_reliable` is the FIRMS × MCD64A1 cross-check the documentation describes, computed
  with the strict criterion (`BurnDate` inside ±15 days, 500 m radius).
- `is_reliable_legacy_median` preserves the superseded column so that anyone who already
  ran an analysis on the earlier file can reproduce it exactly.

No other column changed. The human annotation, the dNBR label, the Painel do Fogo label,
the MapBiomas Fogo label and all 22 features are untouched.

---

## `data/matopiba_states_polygon.geojson`

Single FeatureCollection with four features and 1,633 vertices in total. Each feature carries
three properties, `name`, `color` and `description`, where `color` is a display hex only. There
is no `estado` property and no aggregate region feature.

| `name` | Geometry | Vertices | `description` |
|--------|----------|---------:|---------------|
| Maranhão | MultiPolygon | 871 | MATOPIBA portion of Maranhão. |
| Tocantins | Polygon | 280 | Tocantins, whose whole territory lies within MATOPIBA. |
| Piauí | Polygon | 265 | MATOPIBA portion of Piauí. |
| Bahia | Polygon | 217 | MATOPIBA portion of Bahia. |

Source, as recorded in the FeatureCollection's own `properties` member: Natural Earth 10m
(`admin_1_states_provinces`), clipped to the MATOPIBA bounding box (west -50.0, east -41.5,
south -15.5, north -1.0).

CRS: EPSG:4326 (WGS84 geographic), the GeoJSON default under RFC 7946.
