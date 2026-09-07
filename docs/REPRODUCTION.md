# Reproducing the results

Every command below runs from the repository root, on the three data files in `data/`.
No credential and no network access is needed for anything in the first table.

## Setup

```bash
pip install -r requirements.txt
python prepare_inputs.py
```

`prepare_inputs.py` writes the intermediate files that the pipeline expects. It creates
no new data. It splits the published CSVs back into the layout the research code was
written against, namely a spectral-feature file and copies at the fixed paths that a few
scripts look for without offering a command-line option. Everything it writes is ignored
by git.

## What reproduces from the published data alone

| Command | Reproduces |
|---------|------------|
| `python examples/pairwise_agreement.py` | The four agreement values against the human reference |
| `python src/models/compute_extended_concordance.py --gt-csv data/matopiba_fire_human_gt_v1.csv --painel-csv data/matopiba_fire_full_panel_v1.csv` | The full 10-pair agreement matrix, with p0, kappa, AC1, PI and BI |
| `python src/models/run_decisive_experiment.py --features 22 --gt-csv data/matopiba_fire_human_gt_v1.csv --painel-csv data/matopiba_fire_full_panel_v1.csv` | The main comparison, 6 classifiers by 5 labeling strategies, plus the feature-importance panel |
| `python src/models/run_decisive_experiment.py --features 13 --gt-csv data/matopiba_fire_human_gt_v1.csv --painel-csv data/matopiba_fire_full_panel_v1.csv` | The same comparison restricted to the 13 non-spectral features |
| `python src/models/temporal_holdout_experiment.py --decisive-csv outputs/decisive_experiment/decisive_results.csv` | Temporal holdout. The within-2022 scenario runs. The cross-year scenario cannot, see the limitation below |
| `python src/models/spatial_holdout_experiment.py` | Leave-One-State-Out over the four state polygons. The 13-feature scenario reproduces exactly at n = 453. The 22-feature scenario runs at n = 373, see the limitation below |
| `python src/models/learning_curve.py` | Kappa as a function of training set size |
| `python src/models/evaluate_baselines.py --gt-path data/matopiba_fire_human_gt_v1.csv` | The FIRMS-confidence and dNBR-threshold baselines |
| `python src/models/dawid_skene_label_sources.py --gt-csv data/matopiba_fire_human_gt_v1.csv --painel-csv data/matopiba_fire_full_panel_v1.csv` | Dawid-Skene sensitivity and specificity per label source |
| `python src/models/identify_consensual_errors.py` | The detections that all five classifiers get wrong |
| `python scripts/proporcao_fogo_real_base_comum.py --gt-csv data/matopiba_fire_human_gt_v1.csv --painel-csv data/matopiba_fire_full_panel_v1.csv` | Share classified as real fire per source, over the common base of n = 439 |
| `python scripts/sensibilidade_convencao_rotulagem.py --gt-csv data/matopiba_fire_human_gt_v1.csv --painel-csv data/matopiba_fire_full_panel_v1.csv` | Sensitivity of the agreement matrix to the labeling convention |
| `python scripts/concordancia_ic_matriz.py` | Bootstrap 95% confidence intervals for the pairs involving the human reference |

### A limitation of the published data, measured

The two CSVs were checked column by column. The nine Sentinel-2 spectral features are
present for 471 of the 484 annotated detections of 2022 and for **none** of the 120
annotated detections of 2023 and 2024. Only the FIRMS and ERA5 columns exist for those
two years.

Two consequences follow, and both are visible when the commands above are run:

1. Leave-One-State-Out with 22 features drops from n = 453 to n = 373, because the 80
   detections of 2023 and 2024 that fall inside a state polygon are discarded for want of
   spectral features. The five per-model kappa values shift accordingly. The 13-feature
   scenario is unaffected and reproduces the published values exactly, all five models.
2. The cross-year temporal holdout, which trains on 2022 and tests on 2023 and 2024, ends
   with an empty test set and is skipped by the script, which reports
   `Teste: n=0 ... Pulando`. The within-2022 scenario runs normally.

Everything else in the first table is unaffected, because the agreement matrix and the
main classifier comparison are both anchored on the 2022 subset with complete coverage.

### Verified reproduction

The main comparison was re-run inside this repository, against the run that produced the
dissertation tables. The 6 by 5 result matrix came back with a maximum numeric difference
of `0.0` across all 41 columns. The ten agreement pairs match the published table cell by
cell, and `examples/pairwise_agreement.py` returns the four human-reference values exactly
as printed there.

## What needs more than the published data

| Command | What it additionally needs |
|---------|---------------------------|
| `src/preprocessing/recompute_weak_labels_mcd64a1.py` | An authenticated Google Earth Engine account. It recomputes the strict FIRMS by MCD64A1 crossing that produced the `is_reliable` column already shipped in `data/` |
| `src/preprocessing/extract_sentinel2_features.py`, `extract_mapbiomas_fire.py`, `auto_validate_dnbr.py` | Earth Engine. These built the spectral features and the dNBR label already shipped in `data/` |
| `src/data_ingest/*` | A credential for each provider, read from a local `.env`. Nothing in this repository carries one |
| `src/models/mcd64a1_radius_sensitivity.py` | Earth Engine, for the radius ladder around each detection |
| `src/models/concordancia_base_estendida.py` | The columns `auto_mean_dnbr`, `auto_max_dnbr` and `auto_pct_burned`, which the published package does not carry. The near-equivalent `dnbr_mean`, `dnbr_max` and `pct_burned` are present but are not the same series, so substituting them would silently change the rule-audit reproduction rate this script reports |
| `src/visualization/annotation_app.py` | Earth Engine, to render the three thumbnails live. It is included as the operational definition of the blind protocol |

## Notes on determinism

The classifiers are seeded, and repeated runs on the same input give identical numbers.
The bootstrap confidence intervals are also seeded. Differences in library versions can
move the last decimal place, since LightGBM and XGBoost do not guarantee bit-identical
results across releases.

The label convention is a parameter and not a constant. `src/preprocessing/labeling_convention.py`
is the single source of that projection. The operative convention throughout is
`sensu_sensor`. The alternative is available for inspection and is not the source of any
reported finding.
