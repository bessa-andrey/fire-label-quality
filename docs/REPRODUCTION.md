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
| `python src/models/run_decisive_experiment.py --features 22 --gt-csv data/processed/validation/validation_manual_gt.csv --painel-csv data/matopiba_fire_full_panel_v1.csv` | The main comparison, 6 classifiers by 5 labeling strategies, plus the feature-importance panel. The `--gt-csv` is the copy `prepare_inputs.py` writes, see the note below |
| `python src/models/run_decisive_experiment.py --features 13 --gt-csv data/processed/validation/validation_manual_gt.csv --painel-csv data/matopiba_fire_full_panel_v1.csv` | The same comparison restricted to the 13 non-spectral features |
| `python src/models/temporal_holdout_experiment.py --decisive-csv outputs/decisive_experiment_22feat/decisive_results.csv` | Temporal holdout, both scenarios. The cross-year one trains on 2022 (n = 439) and tests on 2023 and 2024 (n = 95) |
| `python src/models/spatial_holdout_experiment.py` | Leave-One-State-Out over the four state polygons, at n = 453 in both the 13-feature and the 22-feature scenario |
| `python src/models/learning_curve.py` | Kappa as a function of training set size |
| `python src/models/evaluate_baselines.py --gt-path data/matopiba_fire_human_gt_v1.csv` | The FIRMS-confidence and dNBR-threshold baselines |
| `python src/models/dawid_skene_label_sources.py --gt-csv data/matopiba_fire_human_gt_v1.csv --painel-csv data/matopiba_fire_full_panel_v1.csv` | Dawid-Skene sensitivity and specificity per label source |
| `python src/models/identify_consensual_errors.py` | The detections that all five classifiers get wrong |
| `python scripts/proporcao_fogo_real_base_comum.py --gt-csv data/matopiba_fire_human_gt_v1.csv --painel-csv data/matopiba_fire_full_panel_v1.csv` | Share classified as real fire per source, over the common base of n = 439 |
| `python scripts/sensibilidade_convencao_rotulagem.py --gt-csv data/matopiba_fire_human_gt_v1.csv --painel-csv data/matopiba_fire_full_panel_v1.csv` | Sensitivity of the agreement matrix to the labeling convention |
| `python scripts/concordancia_ic_matriz.py` | Bootstrap 95% confidence intervals for the pairs involving the human reference |

### Two human-reference files, and which command takes which

`prepare_inputs.py` writes `data/processed/validation/validation_manual_gt.csv`, a copy of
the published human reference **without** the nine spectral columns. That copy is not a
convenience: it is what fixes the evaluation subset of the main comparison, and the table
above uses it on purpose.

The main comparison merges its spectral features from `data/processed/sentinel2_features.csv`,
which covers the 3,698 detections of 2022. Its 22-feature track is then filtered by
`dnbr_mean.notna()`, which leaves n = 439 of the 535 detections labelled Real or Spurious.
That is the n printed in the dissertation. The merge completes rather than replaces, so
passing `data/matopiba_fire_human_gt_v1.csv` instead would carry the spectral columns of
2023 and 2024 straight into the filter, move it to n = 534, and change all fifteen kappa
values of the 13-vs-22 table, by up to 0.126. Use the copy the setup step writes.

Four other commands read the extended panel, which `prepare_inputs.py` builds from the
panel plus the published human reference, spectral columns included. That is why
Leave-One-State-Out runs at n = 453 with 22 features, the cross-year holdout has a test set
of n = 95, the learning curve runs on the consolidated n = 534, and the consensual-error
analysis finds the 58 detections reported in the dissertation, 43 false negatives and 15
false positives.

Coverage of the spectral block in the published human reference, measured on the file: 573
of the 604 rows carry all eight scene-derived columns, 597 carry at least one, and 601
carry `land_use_class`. A row is incomplete where no cloud-free Sentinel-2 scene was
available in the pre-fire or the post-fire window. Until the September 2026 revision the
120 rows of 2023 and 2024 carried none at all, and both holdout experiments were degraded
by it, one running at n = 373 and the other skipped for an empty test set.

### Verified reproduction

The main comparison was re-run inside this repository, against the run that produced the
dissertation tables. The 6 by 5 result matrix came back with a maximum numeric difference
of `0.0` across all 41 columns. The ten agreement pairs match the published table cell by
cell, and `examples/pairwise_agreement.py` returns the four human-reference values exactly
as printed there.

The three experiments that depend on the spectral block were re-run as well, with the
commands exactly as they appear above. The main comparison evaluates on n = 439 and its
sixteen-column comparison file comes back at a maximum difference of `0.0` against the run
that produced the printed table. Leave-One-State-Out with 22 features runs at n = 453 and
reproduces all twenty cells of the spatial-holdout table. The cross-year holdout tests on
n = 95 and reproduces all fifteen cells of the temporal-holdout table. The learning curve
reproduces both of its variants, ten mean-kappa points and four Wilcoxon p-values, and the
consensual-error analysis returns 58 detections split 43 to 15.

One printed value does not reproduce, and the package is not the reason. The 13-vs-22
table prints 0.202 for Extra Trees in the 19-features-without-dNBR column. The underlying
value is 0.201481, in the run that produced the table and in every run since. It is a
rounding slip in the manuscript, not a difference between the package and the source.

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
