# fire-label-quality

**Labeling strategies for satellite fire hotspot classification in the MATOPIBA region**

Five independent label sources for the same set of NASA FIRMS fire detections, the 22
engineered features that describe each detection, and the pipeline that reproduces the
comparison between them.

This repository accompanies the master's dissertation *Estratégias de Rotulagem para
Classificação de Focos de Incêndio por Satélite na Região MATOPIBA*, PPGEE, Universidade
Federal do Amazonas, 2026.

---

## The question

Whether a FIRMS thermal detection is a real fire or a spurious signal has no single
answer, because it depends on what you compare it against. Multi-sensor agreement,
spectral burn evidence, operational fire records and human visual judgment are used
interchangeably in the literature as if they were the same reference. They are not. Each
one projects the same detections onto a different construct, and a classifier trained
against one of them does not behave like a classifier trained against another.

This repository makes that comparison auditable. It distributes the five label sources
side by side, over the same detections, with the features and the code.

| Source | Column | Mapping to the unified convention |
|--------|--------|-----------------------------------|
| Weak labels, FIRMS crossed with MCD64A1 | `is_reliable` | `y = 1 - is_reliable` |
| dNBR automatic, Sentinel-2 | `manual_label` | `y = manual_label` |
| MapBiomas Fogo | `gt_mapbiomas_fire` | `y = 1 - gt_mapbiomas_fire` |
| Painel do Fogo, CENSIPAM | `gt_painel` | `y = 1 - gt_painel` |
| Human blind annotation | `human_label` | `y = human_label` |

Unified convention: `0` is real fire, `1` is spurious, `-1` is uncertain and excluded.
The target is detection quality, meaning whether combustion occurred, and not fire
typology.

Pairwise agreement between the five sources ranges from a Cohen kappa of 0.05 to 0.35.
The share of detections each source calls real fire ranges from 16.4% to 79.5% over the
same 439 detections.

---

## What is here

```
data/
  matopiba_fire_human_gt_v1.csv     604 detections, 5 label sources, 22 features
  matopiba_fire_full_panel_v1.csv   3,698 detections, 4 automated sources, 22 features
  matopiba_states_polygon.geojson   4 state polygons, used in spatial validation
  DATA_DICTIONARY.md                every column, with unit, domain and provenance
docs/
  ANNOTATION_PROTOCOL.md            the blind protocol and the 10-category taxonomy
  REPRODUCTION.md                   the commands, and what each one reproduces
examples/
  pairwise_agreement.py             minimal example, agreement against the human labels
prepare_inputs.py                   derives the intermediate files the pipeline expects
src/                                the pipeline
scripts/                            complementary analyses
```

## Quick start

```bash
pip install -r requirements.txt
python examples/pairwise_agreement.py
```

```
fonte               kappa   concordancia      n
------------------------------------------------
weak_labels         0.049         34.2%    535
dnbr_auto           0.353         76.3%    535
mapbiomas_fogo      0.081         39.0%    439
painel_fogo         0.098         57.4%    439
```

Those four values are the "versus human annotation" column of the agreement matrix in the
dissertation, reproduced from the published files alone. To reproduce the full matrix and
the classifier comparison, see [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

---

## Licensing

This repository carries **two licenses**, and the boundary between them is by file type,
not by directory.

| What | License | File |
|------|---------|------|
| **Source code**, meaning every `.py` file in `src/`, `scripts/`, `examples/`, and `prepare_inputs.py` itself | MIT | [`LICENSE`](LICENSE) |
| **Data and documentation**, meaning everything in `data/`, everything in `docs/`, this `README.md`, and `CITATION.cff` | CC BY 4.0 | [`LICENSE-DATA`](LICENSE-DATA) |

In plain terms. Take the code and do what you want with it, keeping the copyright notice.
Take the data, the annotations, the derived features and the written documentation, and
do what you want with them, giving credit and saying if you changed anything.

Nothing in this repository is dual-licensed. Every file falls under exactly one of the
two, and the table above decides which.

### Third-party products

The features distributed here are values derived point by point from third-party
products. Those products remain under the licenses of their own providers, and each one
requires the attribution it demands. None of them is redistributed in its original form.

| Provider | What this repository publishes |
|----------|-------------------------------|
| NASA FIRMS | Coordinates, date, brightness, confidence, FRP |
| NASA MODIS MCD64A1 | One binary value per detection, the result of the strict crossing |
| ESA Copernicus Sentinel-2 | NDVI, NBR, dNBR and burned share, aggregated over a 500 m buffer |
| ECMWF ERA5 | Six meteorological variables at the point and date |
| MapBiomas, Fogo and Land Cover | Two derived values per detection |
| CENSIPAM Painel do Fogo | One binary value per detection |
| Natural Earth | The four clipped state polygons, public domain |

---

## What is deliberately absent

The individual annotations from the multi-annotator calibration campaign are not here, in
any form, including under pseudonym. They are third-party personal data tied to individual
performance assessments, and re-identification would be straightforward. The dissertation
reports only the aggregate outcome from that campaign, a Fleiss kappa of 0.36 among the
volunteers.

---

## Citation

```bibtex
@mastersthesis{bessa2026matopiba,
  title  = {Estrat{\'e}gias de Rotulagem para Classifica{\c{c}}{\~a}o de Focos de
            Inc{\^e}ndio por Sat{\'e}lite na Regi{\~a}o MATOPIBA},
  author = {Bessa, Andrey R{\'u}ben Ribeiro},
  year   = {2026},
  school = {Universidade Federal do Amazonas},
  type   = {Disserta{\c{c}}{\~a}o de Mestrado, PPGEE},
  address = {Manaus, Brazil}
}
```

## Acknowledgments

Research advisor: Prof. D.Sc. Celso Barbosa Carvalho, PPGEE, Universidade Federal do
Amazonas. Co-advisor: Prof. D.Sc. André Chaves Mendes, Instituto Politécnico de
Portalegre.

Thanks to NASA FIRMS, ESA Copernicus, ECMWF, MapBiomas and CENSIPAM for the underlying
data products.

## Contact

Andrey Bessa, andrey.bessa@ufam.edu.br
