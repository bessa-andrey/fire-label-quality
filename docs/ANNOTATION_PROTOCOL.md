# Annotation protocol and taxonomy

This document describes how the human labels in `data/matopiba_fire_human_gt_v1.csv`
were produced. It is the operational definition behind the `human_label`,
`human_confidence` and `human_notes` columns.

## What is being labeled

The target is **detection quality**, which asks whether a FIRMS thermal anomaly
corresponds to actual biomass combustion on the ground. It is not fire typology, which
would ask what kind of fire it was. The two questions are separable, and the dataset
answers only the first one.

The unified convention across all five label sources is:

| Value | Meaning |
|------:|---------|
| `0` | Real fire |
| `1` | Spurious detection |
| `-1` | Uncertain, excluded from all agreement computations |

The governing criterion is the **origin of the thermal signal**. A signal produced by
biomass combustion is real. A signal produced by a fixed anthropogenic surface or by a
reflection is spurious. Proximity to an urban perimeter or to an agricultural matrix does
not by itself make a detection spurious. A detection inside such a context that shows
unambiguous burn evidence, meaning a visible scar in the RGB composite together with a
positive dNBR, is labeled `0`.

This is the *sensu sensor* convention: every confirmed combustion counts as real fire,
including planned agricultural burning. It is the convention that aligns the human
reference with the automated sources, all of which detect combustion rather than intent.
An alternative *sensu gestor* convention, under which management burning would count as
spurious, is discussed in the dissertation as future work and is not applied here.

## Two-stage annotation

The 604 samples were annotated by the principal annotator in two stages.

1. **Open annotation** (calibration). Each sample was labeled with full access to the
   coexisting automated labels. This stage is not distributed as a label column and served
   only to stabilize the criteria.
2. **Blind re-annotation** (definitive). Each sample was labeled again with no access to
   any automated label. This stage produced the `human_label` column.

In the blind stage the annotator saw exactly three Sentinel-2 thumbnails and a block of
neutral FIRMS metadata, and nothing else:

- pre-fire RGB composite, from the window of 60 to 30 days before the detection,
- post-fire RGB composite, from the window of 0 to 30 days after the detection,
- dNBR composite over the same two windows,
- latitude, longitude, acquisition date, FIRMS `confidence`, `frp` and `brightness`.

The annotation interface used for this stage is included in this repository as
`src/visualization/annotation_app.py`.

### A documented limitation

The dNBR thumbnail was visible during blind annotation. The dNBR automatic source is
computed from the same quantity, so the human reference and the dNBR source are not
independent. This dependence is acknowledged in the dissertation and is the reason the
comparison includes an ablation of the dNBR-derived features. Any reuse of this dataset
that compares the human reference against the dNBR source inherits the same caveat.

## Taxonomy of ten categories

Whenever a decision had a stated justification, the annotator recorded it in the free-text
field now published as `human_notes`. The field was filled for 205 of the 604 samples,
which means absence of a note carries no information. The standardized categories are:

| # | Category | Visual signature | Class |
|--:|----------|------------------|------:|
| 1 | Sensor false positive | No visual evidence of combustion, no burn scar, non-positive dNBR in the buffer | `1` |
| 2 | Controlled agricultural burning | Scar with straight edges and regular geometric pattern, indicating planned burning of a field. Combustion did occur | `0` |
| 3 | Prescribed burning, fire management | Controlled burn for fuel reduction, typical of protected areas, usually early in the dry season from May to July. Combustion did occur | `0` |
| 4 | Irregular fire over cropland or pasture | Scar with organic irregular edges and no evident geometric pattern, possibly uncontrolled fire advancing over adjacent native vegetation | `0` |
| 5 | Urban perimeter | Fixed anthropogenic heat inside an urban perimeter, no vegetation burn scar visible | `1` |
| 6 | Industrial park | Persistent thermal signature of an industrial complex | `1` |
| 7 | Water body reflection | Specular solar reflection, sun glint, over a water surface | `1` |
| 8 | Fire near an urban area | Real vegetation fire adjacent to inhabited area | `0` |
| 9 | Image compromised by clouds | Cloud cover in the post-fire composite prevents visual verification of the ground | `-1` |
| 10 | No satellite imagery available | No Sentinel-2 imagery in the required window for that location | `-1` |

Two categories in this list, rocky-soil heating and its variants, appear in `human_notes`
under four spellings that reduce to the same case. They are recorded verbatim as typed.

Categories 2, 3, 4 and 8 are the cases where the label depends on the convention. Under
*sensu sensor* they are real fire, because combustion occurred. Under a management-oriented
convention some of them would move.

## Resulting distribution

Over the 604 annotated detections:

| `human_label` | n | share |
|--------------:|--:|------:|
| `0`, real fire | 423 | 70.0% |
| `1`, spurious | 112 | 18.5% |
| `-1`, uncertain | 69 | 11.4% |

The 535 samples with a binary label are the working set for every agreement computation.
The subset that also has complete Sentinel-2 coverage has 439 samples, and that is the
denominator whenever a comparison involves the Painel do Fogo or the MapBiomas Fogo
source. The three sample sizes are distinct and should not be interchanged.

`human_confidence` is the annotator's self-declared certainty on a 1 to 5 scale, recorded
per sample. It is published for transparency and is not used as a weight anywhere in the
analysis.

## Inter-annotator evidence

A separate multi-annotator calibration campaign was run over an 80-sample subset with
three external volunteers. Only the aggregate outcome is reportable, a Fleiss kappa of
0.36 among the volunteers, which is empirical evidence of human variability on this task.
The individual volunteer annotations are third-party personal data and are not
distributed, in any form, including under pseudonym.
