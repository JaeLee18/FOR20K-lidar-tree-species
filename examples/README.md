# Example scans

Four single-tree scans from the FOR-species20K public labeled set, one per LiDAR type plus a broadleaf TLS tree:

| File | Species | Source dataset | LiDAR | Points | Height |
|---|---|---|---|---|---|
| `00415_Picea_abies_TLS.laz` | *Picea abies* (Norway spruce) | Seidl_2021 | terrestrial | 115,626 | 31.3 m |
| `00883_Fagus_sylvatica_TLS.laz` | *Fagus sylvatica* (European beech) | Seidl_2021 | terrestrial | 142,187 | 25.9 m |
| `03450_Quercus_robur_MLS.laz` | *Quercus robur* (pedunculate oak) | LAUTx | mobile | 150,574 | 26.3 m |
| `05554_Pinus_sylvestris_ULS.laz` | *Pinus sylvestris* (Scots pine) | Weiser_2022b | drone | 53,839 | 33.2 m |

```bash
./predict.sh examples/
```

All four are classified correctly by the held-out fold model as well as by the full ensemble. Four of the five fold
models saw these trees during training, so the ensemble's near-100% confidence is not an accuracy estimate.

**Source and license.** Unmodified files from Puliti et al., *FOR-species20K dataset*, Zenodo,
[doi:10.5281/zenodo.13255198](https://doi.org/10.5281/zenodo.13255198), distributed under GPL-3.0-or-later.
