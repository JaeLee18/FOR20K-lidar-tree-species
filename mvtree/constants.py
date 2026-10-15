"""Class list and per-resolution render normalisation statistics."""

# Index order = sklearn LabelEncoder on the `species` column (alphabetical).
# The released weights emit logits in exactly this order.
SPECIES = [
    "Abies_alba", "Acer_campestre", "Acer_pseudoplatanus", "Acer_saccharum",
    "Betula_pendula", "Carpinus_betulus", "Corylus_avellana", "Crataegus_monogyna",
    "Eucalyptus_miniata", "Euonymus_europaeus", "Fagus_sylvatica", "Fraxinus_angustifolia",
    "Fraxinus_excelsior", "Larix_decidua", "Picea_abies", "Picea_glauca",
    "Pinus_contorta", "Pinus_nigra", "Pinus_pinaster", "Pinus_radiata",
    "Pinus_resinosa", "Pinus_sylvestris", "Populus_deltoides", "Populus_tremuloides",
    "Prunus_avium", "Pseudotsuga_menziesii", "Quercus_faginea", "Quercus_ilex",
    "Quercus_petraea", "Quercus_robur", "Quercus_rubra", "Tilia_cordata",
    "Ulmus_laevis",
]
NUM_CLASSES = len(SPECIES)
GENUS = {s: s.split("_")[0] for s in SPECIES}

# Channel statistics of the (depth, density, height) renders over all 17,707
# public labeled trees x 7 views (scripts/data/compute_render_stats.py). The renders
# are geometric maps, so ImageNet statistics do not apply.
RENDER_STATS = {
    256: {"mean": [0.0504, 0.0282, 0.0692], "std": [0.1630, 0.1091, 0.2099]},
    512: {"mean": [0.0317, 0.0130, 0.0429], "std": [0.1314, 0.0713, 0.1682]},
}

DEFAULT_BACKBONE = "convnextv2_base.fcmae_ft_in22k_in1k"
PLATFORMS = ("TLS", "MLS", "ULS")


def render_stats(img_size: int) -> dict:
    if img_size not in RENDER_STATS:
        raise KeyError(f"no render statistics for {img_size}px; run scripts/data/compute_render_stats.py "
                       f"and add them to mvtree.constants.RENDER_STATS")
    return RENDER_STATS[img_size]
