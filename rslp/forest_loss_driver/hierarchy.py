"""Label hierarchy for the forest loss driver model.

ACA's hierarchy has three levels: Level 1 (human-made, natural, non-forest loss), Level
2 (e.g. agriculture, mining, landslide), and Level 3 (e.g. soybean, roads). The model
predicts only the leaves of the tree. A Studio label from any of the annotation
schemes resolves to a set of allowed leaves: a single leaf when the label is specific
enough, or several leaves when it maps to a non-leaf node (e.g. "agriculture") or is
ambiguous under the new hierarchy (e.g. the old "road" can be a road or a logging
road).

The node names in the leaf paths are the label names used by the level1/level2/level3
labelsets in Studio, so that annotations made with the new hierarchy map directly.
"""

from __future__ import annotations

HUMAN_MADE = "human_made"
NATURAL = "natural"
NON_FOREST_LOSS = "non_forest_loss"

# Leaf name -> path of Studio level labels (level1, level2[, level3]).
LEAF_PATHS: dict[str, tuple[str, ...]] = {
    "agriculture_small_scale": (HUMAN_MADE, "agriculture", "small_scale"),
    "agriculture_pasture": (HUMAN_MADE, "agriculture", "pasturecattle"),
    "agriculture_soybean": (HUMAN_MADE, "agriculture", "soybean"),
    "agriculture_oil_palm": (HUMAN_MADE, "agriculture", "oil_palm"),
    "agriculture_rice": (HUMAN_MADE, "agriculture", "rice"),
    "agriculture_other": (HUMAN_MADE, "agriculture", "other_lvl3"),
    "mining_terrace": (HUMAN_MADE, "mining", "terrace_mining"),
    "mining_riverine": (HUMAN_MADE, "mining", "riverine_mining"),
    "airstrip": (HUMAN_MADE, "infrastructures", "airstrips"),
    "road": (HUMAN_MADE, "infrastructures", "roads"),
    "infrastructure_other": (HUMAN_MADE, "infrastructures", "other_lvl3"),
    "logging_road": (HUMAN_MADE, "forest_extraction", "logging_roads"),
    "selective_logging": (HUMAN_MADE, "forest_extraction", "selective_logging"),
    "burned_area": (HUMAN_MADE, "burned_area"),
    "human_other": (HUMAN_MADE, "other_lvl2"),
    "landslide": (NATURAL, "landslide"),
    "blowdown": (NATURAL, "blowdown"),
    "river_dynamics": (NATURAL, "river_dynamics"),
    "flooding": (NATURAL, "flooding"),
    "natural_other": (NATURAL, "other_lvl2"),
    "no_change": (NON_FOREST_LOSS, "no_change"),
    "already_altered": (NON_FOREST_LOSS, "already_altered"),
}
LEAVES: tuple[str, ...] = tuple(LEAF_PATHS)

LEVEL1_NODES: tuple[str, ...] = (HUMAN_MADE, NATURAL, NON_FOREST_LOSS)


def _level2_name(path: tuple[str, ...]) -> str:
    # other_lvl2 appears under both human_made and natural.
    if path[1] == "other_lvl2":
        return f"{path[0].removesuffix('_made')}_other"
    return path[1]


LEVEL2_NODES: tuple[str, ...] = tuple(
    dict.fromkeys(_level2_name(path) for path in LEAF_PATHS.values())
)
# Index of the Level 1 / Level 2 node of each leaf, in LEAVES order.
LEAF_TO_LEVEL1: tuple[int, ...] = tuple(
    LEVEL1_NODES.index(path[0]) for path in LEAF_PATHS.values()
)
LEAF_TO_LEVEL2: tuple[int, ...] = tuple(
    LEVEL2_NODES.index(_level2_name(path)) for path in LEAF_PATHS.values()
)

# The flat categories predicted by the models before the hierarchy was introduced.
LEGACY_CLASSES: tuple[str, ...] = (
    "agriculture",
    "mining",
    "airstrip",
    "road",
    "logging",
    "burned",
    "landslide",
    "hurricane",
    "river",
    "none",
)
# Leaf -> legacy class, for comparing against models trained on the flat categories.
# Leaves that have no legacy equivalent are omitted.
LEAF_TO_LEGACY: dict[str, str] = {
    "agriculture_small_scale": "agriculture",
    "agriculture_pasture": "agriculture",
    "agriculture_soybean": "agriculture",
    "agriculture_oil_palm": "agriculture",
    "agriculture_rice": "agriculture",
    "agriculture_other": "agriculture",
    "mining_terrace": "mining",
    "mining_riverine": "mining",
    "airstrip": "airstrip",
    "road": "road",
    "logging_road": "road",
    "selective_logging": "logging",
    "burned_area": "burned",
    "landslide": "landslide",
    "blowdown": "hurricane",
    "river_dynamics": "river",
    "flooding": "river",
    "no_change": "none",
    "already_altered": "none",
}


def leaves_under(*prefix: str) -> frozenset[str]:
    """Get the leaves under the node identified by a prefix of Studio level labels.

    Args:
        prefix: the node path, e.g. ("human_made", "agriculture").

    Returns:
        the set of leaves whose path starts with the prefix (empty if the prefix does
            not identify a node in the tree).
    """
    return frozenset(
        leaf for leaf, path in LEAF_PATHS.items() if path[: len(prefix)] == prefix
    )


def covering_nodes(leaves: frozenset[str] | set[str]) -> list[str]:
    """Get the largest nodes whose leaves are all in the given set.

    This gives a compact description of a leaf set, e.g. the whole agriculture subtree
    is described as ["human_made/agriculture"].

    Args:
        leaves: a set of leaves.

    Returns:
        the node paths (joined with "/") covering exactly the given leaves, in LEAVES
            order.
    """
    nodes: list[str] = []
    for leaf in LEAVES:
        if leaf not in leaves:
            continue
        path = LEAF_PATHS[leaf]
        for depth in range(1, len(path) + 1):
            if leaves_under(*path[:depth]) <= leaves:
                node = "/".join(path[:depth])
                if node not in nodes:
                    nodes.append(node)
                break
    return nodes


AGRICULTURE = leaves_under(HUMAN_MADE, "agriculture")
# Agriculture over 1 ha: everything except small-scale agriculture.
AGRICULTURE_LARGE_MEDIUM = AGRICULTURE - {"agriculture_small_scale"}
MINING = leaves_under(HUMAN_MADE, "mining")
ROAD_OR_LOGGING_ROAD = frozenset({"road", "logging_road"})
LOGGING = frozenset({"selective_logging", "logging_road"})
RIVER = frozenset({"river_dynamics", "flooding"})
# Burned labels from the old schemes don't say whether the fire was natural.
BURNED = frozenset({"burned_area", "natural_other"})


def _leaf(name: str) -> frozenset[str]:
    assert name in LEAF_PATHS
    return frozenset({name})


# The flat categories (Validatetest, the "validate" field of Validation 2026-09).
FLAT_LABEL_LEAVES: dict[str, frozenset[str]] = {
    "agriculture": AGRICULTURE,
    "mining": MINING,
    "airstrip": _leaf("airstrip"),
    "road": ROAD_OR_LOGGING_ROAD,
    "logging": LOGGING,
    "burned": BURNED,
    "landslide": _leaf("landslide"),
    "hurricane": _leaf("blowdown"),
    "river": RIVER,
    "none": _leaf("no_change"),
    "general_clearance": _leaf("human_other"),
    "other_natural_disturbance": _leaf("natural_other"),
    "already_altered": _leaf("already_altered"),
}

# Original Peru label hierarchy (Forest Loss Driver Peru 7). "unknown" is skipped.
PERU_LEGACY_LABEL_LEAVES: dict[str, frozenset[str]] = FLAT_LABEL_LEAVES | {
    "agriculture-generic": AGRICULTURE,
    "agriculture-small": _leaf("agriculture_small_scale"),
    "agriculture-rice": _leaf("agriculture_rice"),
    "agriculture-mennonite": AGRICULTURE_LARGE_MEDIUM,
    "coca": _leaf("agriculture_small_scale"),
    "flood": RIVER,
    "human": _leaf("human_other"),
    "natural": _leaf("natural_other"),
}

# ACA tag_name hierarchy (Brazil/Colombia 7 and 12, Peru 20260112c). "unknown" and
# "unlabeled" are skipped.
ACA_LABEL_LEAVES: dict[str, frozenset[str]] = {
    "Pasture_praderas_ganaderia": _leaf("agriculture_pasture"),
    "Agriculture-Small": _leaf("agriculture_small_scale"),
    "Agriculture-Medium": AGRICULTURE_LARGE_MEDIUM,
    "Agriculture-Large": AGRICULTURE_LARGE_MEDIUM,
    "Mining": MINING,
    "Mining_in_River": _leaf("mining_riverine"),
    "Roads": _leaf("road"),
    "Logging_roads": _leaf("logging_road"),
    "Airstrips": _leaf("airstrip"),
    "Selective_Logging": _leaf("selective_logging"),
    "Logging_clear-cut": _leaf("human_other"),
    "Urbanization_settlements": _leaf("infrastructure_other"),
    "Burned_areas_Fire": BURNED,
    "Landslides": _leaf("landslide"),
    "Windthrowsblowdowns_Hurricane_winds": _leaf("blowdown"),
    "Flooded_rivers": RIVER,
    "Dry_areas_seasonal": _leaf("natural_other"),
    "General_deforestation_Clearing": _leaf("human_other"),
    "False_positives_not_forest_loss_an_alert_error": _leaf("no_change"),
    "Anthropic_-_Unknown": _leaf("human_other"),
    "Natural_-_Unknown": _leaf("natural_other"),
}

# Monocrop monoculture_tag labels. The monocrop scheme has no small-scale class, so
# other_agriculture may be small-scale.
MONOCROP_LABEL_LEAVES: dict[str, frozenset[str]] = {
    "soybean": _leaf("agriculture_soybean"),
    "mennonites_soybean": _leaf("agriculture_soybean"),
    "oil_palm": _leaf("agriculture_oil_palm"),
    "pastures": _leaf("agriculture_pasture"),
    "rice": _leaf("agriculture_rice"),
    "mennonites_nonsoybean": AGRICULTURE_LARGE_MEDIUM - {"agriculture_soybean"},
    "other_agriculture": frozenset({"agriculture_other", "agriculture_small_scale"}),
}


def parse_levels(
    level1: str | None, level2: str | None, level3: str | None
) -> frozenset[str] | None:
    """Resolve Studio level1/level2/level3 labels to the allowed leaves.

    The deepest prefix of the levels that identifies a node is used, so a level3 value
    that is not a leaf under the given level2 (e.g. soybean under already_altered) is
    dropped. Mennonites are not a leaf (Mennonite colonies are Level 4, recorded as a
    separate attribute), but a Mennonite colony is large/medium agriculture.

    Args:
        level1: the Level 1 label.
        level2: the Level 2 label.
        level3: the Level 3 label.

    Returns:
        the allowed leaves, or None if level1 does not identify a node (e.g. it is
            missing or Unknown).
    """
    if level1 not in LEVEL1_NODES:
        return None
    if (level1, level2, level3) == (HUMAN_MADE, "agriculture", "mennonites"):
        return AGRICULTURE_LARGE_MEDIUM
    prefix = (level1, level2, level3)
    for depth in (3, 2, 1):
        if None in prefix[:depth]:
            continue
        leaves = leaves_under(*prefix[:depth])  # type: ignore[arg-type]
        if leaves:
            return leaves
    return None
