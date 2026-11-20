from rslp.forest_loss_driver import hierarchy


def test_tree_structure() -> None:
    assert len(hierarchy.LEAVES) == 22
    assert len(hierarchy.LEVEL2_NODES) == 13
    assert {hierarchy.LEVEL1_NODES[i] for i in hierarchy.LEAF_TO_LEVEL1} == set(
        hierarchy.LEVEL1_NODES
    )
    assert set(hierarchy.LEAF_TO_LEVEL2) == set(range(len(hierarchy.LEVEL2_NODES)))
    # No leaf is an ancestor of another leaf.
    for leaf, path in hierarchy.LEAF_PATHS.items():
        assert hierarchy.leaves_under(*path) == {leaf}
    assert set(hierarchy.LEAF_TO_LEGACY.values()) == set(hierarchy.LEGACY_CLASSES)


def test_label_maps_cover_hierarchy() -> None:
    label_maps = (
        hierarchy.FLAT_LABEL_LEAVES,
        hierarchy.PERU_LEGACY_LABEL_LEAVES,
        hierarchy.ACA_LABEL_LEAVES,
        hierarchy.MONOCROP_LABEL_LEAVES,
    )
    for label_map in label_maps:
        for leaves in label_map.values():
            assert leaves
            assert leaves <= set(hierarchy.LEAVES)
    for label_map in label_maps[:3]:
        assert "unknown" not in label_map


def test_label_mappings() -> None:
    flat = hierarchy.FLAT_LABEL_LEAVES
    assert flat["agriculture"] == hierarchy.leaves_under("human_made", "agriculture")
    assert flat["road"] == {"road", "logging_road"}
    assert flat["logging"] == {"selective_logging", "logging_road"}
    assert flat["burned"] == {"burned_area", "natural_other"}
    assert flat["river"] == {"river_dynamics", "flooding"}
    assert flat["none"] == {"no_change"}

    aca = hierarchy.ACA_LABEL_LEAVES
    assert "agriculture_small_scale" not in aca["Agriculture-Large"]
    assert aca["Mining_in_River"] == {"mining_riverine"}
    assert aca["Anthropic_-_Unknown"] == {"human_other"}
    assert aca["Natural_-_Unknown"] == {"natural_other"}
    assert aca["Urbanization_settlements"] == {"infrastructure_other"}

    peru = hierarchy.PERU_LEGACY_LABEL_LEAVES
    assert peru["coca"] == {"agriculture_small_scale"}
    assert peru["agriculture-mennonite"] == hierarchy.AGRICULTURE_LARGE_MEDIUM

    monocrop = hierarchy.MONOCROP_LABEL_LEAVES
    assert monocrop["mennonites_soybean"] == {"agriculture_soybean"}
    assert "agriculture_soybean" not in monocrop["mennonites_nonsoybean"]


def test_parse_levels() -> None:
    assert hierarchy.parse_levels("human_made", "agriculture", "soybean") == {
        "agriculture_soybean"
    }
    assert hierarchy.parse_levels("human_made", "other_lvl2", None) == {"human_other"}
    assert hierarchy.parse_levels("natural", "other_lvl2", None) == {"natural_other"}
    assert hierarchy.parse_levels("natural", None, None) == hierarchy.leaves_under(
        "natural"
    )
    assert (
        hierarchy.parse_levels("human_made", "agriculture", "mennonites")
        == hierarchy.AGRICULTURE_LARGE_MEDIUM
    )
    # Level 3 values that are not under the Level 2 node are dropped.
    assert hierarchy.parse_levels("non_forest_loss", "already_altered", "soybean") == {
        "already_altered"
    }
    assert hierarchy.parse_levels("unknown", None, None) is None
    assert hierarchy.parse_levels(None, "agriculture", None) is None


def test_covering_nodes() -> None:
    assert hierarchy.covering_nodes(hierarchy.MINING) == ["human_made/mining"]
    assert hierarchy.covering_nodes({"road", "logging_road"}) == [
        "human_made/infrastructures/roads",
        "human_made/forest_extraction/logging_roads",
    ]
    assert hierarchy.covering_nodes(set(hierarchy.LEAVES)) == [
        "human_made",
        "natural",
        "non_forest_loss",
    ]
