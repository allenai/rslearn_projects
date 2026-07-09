"""Build the RPG CODE_CULTU -> PASTIS-18 semantic class mapping.

PASTIS (Garnot & Landrieu, ICCV 2021) labels 18 crop classes + background(0), derived
from the French RPG. This reproduces that CODE_CULTU -> class_id mapping so it can be
applied nationwide (all of France incl. DROM/Corsica), instead of only the 4 original
Sentinel-2 tiles.

Class indices (must match the existing eval's PASTIS_NAMES):
  0 Background   1 Meadow            2 Soft winter wheat   3 Corn
  4 Winter barley 5 Winter rapeseed  6 Spring barley       7 Sunflower
  8 Grapevine    9 Beet             10 Winter triticale   11 Winter durum wheat
 12 Fruits/vegetables/flowers       13 Potatoes           14 Leguminous fodder
 15 Soybeans     16 Orchard         17 Mixed cereal       18 Sorghum

Approach: a high-confidence CORE dict for the codes that *define* each class (winter/
spring distinctions matter -- PASTIS keeps ORH(4) vs ORP(6) separate, so we map at
CODE_CULTU precision, not CODE_GROUP). The long tail is bucketed by keyword rules
(vegetables/fruits/flowers -> 12, forage legumes -> 14, prairies/grasses -> 1, fruit
trees/olives -> 16). Everything unmatched -> 0 (background), which is also how PASTIS
treats crops outside its 18 (incl. tropical DROM crops like banana/sugarcane/vanilla).

Entries whose PASTIS assignment is genuinely ambiguous are listed in FLAGGED for review
against PASTIS's official Nomenclature table before the mapping is trusted end-to-end.

Run:  python build_pastis_rpg_map.py   # writes pastis_rpg_class_map.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
CSV_PATH = HERE / "data" / "rpg_culture_codes.csv"
OUT_PATH = HERE / "pastis_rpg_class_map.json"

BACKGROUND = 0

PASTIS_NAMES = {
    0: "Background", 1: "Meadow", 2: "Soft winter wheat", 3: "Corn", 4: "Winter barley",
    5: "Winter rapeseed", 6: "Spring barley", 7: "Sunflower", 8: "Grapevine", 9: "Beet",
    10: "Winter triticale", 11: "Winter durum wheat", 12: "Fruits/vegetables/flowers",
    13: "Potatoes", 14: "Leguminous fodder", 15: "Soybeans", 16: "Orchard",
    17: "Mixed cereal", 18: "Sorghum",
    # Extension beyond the original PASTIS 18: the most common French-DROM tropical crops,
    # so overseas parcels get real labels instead of all -> Background. num_classes = 25.
    19: "Sugarcane", 20: "Banana", 21: "Pineapple", 22: "Vanilla",
    23: "Tropical tuber", 24: "Ylang-ylang",
}

# --- High-confidence, class-defining codes (winter/spring precision preserved) ------
CORE: dict[str, int] = {
    "BTH": 2,                       # Blé tendre d'hiver
    "BDH": 11,                      # Blé dur d'hiver
    "ORH": 4,                       # Orge d'hiver
    "ORP": 6,                       # Orge de printemps
    "CZH": 5,                       # Colza d'hiver
    "TRN": 7,                       # Tournesol
    "SOJ": 15,                      # Soja
    "SOG": 18,                      # Sorgho
    "TTH": 10,                      # Triticale d'hiver
    "MIS": 3, "MIE": 3,            # Maïs (grain), Maïs ensilage
    "BTN": 9, "BVF": 9,            # Betterave (sucrière/bette), betterave fourragère [BVF flagged]
    "PTC": 13, "PTF": 13,          # Pomme de terre conso / féculière
    "VRC": 8, "VRN": 8, "VRT": 8,  # Vigne (cuve prod / cuve non-prod / table)
    "MCR": 17,                     # Mélange de céréales -> mixed cereal
    # --- Tropical DROM crops (extension classes 19-24) ---
    "CSA": 19, "CSF": 19, "CSI": 19, "CSP": 19, "CSR": 19,   # Canne à sucre (all tenures)
    "BCA": 20, "BCF": 20, "BCI": 20, "BCP": 20, "BCR": 20,   # Banane créole (fruit/légume)
    "BEA": 20, "BEF": 20, "BEI": 20, "BEP": 20, "BER": 20,   # Banane export
    "ANA": 21,                                               # Ananas (pineapple)
    "VNL": 22, "VNV": 22, "VNB": 22,                         # Vanille (verte / sous bois)
    "TBT": 23,                                               # Tubercule tropical
    "YLA": 24,                                               # Ylang-ylang (Mayotte)
}

# --- Keyword rules for the long tail (checked in order; first match wins) -----------
# Each rule: (class_id, [substrings], optional [exclude substrings]) matched on the
# lowercased French label.
RULES: list[tuple[int, list[str], list[str]]] = [
    # Remaining tropical crops without a dedicated class -> Background(0). The common
    # ones (sugarcane/banana/pineapple/vanilla/tuber/ylang) are handled explicitly in
    # CORE; only minor/unmappable tropical crops fall through here. Checked FIRST so
    # e.g. "café / cacao" doesn't get pulled into another rule.
    (0, ["café", "cacao", "curcuma"], []),
    # Meadow: permanent/temporary grasslands + pure forage grass species + pastoral.
    (1, ["prairie", "ray-grass", "dactyle", "fétuque", "fléole", "paturin", "brôme",
         "festulolium", "graminée fourragère", "moha", "bois pâturé",
         "surface pastorale", "châtaigneraie entretenue", "chênaie entretenue"], []),
    # Leguminous fodder: forage legumes + legume-dominant forage mixes.
    (14, ["luzerne", "trèfle", "sainfoin", "lupin fourrager", "pois fourrager", "vesce",
          "mélilot", "serradelle", "minette", "lotier", "jarosse", "gesse",
          "féverole fourragère", "légumineuses fourragères", "légumineuses déshydratées",
          "légumineuses fourragères prépondérantes", "mélange de légumineuses",
          "dolique", "cornille", "fenugrec"], []),
    # Orchard: fruit/nut trees, olives, vineyards handled in CORE.
    (16, ["verger", "oliveraie", "noix", "noisette", "châtaigne", "prune", "pêche",
          "poire", "cerise", "pistache", "caroube", "truffière", "pépinière",
          "agrume", "avocat"],  # citrus/avocado are tree crops -> orchard
         ["entretenue", "poireau"]),  # exclude 'poireau' (leek): 'poire' is a substring
    # Fruits / vegetables / flowers: field vegetables, berries, melons, ornamentals.
    (12, ["laitue", "batavia", "chou", "carotte", "céleri", "épinard", "epinard",
          "tomate", "aubergine", "courgette", "citrouille", "courge", "potiron",
          "poireau", "oignon", "echalotte", "ail ", "artichaut", "haricot", "flageolet",
          "melon", "pastèque", "fraise", "radis", "navet", "panais", "salsifi",
          "concombre", "cornichon", "poivron", "piment", "endive", "chicorée",
          "scarole", "mâche", "cresson", "roquette", "oseille", "rutabaga",
          "topinambour", "petit fruit rouge", "légume", "fruit annuel", "fruit pérenne",
          "horticulture", "ornementales", "maïs doux", "houblon", "tabac",
          "plante médicinale", "plante à parfum", "sous serre", "sous abri"], []),
]


def classify(code: str, label: str) -> int:
    """Map one CODE_CULTU to a PASTIS class id (0 background if nothing matches)."""
    if code in CORE:
        return CORE[code]
    low = label.lower()
    for class_id, includes, excludes in RULES:
        if any(x in low for x in excludes):
            continue
        if any(x in low for x in includes):
            return class_id
    return BACKGROUND


# Codes whose PASTIS assignment is a judgement call -> review vs official Nomenclature.
FLAGGED = {
    "BVF": "fodder beet -> Beet(9)? PASTIS 'Beet' is likely sugar beet only.",
    "MIE": "silage maize -> Corn(3)? PASTIS may restrict Corn to grain maize (MIS).",
    "MID": "sweet corn -> Fruits/veg(12) via rule, not Corn(3).",
    "MCR": "cereal mix -> Mixed cereal(17); confirm vs méteil-style codes.",
    "CPL": "cereal+protein forage mix -> currently Background; maybe Mixed cereal(17).",
    "PFR": "red berries -> Fruits/veg(12) vs Orchard(16).",
    "grain_legumes": "PHI/PPR/PPO/PCH/LEC/FVL/LDH/LDP (grain pulses) -> Background "
                     "(PASTIS has no pulse-grain class); only *forage* legumes -> 14.",
    "spring_variants": "BTP/BDP/TTP/CZP/SGP/AVP (spring wheat/durum/triticale/rapeseed) "
                       "-> Background (PASTIS keeps only winter variants + Spring barley).",
    "aromatic_ppam": "lavande/thym/menthe etc. -> Background here; PASTIS 'flowers' is "
                     "ornamental, not aromatic/medicinal.",
    "tropical_ext": "Classes 19-24 (sugarcane/banana/pineapple/vanilla/tropical-tuber/"
                    "ylang) EXTEND the original PASTIS 18 to label DROM crops. Minor "
                    "tropical crops (coffee/cacao/curcuma) still -> Background; ACA "
                    "(autre culture non précisée) stays Background (genuinely unspecified).",
}


def main() -> None:
    rows = list(csv.reader(CSV_PATH.open(encoding="utf-8"), delimiter=";"))
    header, data = rows[0], rows[1:]
    assert header[:2] == ["Code", "Libellé"], header

    mapping = {code: classify(code, label) for code, label in data}

    # Per-class breakdown for human review.
    by_class: dict[int, list[str]] = {c: [] for c in PASTIS_NAMES}
    for code, cid in sorted(mapping.items()):
        by_class[cid].append(code)

    OUT_PATH.write_text(json.dumps(
        {
            "_doc": "RPG CODE_CULTU -> PASTIS-18 class id. 0=Background. See "
                    "build_pastis_rpg_map.py for rules; FLAGGED entries need review.",
            "class_names": PASTIS_NAMES,
            "flagged_for_review": FLAGGED,
            "code_to_class": mapping,
        },
        ensure_ascii=False, indent=2,
    ))

    print(f"total codes mapped: {len(mapping)}  -> {OUT_PATH.name}")
    print("per-class counts (n codes):")
    for cid in PASTIS_NAMES:
        codes = by_class[cid]
        preview = ",".join(codes[:14]) + (" ..." if len(codes) > 14 else "")
        print(f"  {cid:>2} {PASTIS_NAMES[cid]:<26} n={len(codes):<4} {preview}")


if __name__ == "__main__":
    main()
