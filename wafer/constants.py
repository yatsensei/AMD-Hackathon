"""The one canonical class list (spec Section 2.3). Never hard-code class names or indices elsewhere."""

CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
           "Near-full", "Random", "Scratch", "none"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NONE_IDX = CLASS_TO_IDX["none"]
DEFECT_IDX = [i for i, c in enumerate(CLASSES) if c != "none"]
CONFUSABLE_PAIRS = [("Edge-Loc", "Scratch"), ("Edge-Loc", "Loc"),
                    ("Loc", "Center"), ("Donut", "Edge-Ring")]

# Raw wafer-map values
NO_DIE, PASS, FAIL = 0, 1, 2

SPLITS = ["train", "val", "test"]
