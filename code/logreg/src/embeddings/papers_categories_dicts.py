class PapersCategories:
    def __init__(self, original_categories_to_categories: dict, categories_to_glove: dict):
        self.original_categories_to_categories = original_categories_to_categories
        self.categories_to_glove = categories_to_glove
        assert sorted(set(self.original_categories_to_categories.keys())) == sorted(
            set(CATEGORIES_ORIGINAL)
        )
        assert sorted(set(self.original_categories_to_categories.values())) == sorted(
            set(self.categories_to_glove.keys())
        )
        self.original_categories_to_categories[None] = None


CATEGORIES_ORIGINAL = [
    "Physics",
    "Computer Science",
    "Mathematics",
    "Astronomy",
    "Biology",
    "Medicine",
    "Engineering",
    "Chemistry",
    "Economics",
    "Psychology",
    "Electrical Engineering",
    "Materials Science",
    "Earth Science",
    "Linguistics",
    "Philosophy",
    "Geography",
    "Sociology",
]

SCIENCE_PROBABILITY = 0.0
DOMAIN_PROBABILITY = 0.5

PAPERS_CATEGORIES_ORIGINAL = PapersCategories(
    original_categories_to_categories={
        original_category: original_category for original_category in CATEGORIES_ORIGINAL
    },
    categories_to_glove={
        "Physics": {"physics": 1.0},
        "Computer Science": {"computer": 1.0 - SCIENCE_PROBABILITY, "science": SCIENCE_PROBABILITY},
        "Mathematics": {"mathematics": 1.0},
        "Astronomy": {"astronomy": 1.0},
        "Biology": {"biology": 1.0},
        "Medicine": {"medicine": 1.0},
        "Engineering": {"engineering": 1.0},
        "Chemistry": {"chemistry": 1.0},
        "Economics": {"economics": 1.0},
        "Psychology": {"psychology": 1.0},
        "Electrical Engineering": {
            "engineering": 1.0 - DOMAIN_PROBABILITY,
            "electrical": DOMAIN_PROBABILITY,
        },
        "Materials Science": {
            "materials": 1.0 - SCIENCE_PROBABILITY,
            "science": SCIENCE_PROBABILITY,
        },
        "Earth Science": {"earth": 1.0 - SCIENCE_PROBABILITY, "science": SCIENCE_PROBABILITY},
        "Linguistics": {"linguistics": 1.0},
        "Philosophy": {"philosophy": 1.0},
        "Geography": {"geography": 1.0},
        "Sociology": {"sociology": 1.0},
    },
)

PAPERS_CATEGORIES = PAPERS_CATEGORIES_ORIGINAL
