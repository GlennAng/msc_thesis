class PapersCategories:
    def __init__(self, original_categories_to_categories : dict, categories_to_glove : dict):
        self.original_categories_to_categories = original_categories_to_categories
        self.categories_to_glove = categories_to_glove
        assert sorted(set(self.original_categories_to_categories.keys())) == sorted(set(CATEGORIES_ORIGINAL))
        assert sorted(set(self.original_categories_to_categories.values())) == sorted(set(self.categories_to_glove.keys()))
        self.original_categories_to_categories[None] = None

CATEGORIES_ORIGINAL = ["physics", "computer_science", "mathematics", "astronomy", "biology", "medicine", "engineering", "chemistry",
                       "economics", "psychology", "electrical_engineering", "materials_science", "earth_science", "linguistics",
                       "philosophy", "geography", "sociology"]

SCIENCE_PROBABILITY = 0.0
DOMAIN_PROBABILITY = 0.5

PAPERS_CATEGORIES_ORIGINAL = PapersCategories(
    original_categories_to_categories = { original_category : original_category for original_category in CATEGORIES_ORIGINAL},
    categories_to_glove = {
        "physics": {"physics" : 1.0},
        "computer_science": {"computer" : 1.0 - SCIENCE_PROBABILITY, "science" : SCIENCE_PROBABILITY},
        "mathematics": {"mathematics" : 1.0},
        "astronomy": {"astronomy" : 1.0},
        "biology": {"biology" : 1.0},
        "medicine": {"medicine" : 1.0},
        "engineering": {"engineering" : 1.0},
        "chemistry": {"chemistry" : 1.0},
        "economics": {"economics" : 1.0},
        "psychology": {"psychology" : 1.0},
        "electrical_engineering": {"engineering" : 1.0 - DOMAIN_PROBABILITY, "electrical" : DOMAIN_PROBABILITY},
        "materials_science": {"materials" : 1.0 - SCIENCE_PROBABILITY, "science" : SCIENCE_PROBABILITY},
        "earth_science": {"earth" : 1.0 - SCIENCE_PROBABILITY, "science" : SCIENCE_PROBABILITY},
        "linguistics": {"linguistics" : 1.0},
        "philosophy": {"philosophy" : 1.0},
        "geography": {"geography" : 1.0},
        "sociology": {"sociology" : 1.0}})

PAPERS_CATEGORIES_GROUPED = PapersCategories(
    original_categories_to_categories = { 
        "physics" : "physics",
        "computer_science" : "computer_science",
        "mathematics" : "mathematics",
        "astronomy" : "physics",
        "biology" : "natural_science",
        "medicine" : "medicine",
        "engineering" : "engineering",
        "chemistry" : "natural_science",
        "economics" : "social_science",
        "psychology" : "social_science",
        "electrical_engineering" : "engineering",
        "materials_science" : "physics",
        "earth_science" : "natural_science",
        "linguistics" : "social_science",
        "philosophy" : "social_science",
        "geography" : "natural_science",
        "sociology" : "social_science"},
    categories_to_glove = {
        "physics": {"physics" : 1.0},
        "computer_science" : {"computer" : 1.0 - SCIENCE_PROBABILITY, "science" : SCIENCE_PROBABILITY},
        "mathematics" : {"mathematics" : 1.0},
        "natural_science" : {"biology" : 0.35, "chemistry" : 0.35, "geography" : 0.15, "science" : 0.15},
        "medicine" : {"medicine" : 1.0},
        "engineering" : {"engineering" : 1.0 - DOMAIN_PROBABILITY, "electrical" : DOMAIN_PROBABILITY},
        "social_science" : {"sociology" : 0.2, "psychology" : 0.2, "economics" : 0.2, "linguistics" : 0.2, "philosophy" : 0.2}})

PAPERS_CATEGORIES = PAPERS_CATEGORIES_ORIGINAL
