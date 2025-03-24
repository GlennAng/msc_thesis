ARXIV_CATEGORIES = ["cs", "math", "cond-mat", "hep", "astro-ph", "physics", "eess", "stat", "nucl", "q-bio", "nlin", "q-fin", "econ"]
ARXIV_RATIOS = [0.0, 0.25, 0.20, 0.20, 0.15, 0.12, 0.0, 0.0, 0.03, 0.02, 0.01, 0.01, 0.01]

def get_arxiv_categories() -> dict:
    return {
        None: "No Category",
    "q-alg": "Quantum Algebra",
    "adap-org": "Adaptation, Noise, and Self-Organizing Systems",
    "alg-geom": "Algebraic Geometry",
    "chao-dyn": "Chaotic Dynamics",
    "cond-mat": "Condensed Matter",
    "dg-ga": "Differential Geometry",
    "funct-an": "Functional Analysis",
    "solv-int": "Exactly Solvable and Integrable Systems",
    "patt-sol": "Pattern Formation and Solitons",
    "cs.ai": "Computer Science - Artificial Intelligence",
    "cs.ar": "Computer Science - Architecture",
    "cs.cc": "Computer Science - Computational Complexity",
    "cs.ce": "Computer Science - Computational Engineering, Finance, and Science",
    "cs.cg": "Computer Science - Computational Geometry",
    "cs.cl": "Computer Science - Computation and Language",
    "cs.cr": "Computer Science - Cryptography and Security",
    "cs.cv": "Computer Science - Computer Vision and Pattern Recognition",
    "cs.cy": "Computer Science - Computers and Society",
    "cs.db": "Computer Science - Databases",
    "cs.dc": "Computer Science - Distributed, Parallel, and Cluster Computing",
    "cs.dl": "Computer Science - Digital Libraries",
    "cs.dm": "Computer Science - Discrete Mathematics",
    "cs.ds": "Computer Science - Data Structures and Algorithms",
    "cs.et": "Computer Science - Emerging Technologies",
    "cs.fl": "Computer Science - Formal Languages and Automata Theory",
    "cs.gl": "Computer Science - General Literature",
    "cs.gr": "Computer Science - Graphics",
    "cs.gt": "Computer Science - Computer Science and Game Theory",
    "cs.hc": "Computer Science - Human-Computer Interaction",
    "cs.ir": "Computer Science - Information Retrieval",
    "cs.it": "Computer Science - Information Theory",
    "cs.lg": "Computer Science - Machine Learning",
    "cs.lo": "Computer Science - Logic in Computer Science",
    "cs.ma": "Computer Science - Multiagent Systems",
    "cs.mm": "Computer Science - Multimedia",
    "cs.ms": "Computer Science - Mathematical Software",
    "cs.na": "Computer Science - Numerical Analysis",
    "cs.ne": "Computer Science - Neural and Evolutionary Computing",
    "cs.ni": "Computer Science - Networking and Internet Architecture",
    "cs.oh": "Computer Science - Other",
    "cs.os": "Computer Science - Operating Systems",
    "cs.pf": "Computer Science - Performance",
    "cs.pl": "Computer Science - Programming Languages",
    "cs.ro": "Computer Science - Robotics",
    "cs.sc": "Computer Science - Symbolic Computation",
    "cs.sd": "Computer Science - Sound",
    "cs.se": "Computer Science - Software Engineering",
    "cs.si": "Computer Science - Social and Information Networks",
    "cs.sy": "Computer Science - Systems and Control",
    "econ.em": "Economics - Econometrics",
    "econ.gn": "Economics - General Economics",
    "econ.th": "Economics - Theoretical Economics",
    "eess.as": "Electrical Engineering and Systems Science - Audio and Speech Processing",
    "eess.iv": "Electrical Engineering and Systems Science - Image and Video Processing",
    "eess.sp": "Electrical Engineering and Systems Science - Signal Processing",
    "eess.sy": "Electrical Engineering and Systems Science - Systems and Control",
    "math.ac": "Mathematics - Commutative Algebra",
    "math.ag": "Mathematics - Algebraic Geometry",
    "math.ap": "Mathematics - Analysis of PDEs",
    "math.at": "Mathematics - Algebraic Topology",
    "math.ca": "Mathematics - Classical Analysis and ODEs",
    "math.co": "Mathematics - Combinatorics",
    "math.ct": "Mathematics - Category Theory",
    "math.cv": "Mathematics - Complex Variables",
    "math.dg": "Mathematics - Differential Geometry",
    "math.ds": "Mathematics - Dynamical Systems",
    "math.fa": "Mathematics - Functional Analysis",
    "math.gm": "Mathematics - General Mathematics",
    "math.gn": "Mathematics - General Topology",
    "math.gr": "Mathematics - Group Theory",
    "math.gt": "Mathematics - Geometric Topology",
    "math.ho": "Mathematics - History and Overview",
    "math.it": "Mathematics - Information Theory",
    "math.kt": "Mathematics - K-Theory and Homology",
    "math.lo": "Mathematics - Logic",
    "math.mg": "Mathematics - Metric Geometry",
    "math.mp": "Mathematics - Mathematical Physics",
    "math.na": "Mathematics - Numerical Analysis",
    "math.nt": "Mathematics - Number Theory",
    "math.oa": "Mathematics - Operator Algebras",
    "math.oc": "Mathematics - Optimization and Control",
    "math.pr": "Mathematics - Probability",
    "math.qa": "Mathematics - Quantum Algebra",
    "math.ra": "Mathematics - Rings and Algebras",
    "math.rt": "Mathematics - Representation Theory",
    "math.sg": "Mathematics - Symplectic Geometry",
    "math.sp": "Mathematics - Spectral Theory",
    "math.st": "Mathematics - Statistics Theory",
    "astro-ph": "Astrophysics",
    "astro-ph.co": "Astrophysics - Cosmology and Nongalactic Astrophysics",
    "astro-ph.ep": "Astrophysics - Earth and Planetary Astrophysics",
    "astro-ph.ga": "Astrophysics - Astrophysics of Galaxies",
    "astro-ph.he": "Astrophysics - High Energy Astrophysical Phenomena",
    "astro-ph.im": "Astrophysics - Instrumentation and Methods for Astrophysics",
    "astro-ph.sr": "Astrophysics - Solar and Stellar Astrophysics",
    "cond-mat.dis-nn": "Condensed Matter - Disordered Systems and Neural Networks",
    "cond-mat.mes-hall": "Condensed Matter - Mesoscale and Nanoscale Physics",
    "cond-mat.mtrl-sci": "Condensed Matter - Materials Science",
    "cond-mat.other": "Condensed Matter - Other",
    "cond-mat.quant-gas": "Condensed Matter - Quantum Gases",
    "cond-mat.soft": "Condensed Matter - Soft Condensed Matter",
    "cond-mat.stat-mech": "Condensed Matter - Statistical Mechanics",
    "cond-mat.str-el": "Condensed Matter - Strongly Correlated Electrons",
    "cond-mat.supr-con": "Condensed Matter - Superconductivity",
    "gr-qc": "General Relativity and Quantum Cosmology",
    "hep-ex": "High Energy Physics - Experiment",
    "hep-lat": "High Energy Physics - Lattice",
    "hep-ph": "High Energy Physics - Phenomenology",
    "hep-th": "High Energy Physics - Theory",
    "math-ph": "Mathematical Physics",
    "nucl-ex": "Nuclear Experiment",
    "nucl-th": "Nuclear Theory",
    "physics.acc-ph": "Physics - Accelerator Physics",
    "physics.app-ph": "Physics - Applied Physics",
    "physics.ao-ph": "Physics - Atmospheric and Oceanic Physics",
    "physics.atom-ph": "Physics - Atomic Physics",
    "physics.bio-ph": "Physics - Biological Physics",
    "physics.chem-ph": "Physics - Chemical Physics",
    "physics.class-ph": "Physics - Classical Physics",
    "physics.atm-clus": "Physics - Atomic and Molecular Clusters",
    "physics.comp-ph": "Physics - Computational Physics",
    "physics.data-an": "Physics - Data Analysis, Statistics and Probability",
    "physics.flu-dyn": "Physics - Fluid Dynamics",
    "physics.gen-ph": "Physics - General Physics",
    "physics.geo-ph": "Physics - Geophysics",
    "physics.ed-ph": "Physics - Physics Education",
    "physics.hist-ph": "Physics - History of Physics",
    "physics.ins-det": "Physics - Instrumentation and Detectors",
    "physics.med-ph": "Physics - Medical Physics",
    "physics.optics": "Physics - Optics",
    "physics.plasm-ph": "Physics - Plasma Physics",
    "physics.pop-ph": "Physics - Popular Physics",
    "physics.soc-ph": "Physics - Physics and Society",
    "physics.space-ph": "Physics - Space Physics",
    "quant-ph": "Quantum Physics",
    "q-bio.bm": "Quantitative Biology - Biomolecules",
    "q-bio.cb": "Quantitative Biology - Cell Behavior",
    "q-bio.gn": "Quantitative Biology - Genomics",
    "q-bio.mn": "Quantitative Biology - Molecular Networks",
    "q-bio.nc": "Quantitative Biology - Neurons and Cognition",
    "q-bio.ot": "Quantitative Biology - Other Quantitative Biology",
    "q-bio.pe": "Quantitative Biology - Populations and Evolution",
    "q-bio.qm": "Quantitative Biology - Quantitative Methods",
    "q-bio.sc": "Quantitative Biology - Subcellular Processes",
    "q-bio.to": "Quantitative Biology - Tissues and Organs",
    "q-fin.cp": "Quantitative Finance - Computational Finance",
    "q-fin.ec": "Quantitative Finance - Economics",
    "q-fin.gn": "Quantitative Finance - General Finance",
    "q-fin.mf": "Quantitative Finance - Mathematical Finance",
    "q-fin.pm": "Quantitative Finance - Portfolio Management",
    "q-fin.pr": "Quantitative Finance - Pricing of Securities",
    "q-fin.rm": "Quantitative Finance - Risk Management",
    "q-fin.st": "Quantitative Finance - Statistical Finance",
    "q-fin.tr": "Quantitative Finance - Trading and Market Microstructure",
    "nlin.ao": "Nonlinear Sciences - Adaptation and Self-Organizing Systems",
    "nlin.cd": "Nonlinear Sciences - Chaotic Dynamics",
    "nlin.cg": "Nonlinear Sciences - Cellular Automata and Lattice Gases",
    "nlin.ps": "Nonlinear Sciences - Pattern Formation and Solitons",
    "nlin.si": "Nonlinear Sciences - Exactly Solvable and Integrable Systems",
    "stat.ap": "Statistics - Applications",
    "stat.co": "Statistics - Computation",
    "stat.me": "Statistics - Methodology",
    "stat.ml": "Statistics - Machine Learning",
    "stat.ot": "Statistics - Other Statistics",
    "stat.th": "Statistics - Statistics Theory"
} 

def get_arxiv_distribution_papers() -> tuple:
    from data_handling import sql_execute
    query = """
    SELECT arxiv_category, count(*) as count
    FROM papers
    GROUP BY arxiv_category
    ORDER BY count DESC;
    """
    result = sql_execute(query)
    categories_counts = {category: 0 for category in ARXIV_CATEGORIES}
    n_total = 0
    for row in result:
        if row[0] is None:
            continue
        for category in ARXIV_CATEGORIES:
            if row[0].startswith(category):
                categories_counts[category] += row[1]
                n_total += row[1]
                break
    categories_counts = {category: count / n_total for category, count in categories_counts.items()}
    return categories_counts, n_total

def get_arxiv_distribution_ratings() -> tuple:
    from data_handling import sql_execute
    query = """
    SELECT p.arxiv_category, COUNT(*) as count
    FROM users_ratings u 
    INNER JOIN papers p ON u.paper_id = p.paper_id
    GROUP BY p.arxiv_category
    ORDER BY count DESC;
    """
    result = sql_execute(query)
    categories = ["cs", "math", "physics", "econ", "eess", "astro-ph", "cond-mat", "hep", "nucl", "q-bio", "q-fin", "nlin", "stat"]
    categories_counts = {category: 0 for category in ARXIV_CATEGORIES}
    n_total = 0
    for row in result:
        if row[0] is None:
            continue
        for category in ARXIV_CATEGORIES:
            if row[0].startswith(category):
                categories_counts[category] += row[1]
                n_total += row[1]
                break
    categories_counts = {category: count / n_total for category, count in categories_counts.items()}
    return categories_counts, n_total


if __name__ == "__main__":
    print(get_arxiv_distribution_papers())
    print(get_arxiv_distribution_ratings())
