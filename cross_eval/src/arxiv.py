from data_handling import *
from embedding import *
import os
embedding_path = "../data/embeddings/after_pca/gte_large_2025-02-23_256"
embedding = Embedding(embedding_path)
arxiv_categories = get_arxiv_categories()

def one_hot_encoding() -> None:
    n_papers = embedding.matrix.shape[0]
    n_categories = len(Arxiv_Category)
    one_hot_matrix = np.zeros((n_papers, n_categories - 1), dtype = embedding.matrix.dtype)
    for paper_id in arxiv_categories.keys():
        arxiv_category = arxiv_categories[paper_id]
        if arxiv_category == Arxiv_Category.none:
            continue
        else:
            arxiv_category_int = arxiv_category.value - 1
            one_hot_matrix[embedding.papers_ids_to_idxs[paper_id], arxiv_category_int] = 1
    one_hot_matrix = np.concatenate((embedding.matrix, one_hot_matrix), axis = 1)
    os.makedirs(embedding_path + "_onehot", exist_ok = True)
    os.system(f"cp {embedding_path}/abs_paper_ids_to_idx.pkl {embedding_path}_onehot/abs_paper_ids_to_idx.pkl")
    np.save(embedding_path + "_onehot/abs_X.npy", one_hot_matrix)

def load_glove_embeddings(dim : int) -> dict:
    from tqdm import tqdm
    glove_path = f"../data/embeddings/glove/glove.6B.{dim}d.txt"
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc = "Loading GloVe embeddings"):
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype = np.float32)
            embeddings[word] = vector
    return embeddings

def normalize_embedding(glove_embedding : np.ndarray, normalization : str = "none") -> np.ndarray:
    if normalization == "none":
        return glove_embedding
    elif normalization == "l2_unit":
        return glove_embedding / np.linalg.norm(glove_embedding)
    elif normalization == "l2_proportional":
        proportionality = glove_embedding.shape[0] / (glove_embedding.shape[0] + embedding.matrix.shape[1])
        return glove_embedding / np.linalg.norm(glove_embedding) * proportionality
    elif normalization == "l2_05":
        return glove_embedding / np.linalg.norm(glove_embedding) * 0.5

def get_glove_category_embeddings(dim : int, normalization : str = "none") -> dict:
    glove_embeddings = load_glove_embeddings(dim)
    glove_category_embeddings = {}
    for arxiv_category in list(Arxiv_Category):
        if arxiv_category == Arxiv_Category.none:
            glove_category_embeddings[arxiv_category] = np.zeros(dim, dtype = embedding.matrix.dtype)
        else:
            if arxiv_category == Arxiv_Category.cs:
                glove_embedding = 0.7 * glove_embeddings["computer"] + 0.3 * glove_embeddings["science"]
            elif arxiv_category == Arxiv_Category.econ:
                glove_embedding = glove_embeddings["economics"]
            elif arxiv_category == Arxiv_Category.eess:
                glove_embedding = 0.5 * glove_embeddings["electrical"] + 0.3 * glove_embeddings["engineering"] + 0.2 * glove_embeddings["systems"]
            elif arxiv_category == Arxiv_Category.math:
                glove_embedding = glove_embeddings["mathematics"]
            elif arxiv_category == Arxiv_Category.physics:
                glove_embedding = glove_embeddings["physics"]
            elif arxiv_category == Arxiv_Category.q_bio:
                glove_embedding = 0.25 * glove_embeddings["quantitative"] + 0.75 * glove_embeddings["biology"]
            elif arxiv_category == Arxiv_Category.q_fin:
                glove_embedding = 0.35 * glove_embeddings["quantitative"] + 0.65 * glove_embeddings["finance"]
            elif arxiv_category == Arxiv_Category.stat:
                glove_embedding = glove_embeddings["statistics"]
            glove_category_embeddings[arxiv_category] = normalize_embedding(glove_embedding, normalization)
    return glove_category_embeddings

def glove_embeddings(dim : int, normalization : str = "none") -> None:
    n_papers = embedding.matrix.shape[0]
    glove_category_embeddings = get_glove_category_embeddings(dim, normalization)
    glove_matrix = np.zeros((n_papers, dim), dtype = embedding.matrix.dtype)
    for paper_id in arxiv_categories.keys():
        arxiv_category = arxiv_categories[paper_id]
        glove_matrix[embedding.papers_ids_to_idxs[paper_id], :] = glove_category_embeddings[arxiv_category]
    glove_matrix = np.concatenate((embedding.matrix, glove_matrix), axis = 1)
    os.makedirs(embedding_path + f"_glove_{dim}_{normalization}", exist_ok = True)
    os.system(f"cp {embedding_path}/abs_paper_ids_to_idx.pkl {embedding_path}_glove_{dim}_{normalization}/abs_paper_ids_to_idx.pkl")
    np.save(f"{embedding_path}_glove_{dim}_{normalization}/abs_X.npy", glove_matrix)

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





def download_glove_embeddings() -> None:
    from tqdm import tqdm
    from zipfile import ZipFile
    import os
    import requests
    save_dir = "../data/embeddings/glove"
    available_dims = [50, 100, 200, 300]
    os.makedirs(save_dir, exist_ok = True)
    for dim in available_dims:
        zip_path = os.path.join(save_dir, f"glove.6B.{dim}d.zip")
        url = f"https://nlp.stanford.edu/data/glove.6B.zip"
        response = requests.get(url, stream = True)
        total_size = int(response.headers.get("content-length", 0))
        print(f"Downloading GloVe embeddings with {dim} dimensions...")
        with open(zip_path, 'wb') as f:
            with tqdm(total = total_size, unit = 'B', unit_scale = True, desc = "Downloading") as pbar:
                for chunk in response.iter_content(chunk_size = 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        print(f"Extracting embeddings to {save_dir}")
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract(f"glove.6B.{dim}d.txt", save_dir)
        os.remove(zip_path)

if __name__ == "__main__":
    glove_embeddings(100, "l2_unit")
