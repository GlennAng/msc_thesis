from pathlib import Path

def get_paths() -> dict:
    base_path = Path(__file__).parent.parent.parent.parent.resolve()
    paths = {
        "base_path": base_path,
        "logreg_path": base_path / "logreg",
        "data_path": base_path / "data"
    }
    paths.update({
        "experiments_path": paths["logreg_path"] / "experiments",
        "outputs_path": paths["logreg_path"] / "outputs",
        "src_path": paths["logreg_path"] / "src",
        "embeddings_path": paths["data_path"] / "embeddings"
        })
    paths.update({
        "data_processing_path": paths["src_path"] / "data_processing",
        "visualizations_path": paths["src_path"] / "visualizations"
    })
    return paths

PATHS = get_paths()