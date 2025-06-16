import sys
from pathlib import Path
try:
    from project_paths import ProjectPaths
except ImportError:
    sys.path.append(str(Path(__file__).parents[2]))
    from project_paths import ProjectPaths
ProjectPaths.add_logreg_paths_to_sys()
ProjectPaths.add_finetuning_paths_to_sys()

import argparse, json, os, pickle, time, torch
import matplotlib.colors as colors, matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from sklearn.metrics import ndcg_score