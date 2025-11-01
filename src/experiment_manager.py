from __future__ import annotations

"""Experiment-level utilities — minimal clean-up keeping API intact."""

from pathlib import Path
import json
import random
from typing import Sequence, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import re
# Optional dependency
try:
    import umap  # type: ignore
except ModuleNotFoundError:
    umap = None  # type: ignore

from src.benchmark.pipeline import benchmark_models
from src.model.model_interface import BaseModel
from src.model.ESN import ESNModel
from src.relative_latent_spaces import evaluate_models, evaluate_models_time
from src.config_manager import ConfigManager

plt.switch_backend("Agg")  # non-interactive backend


# ------------------------------------------------------------------ #
def _list_dirs(path: str | Path) -> list[str]:
    return [str(p) for p in Path(path).iterdir() if p.is_dir()]


# ------------------------------------------------------------------ #
class ExperimentManager:
    def __init__(self, config_manager: ConfigManager, device: str):
        self.config_manager = config_manager
        self.device = device
        self._data_handlers: dict[str, Any] = {}

    # ------------ data-handler cache -------------------------------- #
    def _get_data_handler(self, exp: str):
        if exp not in self._data_handlers:
            cfg = self.config_manager.get_current_dataset_config()
            self._data_handlers[exp] = cfg["data_handler_class"](cfg, exp, True)
        return self._data_handlers[exp]

    # ------------ load models & datasets ---------------------------- #
    def get_models_and_dataset_from_experiment(
        self, exp: str
    ) -> Tuple[list[BaseModel], list[Any]]:
        dh = self._get_data_handler(exp)
        base = Path("results") / exp
        exclude = {base / n for n in [".ipynb_checkpoints", "trajectories", "visualizations"]}

        model_dirs = sorted(d for d in _list_dirs(base) if Path(d) not in exclude)

        models: list[BaseModel] = []
        for d in model_dirs:
            name = Path(d).name.lower()
            if name == "esn" or name.startswith("esn"):
                models.append(ESNModel.load_from_file(f"{d}/model.pkl"))
            else:
                try:
                    models.append(BaseModel.load_from_file(f"{d}/best_current_model.pth"))
                except:
                    models.append(BaseModel.load_from_file(f"{d}/model.pth"))

        datasets = dh.get_datasets([m.hyperparams for m in models])
        datasets = [split[2] for split in datasets]  # take test split
        return models, datasets

    # ------------ wrappers ------------------------------------------ #
    def evalute_latent_spaces_exp(self, exp: str, points: int, anchors: int, random_anchors: bool = False):
        m, d = self.get_models_and_dataset_from_experiment(exp)
        return evaluate_models(
            m,
            d,
            f"results/{exp}/",
            {"device": self.device, "num_samples": points, "num_anchors": anchors,  "random_anchors": random_anchors},
        )
    

    def evaluate_latent_spaces_time(
        self,
        exp: str,
        points: int,
        anchors: int,
        window: int,
        stride: int = 1,
        start: int = 0,
        random_anchors: bool = False,
    ):
        """
        Wrapper for the temporal latent-space evaluation.
        Saves only temporal cosine similarities.

        Args:
            exp   : experiment name
            points: number of time samples (num_samples)
            anchors: number of anchors (num_anchors)
            window : temporal sliding window size
            stride : temporal stride (default 1)
            start  : starting index in dataset (default 0)
            random_anchors: bool (default False)
        """
        m, d = self.get_models_and_dataset_from_experiment(exp)
        return evaluate_models_time(
            m,
            d,
            f"results/{exp}/time_",  # kept distinct from normal evaluate_models output
            {
                "device": self.device,
                "num_samples": points,
                "num_anchors": anchors,
                "random_anchors": random_anchors,
                "temporal_window": window,
                "temporal_stride": stride,
                "temporal_start": start,
            },
        )



    def benchmark_models_exp(self, exp: str):
        m, d = self.get_models_and_dataset_from_experiment(exp)
        benchmark_models(m, d, f"results/{exp}", self.device)
        




    def visualize_relative_latent_spaces(
        self,
        models,
        latent_spaces,
        pairs: Sequence[Tuple[str, Sequence[str]]],
        exp: str,
        *,
        fit_individually=False,
        n_components=2,
        fix_global_limits: bool = False,
    ):
        import re
        import math
        from pathlib import Path
        from typing import Any, Sequence, Tuple, List, Dict
        import numpy as np
        import pandas as pd
        import torch
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        try:
            import umap  # optional
        except Exception:
            umap = None

        if n_components < 2:
            raise ValueError("n_components must be ≥ 2")

        out_dir = Path("results") / exp / "visualizations"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Map raw experiment model names to pretty legend labels (kept as-is)
        def pretty_label(name: str) -> str:
            base = re.sub(r"\d+$", "", name.lower())  # 'mlp1' -> 'mlp'
            mapping = {
                "mlp": "MLP",
                "transformer": "Transformer",
                "rnnautoreg": "Autoregressive RNN",
                "oldrnn": "RNN",
                "koopmantransformer": "Koopman Transformer",
                "nodetransformer": "NODE Transformer",
                "koopmanrnnautoreg": "Koopman RNN",
                "nodernnautoreg": "NODE RNN",
                "nodemlp": "NODE MLP",
                "koopmanmlp": "Koopman MLP",
                "esn": "ESN",
                "none": "True System",
            }
            return mapping.get(base, name)

        reducers = {
            "raw": lambda: None,
            "pca": lambda: PCA(n_components=n_components),
            "tsne": lambda: TSNE(n_components=n_components, random_state=42, init="random"),
        }
        if umap is not None:
            reducers["umap"] = lambda: umap.UMAP(n_components=n_components, random_state=42)

        # -------------------------------------------------------------------------
        # NEW: Family + shade colormap (matches your other script exactly)
        # -------------------------------------------------------------------------
        import colorsys
        from matplotlib import colors as mcolors

        DESIRED_ORDER: List[str] = [
            "True System",
            "MLP", "Koopman MLP", "NODE MLP",
            "RNN", "Autoregressive RNN", "Koopman RNN", "NODE RNN",
            "Transformer", "Koopman Transformer", "NODE Transformer",
            "ESN",
        ]

        INTRA_FAMILY_ORDER: Dict[str, List[str]] = {
            "RNN": ["RNN", "Autoregressive RNN", "NODE RNN", "Koopman RNN"],
            "Transformer": ["Transformer", "NODE Transformer", "Koopman Transformer"],
            "MLP": ["MLP", "NODE MLP", "Koopman MLP"],
            "ESN": ["ESN"],
            "Reference": ["True System"],
        }

        BASE_FAMILY_COLORS: Dict[str, str] = {
            "RNN": "tab:blue",
            "Transformer": "tab:orange",
            "MLP": "tab:green",
            "ESN": "#D1B3FF",
            "Reference": "tab:gray",
        }

        def _to_rgb(c): 
            return mcolors.to_rgb(c)

        def _shade_series(base_rgb, n, darkest=0.25, lightest=0.88):
            """Stable shades per family; index 0 darkest → increasing lightness."""
            if n <= 0:
                return []
            h, l, s = colorsys.rgb_to_hls(*base_rgb)
            s = min(max(s, 0.4), 1.0)
            darkest = max(0.0, min(darkest, 1.0))
            lightest = max(0.0, min(lightest, 1.0))
            if darkest > lightest:
                darkest, lightest = lightest, darkest
            Ls = [l] if n == 1 else np.linspace(darkest, lightest, n)
            return [colorsys.hls_to_rgb(h, float(L), s) for L in Ls]

        def build_colormap(desired_order: List[str]):
            # preserve caller order; group by families and assign fixed shade indices
            unique_labels = list(dict.fromkeys(desired_order))
            fam_to_labels: Dict[str, List[str]] = {fam: [] for fam in INTRA_FAMILY_ORDER.keys()}
            for fam, order in INTRA_FAMILY_ORDER.items():
                fam_to_labels[fam] = [lab for lab in order if lab in unique_labels]

            label_to_color: Dict[str, Tuple[float, float, float, float]] = {}
            for fam, labs in fam_to_labels.items():
                if not labs:
                    continue
                if fam == "Reference":
                    r, g, b = _to_rgb("black")  # or "tab:gray" if you prefer gray
                    for lab in labs:
                        label_to_color[lab] = (r, g, b, 1.0)
                    continue

                base_rgb = _to_rgb(BASE_FAMILY_COLORS[fam])
                full_len = len(INTRA_FAMILY_ORDER[fam])
                shades = _shade_series(base_rgb, full_len)
                for lab in labs:
                    idx = INTRA_FAMILY_ORDER[fam].index(lab)
                    r, g, b = shades[idx]
                    label_to_color[lab] = (r, g, b, 1.0)

            # fallback for unmapped labels
            for lab in unique_labels:
                if lab not in label_to_color:
                    r, g, b = _to_rgb("tab:purple")
                    label_to_color[lab] = (r, g, b, 1.0)

            order_index = {lab: i for i, lab in enumerate(unique_labels)}
            return label_to_color, order_index

        LABEL_TO_COLOR, LABEL_ORDER_INDEX = build_colormap(DESIRED_ORDER)
        # -------------------------------------------------------------------------

        csv_rows: list[list[Any]] = []

        for idx, (rtype, names) in enumerate(pairs, 1):
            if rtype not in reducers:
                raise ValueError(f"Unknown reducer '{rtype}'")
            reducer = reducers[rtype]()
            is_raw = rtype == "raw"

            sel_lat, sel_names_raw, sel_names_pretty = [], [], []
            for mdl, lat in zip(models, latent_spaces):
                raw_name = mdl.hyperparams["model_name"]
                if raw_name in names:
                    sel_names_raw.append(raw_name)
                    sel_names_pretty.append(pretty_label(raw_name))
                    sel_lat.append(lat.cpu().numpy() if isinstance(lat, torch.Tensor) else lat)

            if not sel_lat:
                print(f"[vis-{idx}] no matching models; skip")
                continue

            # dimensionality reduction
            if is_raw:
                red_lat = sel_lat
            else:
                if fit_individually:
                    red_lat = [reducer.fit_transform(arr) for arr in sel_lat]
                else:
                    reducer.fit(sel_lat[0])
                    red_lat = [reducer.transform(arr) for arr in sel_lat]

            # plotting — force white background
            fig = plt.figure(figsize=(8, 6), facecolor="white")
            if n_components == 3:
                ax = fig.add_subplot(111, projection="3d")
                try:
                    ax.xaxis.set_pane_color((1, 1, 1, 1))
                    ax.yaxis.set_pane_color((1, 1, 1, 1))
                    ax.zaxis.set_pane_color((1, 1, 1, 1))
                except Exception:
                    pass
            else:
                ax = fig.add_subplot(111)
            ax.set_facecolor("white")

            if fix_global_limits:
                red_lat_for_plot = []
                for a in red_lat:
                    a = np.asarray(a, dtype=float)
                    mn = np.min(a, axis=0)
                    mx = np.max(a, axis=0)
                    span = mx - mn
                    span[span == 0] = 1.0
                    a01 = (a - mn) / span
                    red_lat_for_plot.append(a01)
                red_lat = red_lat_for_plot

            for pts, raw_name, pretty in zip(red_lat, sel_names_raw, sel_names_pretty):
                color = LABEL_TO_COLOR.get(pretty, (0.5, 0.5, 0.5, 1.0))  # NEW colors
                if n_components == 3:
                    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=8, alpha=0.7, label=pretty, color=color)
                else:
                    ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.7, label=pretty, color=color)
                csv_rows.extend([[raw_name] + p.tolist() for p in pts])

            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            if n_components == 3:
                ax.set_zlabel("Component 3")

            title_prefix = "Absolute" if (is_raw or fit_individually) else "Relative"

            # Legend outside on the right (kept)
            lgd = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=True)
            if lgd and lgd.get_frame():
                lgd.get_frame().set_facecolor("white")

            ax.grid(True)
            fig.tight_layout()

            fig.savefig(
                out_dir / f"{title_prefix.lower()}_{rtype}_{idx}.png",
                dpi=300,
                facecolor="white",
                bbox_inches="tight"
            )
            plt.close(fig)

        if csv_rows:
            max_len = max(len(r) for r in csv_rows)
            for r in csv_rows:
                if len(r) < max_len:
                    r += [np.nan] * (max_len - len(r))
            num_components = max_len - 1
            cols = ["Model"] + [f"Component {i+1}" for i in range(num_components)]
            if fix_global_limits:
                pd.DataFrame(csv_rows, columns=cols).to_csv(out_dir / "abs_latent_spaces.csv", index=False)
            else:
                pd.DataFrame(csv_rows, columns=cols).to_csv(out_dir / "rel_latent_spaces.csv", index=False)
