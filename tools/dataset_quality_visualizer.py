#!/usr/bin/env python3
"""Dataset quality visualizer for NavVLA / OmniVLA-edge training data.

Point it at a dataset folder (the ``data_folder`` of a ``dataset.yaml`` entry,
e.g. ``data/real/tsudanuma/navvla_tsudanuma_junction``) and it shows, per
training sample, *exactly* what the fine-tuning data loader feeds the model:

* the observation context image stack and the sampled goal image,
* the supervised trajectory — predicted waypoints in the global frame and the
  normalized ego-frame target the network actually regresses,
* the language instruction attached to the current frame.

Pass a trained checkpoint with ``--weights`` (e.g. ``training/runs/model_latest.pth``
or ``training/runs/omnivla-edge_tsudanuma8.pth``) and the tool also runs the
model on the *currently displayed* sample and overlays its predicted trajectory
— in both the global path plot and the normalized ego-frame plot — next to the
teacher target, so you can eyeball how well the policy reproduces the data.

With a checkpoint loaded, the language instruction is an editable text box: type
any prompt and hit Enter / "Predict ▶" to see how the policy reacts. This is a
purely inference-time probe — the typed text is CLIP-encoded and swapped into
``feat_text`` only; the teacher targets and the dataset's ``traj_prompt.txt`` are
never modified. "lock across frames" keeps your instruction while you scrub
samples; "↺ dataset" restores the frame's ground-truth prompt.

The loader parameters ``end_slack`` / ``goals_per_obs`` / ``waypoint_spacing``
/ ``metric_waypoint_spacing`` (plus ``context_size`` / ``len_traj_pred`` /
``modality_id``) are editable in the GUI so you can see how each one reshapes
the teacher data and the number of samples.

Fidelity guarantee: :class:`SampleInspector` reuses the real training
:class:`~training.data.dataset.EdgeNavigationDataset`, so its sample index,
sample count, teacher tensors, *and* the inference inputs are identical to what
training consumes (the prediction overlay feeds the model the very same tensors,
just with a deterministic goal time).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Allow ``python tools/dataset_quality_visualizer.py`` (script mode) to import
# the training package by putting the repo root on the path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.data.dataset import MODALITY_USES, EdgeNavigationDataset

# Mirrors training/train.py: trajectories are split 70/30 train/test by sorted
# order. Shown purely as an informational tag in the GUI.
TRAIN_RATIO = 0.7


@dataclass
class VisualizerParams:
    """Loader knobs that determine which teacher data a sample contains."""

    end_slack: int = 3
    goals_per_obs: int = 1
    waypoint_spacing: int = 1
    metric_waypoint_spacing: float = 0.17
    context_size: int = 5
    len_traj_pred: int = 8
    learn_angle: bool = True
    modality_id: int = 7
    image_size: Tuple[int, int] = (96, 96)
    normalize: bool = True
    context_type: str = "temporal"

    @classmethod
    def from_repo_configs(cls, data_folder: Path) -> "VisualizerParams":
        """Best-effort defaults seeded from the repo's training configs.

        Reads ``network.yaml`` for context/horizon and, if ``data_folder``
        matches a ``dataset.yaml`` entry, seeds that entry's loader knobs so the
        tool opens showing the exact configuration training used. Falls back to
        the dataclass defaults when configs or matches are unavailable.
        """
        params = cls()
        repo_root = Path(__file__).resolve().parent.parent
        cfg_dir = repo_root / "training" / "config"
        try:
            import yaml
        except Exception:
            return params

        network_path = cfg_dir / "network.yaml"
        if network_path.exists():
            net = yaml.safe_load(network_path.read_text(encoding="utf-8")) or {}
            params = replace(
                params,
                context_size=int(net.get("context_size", params.context_size)),
                len_traj_pred=int(net.get("len_traj_pred", params.len_traj_pred)),
                learn_angle=bool(net.get("learn_angle", params.learn_angle)),
            )

        dataset_path = cfg_dir / "dataset.yaml"
        if dataset_path.exists():
            dcfg = yaml.safe_load(dataset_path.read_text(encoding="utf-8")) or {}
            if "image_size" in dcfg:
                params = replace(params, image_size=tuple(int(v) for v in dcfg["image_size"]))
            if "normalize" in dcfg:
                params = replace(params, normalize=bool(dcfg["normalize"]))
            target = data_folder.resolve()
            for entry in (dcfg.get("datasets") or {}).values():
                if not isinstance(entry, dict) or "data_folder" not in entry:
                    continue
                entry_folder = (repo_root / "data" / Path(str(entry["data_folder"]))).resolve()
                if entry_folder == target:
                    params = replace(
                        params,
                        end_slack=int(entry.get("end_slack", params.end_slack)),
                        goals_per_obs=int(entry.get("goals_per_obs", params.goals_per_obs)),
                        waypoint_spacing=int(entry.get("waypoint_spacing", params.waypoint_spacing)),
                        metric_waypoint_spacing=float(
                            entry.get("metric_waypoint_spacing", params.metric_waypoint_spacing)
                        ),
                        modality_id=int(entry.get("modality_id", params.modality_id)),
                    )
                    break
        return params


@dataclass
class SampleView:
    """Everything needed to render one training sample."""

    episode: str
    curr_time: int
    goal_time: int
    goal_time_min: int
    goal_time_max: int
    context_times: List[int]
    waypoint_times: List[int]
    positions_global: np.ndarray      # (N, 2) full episode path
    yaw_global: np.ndarray            # (N,)
    context_positions: np.ndarray     # (context_size+1, 2)
    waypoint_positions_global: np.ndarray  # (len_traj_pred+1, 2), incl. current
    goal_position_global: np.ndarray  # (2,)
    actions: np.ndarray               # (len_traj_pred, action_dim) exact target
    goal_pose: np.ndarray             # (4,) exact goal_pose tensor
    prompt: str
    scale: float                      # metric_waypoint_spacing * waypoint_spacing
    normalized: bool


class SampleInspector:
    """Read-only view over an :class:`EdgeNavigationDataset` for visualization.

    Reuses the training dataset so the sample index and teacher tensors match
    training exactly, but never normalizes images or runs CLIP — it derives the
    display geometry straight from the trajectory pickles.
    """

    def __init__(self, data_folder: Path | str, params: VisualizerParams) -> None:
        self.data_folder = Path(data_folder)
        self.params = params
        if not self.data_folder.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.data_folder}")

        traj_names = sorted(
            d.name for d in self.data_folder.iterdir()
            if d.is_dir() and (d / "traj_data.pkl").exists()
        )
        if not traj_names:
            raise ValueError(f"No trajectories (traj_data.pkl) found in {self.data_folder}")

        self.dataset = EdgeNavigationDataset(
            data_folder=self.data_folder,
            traj_names=traj_names,
            dataset_name=self.data_folder.name,
            image_size=params.image_size,
            waypoint_spacing=params.waypoint_spacing,
            len_traj_pred=params.len_traj_pred,
            learn_angle=params.learn_angle,
            context_size=params.context_size,
            context_type=params.context_type,
            end_slack=params.end_slack,
            goals_per_obs=params.goals_per_obs,
            normalize=params.normalize,
            modality_id=params.modality_id,
            metric_waypoint_spacing=params.metric_waypoint_spacing,
        )
        self.episodes: List[str] = list(self.dataset.traj_names)
        self.index_to_data: List[Tuple[str, int, int]] = list(self.dataset.index_to_data)

        self._episode_index: Dict[str, List[int]] = {name: [] for name in self.episodes}
        for base_index, (traj_name, _curr, _goal) in enumerate(self.index_to_data):
            self._episode_index[traj_name].append(base_index)

        n = int(len(self.episodes) * TRAIN_RATIO)
        self._train_episodes = set(self.episodes[:n])

    # -- counts -----------------------------------------------------------------

    @property
    def num_base_samples(self) -> int:
        return len(self.index_to_data)

    @property
    def total_training_samples(self) -> int:
        return len(self.dataset)

    @property
    def modality_uses(self) -> set:
        return set(MODALITY_USES[self.params.modality_id])

    def episode_base_indices(self, episode: str) -> List[int]:
        return list(self._episode_index.get(episode, []))

    def split_of(self, episode: str) -> str:
        return "train" if episode in self._train_episodes else "test"

    def image_path(self, episode: str, time: int) -> Path:
        return self.data_folder / episode / f"{time}.jpg"

    # -- per-sample data --------------------------------------------------------

    def _safe_prompt(self, episode: str, curr_time: int) -> str:
        try:
            return self.dataset.get_prompt(episode, curr_time)
        except FileNotFoundError:
            return "(no traj_prompt.txt)"

    def build_view(self, base_index: int, goal_time: Optional[int] = None) -> SampleView:
        traj_name, curr_time, max_goal_time = self.index_to_data[base_index]
        traj_data = self.dataset.load_trajectory(traj_name)
        positions = self.dataset.read_positions(traj_data)
        yaw = self.dataset.read_yaw(traj_data)
        n = len(positions)

        goal_time_min = curr_time + 1
        goal_time_max = max_goal_time
        if goal_time is None:
            goal_time = goal_time_max
        goal_time = int(np.clip(goal_time, goal_time_min, goal_time_max))

        context_times = list(self.dataset.context_times(curr_time))
        context_positions = positions[[int(np.clip(t, 0, n - 1)) for t in context_times]]

        spacing = self.params.waypoint_spacing
        waypoint_times = [
            min(curr_time + i * spacing, n - 1)
            for i in range(self.params.len_traj_pred + 1)
        ]
        waypoint_positions_global = positions[waypoint_times]
        goal_position_global = positions[min(goal_time, n - 1)]

        actions, goal_pos = self.dataset.compute_actions(traj_data, curr_time, goal_time)
        goal_pose = self.dataset.build_goal_pose(traj_data, curr_time, goal_time, goal_pos)

        return SampleView(
            episode=traj_name,
            curr_time=curr_time,
            goal_time=goal_time,
            goal_time_min=goal_time_min,
            goal_time_max=goal_time_max,
            context_times=context_times,
            waypoint_times=waypoint_times,
            positions_global=positions,
            yaw_global=yaw,
            context_positions=context_positions,
            waypoint_positions_global=waypoint_positions_global,
            goal_position_global=goal_position_global,
            actions=actions.numpy(),
            goal_pose=goal_pose.numpy(),
            prompt=self._safe_prompt(traj_name, curr_time),
            scale=self.params.metric_waypoint_spacing * spacing,
            normalized=self.params.normalize,
        )

    def uses_language(self) -> bool:
        """Whether the active modality conditions the policy on language."""
        return "language" in self.dataset.modality_uses

    def build_model_inputs(
        self,
        base_index: int,
        goal_time: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> Dict[str, "torch.Tensor"]:
        """Build the exact batched (B=1) tensors the model consumes for a sample.

        Delegates to :meth:`EdgeNavigationDataset.build_sample` so the inference
        overlay sees byte-identical teacher inputs to training — only the goal
        time is made deterministic (slider-controlled) instead of randomly drawn.
        Unlike :meth:`build_view`, this normalizes images and (for language
        modalities) runs the CLIP text encoder, since the model needs them.

        ``instruction`` replaces the language conditioning with a custom prompt
        for *inference-capability probing only*: it re-encodes the typed text
        with CLIP and swaps just ``feat_text``. The teacher tensors (actions,
        images, goal pose) are left untouched, and nothing is written back to
        the dataset on disk. The override is ignored for modalities that do not
        use language (``feat_text`` would be masked out anyway).
        """
        traj_name, curr_time, max_goal_time = self.index_to_data[base_index]
        if goal_time is None:
            goal_time = max_goal_time
        goal_time = int(np.clip(goal_time, curr_time + 1, max_goal_time))
        sample = self.dataset.build_sample(traj_name, curr_time, goal_time)

        if instruction is not None and self.uses_language():
            # In-memory CLIP encode (cached on the dataset); no file writes.
            sample["feat_text"] = self.dataset.encode_text(instruction)

        return {key: value.unsqueeze(0) for key, value in sample.items()}


# =============================================================================
# Inference overlay: load a trained OmniVLA-edge checkpoint and predict the
# trajectory for the displayed sample, in the same frame as the teacher data.
# =============================================================================

# network.yaml keys needed to reconstruct the model architecture.
_NETWORK_MODEL_KEYS = (
    "context_size",
    "len_traj_pred",
    "learn_angle",
    "obs_encoder",
    "obs_encoding_size",
    "late_fusion",
    "mha_num_attention_heads",
    "mha_num_attention_layers",
    "mha_ff_dim_factor",
)


def load_network_config(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Read ``training/config/network.yaml`` (the model architecture config)."""
    import yaml

    repo_root = repo_root or Path(__file__).resolve().parent.parent
    network_path = repo_root / "training" / "config" / "network.yaml"
    if not network_path.exists():
        raise FileNotFoundError(f"network.yaml not found: {network_path}")
    cfg = yaml.safe_load(network_path.read_text(encoding="utf-8")) or {}
    missing = [key for key in _NETWORK_MODEL_KEYS if key not in cfg]
    if missing:
        raise ValueError(f"network.yaml missing model keys: {missing}")
    return cfg


def ego_to_global_xy(
    ego_xy: np.ndarray,
    curr_pos: np.ndarray,
    curr_yaw: float,
    scale: float,
    normalized: bool = True,
) -> np.ndarray:
    """Map ego-frame waypoints back to the global frame.

    Exact inverse of ``EdgeNavigationDataset.convert_to_local_coords`` followed
    by the optional ``/ scale`` normalization: the loader produces ego targets
    as ``(p - curr_pos) @ R(yaw)`` (then divided by ``scale``), so the inverse
    is ``(ego * scale) @ R(yaw).T + curr_pos``.
    """
    ego_xy = np.asarray(ego_xy, dtype=np.float64).reshape(-1, 2)
    metric = ego_xy * float(scale) if normalized else ego_xy
    c, s = np.cos(curr_yaw), np.sin(curr_yaw)
    rotmat = np.array([[c, -s], [s, c]], dtype=np.float64)
    return metric.dot(rotmat.T) + np.asarray(curr_pos, dtype=np.float64).reshape(2)


def _select_device(device: Optional[str] = None) -> "torch.device":
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrajectoryPredictor:
    """Run a trained OmniVLA-edge policy on the visualizer's sample tensors.

    The model regresses waypoints in the *same* normalized ego frame as the
    teacher ``actions`` (xy are cumulatively summed, heading dims are unit
    cos/sin), so :meth:`predict` returns an array directly comparable to
    ``SampleView.actions`` and overlay-ready in both visualizer plots.
    """

    def __init__(self, model: "torch.nn.Module", device: "torch.device") -> None:
        self.device = device
        self.model = model.to(device).eval()

    @classmethod
    def from_checkpoint(
        cls,
        weights_path: Path | str,
        network_cfg: Dict[str, Any],
        device: Optional[str] = None,
    ) -> "TrajectoryPredictor":
        from OmniVLA.inference.model_omnivla_edge import OmniVLA_edge

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

        dev = _select_device(device)
        if dev.type == "cpu":
            # OmniVLA_edge.forward indexes masks via obs_img.get_device(), which
            # returns -1 on CPU and raises; the model only runs on CUDA.
            raise RuntimeError(
                "OmniVLA-edge inference requires a CUDA device "
                "(the model is not CPU-compatible). No GPU is available."
            )

        model = OmniVLA_edge(
            context_size=int(network_cfg["context_size"]),
            len_traj_pred=int(network_cfg["len_traj_pred"]),
            learn_angle=bool(network_cfg["learn_angle"]),
            obs_encoder=str(network_cfg["obs_encoder"]),
            obs_encoding_size=int(network_cfg["obs_encoding_size"]),
            late_fusion=bool(network_cfg["late_fusion"]),
            mha_num_attention_heads=int(network_cfg["mha_num_attention_heads"]),
            mha_num_attention_layers=int(network_cfg["mha_num_attention_layers"]),
            mha_ff_dim_factor=int(network_cfg["mha_ff_dim_factor"]),
        )
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = (
            checkpoint.get("state_dict", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        model.load_state_dict(state_dict, strict=True)
        return cls(model, dev)

    @torch.no_grad()
    def predict(self, inputs: Dict[str, "torch.Tensor"]) -> np.ndarray:
        """Return predicted ego-frame actions ``(len_traj_pred, action_dim)``.

        ``inputs`` is the batched dict from
        :meth:`SampleInspector.build_model_inputs`; the dtype casts mirror the
        train/eval forward call exactly.
        """
        device = self.device
        action_pred, _, _ = self.model(
            inputs["obs_images"].to(device),
            inputs["goal_pose"].float().to(device),
            inputs["map_images"].to(device),
            inputs["goal_image"].to(device),
            inputs["goal_mask"].long().to(device),
            inputs["feat_text"].float().to(device),
            inputs["current_img"].to(device),
        )
        return action_pred[0].detach().cpu().numpy()


# =============================================================================
# GUI layer (tkinter + matplotlib). Imported lazily so the inspector stays
# dependency-light for tests and headless use.
# =============================================================================

# Loader knobs exposed as editable entries, with (label, attr, kind).
_PARAM_FIELDS = (
    ("end_slack", "end_slack", int),
    ("goals_per_obs", "goals_per_obs", int),
    ("waypoint_spacing", "waypoint_spacing", int),
    ("metric_waypoint_spacing", "metric_waypoint_spacing", float),
    ("context_size", "context_size", int),
    ("len_traj_pred", "len_traj_pred", int),
    ("modality_id", "modality_id", int),
)

_THUMB_PX = 104


class DatasetQualityVisualizer:
    """Tkinter GUI that renders, per sample, what training actually loads."""

    def __init__(
        self,
        data_folder: Path | str,
        params: Optional[VisualizerParams] = None,
        weights: Optional[Path | str] = None,
        device: Optional[str] = None,
    ) -> None:
        import tkinter as tk

        self.data_folder = Path(data_folder)
        self.params = params or VisualizerParams.from_repo_configs(self.data_folder)
        self.inspector = SampleInspector(self.data_folder, self.params)

        self.predictor: Optional[TrajectoryPredictor] = None
        if weights:
            self.predictor = TrajectoryPredictor.from_checkpoint(
                weights, load_network_config(), device=device
            )

        self.tk = tk
        self.root = tk.Tk()
        title = f"Dataset Quality Visualizer — {self.data_folder.name}"
        if self.predictor is not None:
            title += f"  [weights: {Path(weights).name} on {self.predictor.device}]"
        self.root.title(title)

        self._updating = False           # guard against re-entrant slider callbacks
        self._thumb_refs: list = []       # keep PhotoImage refs alive
        self.current_episode: str = ""
        self.current_view: Optional[SampleView] = None
        self.current_pred: Optional[np.ndarray] = None   # predicted ego actions
        self._pred_error: Optional[str] = None
        self._dataset_prompt: str = ""                   # ground-truth prompt of frame

        self.param_vars: Dict[str, "tk.StringVar"] = {}
        self.info_var = tk.StringVar()
        self.episode_var = tk.StringVar()
        self.sample_var = tk.IntVar(value=0)
        self.goal_var = tk.IntVar(value=0)
        self.sample_label_var = tk.StringVar()
        self.goal_label_var = tk.StringVar()
        self.prompt_var = tk.StringVar()                 # editable instruction (override)
        self.lock_instruction_var = tk.BooleanVar(value=False)
        self.meta_var = tk.StringVar()

        self._build_gui()
        self._select_first_nonempty_episode()

    # -- layout -----------------------------------------------------------------

    def _build_gui(self) -> None:
        self._build_params_panel()
        self._build_nav_panel()
        self._build_media_panel()
        self._build_plot_panel()
        self._build_info_panel()

    def _build_params_panel(self) -> None:
        tk = self.tk
        frame = tk.LabelFrame(self.root, text="Loader parameters (training/config を反映)")
        frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        for col, (label, attr, _kind) in enumerate(_PARAM_FIELDS):
            tk.Label(frame, text=label).grid(row=0, column=col, padx=4, sticky="w")
            var = tk.StringVar(value=str(getattr(self.params, attr)))
            tk.Entry(frame, textvariable=var, width=10).grid(row=1, column=col, padx=4)
            self.param_vars[attr] = var

        tk.Button(frame, text="Apply", command=self.apply_params).grid(
            row=1, column=len(_PARAM_FIELDS), padx=8
        )
        tk.Label(frame, textvariable=self.info_var, fg="#444").grid(
            row=2, column=0, columnspan=len(_PARAM_FIELDS) + 1, sticky="w", padx=4, pady=(2, 0)
        )

    def _build_nav_panel(self) -> None:
        tk = self.tk
        from tkinter import ttk

        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=2)

        tk.Button(frame, text="◀ Prev ep", command=self.prev_episode).grid(row=0, column=0, padx=2)
        self.episode_combo = ttk.Combobox(
            frame, textvariable=self.episode_var, state="readonly", width=28,
            values=self.inspector.episodes,
        )
        self.episode_combo.grid(row=0, column=1, padx=4)
        self.episode_combo.bind("<<ComboboxSelected>>", lambda _e: self.select_episode(self.episode_var.get()))
        tk.Button(frame, text="Next ep ▶", command=self.next_episode).grid(row=0, column=2, padx=2)

        tk.Label(frame, text="sample").grid(row=1, column=0, sticky="e", padx=2)
        self.sample_scale = tk.Scale(
            frame, variable=self.sample_var, orient=tk.HORIZONTAL, from_=0, to=0,
            length=420, showvalue=False, command=self._on_sample_change,
        )
        self.sample_scale.grid(row=1, column=1, columnspan=2, sticky="we", padx=2)
        tk.Label(frame, textvariable=self.sample_label_var, width=34, anchor="w").grid(
            row=1, column=3, sticky="w", padx=4
        )

        tk.Label(frame, text="goal").grid(row=2, column=0, sticky="e", padx=2)
        self.goal_scale = tk.Scale(
            frame, variable=self.goal_var, orient=tk.HORIZONTAL, from_=0, to=0,
            length=420, showvalue=False, command=self._on_goal_change,
        )
        self.goal_scale.grid(row=2, column=1, columnspan=2, sticky="we", padx=2)
        tk.Label(frame, textvariable=self.goal_label_var, width=34, anchor="w").grid(
            row=2, column=3, sticky="w", padx=4
        )

    def _build_media_panel(self) -> None:
        tk = self.tk
        frame = tk.LabelFrame(self.root, text="Observation context  /  Goal image")
        frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        self.context_frame = tk.Frame(frame)
        self.context_frame.pack(side=tk.LEFT, padx=4, pady=2)
        self.goal_frame = tk.Frame(frame)
        self.goal_frame.pack(side=tk.LEFT, padx=16, pady=2)

    def _build_plot_panel(self) -> None:
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self.figure = Figure(figsize=(10.5, 4.2), dpi=100)
        self.ax_global = self.figure.add_subplot(1, 2, 1)
        self.ax_ego = self.figure.add_subplot(1, 2, 2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=self.tk.TOP, fill=self.tk.BOTH, expand=True, padx=6)

    def _build_info_panel(self) -> None:
        tk = self.tk
        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        frame.grid_columnconfigure(1, weight=1)

        tk.Label(frame, text="Language:", fg="#222").grid(row=0, column=0, sticky="w")
        self.prompt_entry = tk.Entry(
            frame, textvariable=self.prompt_var, fg="#0a58ca",
            font=("TkDefaultFont", 11, "bold"),
        )
        self.prompt_entry.grid(row=0, column=1, sticky="we", padx=6)
        self.prompt_entry.bind("<Return>", lambda _e: self.apply_instruction())

        self.predict_button = tk.Button(frame, text="Predict ▶", command=self.apply_instruction)
        self.predict_button.grid(row=0, column=2, padx=2)
        self.reset_prompt_button = tk.Button(
            frame, text="↺ dataset", command=self.reset_instruction
        )
        self.reset_prompt_button.grid(row=0, column=3, padx=2)
        self.lock_check = tk.Checkbutton(
            frame, text="lock across frames", variable=self.lock_instruction_var
        )
        self.lock_check.grid(row=0, column=4, padx=2)

        self.instruction_hint = tk.Label(
            frame,
            text="↑ 自由に書き換えて Enter / Predict で推論を確認 "
                 "(推論のみ・データセットは不変)",
            fg="#888",
        )
        self.instruction_hint.grid(row=1, column=1, columnspan=4, sticky="w", padx=6)

        tk.Label(frame, textvariable=self.meta_var, fg="#444", justify="left", anchor="w").grid(
            row=2, column=0, columnspan=5, sticky="w", pady=(4, 0)
        )

        if self.predictor is None:
            # No model: editing the instruction can't do anything, so disable it.
            self.prompt_entry.configure(state="readonly")
            for widget in (self.predict_button, self.reset_prompt_button, self.lock_check):
                widget.configure(state="disabled")
            self.instruction_hint.configure(
                text="(--weights でチェックポイントを渡すと推論・指示の差し替えが可能)"
            )

    # -- parameter editing ------------------------------------------------------

    def apply_params(self) -> None:
        from tkinter import messagebox

        try:
            new_params = self._read_params()
        except ValueError as exc:
            messagebox.showerror("Invalid parameter", str(exc))
            return
        try:
            inspector = SampleInspector(self.data_folder, new_params)
        except Exception as exc:  # noqa: BLE001 - surface any loader error to the user
            messagebox.showerror("Cannot build dataset", str(exc))
            return

        self.params = new_params
        self.inspector = inspector
        self.episode_combo.configure(values=self.inspector.episodes)
        self._select_first_nonempty_episode()

    def _read_params(self) -> VisualizerParams:
        values = {}
        for _label, attr, kind in _PARAM_FIELDS:
            raw = self.param_vars[attr].get().strip()
            try:
                values[attr] = kind(raw)
            except ValueError:
                raise ValueError(f"'{attr}' must be {kind.__name__}, got '{raw}'")
        return replace(self.params, **values)

    # -- navigation -------------------------------------------------------------

    def _select_first_nonempty_episode(self) -> None:
        total = self.inspector.total_training_samples
        base = self.inspector.num_base_samples
        uses = ", ".join(sorted(self.inspector.modality_uses)) or "(none)"
        self.info_var.set(
            f"episodes={len(self.inspector.episodes)}  base_samples={base}  "
            f"training_pairs={total} (=base×goals_per_obs)  modality_id="
            f"{self.params.modality_id} → uses[{uses}]"
        )
        target = next(
            (ep for ep in self.inspector.episodes if self.inspector.episode_base_indices(ep)),
            self.inspector.episodes[0],
        )
        self.episode_var.set(target)
        self.select_episode(target)

    def select_episode(self, episode: str) -> None:
        self.current_episode = episode
        self.episode_var.set(episode)
        indices = self.inspector.episode_base_indices(episode)

        self._updating = True
        if indices:
            self.sample_scale.configure(from_=0, to=len(indices) - 1, state="normal")
            self.sample_var.set(0)
        else:
            self.sample_scale.configure(from_=0, to=0, state="disabled")
        self._updating = False
        self._on_sample_change(None)

    def prev_episode(self) -> None:
        episodes = self.inspector.episodes
        i = episodes.index(self.current_episode)
        if i > 0:
            self.select_episode(episodes[i - 1])

    def next_episode(self) -> None:
        episodes = self.inspector.episodes
        i = episodes.index(self.current_episode)
        if i < len(episodes) - 1:
            self.select_episode(episodes[i + 1])

    def _current_base_index(self) -> Optional[int]:
        indices = self.inspector.episode_base_indices(self.current_episode)
        if not indices:
            return None
        return indices[int(np.clip(self.sample_var.get(), 0, len(indices) - 1))]

    def _on_sample_change(self, _value) -> None:
        if self._updating:
            return
        base_index = self._current_base_index()
        if base_index is None:
            self._render_empty_episode()
            return
        view = self.inspector.build_view(base_index)  # goal defaults to max

        self._updating = True
        self.goal_scale.configure(from_=view.goal_time_min, to=view.goal_time_max, state="normal")
        self.goal_var.set(view.goal_time)
        self._updating = False

        self._render(view)

    def _on_goal_change(self, _value) -> None:
        if self._updating:
            return
        base_index = self._current_base_index()
        if base_index is None:
            return
        view = self.inspector.build_view(base_index, goal_time=self.goal_var.get())
        self._render(view)

    # -- rendering --------------------------------------------------------------

    def _render_empty_episode(self) -> None:
        self.current_view = None
        for child in self.context_frame.winfo_children():
            child.destroy()
        for child in self.goal_frame.winfo_children():
            child.destroy()
        self.tk.Label(
            self.context_frame,
            text="このパラメータでは学習サンプルが 0 件です\n"
                 "(context_size×spacing + len_traj_pred×spacing + end_slack が軌跡長を超過)",
            fg="#b00", justify="left",
        ).pack()
        self.ax_global.clear()
        self.ax_ego.clear()
        self.canvas.draw_idle()
        self._sync_instruction_to_dataset("")
        split = self.inspector.split_of(self.current_episode)
        self.meta_var.set(f"episode={self.current_episode} [{split}] — no samples")
        self.sample_label_var.set("sample 0/0")
        self.goal_label_var.set("")

    def _render(self, view: SampleView) -> None:
        self.current_view = view
        self._sync_instruction_to_dataset(view.prompt)
        self.current_pred = self._predict_actions(view)
        self._draw_images(view)
        self._draw_plots(view, self.current_pred)
        self._draw_info(view, self.current_pred)

    # -- language instruction override (inference only) -------------------------

    def _sync_instruction_to_dataset(self, dataset_prompt: str) -> None:
        """Track the frame's ground-truth prompt; refill the entry unless locked.

        Navigating to a new sample resets the editable instruction back to that
        frame's dataset prompt, so the default view always shows the real data.
        With "lock across frames" checked, the user's typed instruction is kept
        and re-applied to each frame for probing.
        """
        self._dataset_prompt = dataset_prompt
        if self.predictor is None or not self.lock_instruction_var.get():
            self.prompt_var.set(dataset_prompt)

    def _instruction_override(self) -> Optional[str]:
        """The typed instruction if it differs from the dataset prompt, else None."""
        text = self.prompt_var.get().strip()
        if not text or text == (self._dataset_prompt or "").strip():
            return None
        return text

    def apply_instruction(self) -> None:
        """Re-run inference for the current frame using the typed instruction.

        Only the prediction (plots + info) is refreshed; the teacher images and
        sample stay put, and the dataset on disk is never modified.
        """
        if self.predictor is None or self.current_view is None:
            return
        view = self.current_view
        self.current_pred = self._predict_actions(view)
        self._draw_plots(view, self.current_pred)
        self._draw_info(view, self.current_pred)

    def reset_instruction(self) -> None:
        """Restore the dataset's ground-truth prompt and re-predict."""
        self.prompt_var.set(self._dataset_prompt)
        self.apply_instruction()

    def _predict_actions(self, view: SampleView) -> Optional[np.ndarray]:
        """Run the loaded policy on the displayed sample; None if no/failed model.

        Failures (e.g. GUI params diverging from the checkpoint's architecture)
        are non-fatal: the message is surfaced in the info panel instead.
        """
        if self.predictor is None:
            self._pred_error = None
            return None
        base_index = self._current_base_index()
        if base_index is None:
            return None
        try:
            inputs = self.inspector.build_model_inputs(
                base_index,
                goal_time=view.goal_time,
                instruction=self._instruction_override(),
            )
            pred = self.predictor.predict(inputs)
            self._pred_error = None
            return pred
        except Exception as exc:  # noqa: BLE001 - keep the GUI responsive
            self._pred_error = f"{type(exc).__name__}: {exc}"
            return None

    def _draw_images(self, view: SampleView) -> None:
        for child in self.context_frame.winfo_children():
            child.destroy()
        for child in self.goal_frame.winfo_children():
            child.destroy()
        self._thumb_refs = []

        for col, time in enumerate(view.context_times):
            is_current = time == view.curr_time
            cell = self.tk.Frame(
                self.context_frame,
                highlightthickness=2,
                highlightbackground="#ff8c00" if is_current else "#cccccc",
            )
            cell.grid(row=0, column=col, padx=2)
            self._place_thumb(cell, view.episode, time)
            tag = f"t={time}" + (" (current)" if is_current else "")
            self.tk.Label(cell, text=tag, fg="#ff8c00" if is_current else "#666").pack()

        goal_cell = self.tk.Frame(
            self.goal_frame, highlightthickness=3, highlightbackground="#2e8b57"
        )
        goal_cell.pack()
        self._place_thumb(goal_cell, view.episode, view.goal_time, size=int(_THUMB_PX * 1.4))
        self.tk.Label(goal_cell, text=f"goal t={view.goal_time}", fg="#2e8b57").pack()

    def _place_thumb(self, parent, episode: str, time: int, size: int = _THUMB_PX) -> None:
        from PIL import Image, ImageTk

        path = self.inspector.image_path(episode, time)
        if path.exists():
            img = Image.open(path).convert("RGB").resize((size, size))
            photo = ImageTk.PhotoImage(img)
            self._thumb_refs.append(photo)
            self.tk.Label(parent, image=photo).pack()
        else:
            self.tk.Label(parent, text=f"{time}.jpg\nmissing", width=12, height=6, fg="#b00").pack()

    def _draw_plots(self, view: SampleView, pred: Optional[np.ndarray] = None) -> None:
        ax = self.ax_global
        ax.clear()
        pos = view.positions_global
        ax.plot(pos[:, 0], pos[:, 1], color="0.82", lw=1.0, marker=".", ms=2, zorder=1,
                label="trajectory")

        # goal-sampling window: positions between current and the latest goal.
        win = pos[view.curr_time: view.goal_time_max + 1]
        if len(win) > 1:
            ax.plot(win[:, 0], win[:, 1], color="#ffd28a", lw=4, alpha=0.7, zorder=2,
                    label="goal window")

        ctx = view.context_positions
        ax.scatter(ctx[:, 0], ctx[:, 1], c="#1f77b4", s=22, zorder=3, label="context obs")

        wps = view.waypoint_positions_global
        ax.plot(wps[:, 0], wps[:, 1], color="#d62728", lw=1.8, marker="o", ms=4, zorder=4,
                label="target waypoints")

        cx, cy = pos[view.curr_time]
        yaw = float(view.yaw_global[view.curr_time])

        # predicted trajectory mapped from the ego frame back into the global one.
        if pred is not None:
            pred_global = ego_to_global_xy(pred[:, :2], (cx, cy), yaw, view.scale, view.normalized)
            pred_global = np.vstack([[cx, cy], pred_global])
            ax.plot(pred_global[:, 0], pred_global[:, 1], color="#0033cc", lw=1.8,
                    ls="--", marker="s", ms=4, zorder=8, label="predicted")

        arrow = 0.6 * (np.ptp(pos[:, 0]) + np.ptp(pos[:, 1])) / 20 + 1e-3
        ax.scatter([cx], [cy], c="#ff8c00", s=70, zorder=5, label="current")
        ax.annotate("", xy=(cx + arrow * np.cos(yaw), cy + arrow * np.sin(yaw)), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color="#ff8c00", lw=1.5), zorder=6)

        gx, gy = view.goal_position_global
        ax.scatter([gx], [gy], marker="*", c="#2e8b57", s=160, zorder=7, label="goal")

        ax.set_title(f"Global path — {view.episode}  (t={view.curr_time})")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

        # Ego frame: the literal regression target (normalized when normalize=True).
        ego = self.ax_ego
        ego.clear()
        target_xy = np.vstack([[0.0, 0.0], view.actions[:, :2]])
        ego.plot(target_xy[:, 0], target_xy[:, 1], color="#d62728", lw=1.8, marker="o", ms=4,
                 label="target")
        ego.scatter([0], [0], c="#ff8c00", s=70, zorder=5, label="current (ego origin)")
        if view.actions.shape[1] >= 4:  # learn_angle: draw per-waypoint heading
            step = max(abs(target_xy[:, 0].ptp()), abs(target_xy[:, 1].ptp()), 1e-3) * 0.12
            for (px, py), (c, s) in zip(view.actions[:, :2], view.actions[:, 2:4]):
                ego.annotate("", xy=(px + step * c, py + step * s), xytext=(px, py),
                             arrowprops=dict(arrowstyle="->", color="#9467bd", lw=1.0))

        if pred is not None:
            pred_xy = np.vstack([[0.0, 0.0], pred[:, :2]])
            ego.plot(pred_xy[:, 0], pred_xy[:, 1], color="#0033cc", lw=1.8, ls="--",
                     marker="s", ms=4, zorder=6, label="predicted")
            if pred.shape[1] >= 4:
                step = max(abs(pred_xy[:, 0].ptp()), abs(pred_xy[:, 1].ptp()), 1e-3) * 0.12
                for (px, py), (c, s) in zip(pred[:, :2], pred[:, 2:4]):
                    ego.annotate("", xy=(px + step * c, py + step * s), xytext=(px, py),
                                 arrowprops=dict(arrowstyle="->", color="#0033cc", lw=1.0))

        unit = "normalized ÷{:.3f}".format(view.scale) if view.normalized else "metric [m]"
        ego.set_title(f"Ego-frame target vs prediction ({unit})")
        ego.set_xlabel("forward")
        ego.set_ylabel("left")
        ego.axis("equal")
        ego.grid(True, alpha=0.3)
        ego.legend(fontsize=7, loc="best")

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _draw_info(self, view: SampleView, pred: Optional[np.ndarray] = None) -> None:
        indices = self.inspector.episode_base_indices(view.episode)
        pos_in_ep = indices.index(self._current_base_index()) + 1
        split = self.inspector.split_of(view.episode)

        self.sample_label_var.set(
            f"sample {pos_in_ep}/{len(indices)}  (curr_time={view.curr_time})"
        )
        self.goal_label_var.set(
            f"goal_time={view.goal_time}  range=[{view.goal_time_min}, {view.goal_time_max}]"
        )

        cols = ("dx", "dy", "cosΔψ", "sinΔψ") if view.actions.shape[1] >= 4 else ("dx", "dy")
        header = "  ".join(f"{c:>8}" for c in cols)
        rows = []
        for i, row in enumerate(view.actions, start=1):
            rows.append(f"wp{i:>2}  " + "  ".join(f"{v:8.3f}" for v in row))
        actions_block = f"actions ({header.strip()}):\n" + "\n".join(rows)

        uses = ", ".join(sorted(self.inspector.modality_uses)) or "(none)"
        goal_pose = ", ".join(f"{v:.3f}" for v in view.goal_pose)
        self.meta_var.set(
            f"episode={view.episode} [{split}]   modality uses[{uses}]   "
            f"goal_pose=[{goal_pose}]  (zeros unless pose modality)\n"
            f"normalize={view.normalized}  scale=metric×spacing={view.scale:.4f}\n"
            f"{self._prediction_summary(view, pred)}"
            f"{actions_block}"
        )

    def _prediction_summary(self, view: SampleView, pred: Optional[np.ndarray]) -> str:
        """Instruction status + policy-vs-target error block for the info panel."""
        if self.predictor is None:
            return ""

        override = self._instruction_override()
        if not self.inspector.uses_language():
            instr_line = (
                f"instruction: modality {self.params.modality_id} は言語非依存 "
                f"→ 指示の差し替えは無効"
            )
        elif override is not None:
            instr_line = (
                f'instruction: OVERRIDE → "{override}"   '
                f'(dataset: "{self._dataset_prompt}")'
            )
        else:
            instr_line = "instruction: dataset prompt (override なし)"

        if pred is None:
            return f"{instr_line}\nprediction: FAILED — {self._pred_error}\n"

        n = min(len(pred), len(view.actions))
        metric = view.scale if view.normalized else 1.0
        diff_xy = (pred[:n, :2] - view.actions[:n, :2]) * metric
        per_wp = np.linalg.norm(diff_xy, axis=1)
        mean_err = float(per_wp.mean())
        final_err = float(per_wp[-1])

        heading = ""
        if pred.shape[1] >= 4 and view.actions.shape[1] >= 4:
            dot = np.clip((pred[:n, 2:4] * view.actions[:n, 2:4]).sum(axis=1), -1.0, 1.0)
            heading = f"  heading_err(mean)={np.degrees(np.arccos(dot)).mean():.1f}°"
        # With an override the target still reflects the original instruction, so
        # a larger error here is expected and is the point of the comparison.
        target_tag = "dataset target (元の指示)" if override is not None else "target"
        return (
            f"{instr_line}\n"
            f"prediction vs {target_tag} [m]: mean={mean_err:.3f}  "
            f"final={final_err:.3f}{heading}\n"
        )

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize NavVLA training data the way the data loader sees it.",
    )
    parser.add_argument(
        "dataset_dir", type=Path,
        help="Dataset folder containing episode subdirs (each with traj_data.pkl).",
    )
    parser.add_argument(
        "--weights", type=Path, default=None,
        help="Trained OmniVLA-edge checkpoint (e.g. training/runs/model_latest.pth). "
             "When given, the model's predicted trajectory is overlaid on each sample.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device for inference (default: cuda if available, else cpu).",
    )
    args = parser.parse_args()
    DatasetQualityVisualizer(args.dataset_dir, weights=args.weights, device=args.device).run()


if __name__ == "__main__":
    main()
