"""Tests for the dataset quality visualizer's pure-logic layer.

The visualizer must show *exactly* what training loads, so ``SampleInspector``
is required to reuse :class:`EdgeNavigationDataset` and reproduce its sample
index, sample count, and teacher tensors bit-for-bit. These tests pin that
contract; the tkinter/matplotlib GUI on top is not exercised here.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from tools.dataset_quality_visualizer import (
    SampleInspector,
    TrajectoryPredictor,
    VisualizerParams,
    ego_to_global_xy,
)
from training.data.dataset import EdgeNavigationDataset


# --- synthetic dataset fixture ------------------------------------------------

def _write_episode(ep_dir: Path, n_frames: int, *, with_prompt: bool = True) -> None:
    ep_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(n_frames, dtype=np.float32)
    x = t * 0.2
    y = 0.1 * np.sin(t * 0.3)
    positions = np.stack([x, y], axis=-1).astype(np.float32)
    yaw = np.arctan2(np.gradient(y), np.gradient(x)).astype(np.float32)
    with (ep_dir / "traj_data.pkl").open("wb") as f:
        pickle.dump({"position": positions, "yaw": yaw}, f)
    for i in range(n_frames):
        Image.new("RGB", (8, 8), (i % 255, 0, 0)).save(ep_dir / f"{i}.jpg")
    if with_prompt:
        lines = [f"step {i}" for i in range(n_frames)]
        (ep_dir / "traj_prompt.txt").write_text("\n".join(lines), encoding="utf-8")


@pytest.fixture()
def dataset_dir(tmp_path: Path) -> Path:
    root = tmp_path / "navvla_demo"
    _write_episode(root / "episode01", 24)
    _write_episode(root / "episode02", 30)
    _write_episode(root / "episode03", 18)
    return root


def _params(**overrides) -> VisualizerParams:
    base = dict(
        end_slack=1,
        goals_per_obs=1,
        waypoint_spacing=1,
        metric_waypoint_spacing=0.5,
        context_size=2,
        len_traj_pred=3,
        learn_angle=True,
        modality_id=7,
        image_size=(8, 8),
        normalize=True,
    )
    base.update(overrides)
    return VisualizerParams(**base)


def _reference_dataset(dataset_dir: Path, params: VisualizerParams) -> EdgeNavigationDataset:
    trajs = sorted(
        d.name for d in dataset_dir.iterdir()
        if d.is_dir() and (d / "traj_data.pkl").exists()
    )
    return EdgeNavigationDataset(
        data_folder=dataset_dir,
        traj_names=trajs,
        dataset_name="ref",
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


# --- fidelity to the training data loader ------------------------------------

def test_inspector_reproduces_training_sample_index(dataset_dir: Path) -> None:
    params = _params()
    inspector = SampleInspector(dataset_dir, params)
    ref = _reference_dataset(dataset_dir, params)

    assert inspector.index_to_data == ref.index_to_data
    assert inspector.num_base_samples == len(ref.index_to_data)
    assert inspector.total_training_samples == len(ref)


def test_total_samples_scale_with_goals_per_obs(dataset_dir: Path) -> None:
    single = SampleInspector(dataset_dir, _params(goals_per_obs=1))
    triple = SampleInspector(dataset_dir, _params(goals_per_obs=3))

    assert single.num_base_samples == triple.num_base_samples
    assert triple.total_training_samples == single.num_base_samples * 3


def test_end_slack_monotonically_reduces_samples(dataset_dir: Path) -> None:
    few = SampleInspector(dataset_dir, _params(end_slack=1)).num_base_samples
    more = SampleInspector(dataset_dir, _params(end_slack=4)).num_base_samples
    assert more < few


def test_episode_grouping_covers_all_base_indices(dataset_dir: Path) -> None:
    inspector = SampleInspector(dataset_dir, _params())

    collected: list[int] = []
    for episode in inspector.episodes:
        idxs = inspector.episode_base_indices(episode)
        for base_index in idxs:
            assert inspector.index_to_data[base_index][0] == episode
        collected.extend(idxs)

    assert sorted(collected) == list(range(inspector.num_base_samples))


# --- per-sample teacher data (granularity the user asked to verify) ----------

def test_build_view_shapes_match_training_target(dataset_dir: Path) -> None:
    params = _params()
    inspector = SampleInspector(dataset_dir, params)
    view = inspector.build_view(0)

    assert len(view.context_times) == params.context_size + 1
    assert view.waypoint_positions_global.shape == (params.len_traj_pred + 1, 2)
    # learn_angle -> [dx, dy, cos, sin]
    assert view.actions.shape == (params.len_traj_pred, 4)
    assert view.goal_pose.shape == (4,)
    assert view.goal_time_min <= view.goal_time <= view.goal_time_max


def test_build_view_actions_equal_training_compute_actions(dataset_dir: Path) -> None:
    params = _params()
    inspector = SampleInspector(dataset_dir, params)
    ref = _reference_dataset(dataset_dir, params)

    traj_name, curr_time, max_goal_time = ref.index_to_data[2]
    traj_data = ref.load_trajectory(traj_name)
    ref_actions, _ = ref.compute_actions(traj_data, curr_time, max_goal_time)

    view = inspector.build_view(2, goal_time=max_goal_time)
    np.testing.assert_allclose(view.actions, ref_actions.numpy(), rtol=1e-6, atol=1e-6)


def test_metric_waypoint_spacing_scales_normalized_actions(dataset_dir: Path) -> None:
    half = SampleInspector(dataset_dir, _params(metric_waypoint_spacing=0.5))
    full = SampleInspector(dataset_dir, _params(metric_waypoint_spacing=1.0))

    base_index, goal_time = 3, None
    v_half = half.build_view(base_index, goal_time=goal_time)
    v_full = full.build_view(base_index, goal_time=goal_time)

    # xy targets are divided by metric_waypoint_spacing*waypoint_spacing
    np.testing.assert_allclose(
        v_full.actions[:, :2] * 2.0, v_half.actions[:, :2], rtol=1e-6, atol=1e-6
    )
    # heading dims (cos/sin) are unaffected by the metric scale
    np.testing.assert_allclose(
        v_full.actions[:, 2:], v_half.actions[:, 2:], rtol=1e-6, atol=1e-6
    )


def test_view_prompt_matches_annotation_file(dataset_dir: Path) -> None:
    inspector = SampleInspector(dataset_dir, _params())
    view = inspector.build_view(0)
    expected = f"step {view.curr_time}"
    assert view.prompt == expected


def test_image_path_points_at_existing_frames(dataset_dir: Path) -> None:
    inspector = SampleInspector(dataset_dir, _params())
    view = inspector.build_view(0)
    for time in view.context_times + [view.goal_time]:
        assert inspector.image_path(view.episode, time).exists()


# --- inference overlay layer -------------------------------------------------
# These cover the new "show the model's predicted trajectory on top of the
# teacher data" feature. They use modality_id=6 (image-only) so no CLIP text
# encoder is needed, keeping the unit tests fast and offline.

def _img_params(**overrides) -> VisualizerParams:
    """CLIP-free params (image-only modality) for inference-layer tests."""
    return _params(modality_id=6, **overrides)


def test_build_model_inputs_have_model_ready_shapes(dataset_dir: Path) -> None:
    params = _img_params()
    inspector = SampleInspector(dataset_dir, params)
    inputs = inspector.build_model_inputs(0)

    expected_keys = {
        "obs_images", "goal_pose", "map_images", "goal_image",
        "goal_mask", "feat_text", "current_img", "actions",
    }
    assert expected_keys <= set(inputs)
    # every tensor carries a leading batch dim of 1
    assert inputs["obs_images"].shape[0] == 1
    assert inputs["obs_images"].shape[1] == 3 * (params.context_size + 1)
    assert inputs["map_images"].shape[1] == 9
    assert inputs["goal_pose"].shape == (1, 4)
    assert inputs["goal_mask"].shape == (1,)
    assert inputs["feat_text"].shape == (1, 512)
    assert inputs["actions"].shape == (1, params.len_traj_pred, 4)


def test_build_model_inputs_actions_match_compute_actions(dataset_dir: Path) -> None:
    params = _img_params()
    inspector = SampleInspector(dataset_dir, params)

    base_index = 2
    traj_name, curr_time, max_goal_time = inspector.index_to_data[base_index]
    inputs = inspector.build_model_inputs(base_index, goal_time=max_goal_time)

    traj_data = inspector.dataset.load_trajectory(traj_name)
    ref_actions, _ = inspector.dataset.compute_actions(traj_data, curr_time, max_goal_time)
    np.testing.assert_allclose(
        inputs["actions"][0].numpy(), ref_actions.numpy(), rtol=1e-6, atol=1e-6
    )


def test_ego_to_global_inverts_local_coords(dataset_dir: Path) -> None:
    """ego_to_global_xy must be the exact inverse of the loader's ego transform.

    The supervised ego targets, mapped back through ego_to_global_xy, must land
    on the global waypoint positions the loader sampled them from.
    """
    inspector = SampleInspector(dataset_dir, _img_params())
    view = inspector.build_view(2)

    global_xy = ego_to_global_xy(
        view.actions[:, :2],
        view.positions_global[view.curr_time],
        float(view.yaw_global[view.curr_time]),
        view.scale,
        view.normalized,
    )
    np.testing.assert_allclose(
        global_xy, view.waypoint_positions_global[1:], rtol=1e-4, atol=1e-3
    )


class _StubPolicy(nn.Module):
    """Minimal stand-in for OmniVLA_edge for predictor-logic tests."""

    def __init__(self, len_traj_pred: int, num_params: int) -> None:
        super().__init__()
        self.len_traj_pred = len_traj_pred
        self.num_params = num_params
        self.calls: list = []

    def forward(self, *args):  # noqa: D401 - mirrors OmniVLA_edge's positional API
        self.calls.append(args)
        batch = args[0].shape[0]
        action_pred = torch.arange(
            batch * self.len_traj_pred * self.num_params, dtype=torch.float32
        ).reshape(batch, self.len_traj_pred, self.num_params)
        return action_pred, torch.zeros(batch, 1), args[4]


def test_predictor_passes_inputs_in_model_order_and_returns_ego(dataset_dir: Path) -> None:
    params = _img_params()
    inspector = SampleInspector(dataset_dir, params)
    inputs = inspector.build_model_inputs(0)

    stub = _StubPolicy(params.len_traj_pred, 4)
    predictor = TrajectoryPredictor(stub, torch.device("cpu"))
    pred = predictor.predict(inputs)

    assert isinstance(pred, np.ndarray)
    assert pred.shape == (params.len_traj_pred, 4)
    # forward is called with the 7 positional tensors in OmniVLA_edge's order
    assert len(stub.calls) == 1
    called = stub.calls[0]
    assert len(called) == 7
    assert torch.equal(called[0], inputs["obs_images"])  # obs_images
    assert torch.equal(called[4], inputs["goal_mask"])   # goal_mask (long)


# --- language-instruction override (inference-only) --------------------------
# The user can type a custom instruction to probe the policy. It must change the
# language conditioning (feat_text) only, never the teacher tensors, and must
# never touch the dataset on disk. CLIP is stubbed so these stay fast/offline.

def _stub_text_encoder(monkeypatch, inspector: SampleInspector):
    """Replace CLIP encode_text with a deterministic per-text vector."""
    def fake_encode(text: str) -> torch.Tensor:
        vec = torch.zeros(512, dtype=torch.float32)
        vec[0] = float(abs(hash(text)) % 997 + 1)
        return vec
    monkeypatch.setattr(inspector.dataset, "encode_text", fake_encode)
    return fake_encode


def test_instruction_override_changes_feat_text_only(dataset_dir: Path, monkeypatch) -> None:
    params = _params(modality_id=7)  # language modality
    inspector = SampleInspector(dataset_dir, params)
    fake_encode = _stub_text_encoder(monkeypatch, inspector)

    default = inspector.build_model_inputs(0)
    overridden = inspector.build_model_inputs(0, instruction="go to the red door")

    # teacher tensors are untouched by the override
    assert torch.equal(default["actions"], overridden["actions"])
    assert torch.equal(default["obs_images"], overridden["obs_images"])
    assert torch.equal(default["goal_image"], overridden["goal_image"])
    # only the language conditioning changes, to the typed instruction's encoding
    assert not torch.equal(default["feat_text"], overridden["feat_text"])
    assert torch.equal(overridden["feat_text"][0], fake_encode("go to the red door"))


def test_instruction_override_ignored_without_language_modality(
    dataset_dir: Path, monkeypatch
) -> None:
    params = _img_params()  # modality_id=6, image-only
    inspector = SampleInspector(dataset_dir, params)
    calls: list[str] = []
    monkeypatch.setattr(
        inspector.dataset, "encode_text",
        lambda text: calls.append(text) or torch.zeros(512),
    )

    default = inspector.build_model_inputs(0)
    overridden = inspector.build_model_inputs(0, instruction="should be ignored")

    # non-language modality: feat_text stays as-is and CLIP is never invoked
    assert torch.equal(default["feat_text"], overridden["feat_text"])
    assert calls == []


def test_instruction_override_does_not_touch_dataset_files(
    dataset_dir: Path, monkeypatch
) -> None:
    params = _params(modality_id=7)
    inspector = SampleInspector(dataset_dir, params)
    _stub_text_encoder(monkeypatch, inspector)

    traj_name = inspector.index_to_data[0][0]
    prompt_file = dataset_dir / traj_name / "traj_prompt.txt"
    before_text = prompt_file.read_text(encoding="utf-8")
    before_mtime = prompt_file.stat().st_mtime

    inspector.build_model_inputs(0, instruction="a brand new instruction")

    assert prompt_file.read_text(encoding="utf-8") == before_text
    assert prompt_file.stat().st_mtime == before_mtime
