# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Seamless Animation Transitions — 2026 Enhancement
==================================================
Provides fluid, artifact-free transitions between animation clips.
Inspired by 2026 "dimension-jumping" mechanics (e.g. Black Slate) and the
seamless world-swapping animations enabled by modern hardware.

Features
--------
- **Blend tree** — weighted blending of multiple clips with smooth alpha curves
- **Stitching** — momentum-preserving clip stitching using velocity matching
- **Phase matching** — gait-phase-aligned transitions to eliminate foot pop
- **Inertia continuation** — carries angular/linear momentum into the new clip
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class ClipHandle:
    """A named reference to a loaded animation clip."""

    name: str
    positions: np.ndarray  # [T, J, 3]
    rotations: np.ndarray  # [T, J, 4]  quaternion (w, x, y, z)
    framerate: float
    tags: list[str] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return self.positions.shape[0]

    @property
    def duration(self) -> float:
        return (self.num_frames - 1) / self.framerate


def _slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions."""
    dot = np.sum(q1 * q2, axis=-1, keepdims=True)
    # Ensure shortest path
    q2 = np.where(dot < 0, -q2, q2)
    dot = np.abs(dot)
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    # Fallback to lerp when sin_theta is near zero
    safe = sin_theta > 1e-6
    w1 = np.where(safe, np.sin((1 - t) * theta) / (sin_theta + 1e-12), 1 - t)
    w2 = np.where(safe, np.sin(t * theta) / (sin_theta + 1e-12), t)
    out = w1 * q1 + w2 * q2
    return out / (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-12)


def _smooth_alpha(t: float, curve: str = "cosine") -> float:
    """Return blend alpha in [0, 1] for a given transition progress t in [0, 1]."""
    if curve == "linear":
        return t
    elif curve == "cosine":
        return 0.5 - 0.5 * np.cos(np.pi * t)
    elif curve == "cubic":
        return 3 * t**2 - 2 * t**3
    else:
        return t


class TransitionBlender:
    """
    Blend two animation clips with configurable overlap and easing.

    Parameters
    ----------
    blend_frames : int
        Number of frames over which to cross-fade.
    curve : str
        Easing curve: 'linear', 'cosine', or 'cubic'.
    """

    def __init__(self, blend_frames: int = 15, curve: str = "cosine"):
        self.blend_frames = blend_frames
        self.curve = curve

    def blend(self, clip_a: ClipHandle, clip_b: ClipHandle) -> ClipHandle:
        """
        Produce a stitched clip that transitions from *clip_a* to *clip_b*.

        The last *blend_frames* frames of clip_a are cross-faded with the
        first *blend_frames* frames of clip_b.

        Returns
        -------
        ClipHandle
            Merged clip with a seamless transition.
        """
        bf = min(self.blend_frames, clip_a.num_frames, clip_b.num_frames)

        # Frames before blend region from clip A
        pre_pos = clip_a.positions[: clip_a.num_frames - bf]
        pre_rot = clip_a.rotations[: clip_a.num_frames - bf]

        # Blend region
        blend_pos = np.zeros((bf, clip_a.positions.shape[1], 3))
        blend_rot = np.zeros((bf, clip_a.rotations.shape[1], 4))
        for i in range(bf):
            alpha = _smooth_alpha(i / max(bf - 1, 1), self.curve)
            blend_pos[i] = (1 - alpha) * clip_a.positions[
                clip_a.num_frames - bf + i
            ] + alpha * clip_b.positions[i]
            blend_rot[i] = _slerp(
                clip_a.rotations[clip_a.num_frames - bf + i],
                clip_b.rotations[i],
                alpha,
            )

        # Frames after blend region from clip B
        post_pos = clip_b.positions[bf:]
        post_rot = clip_b.rotations[bf:]

        merged_positions = np.concatenate([pre_pos, blend_pos, post_pos], axis=0)
        merged_rotations = np.concatenate([pre_rot, blend_rot, post_rot], axis=0)

        return ClipHandle(
            name=f"{clip_a.name}_to_{clip_b.name}",
            positions=merged_positions,
            rotations=merged_rotations,
            framerate=clip_a.framerate,
            tags=["transition", *clip_a.tags, *clip_b.tags],
        )


class PhaseMatchedTransition:
    """
    Phase-matched transition that aligns gait cycles before blending.

    For locomotion clips, this avoids the "foot pop" artifact that occurs
    when two clips are at different gait phases at the transition point.

    Parameters
    ----------
    contact_joint_indices : list[int]
        Indices of contact joints (e.g. left/right ankle).
    blend_frames : int
        Blend window size.
    """

    def __init__(
        self,
        contact_joint_indices: list[int] | None = None,
        blend_frames: int = 20,
    ):
        self.contact_joints = contact_joint_indices or [3, 7]  # typical ankle indices
        self.blend_frames = blend_frames
        self.blender = TransitionBlender(blend_frames, curve="cosine")

    def _contact_score(self, positions: np.ndarray, frame: int) -> float:
        """Simple contact score based on foot height proximity to ground."""
        foot_heights = positions[frame, self.contact_joints, 1]
        return float(np.mean(foot_heights))

    def find_best_transition_frame(
        self,
        clip: ClipHandle,
        search_start: int,
        search_end: int,
    ) -> int:
        """Find the frame in *clip* with lowest contact score (feet near ground)."""
        search_end = min(search_end, clip.num_frames - 1)
        best_frame = search_start
        best_score = float("inf")
        for f in range(search_start, search_end + 1):
            score = self._contact_score(clip.positions, f)
            if score < best_score:
                best_score = score
                best_frame = f
        return best_frame

    def transition(
        self,
        clip_a: ClipHandle,
        clip_b: ClipHandle,
        search_window: int = 10,
    ) -> ClipHandle:
        """
        Find best phase match point and blend clips.

        Parameters
        ----------
        clip_a, clip_b : ClipHandle
        search_window : int
            Number of frames around the end of clip_a and start of clip_b
            to search for the best contact alignment.

        Returns
        -------
        ClipHandle
        """
        end_a = self.find_best_transition_frame(
            clip_a,
            max(0, clip_a.num_frames - search_window - 1),
            clip_a.num_frames - 1,
        )
        start_b = self.find_best_transition_frame(
            clip_b,
            0,
            min(search_window, clip_b.num_frames - 1),
        )

        # Trim clips to transition points
        trimmed_a = ClipHandle(
            name=clip_a.name,
            positions=clip_a.positions[: end_a + 1],
            rotations=clip_a.rotations[: end_a + 1],
            framerate=clip_a.framerate,
            tags=clip_a.tags,
        )
        trimmed_b = ClipHandle(
            name=clip_b.name,
            positions=clip_b.positions[start_b:],
            rotations=clip_b.rotations[start_b:],
            framerate=clip_b.framerate,
            tags=clip_b.tags,
        )

        return self.blender.blend(trimmed_a, trimmed_b)


class InertiaTransition:
    """
    Inertia-preserving transition that carries angular and linear momentum
    from the end of clip_a into the start of clip_b.

    Inspired by physics-based systems where environmental forces (waves, wind)
    directly drive character motion — a key trend in 2026 game animation where
    real-world physics dictate how characters and objects move.

    Parameters
    ----------
    inertia_frames : int
        Number of frames to apply inertia correction.
    """

    def __init__(self, inertia_frames: int = 10):
        self.inertia_frames = inertia_frames

    def _compute_velocity(
        self, positions: np.ndarray, frame: int, framerate: float
    ) -> np.ndarray:
        """Finite-difference velocity at a given frame [J, 3]."""
        if frame == 0:
            return (positions[1] - positions[0]) * framerate
        return (positions[frame] - positions[frame - 1]) * framerate

    def transition(self, clip_a: ClipHandle, clip_b: ClipHandle) -> ClipHandle:
        """
        Blend clips while preserving the exit velocity of clip_a.

        Returns
        -------
        ClipHandle
            clip_b with its first *inertia_frames* frames displaced to match
            clip_a's exit momentum, then smoothly corrected back.
        """
        vel_a = self._compute_velocity(
            clip_a.positions, clip_a.num_frames - 1, clip_a.framerate
        )
        vel_b = self._compute_velocity(clip_b.positions, 1, clip_b.framerate)

        # Displacement needed to align velocities
        dt = 1.0 / clip_b.framerate
        n = min(self.inertia_frames, clip_b.num_frames)
        new_positions = clip_b.positions.copy()
        for i in range(n):
            alpha = _smooth_alpha(i / max(n - 1, 1))
            vel_interp = (1 - alpha) * vel_a + alpha * vel_b
            new_positions[i] = clip_b.positions[i] + (vel_interp - vel_b) * dt * (
                n - i
            )

        return ClipHandle(
            name=f"{clip_a.name}_{clip_b.name}_inertia",
            positions=np.concatenate([clip_a.positions, new_positions], axis=0),
            rotations=np.concatenate([clip_a.rotations, clip_b.rotations], axis=0),
            framerate=clip_a.framerate,
            tags=["inertia", *clip_a.tags, *clip_b.tags],
        )


class BlendTree:
    """
    Weighted blend of multiple animation clips at a given time offset.

    Supports dynamic weight assignment enabling style mixing (e.g. combining
    a "walk" clip at 70% with a "limp" clip at 30%) for expressive character
    motion without additional training.

    Parameters
    ----------
    clips : list[ClipHandle]
        Animation clips to blend.
    weights : list[float] or None
        Initial blend weights. Defaults to uniform.
    """

    def __init__(
        self,
        clips: list[ClipHandle],
        weights: list[float] | None = None,
    ):
        self.clips = clips
        if weights is None:
            weights = [1.0 / len(clips)] * len(clips)
        self.weights = np.array(weights, dtype=np.float32)
        self.weights /= self.weights.sum()

    def set_weights(self, weights: list[float]) -> None:
        self.weights = np.array(weights, dtype=np.float32)
        self.weights /= self.weights.sum()

    def sample(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample blended pose at *time* seconds.

        Returns
        -------
        positions : np.ndarray  [J, 3]
        rotations : np.ndarray  [J, 4]  quaternion
        """
        positions = np.zeros_like(self.clips[0].positions[0])
        rotations = np.zeros_like(self.clips[0].rotations[0])
        # Use the first clip as the reference quaternion for slerp chaining
        ref_rot = None

        for clip, w in zip(self.clips, self.weights):
            frame = min(
                int(time * clip.framerate), clip.num_frames - 1
            )
            positions = positions + w * clip.positions[frame]
            if ref_rot is None:
                rotations = clip.rotations[frame]
                ref_rot = rotations
            else:
                rotations = _slerp(rotations, clip.rotations[frame], w)

        return positions, rotations
