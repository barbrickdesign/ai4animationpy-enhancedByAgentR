# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
AI4Animation Web Application — 2026 Enhancement
================================================
A browser-based interface for the AI4AnimationPy framework providing:
 - Motion capture file upload and 3D skeleton preview
 - AI motion inbetweening (fill missing frames between keyframes)
 - Seamless animation transitions between clips
 - Project management (create / save / load / log)
 - Logging dashboard

Run with:
    python webapp/app.py
or via Docker:
    docker compose up
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or from webapp/
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import new modules directly (bypass ai4animation/__init__.py which pulls
# in raylib, pygltflib, etc. that are not needed in web mode)
import importlib.util as _ilu


def _load_module(rel_path: str):
    path = _REPO_ROOT / rel_path
    mod_name = f"_ai4anim_web.{path.stem}"
    spec = _ilu.spec_from_file_location(mod_name, path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_inbetween_mod = _load_module("ai4animation/AI/Inbetweening.py")
_transitions_mod = _load_module("ai4animation/Animation/SeamlessTransitions.py")
_project_mod = _load_module("ai4animation/Projects/ProjectManager.py")

MotionInbetweener = _inbetween_mod.MotionInbetweener
ClipHandle = _transitions_mod.ClipHandle
InertiaTransition = _transitions_mod.InertiaTransition
PhaseMatchedTransition = _transitions_mod.PhaseMatchedTransition
TransitionBlender = _transitions_mod.TransitionBlender
ProjectManager = _project_mod.ProjectManager

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
log = logging.getLogger("ai4animation.webapp")

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
_pm = ProjectManager()
_inbetweener = MotionInbetweener(joint_dim=3, num_joints=22)
_current_session_id = str(uuid.uuid4())[:8]
_loaded_clips: dict[str, ClipHandle] = {}  # clip_name -> ClipHandle


# ---------------------------------------------------------------------------
# BVH loader (minimal, self-contained — avoids raylib dependency in web mode)
# ---------------------------------------------------------------------------

def _load_bvh_minimal(path: str) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Minimal BVH parser that returns positions and quaternions.
    Returns (positions [T, J, 3], rotations [T, J, 4], framerate).
    """
    try:
        # Use ai4animation's BVH importer if available
        from ai4animation.Import.BVHImporter import BVH

        motion = BVH.Load(path)
        positions = np.array(
            [
                [motion.GetLocalPosition(f, j) for j in range(motion.NumJoints)]
                for f in range(motion.NumFrames)
            ],
            dtype=np.float32,
        )
        rotations = np.zeros((motion.NumFrames, motion.NumJoints, 4), dtype=np.float32)
        rotations[..., 0] = 1.0  # identity quaternion w=1
        return positions, rotations, float(motion.Framerate)
    except Exception:
        pass

    # Pure-Python fallback parser (handles basic BVH hierarchy + motion data)
    joint_names: list[str] = []
    offsets: list[list[float]] = []
    framerate = 30.0
    motion_lines: list[str] = []
    in_motion = False

    with open(path, "r") as f:
        lines = f.readlines()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("ROOT") or stripped.startswith("JOINT"):
            joint_names.append(stripped.split()[-1])
        elif stripped.startswith("OFFSET"):
            parts = stripped.split()
            offsets.append([float(p) for p in parts[1:4]])
        elif stripped.startswith("Frame Time:"):
            ft = float(stripped.split(":")[-1].strip())
            framerate = round(1.0 / ft) if ft > 0 else 30.0
        elif in_motion and stripped:
            motion_lines.append(stripped)
        elif stripped == "MOTION":
            in_motion = True

    num_joints = max(len(joint_names), 1)
    offsets_arr = np.array(offsets[:num_joints], dtype=np.float32)
    if offsets_arr.shape[0] < num_joints:
        pad = np.zeros((num_joints - offsets_arr.shape[0], 3), dtype=np.float32)
        offsets_arr = np.concatenate([offsets_arr, pad], axis=0)

    # Parse motion data rows
    all_values: list[list[float]] = []
    for mline in motion_lines:
        try:
            vals = [float(v) for v in mline.split()]
            if vals:
                all_values.append(vals)
        except ValueError:
            continue

    if not all_values:
        # Return a synthetic idle clip
        T = 60
        positions = np.tile(offsets_arr[np.newaxis], (T, 1, 1))
        rotations = np.zeros((T, num_joints, 4), dtype=np.float32)
        rotations[..., 0] = 1.0
        return positions, rotations, framerate

    num_frames = len(all_values)
    cols_per_frame = len(all_values[0])
    motion_arr = np.array(all_values, dtype=np.float32)

    # Place root translation in joint 0, all others use offsets
    positions = np.tile(offsets_arr[np.newaxis], (num_frames, 1, 1))
    if cols_per_frame >= 3:
        positions[:, 0, :] = motion_arr[:, :3] * 0.01  # cm → m scale

    rotations = np.zeros((num_frames, num_joints, 4), dtype=np.float32)
    rotations[..., 0] = 1.0
    return positions, rotations, framerate


def _load_npz(path: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Load a .npz motion file saved by the framework."""
    data = np.load(path, allow_pickle=True)
    positions = data["positions"] if "positions" in data else np.zeros((1, 22, 3))
    rotations = data["rotations"] if "rotations" in data else np.zeros((1, 22, 4))
    if rotations.shape[-1] == 4 and np.allclose(rotations, 0):
        rotations[..., 0] = 1.0
    framerate = float(data["framerate"]) if "framerate" in data else 30.0
    return positions.astype(np.float32), rotations.astype(np.float32), framerate


# ---------------------------------------------------------------------------
# 3-D skeleton viewer
# ---------------------------------------------------------------------------

# Simple skeleton topology (indices into joint array).
# Covers common 22-joint humanoid rigs; any extra joints are ignored.
_SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # spine
    (2, 5), (5, 6), (6, 7), (7, 8),        # left arm
    (2, 9), (9, 10), (10, 11), (11, 12),   # right arm
    (0, 13), (13, 14), (14, 15), (15, 16), # left leg
    (0, 17), (17, 18), (18, 19), (19, 20), # right leg
    (3, 21),                                # head
]


def _build_skeleton_figure(
    positions: np.ndarray,
    frame: int,
    title: str = "Skeleton",
) -> go.Figure:
    """
    Build a Plotly 3-D figure for a single skeleton frame.

    Parameters
    ----------
    positions : np.ndarray  [T, J, 3]  or  [J, 3]
    frame : int
    title : str
    """
    if positions.ndim == 3:
        pos = positions[min(frame, len(positions) - 1)]  # [J, 3]
    else:
        pos = positions  # already [J, 3]

    J = pos.shape[0]

    # Draw joints
    joints_trace = go.Scatter3d(
        x=pos[:, 0],
        y=pos[:, 2],  # swap Y/Z for natural up-axis view
        z=pos[:, 1],
        mode="markers",
        marker=dict(size=5, color="#00CFFF", opacity=0.9),
        name="Joints",
    )

    # Draw bones as line segments
    bone_x, bone_y, bone_z = [], [], []
    for a, b in _SKELETON_EDGES:
        if a < J and b < J:
            bone_x += [pos[a, 0], pos[b, 0], None]
            bone_y += [pos[a, 2], pos[b, 2], None]
            bone_z += [pos[a, 1], pos[b, 1], None]

    bones_trace = go.Scatter3d(
        x=bone_x,
        y=bone_y,
        z=bone_z,
        mode="lines",
        line=dict(color="#FF6B35", width=4),
        name="Bones",
    )

    fig = go.Figure(data=[joints_trace, bones_trace])
    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Z",
            zaxis_title="Y",
            bgcolor="#1A1A2E",
            xaxis=dict(gridcolor="#2D2D44", zerolinecolor="#2D2D44"),
            yaxis=dict(gridcolor="#2D2D44", zerolinecolor="#2D2D44"),
            zaxis=dict(gridcolor="#2D2D44", zerolinecolor="#2D2D44"),
            aspectmode="data",
        ),
        paper_bgcolor="#16213E",
        font=dict(color="#E0E0E0"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=480,
    )
    return fig


# ---------------------------------------------------------------------------
# Helper: format project list as a Markdown table
# ---------------------------------------------------------------------------

def _projects_table() -> str:
    projects = _pm.list_projects()
    if not projects:
        return "_No projects yet. Create one above._"
    rows = ["| Name | Description | Tags | Clips | Last Updated |", "|---|---|---|---|---|"]
    for p in projects:
        updated = time.strftime("%Y-%m-%d %H:%M", time.localtime(p.updated_at))
        rows.append(
            f"| **{p.name}** | {p.description or '—'} | {', '.join(p.tags) or '—'} "
            f"| {len(p.clip_names)} | {updated} |"
        )
    return "\n".join(rows)


def _project_choices() -> list[str]:
    return [f"{p.id[:8]}  {p.name}" for p in _pm.list_projects()]


def _project_id_from_choice(choice: str) -> str | None:
    if not choice:
        return None
    short_id = choice.split()[0]
    for p in _pm.list_projects():
        if p.id.startswith(short_id):
            return p.id
    return None


# ---------------------------------------------------------------------------
# Tab 1 — Motion Viewer
# ---------------------------------------------------------------------------

def upload_motion_file(file_obj, clip_name: str, project_choice: str):
    """Load a BVH or NPZ file, visualise frame 0, and optionally save to project."""
    if file_obj is None:
        return None, "⚠️ No file uploaded.", gr.update()

    clip_name = clip_name.strip() or Path(file_obj.name).stem
    suffix = Path(file_obj.name).suffix.lower()

    try:
        if suffix == ".bvh":
            positions, rotations, framerate = _load_bvh_minimal(file_obj.name)
        elif suffix == ".npz":
            positions, rotations, framerate = _load_npz(file_obj.name)
        else:
            return None, f"⚠️ Unsupported format: {suffix}. Use .bvh or .npz.", gr.update()
    except Exception as exc:
        return None, f"❌ Failed to load file: {exc}", gr.update()

    _loaded_clips[clip_name] = ClipHandle(
        name=clip_name,
        positions=positions,
        rotations=rotations,
        framerate=framerate,
    )

    fig = _build_skeleton_figure(positions, frame=0, title=f"{clip_name} — Frame 0")
    info = (
        f"✅ Loaded **{clip_name}** — {positions.shape[0]} frames, "
        f"{positions.shape[1]} joints @ {framerate:.1f} fps"
    )

    project_id = _project_id_from_choice(project_choice)
    if project_id:
        _pm.save_clip(project_id, clip_name, positions, rotations, framerate)
        _pm.log_event(
            project_id,
            _current_session_id,
            "clip_uploaded",
            data={"clip": clip_name, "frames": int(positions.shape[0])},
        )
        info += f"\n💾 Saved to project."

    return fig, info, gr.update(maximum=max(0, positions.shape[0] - 1))


def scrub_frame(clip_name_dropdown: str, frame: int):
    clip = _loaded_clips.get(clip_name_dropdown)
    if clip is None:
        return None
    return _build_skeleton_figure(
        clip.positions, frame, title=f"{clip_name_dropdown} — Frame {frame}"
    )


def get_loaded_clip_names():
    return list(_loaded_clips.keys())


# ---------------------------------------------------------------------------
# Tab 2 — AI Inbetweening
# ---------------------------------------------------------------------------

def run_inbetweening(
    clip_name: str,
    start_frame: int,
    end_frame: int,
    num_inbetween: int,
    project_choice: str,
):
    clip = _loaded_clips.get(clip_name)
    if clip is None:
        return None, "⚠️ No clip loaded with that name."

    T = clip.positions.shape[0]
    start_frame = int(np.clip(start_frame, 0, T - 1))
    end_frame = int(np.clip(end_frame, 0, T - 1))
    if start_frame >= end_frame:
        return None, "⚠️ Start frame must be less than end frame."

    num_inbetween = max(1, int(num_inbetween))

    # Flatten positions to [J*3] for inbetweening
    s = clip.positions[start_frame].reshape(-1)
    e = clip.positions[end_frame].reshape(-1)
    result = _inbetweener.inbetween(s, e, num_inbetween)  # [num_inbetween+2, J*3]

    J = clip.positions.shape[1]
    inbetween_positions = result.reshape(-1, J, 3)  # [T_new, J, 3]

    # Create new clip from inbetween result
    new_name = f"{clip_name}_inbetween_{start_frame}_{end_frame}"
    new_rotations = np.zeros((inbetween_positions.shape[0], J, 4), dtype=np.float32)
    new_rotations[..., 0] = 1.0
    _loaded_clips[new_name] = ClipHandle(
        name=new_name,
        positions=inbetween_positions,
        rotations=new_rotations,
        framerate=clip.framerate,
        tags=["inbetween"],
    )

    # Visualise middle inbetween frame
    mid = len(inbetween_positions) // 2
    fig = _build_skeleton_figure(
        inbetween_positions, mid, title=f"Inbetween — Mid frame {mid}"
    )
    msg = (
        f"✅ Generated {num_inbetween} inbetween frames → clip **{new_name}** "
        f"({len(inbetween_positions)} total frames)"
    )

    project_id = _project_id_from_choice(project_choice)
    if project_id:
        _pm.save_clip(
            project_id, new_name, inbetween_positions, new_rotations, clip.framerate
        )
        _pm.log_event(
            project_id,
            _current_session_id,
            "inbetweening_applied",
            data={
                "source_clip": clip_name,
                "result_clip": new_name,
                "num_inbetween": num_inbetween,
                "start_frame": start_frame,
                "end_frame": end_frame,
            },
        )

    return fig, msg


# ---------------------------------------------------------------------------
# Tab 3 — Seamless Transitions
# ---------------------------------------------------------------------------

def run_transition(
    clip_a_name: str,
    clip_b_name: str,
    method: str,
    blend_frames: int,
    project_choice: str,
):
    clip_a = _loaded_clips.get(clip_a_name)
    clip_b = _loaded_clips.get(clip_b_name)

    if clip_a is None or clip_b is None:
        return None, "⚠️ Both clips must be loaded."
    if clip_a_name == clip_b_name:
        return None, "⚠️ Select two different clips."

    blend_frames = max(2, int(blend_frames))

    try:
        if method == "Cross-fade":
            blender = TransitionBlender(blend_frames, curve="cosine")
            result_clip = blender.blend(clip_a, clip_b)
        elif method == "Phase-matched":
            transition = PhaseMatchedTransition(blend_frames=blend_frames)
            result_clip = transition.transition(clip_a, clip_b)
        elif method == "Inertia":
            transition = InertiaTransition(inertia_frames=blend_frames)
            result_clip = transition.transition(clip_a, clip_b)
        else:
            return None, f"⚠️ Unknown method: {method}"
    except Exception as exc:
        return None, f"❌ Transition failed: {exc}"

    _loaded_clips[result_clip.name] = result_clip
    mid = result_clip.num_frames // 2
    fig = _build_skeleton_figure(
        result_clip.positions, mid, title=f"Transition — {result_clip.name}"
    )
    msg = (
        f"✅ Created transition clip **{result_clip.name}** "
        f"({result_clip.num_frames} frames, method: {method})"
    )

    project_id = _project_id_from_choice(project_choice)
    if project_id:
        _pm.save_clip(
            project_id,
            result_clip.name,
            result_clip.positions,
            result_clip.rotations,
            result_clip.framerate,
        )
        _pm.log_event(
            project_id,
            _current_session_id,
            "transition_applied",
            data={
                "clip_a": clip_a_name,
                "clip_b": clip_b_name,
                "method": method,
                "result": result_clip.name,
            },
        )

    return fig, msg


# ---------------------------------------------------------------------------
# Tab 4 — Project Management
# ---------------------------------------------------------------------------

def create_project(name: str, description: str, tags: str):
    if not name.strip():
        return "⚠️ Project name cannot be empty.", _projects_table(), gr.update(choices=_project_choices())
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    meta = _pm.create_project(name.strip(), description.strip(), tag_list)
    _pm.log_event(meta.id, _current_session_id, "project_created", data={"name": name})
    msg = f"✅ Created project **{name}** (id: `{meta.id[:8]}…`)"
    return msg, _projects_table(), gr.update(choices=_project_choices())


def refresh_projects():
    return _projects_table(), gr.update(choices=_project_choices())


def load_project_clips(project_choice: str):
    project_id = _project_id_from_choice(project_choice)
    if not project_id:
        return "⚠️ Select a project first."
    clip_names = _pm.list_clips(project_id)
    if not clip_names:
        return "No clips saved in this project yet."
    loaded = []
    for cname in clip_names:
        try:
            positions, rotations, framerate = _pm.load_clip(project_id, cname)
            _loaded_clips[cname] = ClipHandle(
                name=cname,
                positions=positions,
                rotations=rotations,
                framerate=framerate,
            )
            loaded.append(cname)
        except Exception as exc:
            log.warning("Failed to load clip %s: %s", cname, exc)
    return f"✅ Loaded clips: {', '.join(loaded) or 'none'}"


# ---------------------------------------------------------------------------
# Tab 5 — Logs
# ---------------------------------------------------------------------------

def get_logs(project_choice: str, limit: int = 50):
    project_id = _project_id_from_choice(project_choice)
    if not project_id:
        return "⚠️ Select a project first."
    entries = _pm.get_logs(project_id, limit=int(limit))
    if not entries:
        return "_No log entries yet._"
    lines = [
        "| Time | Level | Event | Data |",
        "|---|---|---|---|",
    ]
    for e in entries:
        ts = time.strftime("%H:%M:%S", time.localtime(e["timestamp"]))
        data_str = json.dumps(e["data"])[:80] + ("…" if len(json.dumps(e["data"])) > 80 else "")
        lines.append(f"| {ts} | {e['level']} | {e['event']} | `{data_str}` |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build Gradio interface
# ---------------------------------------------------------------------------

THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate",
)

CSS = """
body { background: #0f0f1a; }
.gradio-container { max-width: 1280px; margin: auto; }
h1, h2, h3 { color: #00CFFF; }
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(theme=THEME, css=CSS, title="AI4Animation Studio") as app:
        gr.Markdown(
            """
# 🎬 AI4Animation Studio — 2026 Edition
**AI-driven character animation in your browser.**
AI motion inbetweening · Seamless transitions · Project memory · Logging
""".strip()
        )

        # Shared project selector (used across tabs)
        with gr.Row():
            project_selector = gr.Dropdown(
                label="Active Project",
                choices=_project_choices(),
                value=None,
                scale=3,
            )
            refresh_btn = gr.Button("🔄 Refresh", scale=1)

        # ----------------------------------------------------------------
        # Tab 1 — Motion Viewer
        # ----------------------------------------------------------------
        with gr.Tab("📂 Motion Viewer"):
            gr.Markdown("Upload a `.bvh` or `.npz` motion capture file to preview it in 3D.")
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(label="Upload Motion File (.bvh / .npz)")
                    clip_name_input = gr.Textbox(
                        label="Clip Name (optional)",
                        placeholder="Auto-detected from filename",
                    )
                    upload_btn = gr.Button("Load File", variant="primary")
                    upload_status = gr.Markdown()

                    gr.Markdown("---")
                    clip_selector = gr.Dropdown(
                        label="Select Loaded Clip",
                        choices=[],
                        value=None,
                    )
                    frame_slider = gr.Slider(
                        label="Frame",
                        minimum=0,
                        maximum=0,
                        step=1,
                        value=0,
                    )
                    scrub_btn = gr.Button("Preview Frame")

                with gr.Column(scale=2):
                    viewer_plot = gr.Plot(label="3D Skeleton Viewer")

            upload_btn.click(
                upload_motion_file,
                inputs=[file_input, clip_name_input, project_selector],
                outputs=[viewer_plot, upload_status, frame_slider],
            ).then(
                lambda: gr.update(choices=get_loaded_clip_names()),
                outputs=[clip_selector],
            )

            scrub_btn.click(
                scrub_frame,
                inputs=[clip_selector, frame_slider],
                outputs=[viewer_plot],
            )

        # ----------------------------------------------------------------
        # Tab 2 — AI Inbetweening
        # ----------------------------------------------------------------
        with gr.Tab("🤖 AI Inbetweening"):
            gr.Markdown(
                """
### AI Motion Inbetweening
Automatically generate smooth, non-repetitive in-between frames between any
two keyframes using a transformer neural network.
Inspired by GDC 2026's AI tools for automated animation.
""".strip()
            )
            with gr.Row():
                with gr.Column(scale=1):
                    ib_clip = gr.Dropdown(
                        label="Source Clip", choices=[], value=None
                    )
                    ib_refresh = gr.Button("🔄 Refresh Clips")
                    ib_start = gr.Number(label="Start Frame (keyframe A)", value=0)
                    ib_end = gr.Number(label="End Frame (keyframe B)", value=30)
                    ib_num = gr.Slider(
                        label="Number of In-between Frames",
                        minimum=1,
                        maximum=120,
                        step=1,
                        value=15,
                    )
                    ib_btn = gr.Button("Generate Inbetweens", variant="primary")
                    ib_status = gr.Markdown()

                with gr.Column(scale=2):
                    ib_plot = gr.Plot(label="Inbetween Result (mid frame preview)")

            ib_refresh.click(
                lambda: gr.update(choices=get_loaded_clip_names()), outputs=[ib_clip]
            )
            ib_btn.click(
                run_inbetweening,
                inputs=[ib_clip, ib_start, ib_end, ib_num, project_selector],
                outputs=[ib_plot, ib_status],
            )

        # ----------------------------------------------------------------
        # Tab 3 — Seamless Transitions
        # ----------------------------------------------------------------
        with gr.Tab("🔀 Seamless Transitions"):
            gr.Markdown(
                """
### Seamless Animation Transitions
Blend two animation clips with physics-aware, artifact-free transitions.
Supports cross-fade, gait-phase alignment, and inertia continuation.
""".strip()
            )
            with gr.Row():
                with gr.Column(scale=1):
                    tr_clip_a = gr.Dropdown(label="Clip A (from)", choices=[], value=None)
                    tr_clip_b = gr.Dropdown(label="Clip B (to)", choices=[], value=None)
                    tr_refresh = gr.Button("🔄 Refresh Clips")
                    tr_method = gr.Radio(
                        label="Transition Method",
                        choices=["Cross-fade", "Phase-matched", "Inertia"],
                        value="Cross-fade",
                    )
                    tr_blend = gr.Slider(
                        label="Blend Window (frames)",
                        minimum=2,
                        maximum=60,
                        step=1,
                        value=15,
                    )
                    tr_btn = gr.Button("Apply Transition", variant="primary")
                    tr_status = gr.Markdown()

                with gr.Column(scale=2):
                    tr_plot = gr.Plot(label="Transition Result (mid frame preview)")

            tr_refresh.click(
                lambda: (
                    gr.update(choices=get_loaded_clip_names()),
                    gr.update(choices=get_loaded_clip_names()),
                ),
                outputs=[tr_clip_a, tr_clip_b],
            )
            tr_btn.click(
                run_transition,
                inputs=[tr_clip_a, tr_clip_b, tr_method, tr_blend, project_selector],
                outputs=[tr_plot, tr_status],
            )

        # ----------------------------------------------------------------
        # Tab 4 — Project Management
        # ----------------------------------------------------------------
        with gr.Tab("📁 Projects"):
            gr.Markdown(
                """
### Project Management
Create and manage animation projects. All clips and log events are
automatically saved and can be reloaded in future sessions.
""".strip()
            )
            with gr.Row():
                with gr.Column():
                    proj_name = gr.Textbox(label="Project Name", placeholder="My Animation Project")
                    proj_desc = gr.Textbox(label="Description", placeholder="Optional description")
                    proj_tags = gr.Textbox(
                        label="Tags (comma-separated)", placeholder="locomotion, biped"
                    )
                    create_btn = gr.Button("Create Project", variant="primary")
                    create_status = gr.Markdown()

                with gr.Column():
                    load_clips_btn = gr.Button("📥 Load Clips from Active Project")
                    load_clips_status = gr.Markdown()

            projects_md = gr.Markdown(_projects_table())

            create_btn.click(
                create_project,
                inputs=[proj_name, proj_desc, proj_tags],
                outputs=[create_status, projects_md, project_selector],
            )
            load_clips_btn.click(
                load_project_clips,
                inputs=[project_selector],
                outputs=[load_clips_status],
            )

        # ----------------------------------------------------------------
        # Tab 5 — Logs
        # ----------------------------------------------------------------
        with gr.Tab("📋 Logs"):
            gr.Markdown(
                """
### Session Logs
Structured event log for the active project.
Tracks every operation (clip uploads, inbetweening, transitions, etc.).
""".strip()
            )
            log_limit = gr.Slider(
                label="Max entries to show", minimum=10, maximum=500, step=10, value=50
            )
            refresh_logs_btn = gr.Button("🔄 Refresh Logs")
            logs_md = gr.Markdown("_Select a project and click Refresh Logs._")

            refresh_logs_btn.click(
                get_logs,
                inputs=[project_selector, log_limit],
                outputs=[logs_md],
            )

        # ----------------------------------------------------------------
        # Global refresh
        # ----------------------------------------------------------------
        refresh_btn.click(
            refresh_projects,
            outputs=[projects_md, project_selector],
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    share = os.environ.get("GRADIO_SHARE", "0") == "1"
    log.info("Starting AI4Animation Studio on port %d …", port)
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        show_api=False,
    )
