# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Project Management System — 2026 Enhancement
=============================================
Provides persistent project storage with structured logging so that
animation projects can be saved, remembered, and resumed across sessions.

Storage layout
--------------
~/.ai4animation/
├── projects.db          — SQLite catalogue of all projects
└── projects/
    └── <project_id>/
        ├── meta.json    — project metadata
        ├── clips/       — saved .npz motion clips
        └── logs/        — per-session event logs (.jsonl)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_ROOT = Path.home() / ".ai4animation"


@dataclass
class ProjectMeta:
    """Metadata record for a single project."""

    id: str
    name: str
    description: str
    created_at: float
    updated_at: float
    tags: list[str] = field(default_factory=list)
    clip_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ProjectMeta":
        return ProjectMeta(**d)


class ProjectManager:
    """
    Manages the lifecycle of animation projects on disk.

    Parameters
    ----------
    root : str or Path, optional
        Root directory for project storage.
        Defaults to ``~/.ai4animation``.
    """

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else _DEFAULT_ROOT
        self._projects_dir = self.root / "projects"
        self._db_path = self.root / "projects.db"
        self._setup_storage()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_storage(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._projects_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    description TEXT,
                    created_at  REAL,
                    updated_at  REAL,
                    tags        TEXT,
                    clip_names  TEXT
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT,
                    session_id TEXT,
                    timestamp  REAL,
                    level      TEXT,
                    event      TEXT,
                    data       TEXT
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    # ------------------------------------------------------------------
    # Project CRUD
    # ------------------------------------------------------------------

    def create_project(
        self,
        name: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> ProjectMeta:
        """
        Create a new project and return its metadata.

        Parameters
        ----------
        name : str
        description : str
        tags : list[str], optional

        Returns
        -------
        ProjectMeta
        """
        now = time.time()
        meta = ProjectMeta(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            created_at=now,
            updated_at=now,
            tags=tags or [],
            clip_names=[],
        )
        self._save_meta(meta)
        project_dir = self._project_dir(meta.id)
        (project_dir / "clips").mkdir(parents=True, exist_ok=True)
        (project_dir / "logs").mkdir(parents=True, exist_ok=True)
        logger.info("Created project '%s' (id=%s)", name, meta.id)
        return meta

    def list_projects(self) -> list[ProjectMeta]:
        """Return all projects ordered by most recently updated."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT id, name, description, created_at, updated_at, tags, clip_names "
                "FROM projects ORDER BY updated_at DESC"
            ).fetchall()
        return [self._row_to_meta(r) for r in rows]

    def get_project(self, project_id: str) -> ProjectMeta | None:
        """Retrieve a project by id. Returns None if not found."""
        with self._connect() as con:
            row = con.execute(
                "SELECT id, name, description, created_at, updated_at, tags, clip_names "
                "FROM projects WHERE id = ?",
                (project_id,),
            ).fetchone()
        return self._row_to_meta(row) if row else None

    def update_project(
        self,
        project_id: str,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> ProjectMeta | None:
        """Update mutable fields of a project."""
        meta = self.get_project(project_id)
        if meta is None:
            return None
        if name is not None:
            meta.name = name
        if description is not None:
            meta.description = description
        if tags is not None:
            meta.tags = tags
        meta.updated_at = time.time()
        self._save_meta(meta)
        return meta

    def delete_project(self, project_id: str) -> bool:
        """Delete a project and all associated data."""
        with self._connect() as con:
            con.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            con.execute("DELETE FROM logs WHERE project_id = ?", (project_id,))
        project_dir = self._project_dir(project_id)
        if project_dir.exists():
            import shutil

            shutil.rmtree(project_dir)
        logger.info("Deleted project %s", project_id)
        return True

    # ------------------------------------------------------------------
    # Clip storage
    # ------------------------------------------------------------------

    def save_clip(
        self,
        project_id: str,
        clip_name: str,
        positions: np.ndarray,
        rotations: np.ndarray,
        framerate: float,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """
        Save a motion clip as a .npz file inside the project.

        Parameters
        ----------
        project_id : str
        clip_name : str
        positions : np.ndarray  [T, J, 3]
        rotations : np.ndarray  [T, J, 4]
        framerate : float
        extra : dict, optional
            Extra arrays to store in the npz.

        Returns
        -------
        Path
            Path to the saved .npz file.
        """
        meta = self.get_project(project_id)
        if meta is None:
            raise ValueError(f"Project {project_id!r} not found")
        clip_dir = self._project_dir(project_id) / "clips"
        clip_dir.mkdir(parents=True, exist_ok=True)
        path = clip_dir / f"{clip_name}.npz"
        arrays = {
            "positions": positions,
            "rotations": rotations,
            "framerate": np.array(framerate),
        }
        if extra:
            arrays.update(extra)
        np.savez_compressed(str(path), **arrays)

        if clip_name not in meta.clip_names:
            meta.clip_names.append(clip_name)
            meta.updated_at = time.time()
            self._save_meta(meta)

        logger.info("Saved clip '%s' to project %s", clip_name, project_id)
        return path

    def load_clip(
        self, project_id: str, clip_name: str
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Load a saved motion clip.

        Returns
        -------
        positions : np.ndarray  [T, J, 3]
        rotations : np.ndarray  [T, J, 4]
        framerate : float
        """
        path = self._project_dir(project_id) / "clips" / f"{clip_name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Clip '{clip_name}' not found in project {project_id}")
        data = np.load(str(path), allow_pickle=True)
        return (
            data["positions"],
            data["rotations"],
            float(data["framerate"]),
        )

    def list_clips(self, project_id: str) -> list[str]:
        """Return clip names for a project."""
        meta = self.get_project(project_id)
        return meta.clip_names if meta else []

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_event(
        self,
        project_id: str,
        session_id: str,
        event: str,
        level: str = "INFO",
        data: dict | None = None,
    ) -> None:
        """
        Record a structured log event for a project.

        Parameters
        ----------
        project_id : str
        session_id : str
        event : str
        level : str
        data : dict, optional
        """
        now = time.time()
        data_str = json.dumps(data or {})
        with self._connect() as con:
            con.execute(
                "INSERT INTO logs (project_id, session_id, timestamp, level, event, data) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (project_id, session_id, now, level, event, data_str),
            )
        # Also append to per-session .jsonl file
        log_dir = self._project_dir(project_id) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{session_id}.jsonl"
        record = {
            "timestamp": now,
            "level": level,
            "event": event,
            "data": data or {},
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def get_logs(
        self,
        project_id: str,
        session_id: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        """
        Retrieve log entries for a project.

        Parameters
        ----------
        project_id : str
        session_id : str, optional
            Filter to a specific session.
        limit : int
            Maximum number of records to return.

        Returns
        -------
        list[dict]
        """
        with self._connect() as con:
            if session_id:
                rows = con.execute(
                    "SELECT timestamp, level, event, data FROM logs "
                    "WHERE project_id = ? AND session_id = ? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (project_id, session_id, limit),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT timestamp, level, event, data FROM logs "
                    "WHERE project_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (project_id, limit),
                ).fetchall()
        return [
            {
                "timestamp": r[0],
                "level": r[1],
                "event": r[2],
                "data": json.loads(r[3]),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _project_dir(self, project_id: str) -> Path:
        return self._projects_dir / project_id

    def _save_meta(self, meta: ProjectMeta) -> None:
        # Write to SQLite
        with self._connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO projects
                    (id, name, description, created_at, updated_at, tags, clip_names)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    meta.id,
                    meta.name,
                    meta.description,
                    meta.created_at,
                    meta.updated_at,
                    json.dumps(meta.tags),
                    json.dumps(meta.clip_names),
                ),
            )
        # Write to meta.json for human readability
        meta_path = self._project_dir(meta.id) / "meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta.to_dict(), f, indent=2)

    @staticmethod
    def _row_to_meta(row: tuple) -> ProjectMeta:
        return ProjectMeta(
            id=row[0],
            name=row[1],
            description=row[2] or "",
            created_at=row[3],
            updated_at=row[4],
            tags=json.loads(row[5] or "[]"),
            clip_names=json.loads(row[6] or "[]"),
        )
