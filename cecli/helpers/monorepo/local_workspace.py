"""
Repo-local multi-project workspaces (``path:`` git roots).

Cecli already supports **clone** workspaces under ``~/.cecli/workspaces/`` with
``repo:`` URLs and paths like ``{project}/main/{file}``. This module adds
**local** layout: projects point at existing directories on disk, and tracked
paths are prefixed as ``{project}/{file}`` (no ``/main/`` segment).

Config file names (at the workspace root — usually the primary project directory):

- ``.cecli.workspaces.yml``
- ``.cecli.workspaces.yaml``

Each project must have exactly one of ``path`` (absolute local git root) or
``repo`` (clone URL; handled by existing clone workspace code).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

WORKSPACE_FILENAMES = (".cecli.workspaces.yml", ".cecli.workspaces.yaml")
METADATA_NAME = ".cecli/.workspace-meta.json"


def find_workspace_config_file(start: Path) -> Path | None:
    """Return the nearest ``.cecli.workspaces.yml`` walking up from *start*."""
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    while True:
        for name in WORKSPACE_FILENAMES:
            candidate = current / name
            if candidate.is_file():
                return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def load_workspace_file(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Workspace file must be a mapping")
    if "name" not in raw:
        raw["name"] = path.parent.name or "workspace"
    if "projects" not in raw:
        raw["projects"] = []
    return raw


def primary_project(config: dict[str, Any]) -> dict[str, Any] | None:
    projects = config.get("projects") or []
    for proj in projects:
        if proj.get("primary"):
            return proj
    if len(projects) == 1:
        return projects[0]
    return projects[0] if projects else None


def project_git_root(workspace_root: Path, project: dict[str, Any], *, layout: str) -> Path | None:
    name = project.get("name")
    if not name:
        return None
    path_val = project.get("path")
    if path_val:
        root = Path(str(path_val)).expanduser().resolve()
        if not root.is_dir():
            return None
        try:
            subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                stderr=subprocess.DEVNULL,
            )
            return root
        except Exception:
            return None
    if layout != "clone":
        return None
    clone_root = workspace_root / name / "main"
    return clone_root if clone_root.is_dir() else None


def project_path_prefix(project: dict[str, Any], *, layout: str) -> str:
    name = str(project.get("name") or "")
    if layout == "clone":
        return f"{name}/main"
    return name


def resolve_workspace_file_path(
    workspace_root: Path,
    workspace_rel: str,
    config: dict[str, Any],
    *,
    layout: str,
) -> tuple[Path, Path, str] | None:
    """
    Map a workspace-relative path to ``(project_git_root, absolute_file, path_in_project_repo)``.
    """
    rel = workspace_rel.replace("\\", "/").lstrip("/")
    if not rel:
        return None
    parts = Path(rel).parts
    if not parts:
        return None
    projects = config.get("projects") or []
    by_name = {str(p.get("name")): p for p in projects if p.get("name")}

    # Clone layout: name/main/rest
    if layout == "clone" and len(parts) >= 2 and parts[1] == "main":
        proj = by_name.get(parts[0])
        if not proj:
            return None
        git_root = project_git_root(workspace_root, proj, layout=layout)
        if not git_root:
            return None
        in_repo = "/".join(parts[2:]) if len(parts) > 2 else ""
        abs_path = git_root / in_repo if in_repo else git_root
        return git_root, abs_path, in_repo

    # Local layout: name/rest or bare path under primary-only tree
    if parts[0] in by_name:
        proj = by_name[parts[0]]
        git_root = project_git_root(workspace_root, proj, layout=layout)
        if not git_root:
            return None
        in_repo = "/".join(parts[1:]) if len(parts) > 1 else ""
        abs_path = git_root / in_repo if in_repo else git_root
        return git_root, abs_path, in_repo

    primary = primary_project(config)
    if primary:
        git_root = project_git_root(workspace_root, primary, layout=layout)
        if git_root:
            in_repo = rel
            return git_root, git_root / in_repo, in_repo
    return None


def union_tracked_files(
    workspace_root: Path,
    config: dict[str, Any],
    *,
    layout: str,
    ignored_file=None,
) -> list[str]:
    """All tracked files as workspace-relative paths."""
    out: list[str] = []
    for proj in config.get("projects") or []:
        name = proj.get("name")
        if not name:
            continue
        git_root = project_git_root(workspace_root, proj, layout=layout)
        if not git_root:
            continue
        prefix = project_path_prefix(proj, layout=layout)
        try:
            lines = subprocess.check_output(
                ["git", "-C", str(git_root), "ls-files"],
                stderr=subprocess.DEVNULL,
                encoding="utf-8",
            ).splitlines()
        except Exception:
            continue
        for line in lines:
            if not line.strip():
                continue
            rel = f"{prefix}/{line}" if prefix else line
            rel = rel.replace("\\", "/")
            if ignored_file and ignored_file(rel):
                continue
            out.append(rel)
    return out


def project_head_shas(
    workspace_root: Path,
    config: dict[str, Any],
    *,
    layout: str,
) -> list[str]:
    shas: list[str] = []
    for proj in config.get("projects") or []:
        name = proj.get("name")
        if not name:
            continue
        git_root = project_git_root(workspace_root, proj, layout=layout)
        if not git_root:
            shas.append(f"{name}:unknown")
            continue
        try:
            sha = subprocess.check_output(
                ["git", "-C", str(git_root), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                encoding="utf-8",
            ).strip()
            shas.append(f"{name}:{sha}")
        except Exception:
            shas.append(f"{name}:unknown")
    return shas


def write_workspace_metadata(workspace_root: Path, config: dict[str, Any], *, layout: str) -> None:
    meta_dir = workspace_root / ".cecli"
    meta_dir.mkdir(parents=True, exist_ok=True)
    payload = {**config, "_layout": layout}
    (meta_dir / ".workspace-meta.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def read_workspace_metadata(workspace_root: Path) -> tuple[dict[str, Any], str] | None:
    legacy = workspace_root / ".cecli-workspace.json"
    modern = workspace_root / METADATA_NAME
    path = modern if modern.is_file() else legacy if legacy.is_file() else None
    if not path:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        layout = data.pop("_layout", "clone")
        return data, layout
    except Exception:
        return None
