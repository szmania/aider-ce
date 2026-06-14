"""Tests for repo-local workspaces (``path:`` git roots, ``.cecli.workspaces.yml``)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from cecli.helpers.monorepo.config import load_workspace_config_file, validate_config
from cecli.helpers.monorepo.local_workspace import (
    find_workspace_config_file,
    load_workspace_file,
    primary_project,
    project_path_prefix,
    read_workspace_metadata,
    resolve_workspace_file_path,
    union_tracked_files,
    write_workspace_metadata,
)
from cecli.io import InputOutput
from cecli.repo import GitRepo
from cecli.utils import make_repo


def _init_git_repo(path: Path, readme: str = "# repo\n") -> None:
    make_repo(path)
    readme_path = path / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init", "--no-gpg-sign"],
        cwd=path,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def two_path_projects(tmp_path: Path):
    """
    Workspace root with ``.cecli.workspaces.yml`` and two sibling git checkouts.

    Layout::

        ws/
          .cecli.workspaces.yml
          app/   (git)
          lib/   (git)
    """
    ws = tmp_path / "ws"
    app = ws / "app"
    lib = ws / "lib"
    app.mkdir(parents=True)
    lib.mkdir(parents=True)
    _init_git_repo(app, "# app\n")
    _init_git_repo(lib, "# lib\n")

    config = {
        "name": "pair",
        "projects": [
            {"name": "app", "path": str(app.resolve()), "primary": True},
            {"name": "lib", "path": str(lib.resolve())},
        ],
    }
    (ws / ".cecli.workspaces.yml").write_text(
        yaml.dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return ws, config, app, lib


class TestValidateConfigPathProjects:
    def test_path_only_project_valid(self):
        validate_config(
            {
                "name": "local",
                "projects": [{"name": "app", "path": "/tmp/app", "primary": True}],
            }
        )

    def test_repo_only_project_valid(self):
        validate_config(
            {
                "name": "clone",
                "projects": [{"name": "p1", "repo": "https://github.com/org/r.git"}],
            }
        )

    def test_missing_path_and_repo(self):
        with pytest.raises(ValueError, match="exactly one of 'path' or 'repo'"):
            validate_config({"name": "test", "projects": [{"name": "p1"}]})

    def test_both_path_and_repo(self):
        with pytest.raises(ValueError, match="exactly one of 'path' or 'repo'"):
            validate_config(
                {
                    "name": "test",
                    "projects": [
                        {
                            "name": "p1",
                            "path": "/tmp/a",
                            "repo": "https://github.com/org/r.git",
                        }
                    ],
                }
            )

    def test_multiple_primary(self):
        with pytest.raises(ValueError, match="Only one project may be marked primary"):
            validate_config(
                {
                    "name": "test",
                    "projects": [
                        {"name": "a", "path": "/a", "primary": True},
                        {"name": "b", "path": "/b", "primary": True},
                    ],
                }
            )


class TestLocalWorkspaceHelpers:
    def test_find_workspace_config_file_walks_up(self, two_path_projects):
        ws, _config, app, _lib = two_path_projects
        expected = (ws / ".cecli.workspaces.yml").resolve()
        assert find_workspace_config_file(ws).resolve() == expected
        # YAML lives at workspace root; project checkout is a subdirectory.
        assert find_workspace_config_file(app).resolve() == expected
        assert find_workspace_config_file(app / "README.md").resolve() == expected

    def test_load_workspace_config_file(self, two_path_projects):
        ws, _config, _app, _lib = two_path_projects
        loaded = load_workspace_config_file(ws / ".cecli.workspaces.yml")
        assert loaded["name"] == "pair"
        assert len(loaded["projects"]) == 2

    def test_union_tracked_files(self, two_path_projects):
        ws, config, _app, _lib = two_path_projects
        files = union_tracked_files(ws, config, layout="local")
        assert "app/README.md" in files
        assert "lib/README.md" in files

    def test_resolve_workspace_file_path_prefixed(self, two_path_projects):
        ws, config, _app, _lib = two_path_projects
        resolved = resolve_workspace_file_path(ws, "lib/README.md", config, layout="local")
        assert resolved is not None
        git_root, abs_path, in_repo = resolved
        assert in_repo == "README.md"
        assert abs_path.name == "README.md"
        assert git_root.name == "lib"

    def test_project_path_prefix_local_vs_clone(self):
        proj = {"name": "app"}
        assert project_path_prefix(proj, layout="local") == "app"
        assert project_path_prefix(proj, layout="clone") == "app/main"

    def test_primary_project_explicit_and_implicit(self):
        cfg = {
            "projects": [
                {"name": "a", "path": "/a"},
                {"name": "b", "path": "/b", "primary": True},
            ]
        }
        assert primary_project(cfg)["name"] == "b"

        single = {"projects": [{"name": "only", "path": "/only"}]}
        assert primary_project(single)["name"] == "only"

    def test_workspace_metadata_roundtrip(self, two_path_projects):
        ws, config, _app, _lib = two_path_projects
        write_workspace_metadata(ws, config, layout="local")
        meta = read_workspace_metadata(ws)
        assert meta is not None
        loaded, layout = meta
        assert layout == "local"
        assert loaded["name"] == config["name"]
        meta_path = ws / ".cecli" / ".workspace-meta.json"
        assert meta_path.is_file()
        on_disk = json.loads(meta_path.read_text(encoding="utf-8"))
        assert on_disk.get("_layout") == "local"


class TestGitRepoLocalWorkspace:
    def test_detects_local_workspace_and_unions_files(self, two_path_projects):
        ws, _config, app, lib = two_path_projects
        io = InputOutput(yes=True)
        repo = GitRepo(io, [str(app / "README.md"), str(lib / "README.md")], None)

        assert repo.is_workspace
        assert repo.workspace_layout == "local"
        assert repo.workspace_path == ws.resolve()

        files = repo.get_workspace_files()
        assert "app/README.md" in files
        assert "lib/README.md" in files

    def test_abs_root_path_resolves_prefixed_path(self, two_path_projects):
        _ws, _config, app, _lib = two_path_projects
        io = InputOutput(yes=True)
        repo = GitRepo(io, [str(app)], None)

        abs_path = Path(repo.abs_root_path("app/README.md"))
        assert abs_path == (app / "README.md").resolve()

    def test_without_workspace_file_multi_repo_fails(self, tmp_path: Path):
        root = tmp_path / "orphan"
        a = root / "a"
        b = root / "b"
        a.mkdir(parents=True)
        b.mkdir(parents=True)
        _init_git_repo(a)
        _init_git_repo(b)
        io = InputOutput(yes=True)
        with pytest.raises(FileNotFoundError):
            GitRepo(io, [str(a / "README.md"), str(b / "README.md")], None)

    def test_load_workspace_file_defaults(self, tmp_path: Path):
        path = tmp_path / ".cecli.workspaces.yml"
        path.write_text("projects: []\n", encoding="utf-8")
        loaded = load_workspace_file(path)
        assert "name" in loaded
        assert loaded["projects"] == []


class TestGitRepoLocalWorkspaceNoYaml:
    def test_single_repo_without_yaml_is_not_local_workspace(self, tmp_path: Path):
        repo_dir = tmp_path / "solo"
        repo_dir.mkdir()
        _init_git_repo(repo_dir)
        io = InputOutput(yes=True)
        repo = GitRepo(io, [str(repo_dir)], None)
        assert find_workspace_config_file(repo_dir) is None
        assert getattr(repo, "workspace_layout", "clone") != "local" or not repo.is_workspace
