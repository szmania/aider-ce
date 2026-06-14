"""
Construction strategies for FileSystemService.

- GitBuilder: uses git ls-files (respects .gitignore)
- ScandirBuilder: uses os.walk with ignore filtering
"""

import hashlib
import os
import subprocess
from pathlib import Path

from .ignore import FileIgnoreFilter


class GitBuilder:
    """Build file list from git repository."""

    @staticmethod
    def collect(root: str) -> list[str]:
        """
        Collect all tracked files relative to root.

        Uses ``git ls-files -z`` for null-delimited, encoding-safe output.
        Respects .gitignore naturally (git's built-in behavior).

        Args:
            root: Root directory of the git repository

        Returns:
            Sorted list of root-relative paths
        """
        try:
            result = subprocess.check_output(
                ["git", "ls-files", "-z"],
                cwd=root,
                text=True,
            )
            paths = [p for p in result.split("\0") if p]
            return sorted(paths)
        except subprocess.CalledProcessError:
            return []
        except FileNotFoundError:
            return []
        return sorted(paths)

    @staticmethod
    def staged_only(root: str) -> list[str]:
        """
        Get only staged files (currently staged but not committed).

        Args:
            root: Root directory of the git repository

        Returns:
            List of staged file paths relative to root
        """
        result = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "-z"],
            cwd=root,
            text=True,
        )
        return [p for p in result.split("\0") if p]

    @staticmethod
    def get_head_sha(root: str) -> str:
        """
        Get current HEAD SHA for cache invalidation.

        Args:
            root: Root directory of the git repository

        Returns:
            HEAD commit SHA string, or empty string on failure
        """
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except subprocess.CalledProcessError:
            return ""

    @staticmethod
    def get_cache_key(root: str) -> str:
        """
        Generate a cache key combining HEAD SHA and staged file paths.

        Combines the current HEAD commit hash with the list of staged
        (but not yet committed) files so that new files appear in the
        index even before they are committed.

        Args:
            root: Root directory of the git repository

        Returns:
            SHA-256 hex digest as cache key
        """
        sha = GitBuilder.get_head_sha(root)
        staged = GitBuilder.staged_only(root)
        combined = sha + "|" + "".join(sorted(staged))
        return hashlib.sha256(combined.encode()).hexdigest()


class ScandirBuilder:
    """Build file list by walking filesystem."""

    @staticmethod
    def collect(
        root: str,
        ignore_filter: FileIgnoreFilter | None = None,
    ) -> list[str]:
        """
        Walk filesystem collecting files relative to root.

        Prunes ignored directories during walk (never descends into them),
        avoiding wasted I/O on subtrees like ``node_modules/``.

        Args:
            root: Root directory to scan
            ignore_filter: Optional ignore rules for filtering

        Returns:
            Sorted list of root-relative paths (POSIX-style)
        """
        paths = []
        root_path = Path(root).resolve()

        for dirpath, dirnames, filenames in os.walk(root_path):
            rel_dir = os.path.relpath(dirpath, root)

            # Prune ignored directories in-place (os.walk obeys this)
            if ignore_filter:
                dirnames[:] = [
                    d
                    for d in dirnames
                    if not ignore_filter.is_dir_ignored(os.path.join(rel_dir, d))
                ]

            for filename in filenames:
                rel_path = os.path.join(rel_dir, filename) if rel_dir != "." else filename
                if ignore_filter and ignore_filter.is_file_ignored(rel_path):
                    continue
                # Normalize to POSIX-style paths
                paths.append(rel_path.replace(os.sep, "/"))

        return sorted(paths)

    @staticmethod
    def get_cache_key(root: str) -> str:
        """
        Generate a cache key from directory mtime.

        Hashes the mtimes of all immediate children under root to detect
        filesystem changes. A more robust approach would hash mtimes of
        all nested entries, but this provides a good trade-off for speed.

        Args:
            root: Root directory to scan

        Returns:
            SHA-256 hex digest as cache key
        """
        mtimes = []
        for entry in os.scandir(root):
            try:
                mtimes.append(str(entry.stat().st_mtime))
            except OSError:
                continue
        return hashlib.sha256("|".join(sorted(mtimes)).encode()).hexdigest()
