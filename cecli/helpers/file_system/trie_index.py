"""
marisa-trie wrapper with warm cache and serialization helpers.

Provides convenience methods for working with marisa_trie.Trie instances,
including prefix-based key iteration and memory-mapped loading.
"""

import marisa_trie


class TrieIndex:
    """
    Lightweight wrapper around marisa_trie.Trie.

    Provides typed convenience methods for common operations
    like prefix listing and existence checks.
    """

    def __init__(self, paths: list[str] | None = None):
        """
        Initialize the trie, optionally with an initial set of paths.

        Args:
            paths: Initial list of file paths to index
        """
        self._trie: marisa_trie.Trie | None = None
        if paths is not None:
            self._trie = marisa_trie.Trie(paths)

    def build(self, paths: list[str]) -> None:
        """
        Build or rebuild the trie from a list of paths.

        Args:
            paths: Full list of file paths to index
        """
        self._trie = marisa_trie.Trie(paths)

    def exists(self, path: str) -> bool:
        """
        Check if a path exists in the trie (O(path_depth)).

        Args:
            path: The path to check

        Returns:
            True if the path is in the trie
        """
        if self._trie is None:
            return False
        return path in self._trie

    def list_prefix(self, prefix: str = "") -> list[str]:
        """
        List all keys with the given prefix.

        Leverages the compressed trie structure for efficient
        prefix matching without iteration over unrelated keys.

        Args:
            prefix: String prefix to filter by (empty = all keys)

        Returns:
            List of matching keys
        """
        if self._trie is None:
            return []
        return list(self._trie.keys(prefix))

    def list_all(self) -> list[str]:
        """Return all keys in the trie."""
        return self.list_prefix("")

    def count(self) -> int:
        """Return the number of keys in the trie."""
        if self._trie is None:
            return 0
        return len(self._trie)

    def prefixes(self, path: str) -> list[str]:
        """
        Return all prefix components of a path that exist in the trie.

        Args:
            path: A file path to decompose into prefixes

        Returns:
            List of matching prefixes (from shortest to longest)
        """
        if self._trie is None:
            return []
        return list(self._trie.prefixes(path))

    def save(self, path: str) -> None:
        """
        Serialize the trie to disk in marisa-trie binary format.

        Args:
            path: File path for the .marisa binary output
        """
        if self._trie is not None:
            self._trie.save(path)

    def load(self, path: str) -> bool:
        """
        Load a trie from a .marisa binary file using mmap.

        Args:
            path: File path to the .marisa binary

        Returns:
            True if loading succeeded, False otherwise
        """
        try:
            self._trie = marisa_trie.Trie()
            self._trie.mmap(path)
            return True
        except (IOError, OSError, TypeError, RuntimeError):
            self._trie = None
            return False

    @property
    def trie(self) -> marisa_trie.Trie | None:
        """Access the underlying marisa_trie.Trie instance."""
        return self._trie
