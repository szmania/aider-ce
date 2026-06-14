"""
RapidFuzz-based fuzzy file path matching engine.

All searches run against the trie index's path list (via list_all())
to avoid duplicating the full file list in memory. Falls back to an
internally stored list when no trie reference is available.
"""

from ngram import NGram
from rapidfuzz import fuzz, process

# Threshold presets for different use cases (0-1 scale, converted to 0-100 internally)
THRESHOLD_MENTION_DETECTION: float = 0.5
THRESHOLD_FUZZY_SEARCH: float = 0.3
THRESHOLD_BASENAME_SEARCH: float = 0.6


class TrigramIndex:
    """
    RapidFuzz-based fuzzy file path matcher.

    Searches against a trie index's path list to avoid duplicating
    the full file list in memory. Falls back to an internally stored
    list when no trie reference is available (e.g., after serialization).
    """

    def __init__(
        self,
        items: list[str] | None = None,
        trie_index=None,
    ):
        """
        Initialize the trigram index.

        Args:
            items: Initial list of file paths to index (fallback)
            trie_index: A TrieIndex whose list_all() provides the path list
        """
        self._items = items
        self._trie_index = trie_index

    def _get_items(self) -> list[str]:
        """Resolve the path list from the trie or fall back to stored items."""
        if self._trie_index is not None:
            return self._trie_index.list_all()
        return self._items or []

    def build(self, items: list[str]) -> None:
        """
        Build or rebuild the index from a list of items.

        Stores the full list for rapidfuzz to search against.
        Detaches any trie reference.

        Args:
            items: Full list of file paths to index
        """
        self._items = items
        self._trie_index = None

    def search(
        self,
        query: str,
        threshold: float = THRESHOLD_FUZZY_SEARCH,
        max_results: int = 20,
    ) -> list[str]:
        """
        Fuzzy search for items matching a query string.

        Uses rapidfuzz.process.extract() with partial_ratio scorer for
        fast C++-backed fuzzy matching against the trie's path list.
        When fewer than 100 results are returned, re-ranks them using
        the ngram library's trigram similarity to better match human
        expectations for ordering.

        Args:
            query: Search string
            threshold: Minimum similarity score (0.0-1.0)
            max_results: Maximum number of results to return

        Returns:
            List of matching items, sorted by relevance
        """
        items = self._get_items()

        if not items or not query:
            return []
        score_cutoff = int(threshold * 100)
        results = process.extract(
            query,
            items,
            scorer=fuzz.partial_ratio,
            limit=max_results,
            score_cutoff=score_cutoff,
        )
        match_names = [match for match, score, _ in results]

        # Re-rank with ngram trigram similarity when result set is small
        # enough to benefit from the more human-like ordering.
        if len(match_names) < 100:
            ng = NGram(match_names, N=3)
            reranked = ng.search(query, threshold=0.0)
            match_names = [item for item, score in reranked]

        return match_names

    def search_with_scores(
        self,
        query: str,
        threshold: float = THRESHOLD_FUZZY_SEARCH,
        max_results: int = 20,
    ) -> list[tuple[str, float]]:
        """
        Fuzzy search returning items with their similarity scores.

        Args:
            query: Search string
            threshold: Minimum similarity score (0.0-1.0)
            max_results: Maximum number of results to return

        Returns:
            List of (item, score) tuples, sorted by score descending
        """
        items = self._get_items()
        if not items or not query:
            return []
        score_cutoff = int(threshold * 100)
        results = process.extract(
            query,
            items,
            scorer=fuzz.WRatio,
            limit=max_results,
            score_cutoff=score_cutoff,
        )
        return [(match, score / 100.0) for match, score, _ in results]

    def count(self) -> int:
        """Return the number of items in the index."""
        return len(self._get_items())

    def clear(self) -> None:
        """Clear the index."""
        self._items = None
        self._trie_index = None

    @property
    def ngram(self):
        """Backward-compatibility shim — always returns None."""
        return True

    @property
    def configuration(self) -> dict:
        """Return the current configuration parameters."""
        return {
            "engine": "rapidfuzz",
            "scorer": "fuzz.WRatio",
        }

    # --- Pickle support ---

    def __getstate__(self):
        """Resolve items before serialization so the trie ref is not pickled."""
        state = self.__dict__.copy()
        state["_items"] = self._get_items()
        state["_trie_index"] = None
        return state

    def __setstate__(self, state):
        """Restore from pickled state."""
        self.__dict__.update(state)
