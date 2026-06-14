import re

import xxhash


class HashPos:
    B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789~_"
    # Regex pattern for HashPos format: {4-char-hash}::
    HASH_PREFIX_RE = re.compile(r"^([0-9a-zA-Z\~_@]{4})::")
    # Regex for normalization: 4 hash chars optionally followed by '::'
    NORMALIZE_RE = re.compile(r"^([0-9a-zA-Z\~_@]{4})(?:)?::")
    # Regex for a raw 4-character fragment
    FRAGMENT_RE = re.compile(r"^[0-9a-zA-Z\~_@]{4}$")

    def __init__(self, source_text: str = ""):
        self.lines = source_text.splitlines()
        self.total = len(self.lines)

    def _get_region_bits(self, line_idx: int) -> tuple[int, int]:
        """
        Uses line_idx modulo 16 (4 bits) to get two 2-bit flags (b1, b2).
        This guarantees up to 16 consecutive repeating lines get unique spatial anchors.
        """
        mod_val = line_idx % 16

        # Split the 4-bit modulo value into two separate 2-bit flags
        b1 = (mod_val >> 2) & 3  # Top 2 bits (mask with 0b11)
        b2 = mod_val & 3  # Bottom 2 bits
        return b1, b2

    def _get_neighborhood_hash(self, line_idx: int) -> int:
        """
        Creates a 20-bit digest using the current line and the 3 lines
        before and after it.
        """
        start = max(0, line_idx - 3)
        end = min(self.total, line_idx + 4)

        context_window = "\n".join(self.lines[start:end])
        full_hash = xxhash.xxh3_64_intdigest(context_window.encode("utf-8"))

        # Isolate exactly 20 bits
        return full_hash & 0xFFFFF

    def generate_private_id(self, text: str) -> str:
        """
        Generates a fast 12-bit (3 hex chars) hash based purely on the line text.
        """
        bits = xxhash.xxh3_64_intdigest(text.encode("utf-8")) & 0xFFF
        return f"{bits:03x}"

    def generate_public_id(self, text: str, line_idx: int) -> str:
        """
        Generates a 4-char Base64 ID combining modulo buckets and context hash.
        Layout: [2-bit b1] [2-bit b2] [10-bit Hash A] [10-bit Hash B]
        """
        b1, b2 = self._get_region_bits(line_idx)
        neighborhood_hash = self._get_neighborhood_hash(line_idx)

        # Split the 20-bit hash into two 10-bit halves
        hash_a = (neighborhood_hash >> 10) & 0x3FF
        hash_b = neighborhood_hash & 0x3FF

        # Construct the mixed 24-bit integer
        packed = (b1 << 22) | (b2 << 20) | (hash_a << 10) | hash_b
        res = ""
        for _ in range(4):
            res += self.B64[packed % 64]
            packed //= 64
        return res

    def unpack_public_id(self, public_id: str) -> tuple[int, int]:
        """
        Reverses the Public ID back into its (Modulo 16, Neighborhood Hash) values.
        """
        packed = 0
        for i, char in enumerate(public_id):
            packed |= self.B64.index(char) << (6 * i)

        b1 = (packed >> 22) & 3
        b2 = (packed >> 20) & 3
        hash_a = (packed >> 10) & 0x3FF
        hash_b = packed & 0x3FF
        mod_val = (b1 << 2) | b2
        neighborhood_hash = (hash_a << 10) | hash_b

        return mod_val, neighborhood_hash

    def format_content(self, use_private_ids: bool = False, start_line: int = 1) -> str:
        formatted_lines = []
        for i, line in enumerate(self.lines):
            prefix = (
                self.generate_private_id(line)
                if use_private_ids
                else self.generate_public_id(line, i)
            )
            formatted_lines.append(f"{prefix}::{line}")
        return "\n".join(formatted_lines)

    def resolve_to_lines(self, public_id: str, start_line: int = 1) -> list[int]:
        target_mod, target_hash = self.unpack_public_id(public_id)
        matches = []

        # Find all lines whose neighborhood hash matches our target
        for i, line in enumerate(self.lines):
            if self._get_neighborhood_hash(i) == target_hash:
                matches.append(i)

        if not matches:
            return []

        # If perfectly unique, return it immediately
        if len(matches) == 1:
            return matches

        # Distance Heuristic: If multiple matches exist (e.g. repeated code blocks),
        # prioritize the one whose modulo is closest to the target modulo.
        # We use circular distance since mod 16 wraps around (0 is adjacent to 15).
        def modulo_distance(idx: int) -> int:
            current_mod = idx % 16
            dist = abs(current_mod - target_mod)
            return min(dist, 16 - dist)

        matches.sort(key=modulo_distance)

        return matches

    def resolve_range(self, start_id: str, end_id: str) -> tuple[int, int]:
        """
        Resolves a block range from two Public IDs.

        Logic:
        1. Resolve all candidates for both IDs (sorted by best match).
        2. Find the pair of (start, end) that are logically ordered.
        3. Returns (start_index, end_index)
        """
        starts = self.resolve_to_lines(start_id)
        ends = self.resolve_to_lines(end_id)

        if not starts or not ends:
            raise ValueError(f"Could not resolve IDs: {start_id}..{end_id}")

        for s in starts:
            for e in ends:
                if s <= e:
                    return s, e

        raise ValueError(
            f"Found matches for {start_id} and {end_id}, but no logically ordered range or unique"
            " matches."
        )

    @staticmethod
    def strip_prefix(text: str) -> str:
        r"""
        Remove HashPos prefixes from the start of every line.

        Removes prefixes that match the pattern: "{4-char-hash}"
        where the hash is exactly 4 characters from the set [0-9a-zA-Z\~_@] followed by '::'.

        Args:
            text: Input text with HashPos prefixes

        Returns:
            String with HashPos prefixes removed from each line
        """
        lines = text.splitlines(keepends=True)
        result_lines = []
        for line in lines:
            # Remove the HashPos prefix if present
            stripped_line = HashPos.HASH_PREFIX_RE.sub("", line, count=1)
            result_lines.append(stripped_line)

        return "".join(result_lines)

    @staticmethod
    def extract_prefix(line: str) -> str:
        """
        Extract the hash prefix from a line if it has a HashPos prefix.

        Args:
            line: A line of text that may contain a HashPos prefix

        Returns:
            The hash prefix (4 characters) if found, otherwise empty string
        """
        match = HashPos.HASH_PREFIX_RE.match(line)
        if match:
            return match.group(1)
        return ""

    @staticmethod
    def normalize(hashpos_str: str) -> str:
        """
        Normalize a HashPos string to the 4-character hash fragment.

        Accepts HashPos strings in "{hash_prefix}::" format or a raw "{hash_prefix}" fragment.
        Also extracts HashPos from strings that contain content after the HashPos,
        e.g., "H7M5::Line 1"

        Args:
            hashpos_str: HashPos string in various formats

        Returns:
            str: The 4-character hash fragment

        Raises:
            ValueError: If format is invalid
        """
        if hashpos_str is None:
            raise ValueError("HashPos string cannot be None")

        # Check if it's already a raw fragment
        if HashPos.FRAGMENT_RE.match(hashpos_str):
            return hashpos_str

        match = HashPos.NORMALIZE_RE.match(hashpos_str)
        if match:
            return match.group(1)

        # If no pattern matches, raise error
        raise ValueError(
            f"Invalid HashPos format '{hashpos_str}'. "
            r"Expected \"{content ID}\" "
            r"where content ID is exactly 4 characters from the set [0-9a-zA-Z\~_@]."
        )
