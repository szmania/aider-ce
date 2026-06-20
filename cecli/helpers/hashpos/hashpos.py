import re

import xxhash


class HashPos:
    B256 = (
        "ABCDEFGHIJKLMNOP"
        "QRSTUVWXYZabcdef"
        "ghijklmnopqrstuv"
        "wxyz0123456789~_"
        "áéíóúñüöäßåøæçèà"
        "ùîôûбгджзийлпфцч"
        "шщъыьэюя的是不了人我在有"
        "他这为之大来以个中上们到说国和学"
        "あいうえおかきくけこさしすせそた"
        "ちつてとアイウエオカキクケコサシ"
        "スセソタチツテトαβγδεζηθ"
        "ικλμνξπ要会出就道也时年得"
        "生自下而过能可对行没发用天作方成"
        "者多日都三小机把理实心看起样好当"
        "点本民事其然想经去种动全意面前所"
        "业定现将法新明问度但最美月手走信"
    )

    # We use a regex-safe character class string for compiling patterns
    _B256_REGEX_SET = (
        "A-Za-z0-9~_"
        "áéíóúñüöäßåøæçèà"
        "ùîôûбгджзийлпфцч"
        "шщъыьэюя的是不了人我在有"
        "他这为之大来以个中上们到说国和学"
        "あいうえおかきくけこさしすせそた"
        "ちつてとアイウエオカキクケコサシ"
        "スセソタチツテトαβγδεζηθ"
        "ικλμνξπ要会出就道也时年得"
        "生自下而过能可对行没发用天作方成"
        "者多日都三小机把理实心看起样好当"
        "点本民事其然想经去种动全意面前所"
        "业定现将法新明问度但最美月手走信"
    )

    # Regex pattern for HashPos format: {3-char-hash}::
    HASH_PREFIX_RE = re.compile(rf"^([{_B256_REGEX_SET}]{{3}})::")
    # Regex for normalization: 3 hash chars optionally followed by '::'
    NORMALIZE_RE = re.compile(rf"^([{_B256_REGEX_SET}]{{3}})(?:)?::")
    # Regex for a raw 3-character fragment
    FRAGMENT_RE = re.compile(rf"^[{_B256_REGEX_SET}]{{3}}$")

    def __init__(self, source_text: str = ""):
        self.lines = source_text.splitlines()
        self.total = len(self.lines)

    def _get_region_val(self, line_idx: int) -> int:
        """
        Uses line_idx modulo 16 (4 bits).
        Guarantees up to 16 consecutive repeating lines get unique spatial anchors.
        """
        return line_idx % 16

    def _get_neighborhood_hash(self, line_idx: int) -> int:
        """
        Creates a 20-bit digest using the current line and the 3 lines
        before and after it.
        """
        start = max(0, line_idx - 2)
        end = min(self.total, line_idx + 3)

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
        Generates a 3-char Base256 ID combining a 4-bit modulo bucket and a 20-bit context hash.
        Layout: [4-bit Region] [20-bit Neighborhood Hash] = 24 bits total.
        Each Base256 char holds 8 bits (3 chars * 8 = 24 bits).
        """
        region_val = self._get_region_val(line_idx)
        neighborhood_hash = self._get_neighborhood_hash(line_idx)

        # Pack the 24-bit integer
        packed = (region_val << 20) | neighborhood_hash

        res = ""
        for _ in range(3):
            res += self.B256[packed % 256]
            packed //= 256
        return res

    def unpack_public_id(self, public_id: str) -> tuple[int, int]:
        """
        Reverses the Public ID back into its (Modulo 16, Neighborhood Hash) values.
        """
        packed = 0
        for i, char in enumerate(public_id):
            packed |= self.B256.index(char) << (8 * i)

        # Extract the 4-bit region (mask 0xF) and 20-bit hash (mask 0xFFFFF)
        region_val = (packed >> 20) & 0xF
        neighborhood_hash = packed & 0xFFFFF

        return region_val, neighborhood_hash

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
            f"Found matches for {start_id} and {end_id}, but no logically ordered range or unique matches."
        )

    @staticmethod
    def strip_prefix(text: str) -> str:
        """
        Remove HashPos prefixes from the start of every line.
        """
        lines = text.splitlines(keepends=True)
        result_lines = []
        for line in lines:
            stripped_line = HashPos.HASH_PREFIX_RE.sub("", line, count=1)
            result_lines.append(stripped_line)

        return "".join(result_lines)

    @staticmethod
    def extract_prefix(line: str) -> str:
        """
        Extract the hash prefix from a line if it has a HashPos prefix.
        """
        match = HashPos.HASH_PREFIX_RE.match(line)
        if match:
            return match.group(1)
        return ""

    @staticmethod
    def normalize(hashpos_str: str) -> str:
        """
        Normalize a HashPos string to the 3-character hash fragment.
        """
        if hashpos_str is None:
            raise ValueError("HashPos string cannot be None")

        if HashPos.FRAGMENT_RE.match(hashpos_str):
            return hashpos_str

        match = HashPos.NORMALIZE_RE.match(hashpos_str)
        if match:
            return match.group(1)

        raise ValueError(
            f"Invalid HashPos format '{hashpos_str}'. "
            r"Expected a 3-character string from the Base256 character set."
        )
