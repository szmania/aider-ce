import re

import xxhash


class HashPos:
    # -------------------------------------------------------------------------
    # TOKEN-OPTIMIZED PREFIX-FREE ENCODING SETUP
    # -------------------------------------------------------------------------
    # flake8: noqa
    # fmt: off
    _TOKEN_LIST =   [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'eb', 'ec', # noqa
        'ed', 'eg', 'ej', 'ek', 'el', 'em', 'en', 'ep', 'eq', 'er', 'es', 'et', 'eu', 'ev', 'ex', 'ey', # noqa
        'fa', 'fd', 'fe', 'fi', 'fl', 'fn', 'fo', 'fp', 'fr', 'fs', 'ft', 'ga', 'ge', 'gi', 'gl', 'go', # noqa
        'gr', 'gu', 'gy', 'ha', 'he', 'hi', 'ho', 'hr', 'hw', 'hy', 'ib', 'ic', 'id', 'ie', 'if', 'ig', # noqa
        'ih', 'ii', 'ik', 'il', 'im', 'in', 'io', 'ip', 'ir', 'is', 'it', 'iv', 'iz', 'ja', 'je', 'ji', # noqa
        'jo', 'js', 'ju', 'ka', 'ke', 'kg', 'ki', 'kl', 'km', 'kn', 'ko', 'kw', 'la', 'lb', 'le', 'li', # noqa
        'll', 'lo', 'lp', 'lu', 'ly', 'ma', 'mb', 'md', 'me', 'mg', 'mi', 'mk', 'ml', 'mm', 'mo', 'mp', # noqa
        'ms', 'mt', 'mu', 'mx', 'my', 'na', 'nd', 'ne', 'ng', 'ni', 'nl', 'nn', 'no', 'np', 'ns', 'nu', # noqa
        'nv', 'ny', 'ob', 'oc', 'od', 'of', 'og', 'oh', 'ok', 'ol', 'om', 'on', 'op', 'or', 'os', 'ot', # noqa
        'ou', 'ov', 'ox', 'oz', 'pa', 'pb', 'pc', 'pd', 'pe', 'pg', 'ph', 'pi', 'pl', 'pm', 'po', 'pp', # noqa
        'pr', 'ps', 'pt', 'pu', 'px', 'py', 'qu', 'ra', 'rb', 'rc', 're', 'rh', 'ri', 'rm', 'ro', 'rs', # noqa
        'rt', 'ru', 'rv', 'rx', 'ry', 'sa', 'sb', 'sc', 'sd', 'se', 'sf', 'sh', 'si', 'sk', 'sl', 'sm', # noqa
        'sn', 'so', 'sp', 'sq', 'ss', 'st', 'su', 'sv', 'sw', 'sy', 'ta', 'tb', 'tc', 'td', 'te', 'tf', # noqa
        'th', 'ti', 'tm', 'to', 'tp', 'tr', 'ts', 'tt', 'tu', 'tv', 'tw', 'tx', 'ty', 'ub', 'ud', 'ui', # noqa
        'uk', 'ul', 'um', 'un', 'up', 'ur', 'us', 'ut', 'uz', 'va', 've', 'vi', 'vm', 'vo', 'vr', 'vs', # noqa
        'vy', 'wa', 'we', 'wh', 'wo', 'wp', 'wr', 'wx', 'wy', 'ya', 'ye', 'yo', 'yy', 'za', 'ze', 'zo' # noqa
    ]
    # fmt: on
    # flake8: qa

    # Quick lookups for the 256 bytes
    ENCODE_MAP = {i: token for i, token in enumerate(_TOKEN_LIST)}
    DECODE_MAP = {token: i for i, token in enumerate(_TOKEN_LIST)}

    # Because all 2-char tokens start with letters e-z (except x), which are never used as standalone
    # 1-char tokens, we can cleanly split them.
    _PREFIX_CHARS = set("efghijklmnopqrstuvwyz")

    # Regex building blocks dynamically matching the list logic.
    # Single chars are 0-9 and a-d. Two-char tokens are e-z (except x) followed by a letter.
    _BYTE_REGEX = r"(?:[0-9a-d]|[efghijklmnopqrstuvwyz][A-Za-z])"
    # Regex for HashPos format: {3 encoded bytes}::
    HASH_PREFIX_RE = re.compile(rf"^({_BYTE_REGEX}{{3}})::")
    # Regex for normalization: 3 encoded bytes optionally followed by '::'
    NORMALIZE_RE = re.compile(rf"^({_BYTE_REGEX}{{3}})(?:::.*| .*)?$")
    # Regex for a raw 3-byte encoded fragment
    FRAGMENT_RE = re.compile(rf"^{_BYTE_REGEX}{{3}}$")

    def __init__(self, source_text: str = ""):
        self.lines = source_text.splitlines()
        self.total = len(self.lines)

    def _get_region_val(self, line_idx: int) -> int:
        """
        Maps the line to one of 16 proportional vertical buckets (4 bits).
        This acts as a binary space partition:
        - bit 3 is top/bottom half
        - bit 2 is top/bottom quarter
        - bit 1 is top/bottom eighth
        - bit 0 is top/bottom sixteenth
        """
        if self.total == 0:
            return 0

        # Calculate which 16th of the file the line falls into
        region = (line_idx * 16) // self.total

        # Clamp to 15 to handle edge cases safely
        return min(region, 15)

    def _get_neighborhood_hash(self, line_idx: int) -> int:
        """
        Creates a 20-bit digest using the current line and the 2 lines
        before and after it.
        """
        start = max(0, line_idx - 2)
        end = min(self.total, line_idx + 3)

        context_window = "\n".join(self.lines[start:end])
        full_hash = xxhash.xxh3_64_intdigest(context_window.encode("utf-8"))

        # Isolate exactly 20 bits
        return full_hash & 0xFFFFF

    def generate_public_id(self, text: str, line_idx: int) -> str:
        """
        Generates a 3-to-6 char ID using the token-optimized prefix-free encoding.
        Layout: [20-bit Neighborhood Hash] [4-bit Region] = 24 bits total.
        """
        neighborhood_hash = self._get_neighborhood_hash(line_idx)
        region_val = self._get_region_val(line_idx)

        # Pack the 24-bit integer
        packed = (neighborhood_hash << 4) | region_val

        # Encode 3 bytes using the prefix-free map
        res = ""
        for _ in range(3):
            byte_segment = packed % 256
            res += self.ENCODE_MAP[byte_segment]
            packed //= 256

        return res

    def unpack_public_id(self, public_id: str) -> tuple[int, int]:
        """
        Reverses the Public ID back into its (Neighborhood Hash, Region Value) values.
        Reads the prefix-free string left-to-right to unambiguously decode the bytes.
        """
        packed = 0
        byte_shift = 0
        i = 0

        while i < len(public_id):
            char = public_id[i]

            # The S-Z characters explicitly signal a two-character sequence
            if char in self._PREFIX_CHARS:
                seq = public_id[i : i + 2]
                i += 2
            else:
                seq = char
                i += 1

            byte_val = self.DECODE_MAP[seq]
            packed |= byte_val << byte_shift
            byte_shift += 8

        # Extract the 20-bit hash (shift right by 4, mask 0xFFFFF)
        neighborhood_hash = (packed >> 4) & 0xFFFFF

        # Extract the 4-bit region from the lowest bits
        region_val = packed & 0xF

        return neighborhood_hash, region_val

    def format_content(self, start_line: int = 1) -> str:
        formatted_lines = []
        for i, line in enumerate(self.lines):
            prefix = self.generate_public_id(line, i)
            if line.strip():
                formatted_lines.append(f"{prefix}::{line}")
            else:
                formatted_lines.append(f"{line}")

        return "\n".join(formatted_lines)

    def resolve_to_lines(self, public_id: str, start_line: int = 1) -> list[int]:
        target_hash, target_region = self.unpack_public_id(public_id)
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
        # prioritize the one whose current binary region is closest to the target region.
        def region_distance(idx: int) -> int:
            current_region = self._get_region_val(idx)
            # Linear distance because proportional regions don't wrap around
            return abs(current_region - target_region)

        matches.sort(key=region_distance)

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
        Normalize a HashPos string to the exact matched prefix fragment.
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
            r"Expected a valid content ID containing [a-z0-9] followed by `::`"
        )
