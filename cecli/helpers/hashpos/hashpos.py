import re

import xxhash


class HashPos:
    # 1024-character Base1024 corpus
    B1024 = (
        "0123456789ABCDEFGHIJ"
        "KLMNOPQRSTUVWXYZabcd"
        "efghijklmnopqrstuvwx"
        "yz¡£§©«®°±·»¿×ßæðøĐđ"
        "ıłœəαβγδεηθικλμνοπρς"
        "στυφχωАБВГДЕЗИКЛМНОП"
        "РСТУФЦЧЭЯабвгдежзикл"
        "мнопрстуфхцчшщъыьэюя"
        "іאבדהוחילמנערשת،ابةت"
        "ثجحخدذرزسشصضطظعغفقكل"
        "منهوىيپکگیकतनपमरलसहন"
        "রกขคงจชณดตถทนบปผพมยร"
        "ลวสหอะาเแใไ‐–—―‘’“”„"
        "†•′※€←→−─│█■●★☆♥♪、。《"
        "》「」『』【】〜あいうえおかきくけこさし"
        "すせそたちっつてとなにのはまみめもやよら"
        "りるれろわをんアィイウェエオカキクコサシ"
        "スセタチッテトナニフマムメャュョラリルレ"
        "ロン・ー一万三上下不与专业东两个中为主么"
        "义之也书了事二于五些交产享京人亿今介从他"
        "付代以们件价任份企会传但位体何余作你使例"
        "供価保信修倍值停像元先入全公共关其具内円"
        "册再写出击分列则初利别到制前力功加务动動"
        "包化北区十午华单南即历原去县参及友反发取"
        "变口只可台右号司合同名后向否含听启告员周"
        "命和品哈商問器四回因国图土在地场型城基報"
        "場增声处备复外多大天失头女好如始子字存学"
        "安完定实客家容密对导将小少尔就局展山岁州"
        "工左已市布常平年并广序库应店度建开式引张"
        "当录形影径待後得微心必志态思性总息您情意"
        "感成我或户所手打技投报拉持指按换据排接推"
        "提播支收改放政效数整文料断新方族无日时明"
        "易星是時景更最月有服期木未本机权束条来板"
        "构果查标样格案模次款止正此步歳段每比民気"
        "水求江没治法注活流海消清游源火点無然片版"
        "物特率环现球理生用由电男画界番登的目直相"
        "省看県真知码确示社票私种科秒称移程税空立"
        "站章端笑符第等简算管米类系素索约级线组经"
        "结给统编网置美老考者而联能自至色节英藏行"
        "表装西要見见规视角解言計記話読计认议记论"
        "设证评试话该语误说请读调象责败货费资起超"
        "路身车转载辑输达过运近还这进连述退送选通"
        "速造連道部都配释里重量金错长開間関门闭问"
        "间队阳陆限院除雅集雷需非面音页项题首验高"
        "黑가간개거게결경고공과구그글기나내는능니"
        "다당대도동되된드든들디라래러력로록료류른"
        "를름리만메면명목문미버번보복부분비사산상"
        "색생서성세션소수스습시식신아야어에여열오"
        "와요용우운원위으은을음의이인일임입자작장"
        "재적전정제져조주지진째체출치크태터턴트하"
        "한할함해호화환회�ª²³´µ¹º¼½ÀÁ"
        "ÂÃÄÇ"
    )

    # We escape every individual character just to be completely safe from regex metacharacters
    _B1024_REGEX_SET = "".join(re.escape(c) for c in B1024)

    # Regex pattern for HashPos format: {3-char-hash}::
    HASH_PREFIX_RE = re.compile(rf"^([{_B1024_REGEX_SET}]{{3}})::")
    # Regex for normalization: 3 hash chars optionally followed by '::'
    NORMALIZE_RE = re.compile(rf"^([{_B1024_REGEX_SET}]{{3}})(?:)?::")
    # Regex for a raw 3-character fragment
    FRAGMENT_RE = re.compile(rf"^[{_B1024_REGEX_SET}]{{3}}$")

    # Looser pattern: any 3 chars with at least one non-ASCII followed by ::
    _LOOSE_PREFIX_RE = re.compile(r"^(?=.{0,2}[^\x00-\x7f]).{3}::")

    def __init__(self, source_text: str = ""):
        self.lines = source_text.splitlines()
        self.total = len(self.lines)

    def _get_line_hash(self, text: str) -> int:
        """
        Creates a 20-bit digest of the current line's text.
        """
        return xxhash.xxh3_64_intdigest(text.encode("utf-8")) & 0xFFFFF

    def _get_adjacent_hash(self, line_idx: int) -> int:
        """
        Creates a 10-bit digest of specific surrounding lines at offsets
        -7, -5, -3, -2, +2, +3, +5, +7 to provide local context.
        """
        offsets = [-7, -5, -3, -2, 2, 3, 5, 7]
        adjacent_lines = []

        for offset in offsets:
            target_idx = line_idx + offset
            if 0 <= target_idx < self.total:
                adjacent_lines.append(self.lines[target_idx])

        context = "\n".join(adjacent_lines)
        return xxhash.xxh3_64_intdigest(context.encode("utf-8")) & 0x3FF

    def generate_private_id(self, text: str) -> str:
        """
        Generates a fast 12-bit (3 hex chars) hash based purely on the line text.
        """
        bits = xxhash.xxh3_64_intdigest(text.encode("utf-8")) & 0xFFF
        return f"{bits:03x}"

    def generate_public_id(self, text: str, line_idx: int) -> str:
        """
        Generates a 3-character Base1024 ID.
        Layout: [20-bit Line Hash] [10-bit Adjacent Hash] = 30 bits total.
        Each Base1024 character holds 10 bits.
        """
        line_hash = self._get_line_hash(text)
        adj_hash = self._get_adjacent_hash(line_idx)

        # Pack the 30-bit integer
        packed = (line_hash << 10) | adj_hash

        res = ""
        for _ in range(3):
            # Extract 10 bits at a time using modulo 1024
            res += self.B1024[packed % 1024]
            packed //= 1024
        return res

    def unpack_public_id(self, public_id: str) -> tuple[int, int]:
        """
        Reverses the Public ID back into its (Line Hash, Adjacent Hash) values.
        """
        packed = 0
        for i, char in enumerate(public_id):
            # Each character restores 10 bits
            packed |= self.B1024.index(char) << (10 * i)

        # Extract bits based on layout
        line_hash = (packed >> 10) & 0xFFFFF
        adj_hash = packed & 0x3FF

        return line_hash, adj_hash

    def format_content(self, use_private_ids: bool = False, start_line: int = 1) -> str:
        formatted_lines = []
        for i, line in enumerate(self.lines):
            prefix = (
                self.generate_private_id(line)
                if use_private_ids
                else self.generate_public_id(line, i)
            )
            if line.strip():
                formatted_lines.append(f"{prefix}::{line}")
            else:
                formatted_lines.append(f"{line}")

        return "\n".join(formatted_lines)

    def resolve_to_lines(self, public_id: str, start_line: int = 1) -> list[int]:
        target_line_hash, target_adj_hash = self.unpack_public_id(public_id)
        matches = []

        # 1. Primary Filter: Find all lines whose 20-bit line content hash matches
        for i, line in enumerate(self.lines):
            if self._get_line_hash(line) == target_line_hash:
                matches.append(i)

        if not matches:
            return []

        # If perfectly unique (highly likely given 20 bits of line entropy), return immediately
        if len(matches) == 1:
            return matches

        # 2. Tie-Breaking Heuristic:
        # If multiple identical lines exist, score them based on adjacency match.
        def score_match(idx: int) -> int:
            # Adjacency match: 0 means exact match, 1 means mismatch (we want lower scores)
            return 0 if self._get_adjacent_hash(idx) == target_adj_hash else 1

        matches.sort(key=score_match)

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
        Also strips any 3-char sequence with at least one non-ASCII char followed by ::.
        """
        lines = text.splitlines(keepends=True)
        result_lines = []
        for line in lines:
            stripped_line = HashPos.HASH_PREFIX_RE.sub("", line, count=1)
            if stripped_line == line:
                stripped_line = HashPos._LOOSE_PREFIX_RE.sub("", line, count=1)
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
            r"Expected a 3-character string from the Base1024 character set."
        )
