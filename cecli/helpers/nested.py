import copy
import json
import os
from typing import Any, Dict, List, Union


def arg_resolver(obj: Union[List[Any], Dict[str, Any], Any], key: str, default: Any = None) -> Any:
    """
    Resolves a single key or index from an object with dash/underscore flexibility.
    """
    # 1. Handle List/Sequence access
    if isinstance(obj, (list, tuple)):
        if str(key).isdigit():
            idx = int(key)
            return obj[idx] if idx < len(obj) else default
        return default

    # 2. Handle Dict access
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        # Test underscore and hyphen versions directly
        key_str = str(key)
        # Check underscore version
        if "-" in key_str:
            underscore_key = key_str.replace("-", "_")
            if underscore_key in obj:
                return obj[underscore_key]
        # Check hyphen version
        if "_" in key_str:
            hyphen_key = key_str.replace("_", "-")
            if hyphen_key in obj:
                return obj[hyphen_key]
        return default

    # 3. Handle Object attribute access
    if hasattr(obj, "__dict__") or hasattr(obj, "__slots__"):
        if hasattr(obj, str(key)):
            return getattr(obj, key)
        # Test underscore and hyphen versions directly
        key_str = str(key)
        # Check underscore version
        if "-" in key_str:
            underscore_key = key_str.replace("-", "_")
            if hasattr(obj, underscore_key):
                return getattr(obj, underscore_key)
        # Check hyphen version
        if "_" in key_str:
            hyphen_key = key_str.replace("_", "-")
            if hasattr(obj, hyphen_key):
                return getattr(obj, hyphen_key)
        return default

    return default


def getter(
    data: Union[List[Any], Dict[str, Any], Any], path: Union[str, List[str]], default: Any = None
) -> Any:
    """Safely access nested dicts, lists, and objects using normalized dot-notation."""

    if data is None:
        return default

    # Handle single path string
    if isinstance(path, str):
        paths = [path]
    else:
        paths = path

    # Try each path, return first valid result
    for path_str in paths:
        current = data
        parts = path_str.split(".")
        found = True

        for part in parts:
            current = arg_resolver(current, part, default=default)
            if current is default:
                found = False
                break

        if found:
            return current

    return default


DEEP_MERGE_LIST_FIELDS: frozenset[str] = frozenset(
    {
        # ── Top-level argparse action="append" args ──────────────────
        # These are excluded from configargparse for .cecli.conf.yml files
        # (to prevent shallow overwrite), so they must be deep-merged here.
        "rules",
        "file",
        "read",
        "mcp_servers_files",
        "set_env",
        "api_key",
        "alias",
        "exempt_paths",
        "lint_cmd",
        # ── Nested agent-config array fields ─────────────────────────
        "skills_paths",
        "skills_includelist",
        "skills_excludelist",
        "skills_init",
        "subagent_paths",
        "tools_paths",
        "tools_includelist",
        "tools_excludelist",
        "servers_includelist",
        "servers_excludelist",
        "allowed_commands",
    }
)

DEEP_MERGE_JSON_FIELDS: frozenset[str] = frozenset(
    {
        "agent_config",
        "mcp_servers",
        "hooks",
        "model_providers",
        "security_config",
        "retries",
        "custom",
        "tui_config",
    }
)


def _normalize_keys(obj: Any) -> Any:
    """
    Recursively convert hyphenated dict keys to underscore-separated keys.

    YAML config files may use hyphenated keys (e.g. ``agent-config``,
    ``mcp-servers-files``), but argparse attributes and the
    ``DEEP_MERGE_*_FIELDS`` frozensets use underscores (``agent_config``,
    ``mcp_servers_files``).  This helper normalizes loaded YAML dicts so
    that key lookups against the frozensets succeed.

    Lists, scalars, and non-dict values are returned unchanged.
    """
    if isinstance(obj, dict):
        return {key.replace("-", "_"): _normalize_keys(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_normalize_keys(item) for item in obj]
    return obj


def _deduplicate_list(merged_list: list, new_list: list) -> list:
    """
    Append elements from new_list to merged_list, skipping duplicates.

    Deduplication is value-based:
    - Primitives (str, int, float, bool, None): compared with ``==``
    - Dicts: compared via ``json.dumps(obj, sort_keys=True)`` for stable
      structural equality
    - Other types: compared with ``==``

    First-occurrence order is preserved: elements already in merged_list
    keep their position; new unique elements are appended at the end.

    Each appended element is deep-copied via ``copy.deepcopy()`` to
    prevent reference sharing between the merged result and the source
    lists.
    """
    # Build a set of canonical representations for fast lookup
    seen: set = set()
    for item in merged_list:
        seen.add(_canonical_repr(item))

    for item in new_list:
        key = _canonical_repr(item)
        if key not in seen:
            seen.add(key)
            merged_list.append(copy.deepcopy(item))

    return merged_list


def _canonical_repr(item: Any) -> Any:
    """
    Return a hashable, equality-comparable canonical representation
    of *item* for deduplication purposes.

    - Primitives (str, int, float, bool, None, tuple): returned as-is
    - Dicts: serialized to a JSON string with sorted keys
    - Lists: recursively converted to a tuple of canonical representations
    - Everything else: returned as-is (falls back to ``==``)
    """
    if isinstance(item, dict):
        return json.dumps(item, sort_keys=True, default=str)
    if isinstance(item, list):
        return tuple(_canonical_repr(e) for e in item)
    return item


def deep_merge(dict1, dict2, deep_merge_arrays=True):
    """
    Recursively merges *dict2* into *dict1*.

    Parameters
    ----------
    dict1 : dict
        The base dictionary (earlier / lower-precedence config).
    dict2 : dict
        The overlay dictionary (later / higher-precedence config).
    deep_merge_arrays : bool, optional
        When ``True`` (the default), list values are concatenated with
        deduplication instead of being overwritten.  When ``False``,
        the existing shallow-overwrite behaviour is preserved.

    Returns
    -------
    dict
        A new dictionary (deep-copied from *dict1*) with *dict2* merged
        in.  Neither *dict1* nor *dict2* is mutated.

    Merge rules (per key)
    ---------------------
    * Both values are dicts → recursively ``deep_merge`` the sub-dicts.
    * Both values are lists **and** ``deep_merge_arrays=True`` →
      concatenate with deduplication (first-occurrence order preserved).
    * Otherwise → *dict2*'s value overwrites *dict1*'s value.
    """
    merged = copy.deepcopy(dict1)
    for key, value in dict2.items():
        # When deep_merge_arrays is enabled, a None value in dict2 means
        # "not set" — skip it to preserve dict1's value.  This prevents
        # empty/null YAML keys (e.g. ``subagent_paths:`` with no value)
        # from overwriting populated arrays from lower-precedence configs.
        if deep_merge_arrays and value is None:
            continue
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value, deep_merge_arrays=deep_merge_arrays)
        elif (
            deep_merge_arrays
            and key in merged
            and isinstance(merged[key], list)
            and isinstance(value, list)
        ):
            merged[key] = _deduplicate_list(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def is_cecli_conf_file(filepath: str) -> bool:
    """
    Return ``True`` if *filepath* refers to a ``.cecli.conf.yml`` file
    (which should receive deep-merge treatment), ``False`` otherwise
    (e.g. ``.cecli/conf.yml`` or unknown patterns).

    The check is based solely on the basename of the path.
    """
    basename = os.path.basename(filepath)
    return basename == ".cecli.conf.yml"


def deep_merge_config_dicts(
    config_dicts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Cumulatively deep-merge a list of raw config dictionaries.

    Each successive dict is merged into the accumulator using
    ``deep_merge(acc, next_dict, deep_merge_arrays=True)``.  The
    accumulator is deep-copied before each merge to prevent mutation
    of intermediate results.

    Returns the final merged dictionary.  If *config_dicts* is empty,
    returns an empty dict.
    """
    if not config_dicts:
        return {}

    merged = copy.deepcopy(config_dicts[0])
    for next_dict in config_dicts[1:]:
        merged = deep_merge(merged, next_dict, deep_merge_arrays=True)
    return merged
