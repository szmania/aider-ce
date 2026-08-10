import json
import os
import tempfile

import yaml

from cecli.args import get_parser
from cecli.helpers import config_utils
from cecli.main import convert_yaml_to_json_string, merge_agent_config


def test_cli_agent_config_merges_with_config_file():
    """CLI --agent-config must deep-merge with the config-file agent-config
    instead of replacing it wholesale (regression: file-only keys dropped)."""
    file_ac = {
        "command_timeout": 0,
        "skip_cli_confirmations": True,
        "tools_paths": ["/tmp/mytools"],
        "skills_paths": ["/tmp/skills"],
    }
    cli = json.dumps({"command_timeout": 120})
    merged = json.loads(merge_agent_config(cli, file_ac))
    assert merged["command_timeout"] == 120  # CLI wins per-key
    assert merged["skip_cli_confirmations"] is True  # file-only key preserved
    assert merged["tools_paths"] == ["/tmp/mytools"]  # file-only key preserved
    assert merged["skills_paths"] == ["/tmp/skills"]  # file-only key preserved


def test_file_agent_config_given_as_json_string():
    """Merged config from read_and_merge_all_configs holds agent-config as a
    JSON string (YAML block scalar) - the helper must handle that form."""
    file_ac = '{"command_timeout": 0, "skills_paths": ["/tmp/skills"]}'
    merged = json.loads(merge_agent_config('{"skip_cli_confirmations": true}', file_ac))
    assert merged["skip_cli_confirmations"] is True
    assert merged["command_timeout"] == 0
    assert merged["skills_paths"] == ["/tmp/skills"]


def test_no_file_agent_config_returns_cli_unchanged():
    """Without a config-file agent-config the merge must be a no-op."""
    cli = '{"command_timeout": 120}'
    assert merge_agent_config(cli, None) == cli
    assert merge_agent_config(cli, {}) == cli
    assert merge_agent_config(cli, "not json") == cli


def test_nested_dict_keys_deep_merged():
    """Nested dicts under agent-config are merged recursively, CLI wins."""
    file_ac = {"nested": {"a": 1, "b": 2}}
    merged = json.loads(merge_agent_config('{"nested": {"b": 9, "c": 3}}', file_ac))
    assert merged["nested"] == {"a": 1, "b": 9, "c": 3}


def test_cli_array_replaces_file_array():
    """Arrays are not deep-merged: CLI list values win wholesale."""
    file_ac = {"skills_paths": ["/tmp/file-skills"]}
    merged = json.loads(merge_agent_config('{"skills_paths": ["/tmp/cli-skills"]}', file_ac))
    assert merged["skills_paths"] == ["/tmp/cli-skills"]


def test_main_async_pipeline_preserves_file_keys(tmp_path):
    """Replicates main_async: merge config files -> temp yaml -> parser ->
    CLI parse -> temp file deleted -> convert + merge against merged_config.

    Guards against regressing to the broken baseline (re-parsing argv after
    the temp config file was already unlinked yields no config-file values).
    """
    conf = tmp_path / ".cecli.conf.yml"
    conf.write_text(
        "agent-config: |\n"
        '  {"skills_paths": ["./.cecli/skills"], "skills_init": ["android-cli"]}\n'
    )
    paths = [str(conf)]
    merged_config = config_utils.read_and_merge_all_configs(paths, [], paths)

    fd, tmp = tempfile.mkstemp(suffix=".yml", prefix="cecli_merged_")
    os.close(fd)
    with open(tmp, "w") as f:
        yaml.dump(merged_config, f)
    try:
        parser = get_parser([tmp], None)
        argv = ['--agent-config={"command_timeout": 0}']
        args, _ = parser.parse_known_args(argv)
    finally:
        os.unlink(tmp)  # main_async deletes the temp file before the merge point

    args.agent_config = convert_yaml_to_json_string(args.agent_config)
    file_ac = merged_config.get("agent-config")
    args.agent_config = merge_agent_config(args.agent_config, file_ac)

    merged = json.loads(args.agent_config)
    assert merged["command_timeout"] == 0
    assert merged["skills_paths"] == ["./.cecli/skills"]
    assert merged["skills_init"] == ["android-cli"]
