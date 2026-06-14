import pytest

from cecli.helpers.monorepo.config import validate_config


def test_validate_config_empty():
    # Should not raise
    validate_config({})


def test_validate_config_no_name():
    with pytest.raises(ValueError, match="Workspace configuration must include a 'name'"):
        validate_config({"projects": []})


def test_validate_config_invalid_project_missing_source():
    with pytest.raises(ValueError, match="exactly one of 'path' or 'repo'"):
        validate_config({"name": "test", "projects": [{"name": "p1"}]})


def test_validate_config_invalid_project_both_sources():
    with pytest.raises(ValueError, match="exactly one of 'path' or 'repo'"):
        validate_config(
            {
                "name": "test",
                "projects": [
                    {
                        "name": "p1",
                        "path": "/tmp/p1",
                        "repo": "https://github.com/org/r.git",
                    }
                ],
            }
        )


def test_validate_config_path_project():
    validate_config(
        {
            "name": "local",
            "projects": [{"name": "app", "path": "/abs/app", "primary": True}],
        }
    )


def test_validate_config_duplicate_project():
    with pytest.raises(ValueError, match="Duplicate project name: p1"):
        validate_config(
            {
                "name": "test",
                "projects": [{"name": "p1", "repo": "url1"}, {"name": "p1", "repo": "url2"}],
            }
        )


def test_validate_config_valid():
    config = {"name": "test", "projects": [{"name": "p1", "repo": "url1"}]}
    validate_config(config)
    assert config["projects"][0]["name"] == "p1"


def test_load_workspace_config_json_string():
    from cecli.helpers.monorepo.config import load_workspace_config

    config_str = '{"name": "json-ws", "projects": []}'
    config = load_workspace_config(config_str)
    assert config["name"] == "json-ws"


def test_load_workspace_config_yaml_string():
    from cecli.helpers.monorepo.config import load_workspace_config

    config_str = "name: yaml-ws\nprojects: []"
    config = load_workspace_config(config_str)
    assert config["name"] == "yaml-ws"
