# flake8: noqa: F401
# Import tool modules into the cecli.tools namespace

from . import (
    _yield,
    command,
    delegate,
    edit_file,
    explore_code,
    git_branch,
    git_diff,
    git_log,
    git_remote,
    git_show,
    git_status,
    grep,
    ls,
    orchestrate,
    read_file,
    resource_manager,
    thinking,
    undo_change,
    update_todo_list,
)

# List of all available tool modules for dynamic discovery
TOOL_MODULES = [
    command,
    delegate,
    edit_file,
    explore_code,
    _yield,
    git_branch,
    git_diff,
    git_log,
    git_remote,
    git_show,
    git_status,
    grep,
    ls,
    orchestrate,
    read_file,
    resource_manager,
    thinking,
    undo_change,
    update_todo_list,
]
