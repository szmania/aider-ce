# flake8: noqa: F401
# Import tool modules into the cecli.tools namespace

# Import all tool modules
from . import (
    _yield,
    command,
    command_interactive,
    context_manager,
    delegate,
    edit_text,
    explore_code,
    git_branch,
    git_diff,
    git_log,
    git_remote,
    git_show,
    git_status,
    grep,
    load_skill,
    ls,
    read_range,
    remove_skill,
    thinking,
    undo_change,
    update_todo_list,
)

# List of all available tool modules for dynamic discovery
TOOL_MODULES = [
    command,
    command_interactive,
    context_manager,
    delegate,
    edit_text,
    explore_code,
    _yield,
    git_branch,
    git_diff,
    git_log,
    git_remote,
    git_show,
    git_status,
    grep,
    load_skill,
    ls,
    read_range,
    remove_skill,
    thinking,
    undo_change,
    update_todo_list,
]
