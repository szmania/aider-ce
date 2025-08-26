import json
import logging
import os
import sys

import configargparse

from aider import models
from aider.args import get_parser
from aider.coders import Coder
from aider.io import InputOutput
from aider.mcp.server import McpServer
from aider.repo import GitRepo
from aider.utils import find_common_root

logging.basicConfig()
log = logging.getLogger("aider")


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    # Pre-parse for --env-file and --config
    preparser = configargparse.ArgumentParser(add_help=False)
    preparser.add_argument("-c", "--config", is_config_file=True)
    preparser.add_argument("--env-file")
    pargs, remaining_argv = preparser.parse_known_args(args)

    if pargs.env_file:
        from dotenv import load_dotenv

        load_dotenv(pargs.env_file, override=True)

    # Figure out the git repo root
    try:
        repo = GitRepo()
        git_root = repo.root
    except FileNotFoundError:
        git_root = None
        repo = None

    default_config_files = []
    if git_root:
        default_config_files.append(os.path.join(git_root, ".aider.conf.yml"))
    default_config_files.append(os.path.expanduser("~/.aider.conf.yml"))

    if pargs.config:
        default_config_files.insert(0, pargs.config)

    parser = get_parser(default_config_files, git_root)

    try:
        args = parser.parse_args(remaining_argv)
    except (configargparse.ArgumentError, SystemExit) as e:
        # a bit of a hack to not exit with 0 on --help
        if hasattr(e, "code") and e.code == 0:
            return 0
        print(e)
        return 1

    # THE CHANGE IS HERE
    models.RETRY_TIMEOUT = args.retry_timeout
    models.RETRY_BACKOFF_FACTOR = args.retry_backoff_factor

    if args.verbose:
        log.setLevel(logging.INFO)
    else:
        log.setLevel(logging.WARNING)

    io = InputOutput(
        pretty=args.pretty,
        yes=args.yes_always,
        chat_history_file=args.chat_history_file,
        input_history_file=args.input_history_file,
        user_input_color=args.user_input_color,
        tool_output_color=args.tool_output_color,
        tool_error_color=args.tool_error_color,
        tool_warning_color=args.tool_warning_color,
        assistant_output_color=args.assistant_output_color,
        code_theme=args.code_theme,
        stream=args.stream,
        llm_history_file=args.llm_history_file,
        encoding=args.encoding,
        line_endings=args.line_endings,
        vim_mode=args.vim,
        multiline_mode=args.multiline,
        notifications=args.notifications,
        notification_command=args.notifications_command,
        fancy_input=args.fancy_input,
        placeholder=args.message or args.message_file,
    )

    fnames = args.files
    if args.file:
        fnames.extend(args.file)

    if not repo and args.git:
        # Coder will create a GitRepo if it's not provided
        pass
    elif not args.git:
        repo = None

    if not repo and fnames:
        common_root = find_common_root(fnames)
        if common_root:
            os.chdir(common_root)
            log.info(f"Changed CWD to {common_root}")

    if args.model_settings_file:
        models.register_models([args.model_settings_file])
    if args.model_metadata_file:
        models.register_litellm_models([args.model_metadata_file])

    try:
        main_model = models.Model(
            args.model,
            weak_model=args.weak_model,
            editor_model=args.editor_model,
            editor_edit_format=args.editor_edit_format,
            verbose=args.verbose,
        )
    except ValueError as e:
        io.tool_error(str(e))
        return 1

    if args.show_model_warnings:
        if models.sanity_check_models(io, main_model):
            io.tool_output(models.urls.model_warnings)

    lint_cmds = dict()
    if args.lint_cmd:
        for lint_cmd in args.lint_cmd:
            lang, cmd = lint_cmd.split(":", 1)
            lint_cmds[lang.strip()] = cmd.strip()

    mcp_servers = []
    if args.mcp_servers or args.mcp_servers_file:
        mcp_configs = []
        if args.mcp_servers:
            mcp_configs.extend(json.loads(args.mcp_servers))
        if args.mcp_servers_file:
            with open(args.mcp_servers_file, "r") as f:
                mcp_configs.extend(json.load(f))
        for config in mcp_configs:
            mcp_servers.append(McpServer(**config))

    coder = Coder.create(
        main_model=main_model,
        edit_format=args.edit_format,
        io=io,
        repo=repo,
        fnames=fnames,
        add_gitignore_files=args.add_gitignore_files,
        read_only_fnames=args.read,
        show_diffs=args.show_diffs,
        auto_commits=args.auto_commits,
        dirty_commits=args.dirty_commits,
        dry_run=args.dry_run,
        map_tokens=args.map_tokens,
        verbose=args.verbose,
        stream=args.stream,
        use_git=args.git,
        restore_chat_history=args.restore_chat_history,
        auto_lint=args.auto_lint,
        auto_test=args.auto_test,
        lint_cmds=lint_cmds,
        test_cmd=args.test_cmd,
        map_mul_no_files=args.map_multiplier_no_files,
        map_max_line_length=args.map_max_line_length,
        map_refresh=args.map_refresh,
        cache_prompts=args.cache_prompts,
        num_cache_warming_pings=args.cache_keepalive_pings,
        suggest_shell_commands=args.suggest_shell_commands,
        chat_language=args.chat_language,
        commit_language=args.commit_language,
        detect_urls=args.detect_urls,
        auto_accept_architect=args.auto_accept_architect,
        mcp_servers=mcp_servers,
        enable_context_compaction=args.enable_context_compaction,
        context_compaction_max_tokens=args.context_compaction_max_tokens,
        context_compaction_summary_tokens=args.context_compaction_summary_tokens,
        map_cache_dir=args.map_cache_dir,
    )

    if args.show_prompts:
        coder.show_prompts()
        return 0

    if args.show_repo_map:
        if not coder.repo_map:
            io.tool_error("Repo map is not available.")
            return 1
        io.tool_output(coder.repo_map.get_repo_map(coder.abs_fnames, []))
        return 0

    message = args.message
    if args.message_file:
        with open(args.message_file, "r") as f:
            message = f.read()

    coder.run(with_message=message)
    return 0


if __name__ == "__main__":
    status = main()
    sys.exit(status)
