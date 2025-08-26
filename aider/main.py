import json
import os
import re
import sys
import threading
import traceback
import webbrowser
from dataclasses import fields
from pathlib import Path

try:
    import git
except ImportError:
    git = None

import importlib_resources
import shtab
from dotenv import load_dotenv
from prompt_toolkit.enums import EditingMode

from aider import __version__, models, urls, utils
from aider.analytics import Analytics
from aider.args import get_parser
from aider.coders import Coder
from aider.coders.base_coder import UnknownEditFormat
from aider.commands import Commands, SwitchCoder
from aider.copypaste import ClipboardWatcher
from aider.deprecated import handle_deprecated_model_args
from aider.format_settings import format_settings, scrub_sensitive_info
from aider.history import ChatSummary
from aider.io import InputOutput
from aider.llm import litellm  # noqa: F401; properly init litellm on launch
from aider.mcp import load_mcp_servers
from aider.models import ModelSettings
from aider.onboarding import offer_openrouter_oauth, select_default_model
from aider.repo import ANY_GIT_ERROR, GitRepo
from aider.report import report_uncaught_exceptions
from aider.versioncheck import check_version, install_from_main_branch, install_upgrade
from aider.watch import FileWatcher

from .dump import dump  # noqa: F401


def check_config_files_for_yes(config_files):
    found = False
    for config_file in config_files:
        if Path(config_file).exists():
            try:
                with open(config_file, "r") as f:
                    for line in f:
                        if line.strip().startswith("yes:"):
                            print("Configuration error detected.")
                            print(f"The file {config_file} contains a line starting with 'yes:'")
                            print("Please replace 'yes:' with 'yes-always:' in this file.")
                            found = True
            except Exception:
                pass
    return found


def get_git_root():
    """Try and guess the git repo, since the conf.yml can be at the repo root"""
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.working_tree_dir
    except (git.InvalidGitRepositoryError, FileNotFoundError):
        return None


def guessed_wrong_repo(io, git_root, fnames, git_dname):
    """After we parse the args, we can determine the real repo. Did we guess wrong?"""

    try:
        check_repo = Path(GitRepo(io, fnames, git_dname).root).resolve()
    except (OSError,) + ANY_GIT_ERROR:
        return

    # we had no guess, rely on the "true" repo result
    if not git_root:
        return str(check_repo)

    git_root = Path(git_root).resolve()
    if check_repo == git_root:
        return

    return str(check_repo)


def make_new_repo(git_root, io):
    try:
        repo = git.Repo.init(git_root)
        check_gitignore(git_root, io, False)
    except ANY_GIT_ERROR as err:  # issue #1233
        io.tool_error(f"Unable to create git repo in {git_root}")
        io.tool_output(str(err))
        return

    io.tool_output(f"Git repository created in {git_root}")
    return repo


def setup_git(git_root, io):
    if git is None:
        return

    try:
        cwd = Path.cwd()
    except OSError:
        cwd = None

    repo = None

    if git_root:
        try:
            repo = git.Repo(git_root)
        except ANY_GIT_ERROR:
            pass
    elif cwd == Path.home():
        io.tool_warning(
            "You should probably run aider in your project's directory, not your home dir."
        )
        return
    elif cwd and io.confirm_ask(
        "No git repo found, create one to track aider's changes (recommended)?"
    ):
        git_root = str(cwd.resolve())
        repo = make_new_repo(git_root, io)

    if not repo:
        return

    try:
        user_name = repo.git.config("--get", "user.name") or None
    except git.exc.GitCommandError:
        user_name = None

    try:
        user_email = repo.git.config("--get", "user.email") or None
    except git.exc.GitCommandError:
        user_email = None

    if user_name and user_email:
        return repo.working_tree_dir

    with repo.config_writer() as git_config:
        if not user_name:
            git_config.set_value("user", "name", "Your Name")
            io.tool_warning('Update git name with: git config user.name "Your Name"')
        if not user_email:
            git_config.set_value("user", "email", "you@example.com")
            io.tool_warning('Update git email with: git config user.email "you@example.com"')

    return repo.working_tree_dir


def check_gitignore(git_root, io, ask=True):
    if not git_root:
        return

    try:
        repo = git.Repo(git_root)
        patterns_to_add = []

        if not repo.ignored(".aider"):
            patterns_to_add.append(".aider*")

        env_path = Path(git_root) / ".env"
        if env_path.exists() and not repo.ignored(".env"):
            patterns_to_add.append(".env")

        if not patterns_to_add:
            return

        gitignore_file = Path(git_root) / ".gitignore"
        if gitignore_file.exists():
            try:
                content = io.read_text(gitignore_file)
                if content is None:
                    return
                if not content.endswith("\n"):
                    content += "\n"
            except OSError as e:
                io.tool_error(f"Error when trying to read {gitignore_file}: {e}")
                return
        else:
            content = ""
    except ANY_GIT_ERROR:
        return

    if ask:
        io.tool_output("You can skip this check with --no-gitignore")
        if not io.confirm_ask(f"Add {', '.join(patterns_to_add)} to .gitignore (recommended)?"):
            return

    content += "\n".join(patterns_to_add) + "\n"

    try:
        io.write_text(gitignore_file, content)
        io.tool_output(f"Added {', '.join(patterns_to_add)} to .gitignore")
    except OSError as e:
        io.tool_error(f"Error when trying to write to {gitignore_file}: {e}")
        io.tool_output(
            "Try running with appropriate permissions or manually add these patterns to .gitignore:"
        )
        for pattern in patterns_to_add:
            io.tool_output(f"  {pattern}")


def check_streamlit_install(io):
    return utils.check_pip_install_extra(
        io,
        "streamlit",
        "You need to install the aider browser feature",
        ["aider-chat[browser]"],
    )


def write_streamlit_credentials():
    from streamlit.file_util import get_streamlit_file_path

    # See https://github.com/Aider-AI/aider/issues/772

    credential_path = Path(get_streamlit_file_path()) / "credentials.toml"
    if not os.path.exists(credential_path):
        empty_creds = '[general]\nemail = ""\n'

        os.makedirs(os.path.dirname(credential_path), exist_ok=True)
        with open(credential_path, "w") as f:
            f.write(empty_creds)
    else:
        print("Streamlit credentials already exist.")


def launch_gui(args):
    from streamlit.web import cli

    from aider import gui

    print()
    print("CONTROL-C to exit...")

    # Necessary so streamlit does not prompt the user for an email address.
    write_streamlit_credentials()

    target = gui.__file__

    st_args = ["run", target]

    st_args += [
        "--browser.gatherUsageStats=false",
        "--runner.magicEnabled=false",
        "--server.runOnSave=false",
    ]

    # https://github.com/Aider-AI/aider/issues/2193
    is_dev = "-dev" in str(__version__)

    if is_dev:
        print("Watching for file changes.")
    else:
        st_args += [
            "--global.developmentMode=false",
            "--server.fileWatcherType=none",
            "--client.toolbarMode=viewer",  # minimal?
        ]

    st_args += ["--"] + args

    cli.main(st_args)

    # from click.testing import CliRunner
    # runner = CliRunner()
    # from streamlit.web import bootstrap
    # bootstrap.load_config_options(flag_options={})
    # cli.main_run(target, args)
    # sys.argv = ['streamlit', 'run', '--'] + args


def parse_lint_cmds(lint_cmds, io):
    err = False
    res = dict()
    for lint_cmd in lint_cmds:
        if re.match(r"^[a-z]+:.*", lint_cmd):
            pieces = lint_cmd.split(":")
            lang = pieces[0]
            cmd = lint_cmd[len(lang) + 1 :]
            lang = lang.strip()
        else:
            lang = None
            cmd = lint_cmd

        cmd = cmd.strip()

        if cmd:
            res[lang] = cmd
        else:
            io.tool_error(f'Unable to parse --lint-cmd "{lint_cmd}"')
            io.tool_output('The arg should be "language: cmd --args ..."')
            io.tool_output('For example: --lint-cmd "python: flake8 --select=E9"')
            err = True
    if err:
        return
    return res


def generate_search_path_list(default_file, git_root, command_line_file):
    files = []
    files.append(Path.home() / default_file)  # homedir
    if git_root:
        files.append(Path(git_root) / default_file)  # git root
    files.append(default_file)
    if command_line_file:
        files.append(command_line_file)

    resolved_files = []
    for fn in files:
        try:
            resolved_files.append(Path(fn).resolve())
        except OSError:
            pass

    files = resolved_files
    files.reverse()
    uniq = []
    for fn in files:
        if fn not in uniq:
            uniq.append(fn)
    uniq.reverse()
    files = uniq
    files = list(map(str, files))
    files = list(dict.fromkeys(files))

    return files


def register_models(git_root, model_settings_fname, io, verbose=False):
    model_settings_files = generate_search_path_list(
        ".aider.model.settings.yml", git_root, model_settings_fname
    )

    try:
        files_loaded = models.register_models(model_settings_files)
        if len(files_loaded) > 0:
            if verbose:
                io.tool_output("Loaded model settings from:")
                for file_loaded in files_loaded:
                    io.tool_output(f"  - {file_loaded}")  # noqa: E221
        elif verbose:
            io.tool_output("No model settings files loaded")
    except Exception as e:
        io.tool_error(f"Error loading aider model settings: {e}")
        return 1

    if verbose:
        io.tool_output("Searched for model settings files:")
        for file in model_settings_files:
            io.tool_output(f"  - {file}")

    return None


def load_dotenv_files(git_root, dotenv_fname, encoding="utf-8"):
    # Standard .env file search path
    dotenv_files = generate_search_path_list(
        ".env",
        git_root,
        dotenv_fname,
    )

    # Explicitly add the OAuth keys file to the beginning of the list
    oauth_keys_file = Path.home() / ".aider" / "oauth-keys.env"
    if oauth_keys_file.exists():
        # Insert at the beginning so it's loaded first (and potentially overridden)
        dotenv_files.insert(0, str(oauth_keys_file.resolve()))
        # Remove duplicates if it somehow got included by generate_search_path_list
        dotenv_files = list(dict.fromkeys(dotenv_files))

    loaded = []
    for fname in dotenv_files:
        try:
            if Path(fname).exists():
                load_dotenv(fname, override=True, encoding=encoding)
                loaded.append(fname)
        except OSError as e:
            print(f"OSError loading {fname}: {e}")
        except Exception as e:
            print(f"Error loading {fname}: {e}")
    return loaded


def register_litellm_models(git_root, model_metadata_fname, io, verbose=False):
    model_metadata_files = []

    # Add the resource file path
    resource_metadata = importlib_resources.files("aider.resources").joinpath("model-metadata.json")
    model_metadata_files.append(str(resource_metadata))

    model_metadata_files += generate_search_path_list(
        ".aider.model.metadata.json", git_root, model_metadata_fname
    )

    try:
        model_metadata_files_loaded = models.register_litellm_models(model_metadata_files)
        if len(model_metadata_files_loaded) > 0 and verbose:
            io.tool_output("Loaded model metadata from:")
            for model_metadata_file in model_metadata_files_loaded:
                io.tool_output(f"  - {model_metadata_file}")  # noqa: E221
    except Exception as e:
        io.tool_error(f"Error loading model metadata models: {e}")
        return 1


def sanity_check_repo(repo, io):
    if not repo:
        return True

    if not repo.repo.working_tree_dir:
        io.tool_error("The git repo does not seem to have a working tree?")
        return False

    bad_ver = False
    try:
        repo.get_tracked_files()
        if not repo.git_repo_error:
            return True
        error_msg = str(repo.git_repo_error)
    except UnicodeDecodeError as exc:
        error_msg = (
            "Failed to read the Git repository. This issue is likely caused by a path encoded "
            f'in a format different from the expected encoding "{sys.getfilesystemencoding()}".\n'
            f"Internal error: {str(exc)}"
        )
    except ANY_GIT_ERROR as exc:
        error_msg = str(exc)
        bad_ver = "version in (1, 2)" in error_msg
    except AssertionError as exc:
        error_msg = str(exc)
        bad_ver = "version in (1, 2)" in error_msg

    io.tool_error(f"Error reading git repo: {error_msg}")
    if bad_ver:
        io.tool_error("Try running: git config core.repositoryformatversion 0")

    return False


def main(main_args=None, input_stream=None, output_stream=None):
    report_uncaught_exceptions()

    if main_args is None:
        main_args = sys.argv[1:]

    # We need to parse --env-file before we create the main parser,
    # because the main parser's defaults depend on the git_root, which
    # can be affected by the env file.
    preparser = get_parser([], None)
    pre_args, _ = preparser.parse_known_args(args=main_args)

    git_root = get_git_root()
    if git_root:
        os.environ["AIDER_GIT_ROOT"] = git_root

    # Load environment variables from file
    loaded_dotenv_files = load_dotenv_files(git_root, pre_args.env_file)

    # Now that we have the git_root, we can determine the default config files
    default_config_files = generate_search_path_list(".aider.conf.yml", git_root, None)

    # Check for deprecated 'yes: true' in config files
    if check_config_files_for_yes(default_config_files):
        return 1

    parser = get_parser(default_config_files, git_root)
    args = parser.parse_args(args=main_args)

    if args.retry_timeout:
        models.retry_timeout = args.retry_timeout
    if args.retry_backoff_factor:
        models.retry_backoff_factor = args.retry_backoff_factor

    # You can't have both dark and light mode
    if args.dark_mode and args.light_mode:
        print("Error: You can't specify both --dark-mode and --light-mode.")
        return 1

    if args.shell_completions:
        parser.prog = "aider"
        print(shtab.complete(parser, shell=args.shell_completions))
        return 0

    if args.upgrade:
        install_upgrade()
        return 0

    if args.install_main_branch:
        install_from_main_branch()
        return 0

    if args.just_check_update:
        return check_version(just_check=True)

    if args.gui:
        if not check_streamlit_install(InputOutput()):
            return 1
        launch_gui(main_args)
        return 0

    # Class to handle all user input and output
    io = InputOutput(
        pretty=args.pretty,
        yes=args.yes_always,
        input_history_file=args.input_history_file,
        chat_history_file=args.chat_history_file,
        llm_history_file=args.llm_history_file,
        input_stream=input_stream,
        output_stream=output_stream,
        encoding=args.encoding,
        user_input_color=args.user_input_color,
        tool_output_color=args.tool_output_color,
        tool_error_color=args.tool_error_color,
        tool_warning_color=args.tool_warning_color,
        assistant_output_color=args.assistant_output_color,
        code_theme=args.code_theme,
        dark_mode=args.dark_mode,
        light_mode=args.light_mode,
        line_endings=args.line_endings,
        vim_mode=args.vim,
        multiline_mode=args.multiline,
        notifications=args.notifications,
        notification_command=args.notifications_command,
        fancy_input=args.fancy_input,
        completion_menu_color=args.completion_menu_color,
        completion_menu_bg_color=args.completion_menu_bg_color,
        completion_menu_current_color=args.completion_menu_current_color,
        completion_menu_current_bg_color=args.completion_menu_current_bg_color,
        editor=args.editor,
    )

    if args.verbose:
        io.tool_output("Loaded .env files:")
        for fname in loaded_dotenv_files:
            io.tool_output(f"- {fname}")

    # Set up analytics
    analytics = Analytics(
        args.analytics,
        args.analytics_log,
        args.analytics_disable,
        args.analytics_posthog_host,
        args.analytics_posthog_project_api_key,
    )
    analytics.event("main", "start")

    # Set SSL verification
    if not args.verify_ssl:
        os.environ["REQUESTS_CA_BUNDLE"] = ""
        os.environ["SSL_CERT_FILE"] = ""
        litellm.ssl_verify = False
        models.model_info_manager.set_verify_ssl(False)

    # Set timeout
    if args.timeout:
        models.request_timeout = args.timeout

    # Set environment variables
    for env_var in args.set_env:
        try:
            name, value = env_var.split("=", 1)
            os.environ[name] = value
        except ValueError:
            io.tool_error(f"Invalid format for --set-env: {env_var}. Use NAME=value.")
            return 1

    # Set API keys
    for api_key_arg in args.api_key:
        try:
            provider, key = api_key_arg.split("=", 1)
            env_var_name = f"{provider.upper()}_API_KEY"
            os.environ[env_var_name] = key
        except ValueError:
            io.tool_error(f"Invalid format for --api-key: {api_key_arg}. Use PROVIDER=key.")
            return 1

    # Handle deprecated model arguments
    handle_deprecated_model_args(args, io)

    # Register models
    if register_models(git_root, args.model_settings_file, io, args.verbose):
        return 1

    # Register litellm models
    if register_litellm_models(git_root, args.model_metadata_file, io, args.verbose):
        return 1

    # Handle model aliases
    if args.alias:
        for alias_arg in args.alias:
            try:
                alias, model_name = alias_arg.split(":", 1)
                models.MODEL_ALIASES[alias] = model_name
            except ValueError:
                io.tool_error(f"Invalid format for --alias: {alias_arg}. Use ALIAS:MODEL_NAME.")
                return 1

    if args.list_models:
        models.print_matching_models(io, args.list_models)
        return 0

    if args.check_update:
        check_version(io)

    if args.show_release_notes is not None:
        io.user_settings.set("show_release_notes", args.show_release_notes)

    if args.yes_always is not None:
        io.user_settings.set("yes_always", args.yes_always)

    if args.dark_mode:
        io.user_settings.set("color_scheme", "dark")
    elif args.light_mode:
        io.user_settings.set("color_scheme", "light")

    if args.vim:
        io.user_settings.set("editing_mode", EditingMode.VI)
    else:
        io.user_settings.set("editing_mode", EditingMode.EMACS)

    if args.multiline:
        io.user_settings.set("multiline_mode", True)
    else:
        io.user_settings.set("multiline_mode", False)

    if args.notifications:
        io.user_settings.set("notifications", True)
    else:
        io.user_settings.set("notifications", False)

    if args.notification_command:
        io.user_settings.set("notification_command", args.notification_command)

    if args.fancy_input:
        io.user_settings.set("fancy_input", True)
    else:
        io.user_settings.set("fancy_input", False)

    if args.editor:
        io.user_settings.set("editor", args.editor)

    if args.verbose:
        io.tool_output(f"Aider v{__version__}")
        io.tool_output(f"Default config files: {default_config_files}")
        io.tool_output(f"Loaded config file: {parser.get_config_file_paths()}")
        io.tool_output(format_settings(args, scrub_sensitive=False))

    fnames = args.files
    if args.file:
        fnames += args.file

    if not fnames and (args.lint or args.commit):
        args.git = True

    if args.git:
        git_dname = setup_git(git_root, io)
        if not git_dname:
            args.git = False
        else:
            git_root = git_dname

    # Now that we have the real git_root, check if our guess was wrong
    # and we need to reload the config file.
    wrong_repo = guessed_wrong_repo(io, git_root, fnames, git_dname)
    if wrong_repo:
        git_root = wrong_repo
        os.environ["AIDER_GIT_ROOT"] = git_root
        default_config_files = generate_search_path_list(".aider.conf.yml", git_root, None)
        parser = get_parser(default_config_files, git_root)
        args = parser.parse_args(args=main_args)

    if args.git and args.gitignore:
        check_gitignore(git_root, io)

    if not args.model:
        args.model = select_default_model(args, io, analytics)
        if not args.model:
            offer_openrouter_oauth(io, analytics)
            return 1

    try:
        main_model = models.Model(
            args.model,
            args.weak_model,
            args.editor_model,
            args.editor_edit_format,
            verbose=args.verbose,
        )
    except ValueError as e:
        io.tool_error(str(e))
        return 1

    if args.show_model_warnings:
        if models.sanity_check_models(io, main_model):
            io.tool_output("Use --no-show-model-warnings to proceed.")
            return 1

    if args.check_model_accepts_settings:
        if args.reasoning_effort and "reasoning_effort" not in main_model.accepts_settings:
            io.tool_error(
                f"Model {main_model.name} does not support reasoning_effort. Use"
                " --no-check-model-accepts-settings to proceed."
            )
            return 1
        if args.thinking_tokens and "thinking_tokens" not in main_model.accepts_settings:
            io.tool_error(
                f"Model {main_model.name} does not support thinking_tokens. Use"
                " --no-check-model-accepts-settings to proceed."
            )
            return 1

    if args.reasoning_effort:
        main_model.set_reasoning_effort(args.reasoning_effort)

    if args.thinking_tokens:
        main_model.set_thinking_tokens(args.thinking_tokens)

    if args.max_chat_history_tokens:
        main_model.max_chat_history_tokens = args.max_chat_history_tokens

    if args.map_tokens is None:
        args.map_tokens = main_model.get_repo_map_tokens()

    repo = None
    if args.git:
        try:
            repo = GitRepo(
                io,
                fnames,
                git_root,
                aider_ignore_file=args.aiderignore,
                models=main_model.commit_message_models(),
                attribute_author=args.attribute_author,
                attribute_committer=args.attribute_committer,
                attribute_commit_message_author=args.attribute_commit_message_author,
                attribute_commit_message_committer=args.attribute_commit_message_committer,
                attribute_co_authored_by=args.attribute_co_authored_by,
                commit_prompt=args.commit_prompt,
                git_commit_verify=args.git_commit_verify,
                subtree_only=args.subtree_only,
            )
            if not args.skip_sanity_check_repo and not sanity_check_repo(repo, io):
                return 1
        except (OSError,) + ANY_GIT_ERROR as e:
            io.tool_error(f"Error setting up git repo: {e}")
            return 1

    if args.lint:
        if not fnames and repo:
            fnames = repo.get_dirty_files()
            if not fnames:
                io.tool_output("No dirty files to lint.")
                return 0

        if not fnames:
            io.tool_error("No files to lint.")
            return 1

    if args.test:
        if not args.test_cmd:
            io.tool_error("No test command provided. Use --test-cmd to specify.")
            return 1

    if args.commit:
        if not repo:
            io.tool_error("Not in a git repo, cannot commit.")
            return 1
        repo.commit(coder=None, io=io)
        return 0

    if args.verbose:
        io.tool_output(main_model.commit_message_models())

    summarizer = ChatSummary(
        [main_model.weak_model, main_model], main_model.max_chat_history_tokens
    )

    # Initialize MCP servers
    mcp_servers = None
    if args.mcp_servers or args.mcp_servers_file:
        mcp_servers = load_mcp_servers(
            args.mcp_servers, args.mcp_servers_file, args.mcp_transport, io
        )

    # Initialize file watcher
    file_watcher = None
    if args.watch_files:
        file_watcher = FileWatcher()

    coder = Coder.create(
        main_model,
        args.edit_format,
        io,
        repo,
        fnames,
        add_gitignore_files=args.add_gitignore_files,
        read_only_fnames=args.read,
        show_diffs=args.show_diffs,
        auto_commits=args.auto_commits,
        dirty_commits=args.dirty_commits,
        dry_run=args.dry_run,
        map_tokens=args.map_tokens,
        map_mul_no_files=args.map_multiplier_no_files,
        map_max_line_length=args.map_max_line_length,
        verbose=args.verbose,
        stream=args.stream,
        use_git=args.git,
        summarizer=summarizer,
        analytics=analytics,
        map_refresh=args.map_refresh,
        cache_prompts=args.cache_prompts,
        num_cache_warming_pings=args.cache_keepalive_pings,
        suggest_shell_commands=args.suggest_shell_commands,
        chat_language=args.chat_language,
        commit_language=args.commit_language,
        detect_urls=args.detect_urls,
        file_watcher=file_watcher,
        auto_copy_context=args.copy_paste,
        auto_accept_architect=args.auto_accept_architect,
        mcp_servers=mcp_servers,
        enable_context_compaction=args.enable_context_compaction,
        context_compaction_max_tokens=args.context_compaction_max_tokens,
        context_compaction_summary_tokens=args.context_compaction_summary_tokens,
        map_cache_dir=args.map_cache_dir,
    )

    if args.apply:
        content = io.read_text(args.apply)
        if content is None:
            return 1
        coder.apply_updates(content)
        return 0

    if args.apply_clipboard_edits:
        try:
            import pyperclip
        except ImportError:
            io.tool_error(
                "To use clipboard edits, you need to install pyperclip: pip install pyperclip"
            )
            return 1
        content = pyperclip.paste()
        if not content:
            io.tool_error("Clipboard is empty.")
            return 1
        coder.partial_response_content = content
        coder.apply_updates()
        return 0

    if args.show_repo_map:
        coder.get_repo_map()
        return 0

    if args.show_prompts:
        coder.format_chat_chunks()
        return 0

    if args.lint:
        lint_cmds = parse_lint_cmds(args.lint_cmd, io)
        if lint_cmds is None:
            return 1
        coder.setup_lint_cmds(lint_cmds)
        coder.lint_edited(fnames)
        return 0

    if args.test:
        coder.reflected_message = coder.commands.cmd_test(args.test_cmd)
        if coder.reflected_message:
            io.tool_output("Fixing test errors...")
            coder.run()
        return 0

    if args.load:
        try:
            with open(args.load, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    io.tool_output(f"> {line}")
                    coder.run(line)
        except FileNotFoundError:
            io.tool_error(f"Could not find file to load: {args.load}")
            return 1

    if args.exit:
        return 0

    coder.show_announcements()

    if args.message or args.message_file:
        if args.message:
            message = args.message
        else:
            try:
                message = Path(args.message_file).read_text()
            except FileNotFoundError:
                io.tool_error(f"Could not find file: {args.message_file}")
                return 1

        io.tool_output()
        coder.run(with_message=message)
        return 0

    # Start the file watcher if enabled
    if file_watcher:
        file_watcher.start()

    # Start the clipboard watcher if enabled
    clipboard_watcher = None
    if args.copy_paste:
        clipboard_watcher = ClipboardWatcher(io)
        clipboard_watcher.start()

    try:
        coder.run()
    except SwitchCoder as e:
        e.kwargs["fnames"] = list(coder.abs_fnames)
        e.kwargs["read_only_fnames"] = list(coder.abs_read_only_fnames)
        e.kwargs["done_messages"] = coder.done_messages
        e.kwargs["cur_messages"] = coder.cur_messages
        e.kwargs["aider_commit_hashes"] = coder.aider_commit_hashes
        e.kwargs["commands"] = coder.commands.clone()
        e.kwargs["total_cost"] = coder.total_cost
        e.kwargs["file_watcher"] = file_watcher

        coder = Coder.create(
            from_coder=coder,
            **e.kwargs,
        )
        coder.run()
    except Exception:
        # Display the traceback
        traceback.print_exc()
        # Optionally, you can log the exception to a file or another service
        # for further analysis.
        # For example:
        # with open("error_log.txt", "a") as f:
        #     traceback.print_exc(file=f)
    finally:
        # Stop the file watcher if it was started
        if file_watcher:
            file_watcher.stop()
        # Stop the clipboard watcher if it was started
        if clipboard_watcher:
            clipboard_watcher.stop()

        analytics.event("main", "exit")

    return 0


if __name__ == "__main__":
    status = main()
    sys.exit(status)
