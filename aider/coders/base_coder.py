#!/usr/bin/env python

import asyncio
import base64
import hashlib
import json
import locale
import math
import mimetypes
import os
import platform
import re
import sys
import threading
import time
import traceback
import importlib.util
import inspect
from collections import defaultdict
from datetime import datetime

# Optional dependency: used to convert locale codes (eg ``en_US``)
# into human-readable language names (eg ``English``).
try:
    from babel import Locale  # type: ignore
except ImportError:  # Babel not installed – we will fall back to a small mapping
    Locale = None
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import List

from litellm import experimental_mcp_client
from rich.console import Console

from aider import __version__, models, prompts, urls, utils
from aider.analytics import Analytics
from aider.commands import Commands
from aider.exceptions import EmptyLLMResponseError, LiteLLMExceptions
from aider.history import ChatSummary
from aider.io import ConfirmGroup, InputOutput
from aider.linter import Linter
from aider.llm import litellm
from aider.mcp.server import LocalServer
from aider.models import RETRY_TIMEOUT
from aider.reasoning_tags import (
    REASONING_TAG,
    format_reasoning_content,
    remove_reasoning_content,
    replace_reasoning_tags,
)
from aider.repo import ANY_GIT_ERROR
from aider.repomap import RepoMap
from aider.run_cmd import run_cmd
from aider.utils import format_content, format_messages, format_tokens, is_image_file
from aider.waiting import WaitingSpinner

from ..dump import dump  # noqa: F401
from .chat_chunks import ChatChunks


class UnknownEditFormat(ValueError):
    def __init__(self, edit_format, valid_formats):
        self.edit_format = edit_format
        self.valid_formats = valid_formats
        super().__init__(
            f"Unknown edit format {edit_format}. Valid formats are: {', '.join(valid_formats)}"
        )


class MissingAPIKeyError(ValueError):
    pass


class FinishReasonLength(Exception):
    pass


def wrap_fence(name):
    return f"<{name}>", f"</{name}>"


all_fences = [
    ("`" * 3, "`" * 3),
    ("`" * 4, "`" * 4),  # LLMs ignore and revert to triple-backtick, causing #2879
    wrap_fence("source"),
    wrap_fence("code"),
    wrap_fence("pre"),
    wrap_fence("codeblock"),
    wrap_fence("sourcecode"),
]


class Coder:
    abs_fnames = None
    abs_read_only_fnames = None
    repo = None
    last_aider_commit_hash = None
    aider_edited_files = None
    last_asked_for_commit_time = 0
    repo_map = None
    functions = None
    num_exhausted_context_windows = 0
    num_malformed_responses = 0
    last_keyboard_interrupt = None
    num_reflections = 0
    max_reflections = 3
    num_tool_calls = 0
    max_tool_calls = 25
    edit_format = None
    yield_stream = False
    temperature = None
    auto_lint = True
    auto_test = False
    test_cmd = None
    lint_outcome = None
    test_outcome = None
    multi_response_content = ""
    partial_response_content = ""
    partial_response_tool_call = []
    commit_before_message = []
    message_cost = 0.0
    add_cache_headers = False
    cache_warming_thread = None
    num_cache_warming_pings = 0
    suggest_shell_commands = True
    detect_urls = True
    ignore_mentions = None
    chat_language = None
    commit_language = None
    file_watcher = None
    mcp_servers = None
    mcp_tools = None
    did_discover_tools = False # New class attribute for session-wide tool discovery

    # Context management settings (for all modes)
    context_management_enabled = False  # Disabled by default except for navigator mode
    large_file_token_threshold = (
        25000  # Files larger than this will be truncated when context management is enabled
    )

    @classmethod
    def create(
        self,
        main_model=None,
        edit_format=None,
        io=None,
        from_coder=None,
        summarize_from_coder=True,
        **kwargs,
    ):
        import aider.coders as coders

        if not main_model:
            if from_coder:
                main_model = from_coder.main_model
            else:
                main_model = models.Model(models.DEFAULT_MODEL_NAME)

        if edit_format == "code":
            edit_format = None
        if edit_format is None:
            if from_coder:
                edit_format = from_coder.edit_format
            else:
                edit_format = main_model.edit_format

        if not io and from_coder:
            io = from_coder.io

        if from_coder:
            use_kwargs = dict(from_coder.original_kwargs)  # copy orig kwargs

            # If the edit format changes, we can't leave old ASSISTANT
            # messages in the chat history. The old edit format will
            # confused the new LLM. It may try and imitate it, disobeying
            # the system prompt.
            done_messages = from_coder.done_messages
            if edit_format != from_coder.edit_format and done_messages and summarize_from_coder:
                try:
                    done_messages = from_coder.summarizer.summarize_all(done_messages)
                except ValueError:
                    # If summarization fails, keep the original messages and warn the user
                    io.tool_warning(
                        "Chat history summarization failed, continuing with full history"
                    )

            # Bring along context from the old Coder
            update = dict(
                fnames=list(from_coder.abs_fnames),
                # Copy read-only files
                read_only_fnames=list(from_coder.abs_read_only_fnames),
                done_messages=done_messages,
                cur_messages=from_coder.cur_messages,
                aider_commit_hashes=from_coder.aider_commit_hashes,
                commands=from_coder.commands.clone(),
                total_cost=from_coder.total_cost,
                ignore_mentions=from_coder.ignore_mentions,
                total_tokens_sent=from_coder.total_tokens_sent,
                total_tokens_received=from_coder.total_tokens_received,
                file_watcher=from_coder.file_watcher,
                mcp_servers=from_coder.mcp_servers,
                mcp_tools=from_coder.mcp_tools,
                did_discover_tools=from_coder.did_discover_tools, # Pass the new flag
            )
            use_kwargs.update(update)  # override to complete the switch
            use_kwargs.update(kwargs)  # override passed kwargs

            kwargs = use_kwargs
            from_coder.ok_to_warm_cache = False

        for coder in coders.__all__:
            if hasattr(coder, "edit_format") and coder.edit_format == edit_format:
                res = coder(main_model, io, **kwargs)
                res.original_kwargs = dict(kwargs)
                return res

        valid_formats = [
            str(c.edit_format)
            for c in coders.__all__
            if hasattr(c, "edit_format") and c.edit_format is not None
        ]
        raise UnknownEditFormat(edit_format, valid_formats)

    def clone(self, **kwargs):
        new_coder = Coder.create(from_coder=self, **kwargs)
        return new_coder

    def get_announcements(self):
        lines = []
        lines.append(f"Aider v{__version__}")

        # Model
        main_model = self.main_model
        weak_model = main_model.weak_model

        if weak_model is not main_model:
            prefix = "Main model"
        else:
            prefix = "Model"

        output = f"{prefix}: {main_model.name} with {self.edit_format} edit format"

        # Check for thinking token budget
        thinking_tokens = main_model.get_thinking_tokens()
        if thinking_tokens:
            output += f", {thinking_tokens} think tokens"

        # Check for reasoning effort
        reasoning_effort = main_model.get_reasoning_effort()
        if reasoning_effort:
            output += f", reasoning {reasoning_effort}"

        if self.add_cache_headers or main_model.caches_by_default:
            output += ", prompt cache"
        if main_model.info.get("supports_assistant_prefill"):
            output += ", infinite output"

        lines.append(output)

        if self.edit_format == "architect":
            output = (
                f"Editor model: {main_model.editor_model.name} with"
                f" {main_model.editor_edit_format} edit format"
            )
            lines.append(output)

        if weak_model is not main_model:
            output = f"Weak model: {weak_model.name}"
            lines.append(output)

        # Repo
        if self.repo:
            rel_repo_dir = self.repo.get_rel_repo_dir()
            num_files = len(self.repo.get_tracked_files())

            lines.append(f"Git repo: {rel_repo_dir} with {num_files:,} files")
            if num_files > 1000:
                lines.append(
                    "Warning: For large repos, consider using --subtree-only and .aiderignore"
                )
                lines.append(f"See: {urls.large_repos}")
        else:
            lines.append("Git repo: none")

        # Repo-map
        if self.repo_map:
            map_tokens = self.repo_map.max_map_tokens
            if map_tokens > 0:
                refresh = self.repo_map.refresh
                lines.append(f"Repo-map: using {map_tokens} tokens, {refresh} refresh")
                max_map_tokens = self.main_model.get_repo_map_tokens() * 2
                if map_tokens > max_map_tokens:
                    lines.append(
                        f"Warning: map-tokens > {max_map_tokens} is not recommended. Too much"
                        " irrelevant code can confuse LLMs."
                    )
            else:
                lines.append("Repo-map: disabled because map_tokens == 0")
        else:
            lines.append("Repo-map: disabled")

        # Files
        for fname in self.get_inchat_relative_files():
            lines.append(f"Added {fname} to the chat.")

        for fname in self.abs_read_only_fnames:
            rel_fname = self.get_rel_fname(fname)
            lines.append(f"Added {rel_fname} to the chat (read-only).")

        if self.done_messages:
            lines.append("Restored previous conversation history.")

        if self.io.multiline_mode:
            lines.append("Multiline mode: Enabled. Enter inserts newline, Alt-Enter submits text")

        return lines

    ok_to_warm_cache = False

    def __init__(
        self,
        main_model,
        io,
        repo=None,
        fnames=None,
        add_gitignore_files=False,
        read_only_fnames=None,
        show_diffs=False,
        auto_commits=True,
        dirty_commits=True,
        dry_run=False,
        map_tokens=1024,
        verbose=False,
        stream=True,
        use_git=True,
        cur_messages=None,
        done_messages=None,
        restore_chat_history=False,
        auto_lint=True,
        auto_test=False,
        lint_cmds=None,
        test_cmd=None,
        aider_commit_hashes=None,
        map_mul_no_files=8,
        map_max_line_length=100,
        commands=None,
        summarizer=None,
        total_cost=0.0,
        analytics=None,
        map_refresh="auto",
        cache_prompts=False,
        num_cache_warming_pings=0,
        suggest_shell_commands=True,
        chat_language=None,
        commit_language=None,
        detect_urls=True,
        ignore_mentions=None,
        total_tokens_sent=0,
        total_tokens_received=0,
        file_watcher=None,
        auto_copy_context=False,
        auto_accept_architect=True,
        mcp_servers=None,
        mcp_tools=None,
        enable_context_compaction=False,
        context_compaction_max_tokens=None,
        context_compaction_summary_tokens=8192,
        map_cache_dir=".",
        retry_timeout=RETRY_TIMEOUT,
        retry_backoff_factor=2.0,
        retry_on_unavailable=False,
        did_discover_tools=False, # New parameter for session-wide tool discovery
    ):
        # initialize from args.map_cache_dir
        self.map_cache_dir = map_cache_dir
        self.retry_timeout = retry_timeout
        self.retry_backoff_factor = retry_backoff_factor
        self.retry_on_unavailable = retry_on_unavailable

        # Fill in a dummy Analytics if needed, but it is never .enable()'d
        self.analytics = analytics if analytics is not None else Analytics()

        self.event = self.analytics.event
        self.chat_language = chat_language
        self.commit_language = commit_language
        self.commit_before_message = []
        self.aider_commit_hashes = set()
        self.rejected_urls = set()
        self.abs_root_path_cache = {}

        self.auto_copy_context = auto_copy_context
        self.auto_accept_architect = auto_accept_architect

        self.ignore_mentions = ignore_mentions
        if not self.ignore_mentions:
            self.ignore_mentions = set()

        self.file_watcher = file_watcher
        if self.file_watcher:
            self.file_watcher.coder = self

        self.suggest_shell_commands = suggest_shell_commands
        self.detect_urls = detect_urls

        self.num_cache_warming_pings = num_cache_warming_pings
        self.mcp_servers = mcp_servers
        self.mcp_tools = mcp_tools
        self.enable_context_compaction = enable_context_compaction

        self.context_compaction_max_tokens = context_compaction_max_tokens
        self.context_compaction_summary_tokens = context_compaction_summary_tokens

        if not fnames:
            fnames = []

        if io is None:
            io = InputOutput()

        if aider_commit_hashes:
            self.aider_commit_hashes = aider_commit_hashes
        else:
            self.aider_commit_hashes = set()

        self.chat_completion_call_hashes = []
        self.chat_completion_response_hashes = []
        self.need_commit_before_edits = set()

        self.total_cost = total_cost
        self.total_tokens_sent = total_tokens_sent
        self.total_tokens_received = total_tokens_received
        self.message_tokens_sent = 0
        self.message_tokens_received = 0

        self.verbose = verbose
        self.abs_fnames = set()
        self.abs_read_only_fnames = set()
        self.add_gitignore_files = add_gitignore_files

        if cur_messages:
            self.cur_messages = cur_messages
        else:
            self.cur_messages = []

        if done_messages:
            self.done_messages = done_messages
        else:
            self.done_messages = []

        self.io = io

        self.shell_commands = []

        if not auto_commits:
            dirty_commits = False

        self.auto_commits = auto_commits
        self.dirty_commits = dirty_commits

        self.dry_run = dry_run
        self.pretty = self.io.pretty

        self.main_model = main_model
        # Set the reasoning tag name based on model settings or default
        self.reasoning_tag_name = (
            self.main_model.reasoning_tag if self.main_model.reasoning_tag else REASONING_TAG
        )

        self.stream = stream and main_model.streaming

        if cache_prompts and self.main_model.cache_control:
            self.add_cache_headers = True

        self.show_diffs = show_diffs

        self.commands = commands or Commands(self.io, self)
        self.commands.coder = self

        self.repo = repo
        if use_git and self.repo is None:
            try:
                self.repo = GitRepo(
                    self.io,
                    fnames,
                    None,
                    models=main_model.commit_message_models(),
                )
            except FileNotFoundError:
                pass

        if self.repo:
            self.root = self.repo.root

        for fname in fnames:
            fname = Path(fname)
            if self.repo and self.repo.git_ignored_file(fname) and not self.add_gitignore_files:
                self.io.tool_warning(f"Skipping {fname} that matches gitignore spec.")
                continue

            if self.repo and self.repo.ignored_file(fname):
                self.io.tool_warning(f"Skipping {fname} that matches aiderignore spec.")
                continue

            if not fname.exists():
                if utils.touch_file(fname):
                    self.io.tool_output(f"Creating empty file {fname}")
                else:
                    self.io.tool_warning(f"Can not create {fname}, skipping.")
                    continue

            if not fname.is_file():
                self.io.tool_warning(f"Skipping {fname} that is not a normal file.")
                continue

            fname = str(fname.resolve())

            self.abs_fnames.add(fname)
            self.check_added_files()

        if not self.repo:
            self.root = utils.find_common_root(self.abs_fnames)

        if read_only_fnames:
            self.abs_read_only_fnames = set()
            for fname in read_only_fnames:
                abs_fname = self.abs_root_path(fname)
                if os.path.exists(abs_fname):
                    self.abs_read_only_fnames.add(abs_fname)
                else:
                    self.io.tool_warning(f"Error: Read-only file {fname} does not exist. Skipping.")

        if map_tokens is None:
            use_repo_map = main_model.use_repo_map
            map_tokens = 1024
        else:
            use_repo_map = map_tokens > 0

        max_inp_tokens = self.main_model.info.get("max_input_tokens") or 0

        has_map_prompt = hasattr(self, "gpt_prompts") and self.gpt_prompts.repo_content_prefix

        if use_repo_map and self.repo and has_map_prompt:
            self.repo_map = RepoMap(
                map_tokens,
                self.map_cache_dir,
                self.main_model,
                io,
                self.gpt_prompts.repo_content_prefix,
                self.verbose,
                max_inp_tokens,
                map_mul_no_files=map_mul_no_files,
                refresh=map_refresh,
                max_code_line_length=map_max_line_length,
            )

        self.summarizer = summarizer or ChatSummary(
            [self.main_model.weak_model, self.main_model],
            self.main_model.max_chat_history_tokens,
        )

        self.summarizer_thread = None
        self.summarized_done_messages = []
        self.summarizing_messages = None

        if not self.done_messages and restore_chat_history:
            history_md = self.io.read_text(self.io.chat_history_file)
            if history_md:
                self.done_messages = utils.split_chat_history_markdown(history_md)
                self.summarize_start()

        # Linting and testing
        self.linter = Linter(root=self.root, encoding=io.encoding)
        self.auto_lint = auto_lint
        self.setup_lint_cmds(lint_cmds)
        self.lint_cmds = lint_cmds
        self.auto_test = auto_test
        self.test_cmd = test_cmd

        # Instantiate MCP tools
        if self.mcp_servers:
            self.initialize_mcp_tools()
        # validate the functions jsonschema
        if self.functions:
            from jsonschema import Draft7Validator

            for function in self.functions:
                Draft7Validator.check_schema(function)

            if self.verbose:
                self.io.tool_output("JSON Schema:")
                self.io.tool_output(json.dumps(self.functions, indent=4))

        self.did_discover_tools = did_discover_tools # Set instance attribute
