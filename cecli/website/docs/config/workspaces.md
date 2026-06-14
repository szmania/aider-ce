---
parent: Configuration
nav_order: 41
description: Workspaces allow you to work across multiple related repositories simultaneously
---
# Workspaces

Workspaces allow you to manage multiple git repositories within a single monorepo-like folder structure, enabling development across multiple related projects. `cecli` supports two workspace modes: 

**clone** workspaces (remote `repo:` URLs cloned into `~/.cecli/workspaces/`)

**local** workspaces (existing on-disk git roots referenced by absolute `path:`)

## Configuration

You can configure workspaces in multiple locations. `cecli` searches for configurations in the following order:

1. **CLI Argument**: Via a JSON/YAML configuration or file path passed to the `--workspaces` argument.
2. **Local Workspaces File**: `.cecli.workspaces.yml` or `.cecli.workspaces.yaml` in the current directory.
3. **Global Workspaces File**: `~/.cecli/workspaces.yml` or `~/.cecli/workspaces.yaml`.

4. **Repo-Local Config File**: `.cecli.workspaces.yml` or `.cecli.workspaces.yaml` placed at a common ancestor of your project directories. `cecli` discovers this file by walking up from any project path, enabling a **local** workspace layout without cloning into `~/.cecli/workspaces/`.

### Example Configuration

```yaml
workspaces:
  name: "my-workspace"
  projects:
    - name: "frontend"
      repo: "https://github.com/user/frontend.git"
      branch: "main"
      worktrees:
        - name: "feature-auth"
          branch: "feature/auth"
    - name: "backend"
      repo: "https://github.com/user/backend.git"
      branch: "develop"
      use_current_branch: true  # Default: true. Set to false to force branch switching on init.
      ignore: "~/.cecli/backend.ignore" # Optional: Path to a custom ignore file for this project
```

### Local Workspace Configuration

For **local** workspaces, place a `.cecli.workspaces.yml` file at a common ancestor of your project directories. Each project references an existing git root via `path:` instead of a remote `repo:` URL.

```yaml
# .cecli.workspaces.yml
name: my-workspace
projects:
  - name: app
    path: /abs/path/to/app
    primary: true        # At most one project can be primary
  - name: lib
    path: /abs/path/to/lib
    readonly: true       # Prevents commits to this project
```

**Validation rules:**

- Each project must have a `name` and **exactly one** of `path` (local git root) or `repo` (clone URL).
- At most one project can be marked `primary: true`.
- Projects with `readonly: true` are excluded from commits.

### Path Layout

The workspace layout determines how file paths are structured within the workspace:

| Layout | Prefix | Example |
|--------|--------|--------|
| **clone** (repo-based) | `{project}/main/{file}` | `app/main/src/main.py` |
| **local** (path-based) | `{project}/{file}` | `app/src/main.py` |


### Multiple Workspaces

You can define a list of workspaces. Use the `active: true` flag to specify which one should be used by default when running `cecli` without the `--workspace-name` argument. **Note: At most one workspace can be marked as active.**
```yaml
workspaces:
  - name: "project-a"
    active: true
    projects:
      - name: "app"
        repo: "https://github.com/user/app.git"
  - name: "project-b"
    projects:
      - name: "api"
        repo: "https://github.com/user/api.git"
```


## Usage

To use a workspace:

```bash
cecli --workspace-name my-workspace
# OR if using a specific config file
cecli --workspaces path/to/workspaces.yml --workspace-name my-workspace
```

If the workspace does not exist, `cecli` will create the directory structure at `~/.cecli/workspaces/my-workspace/` and clone the configured repositories. For **local** workspaces, the configured `path:` directories are used in-place — no cloning occurs.

### Clone Workspace Structure

```
~/.cecli/workspaces/
└── workspace-name/
    ├── .cecli-workspace.json
    └── project-name/
        ├── main/        # Main repository clone
        └── worktrees/   # Additional worktrees
```

### Local Workspace Structure

Local workspaces do **not** create a `~/.cecli/workspaces/` directory. Instead, the config file directory itself serves as the workspace root, with metadata stored at:

```
.cecli/
└── .workspace-meta.json
```

The project directories exist at their configured `path:` locations on disk.

## Arguments

`--workspaces <file>`: Provide a JSON/YAML configuration or file path for workspace initialization.

`--workspace-name <name>`: Specify the workspace name to activate from the configuration.
