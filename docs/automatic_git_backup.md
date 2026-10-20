# Automatic Git backup in VS Code

Opening this trusted workspace starts **Auto commit and push (10 min)**.
The first check runs after 10 minutes; subsequent checks run every 10 minutes.
Only saved files are backed up. Closing the workspace stops its task.
This uses macOS `/usr/bin/python3` and `/usr/bin/git`, independently of Conda.

The task commits changes on `main` and pushes to its existing `origin/main`
upstream. Commit messages use `backup: YYYY-MM-DD HH:MM:SS +ZZZZ`.
It includes `src/`, `tests/`, `docs/`, `scripts/`, `Paper/`, `.vscode/`,
`README.md`, `environment.yml`, `pyproject.toml`, and `.gitignore`.
Git ignore rules apply to untracked files. Experiment outputs under `results/`,
archives, build directories, LaTeX build products, and Overleaf metadata are
excluded. PDF figures inside a `figures/` directory are included.

When changes are manually staged, or a merge/rebase/conflict is in progress,
the task skips the check. It also skips other branches or a changed upstream.
Push failures are retried on the next check, including when no new files change.
It never pulls, rebases, or force pushes automatically; resolve remote divergence
manually. A failed commit hook can leave the task's changes staged: review and
commit or unstage those changes before automatic backup can continue.

To start immediately after configuring it, run **Tasks: Run Task** from the
Command Palette and select **Auto commit and push (10 min)**, or reopen the
project. View its messages in the task's terminal. To stop the current task,
use **Tasks: Terminate Task**. To disable future startup, remove
`"runOn": "folderOpen"` from this task in `.vscode/tasks.json`.

Preview the next commit without changing Git or contacting the remote:

```sh
/usr/bin/python3 scripts/auto-git-sync.py --dry-run
```

VS Code task startup behavior is documented in the
[official Tasks guide](https://code.visualstudio.com/docs/debugtest/tasks#_run-behavior).
