#!/usr/bin/env python3
"""Back up saved project sources to origin/main from a VS Code background task."""

import argparse
import datetime
import fcntl
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = {"src", "tests", "docs", "scripts", "Paper", ".vscode"}
ROOT_FILES = {"README.md", "environment.yml", "pyproject.toml", ".gitignore"}
EXCLUDED_DIRS = {"build", "dist", "__pycache__", ".overleaf", ".venv"}
EXCLUDED_SUFFIXES = {
    ".zip", ".tar", ".gz", ".7z", ".rar", ".aux", ".bbl", ".blg",
    ".fdb_latexmk", ".fls", ".log", ".nav", ".out", ".snm", ".synctex",
    ".toc", ".vrb", ".xdv", ".dvi", ".pyc", ".tmp", ".bak",
}
ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}


def log(message):
    print("[{}] {}".format(datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                          message), flush=True)


def git(*args, check=True):
    return subprocess.run(
        ["/usr/bin/git", *args], cwd=ROOT, env=ENV, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120, check=check,
    )


def source_file(name):
    path = Path(name)
    if path.parts[0] not in SOURCE_DIRS and name not in ROOT_FILES:
        return False
    if EXCLUDED_DIRS.intersection(path.parts) or path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if path.suffix.lower() == ".pdf" and "figures" not in path.parts:
        return False
    return not path.name.startswith(".env")


def changed_sources():
    tracked = git("diff", "--name-only", "--no-renames", "-z", "HEAD").stdout
    untracked = git("ls-files", "--others", "--exclude-standard", "-z").stdout
    return sorted({os.fsdecode(name) for name in (tracked + untracked).split(b"\0")
                   if name and source_file(os.fsdecode(name))})


def skip_reason():
    if git("symbolic-ref", "--quiet", "--short", "HEAD", check=False).stdout.strip() != b"main":
        return "only main is enabled"
    upstream = git("rev-parse", "--abbrev-ref", "@{upstream}", check=False).stdout.strip()
    if upstream != b"origin/main":
        return "main must track origin/main"
    for marker in ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-merge",
                   "rebase-apply", "sequencer", "BISECT_LOG", "index.lock"):
        location = os.fsdecode(git("rev-parse", "--git-path", marker).stdout).strip()
        if (ROOT / location).exists():
            return "another Git operation is in progress ({})".format(marker)
    if git("ls-files", "--unmerged", "-z").stdout:
        return "unresolved conflicts"
    if git("diff", "--cached", "--quiet", check=False).returncode != 0:
        return "manually staged changes; commit or unstage them first"
    return None


def sync(dry_run=False):
    reason = skip_reason()
    if reason:
        log("Skipped: " + reason)
        return
    paths = changed_sources()
    if dry_run:
        log("Preview: {} source files would be committed:".format(len(paths)))
        for name in paths:
            print("  " + name, flush=True)
        log("Push destination: origin, main -> main (no changes made)")
        return
    if paths:
        # Do not stage unrelated output files or manually staged work.
        for offset in range(0, len(paths), 100):
            git("--literal-pathspecs", "add", "--all", "--", *paths[offset:offset + 100])
        if git("diff", "--cached", "--quiet", check=False).returncode != 0:
            message = "backup: " + datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
            result = git("--literal-pathspecs", "commit", "--only", "-m", message, "--", *paths)
            log(os.fsdecode(result.stdout).strip())
    # Retry a previous failed push even if there are no new file changes.
    if int(git("rev-list", "--count", "@{upstream}..HEAD").stdout):
        result = git("push", "origin", "HEAD:refs/heads/main")
        log("Push succeeded. " + os.fsdecode(result.stderr).strip())
    else:
        log("No commits to push.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="repeat until VS Code closes the task")
    parser.add_argument("--interval", type=int, default=600, help="seconds between checks")
    parser.add_argument("--dry-run", action="store_true", help="preview without staging, committing or pushing")
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be positive")
    if args.dry_run:
        sync(dry_run=True)
        return
    lock_path = ROOT / os.fsdecode(git("rev-parse", "--git-path", "maas-auto-sync.lock").stdout).strip()
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("Another automatic sync task is already running.")
            return
        log("Automatic Git backup ready: origin/main, every {} seconds.".format(args.interval))
        while True:
            if args.watch:
                time.sleep(args.interval)
            try:
                sync()
            except subprocess.CalledProcessError as error:
                log("Git failed; no pull or force push attempted: " + os.fsdecode(error.stderr).strip())
            except subprocess.TimeoutExpired:
                log("Git timed out; retrying on the next check.")
            if not args.watch:
                break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Automatic Git backup stopped.")
