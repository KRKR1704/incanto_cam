"""Install the repo's git hooks (scripts/hooks/*) into .git/hooks. Run once per clone."""
from __future__ import annotations

import shutil
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOOKS_SRC = ROOT / "scripts" / "hooks"


def main() -> int:
    git_hooks = subprocess.run(
        ["git", "rev-parse", "--git-path", "hooks"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    dest_dir = (ROOT / git_hooks).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(p for p in HOOKS_SRC.iterdir() if p.is_file()):
        dest = dest_dir / src.name
        # Normalize to LF: Git for Windows runs hooks with its own sh, which chokes on CRLF.
        dest.write_bytes(src.read_bytes().replace(b"\r\n", b"\n"))
        dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"installed {src.name} -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
