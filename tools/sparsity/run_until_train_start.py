#!/usr/bin/env python3
"""Run a training command and exit successfully once LibCity logs training start."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.sparsity.train_job_plan import training_started_in_text


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a command until training-start log line appears")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="Command after --")
    args = ap.parse_args()
    cmd = args.cmd
    if cmd[:1] == ["--"]:
        cmd = cmd[1:]
    if not cmd:
        ap.error("missing command after --")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    captured: list[str] = []
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            captured.append(line)
            if training_started_in_text(line):
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
                sys.exit(0)
        rc = proc.wait()
        tail = "".join(captured[-40:])
        print(
            f"Command exited with code {rc} before training start marker was seen.\n"
            f"Last log lines:\n{tail}",
            file=sys.stderr,
        )
        sys.exit(rc if rc != 0 else 1)
    except KeyboardInterrupt:
        proc.terminate()
        raise


if __name__ == "__main__":
    main()
