#!/usr/bin/env python3
"""Poll launcher run logs until LibCity prints its training-start marker."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.sparsity.train_job_plan import TRAINING_STARTED_MARKERS, training_started_in_text


def _load_job_names(generated_path: Path) -> list[str]:
    raw = json.loads(generated_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{generated_path} must contain a JSON list")
    return [str(item["name"]) for item in raw]


def wait_for_training_started(
    *,
    logs_dir: Path,
    job_names: list[str],
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, str]:
    """
    Block until every job log contains a training-start marker, or raise on timeout/failure.

    Returns a mapping job_name -> log excerpt around the start marker.
    """
    pending = set(job_names)
    started: dict[str, str] = {}
    deadline = time.time() + timeout_seconds

    while pending and time.time() < deadline:
        for name in list(pending):
            log_path = logs_dir / f"{name}.log"
            status_path = logs_dir / f"{name}.status.json"

            if status_path.is_file():
                try:
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    status = {}
                exit_code = status.get("exit_code")
                if exit_code is not None and int(exit_code) != 0:
                    tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:] if log_path.is_file() else ""
                    raise RuntimeError(
                        f"{name} exited with code {exit_code} before training started.\n"
                        f"Log tail ({log_path}):\n{tail}"
                    )

            if not log_path.is_file():
                continue

            text = log_path.read_text(encoding="utf-8", errors="replace")
            if training_started_in_text(text):
                for marker in TRAINING_STARTED_MARKERS:
                    idx = text.find(marker)
                    if idx >= 0:
                        excerpt = text[max(0, idx - 120) : idx + len(marker) + 80].strip()
                        started[name] = excerpt
                        break
                else:
                    started[name] = "Start training"
                pending.discard(name)

        if pending:
            time.sleep(poll_seconds)

    if pending:
        details = []
        for name in sorted(pending):
            log_path = logs_dir / f"{name}.log"
            if log_path.is_file():
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
                details.append(f"{name} (log exists, no start marker yet):\n{tail}")
            else:
                details.append(f"{name}: log not created yet ({log_path})")
        raise TimeoutError(
            f"Timed out after {timeout_seconds:.0f}s waiting for training start on: "
            f"{', '.join(sorted(pending))}\n\n" + "\n\n".join(details)
        )

    return started


def main() -> None:
    ap = argparse.ArgumentParser(description="Wait until all retrain jobs log 'Start training'")
    ap.add_argument("--logs-dir", required=True, help="Launcher logs directory")
    ap.add_argument("--generated", required=True, help="Generated runs JSON from the launcher")
    ap.add_argument("--timeout-seconds", type=float, default=900.0)
    ap.add_argument("--poll-seconds", type=float, default=10.0)
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir).resolve()
    generated_path = Path(args.generated).resolve()
    job_names = _load_job_names(generated_path)

    print(f"Waiting for training start on {len(job_names)} jobs (timeout={args.timeout_seconds:.0f}s)...")
    try:
        started = wait_for_training_started(
            logs_dir=logs_dir,
            job_names=job_names,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
    except (TimeoutError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    for name in job_names:
        print(f"OK  {name}")
        print(f"    {started[name][:200]}...")
    print(f"All {len(job_names)} jobs reached training start.")


if __name__ == "__main__":
    main()
