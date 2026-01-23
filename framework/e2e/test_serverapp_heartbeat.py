"""Test run heartbeat functionality."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from flwr.common.constant import (
    SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS,
    SIMULATIONIO_API_DEFAULT_CLIENT_ADDRESS,
)

# Use absolute path to ensure cleanup works regardless of working directory
DATABASE_FILE = Path(__file__).parent / "tmp.db"

use_sim = sys.argv[1] == "simulation" if len(sys.argv) > 1 else False
plugin_type_arg = "simulation" if use_sim else "serverapp"
address_arg = (
    SIMULATIONIO_API_DEFAULT_CLIENT_ADDRESS
    if use_sim
    else SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS
)
app_cmd = "flwr-simulation" if use_sim else "flwr-serverapp"


def run_superlink() -> subprocess.Popen:
    """Run the SuperLink."""
    cmd = ["flower-superlink", "--insecure"]
    cmd += ["--database", str(DATABASE_FILE)]
    cmd += ["--isolation", "process"]
    if use_sim:
        cmd += ["--simulation"]

    return subprocess.Popen(cmd)


def run_superexec() -> subprocess.Popen:
    """Run the SuperExec."""
    cmd = ["flower-superexec", "--insecure"]
    cmd += ["--appio-api-address", address_arg]
    cmd += ["--plugin-type", plugin_type_arg]
    return subprocess.Popen(cmd)


def flwr_run() -> str:
    """Run the `flwr run` command and return `run_id`."""
    # Run the command
    result = subprocess.run(
        ["flwr", "run", ".", "e2e", "--format", "json"],
        capture_output=True,
        text=True,
        check=True,
    )

    # Parse JSON output and ensure the command succeeded
    data = json.loads(result.stdout)
    assert data["success"], "flwr run failed\n" + str(data)

    # Return the run ID
    return data["run-id"]


def flwr_ls() -> dict[str, str]:
    """Run `flwr ls` command and return a mapping of run_id to status.

    Returns
    -------
    dict[str, str]
        A dictionary where keys are run IDs and values are their statuses.
    """
    # Run the command
    result = subprocess.run(
        ["flwr", "ls", "e2e", "--format", "json"],
        capture_output=True,
        text=True,
        check=True,
    )

    # Parse JSON output and ensure the command succeeded
    data = json.loads(result.stdout)
    assert data["success"], "flwr ls failed"

    # Return a dictionary mapping run_id to status
    return {entry["run-id"]: entry["status"] for entry in data["runs"]}


def get_pids(command: str) -> list[int]:
    """Get the PIDs of a running command."""
    result = subprocess.run(
        ["pgrep", "-f", command],
        capture_output=True,
        text=True,
        check=False,
    )
    pids = result.stdout.strip().split("\n")
    return [int(pid) for pid in pids if pid]


def kill_orphan_processes() -> None:
    """Kill any orphan SuperLink or SuperExec processes from previous runs."""
    for process_name in ["flower-superlink", "flower-superexec"]:
        pids = get_pids(process_name)
        for pid in pids:
            try:
                print(f"Killing orphan {process_name} process (PID: {pid})...")
                os.kill(pid, signal.SIGKILL)
                time.sleep(0.1)
            except ProcessLookupError:
                pass  # Process already gone


def cleanup_database() -> None:
    """Remove the database file if it exists."""
    if DATABASE_FILE.exists():
        DATABASE_FILE.unlink()
    # Also remove SQLite journal/wal files if they exist
    for suffix in ["-journal", "-wal", "-shm"]:
        journal_file = DATABASE_FILE.with_suffix(DATABASE_FILE.suffix + suffix)
        if journal_file.exists():
            journal_file.unlink()


def kill_process(proc: subprocess.Popen, name: str) -> None:
    """Forcefully kill a process and wait for it to terminate."""
    if proc.poll() is None:  # Process is still running
        print(f"Killing {name}...")
        proc.kill()  # SIGKILL for forceful termination
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"Warning: {name} did not terminate within timeout")


def check_superlink_running() -> bool:
    """Check if SuperLink process is running and print status."""
    pids = get_pids("flower-superlink")
    if pids:
        print(f"✓ SuperLink is running (PIDs: {pids})")
        return True
    else:
        print("✗ SuperLink is NOT running")
        return False


def main() -> None:
    """."""
    # Show version and initialize Flower config
    subprocess.run(["flwr", "--version"], check=True)

    # Check initial state
    print("\n=== Initial State ===")
    check_superlink_running()


if __name__ == "__main__":
    main()
