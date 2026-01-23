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
    Status,
    SubStatus,
)

DATABASE_FILE = Path("tmp.db")

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


def main() -> None:
    """."""
    # Show version and initialize Flower config
    subprocess.run(["flwr", "--version"], check=True)

    # Kill any orphan processes from previous failed runs
    kill_orphan_processes()

    # Clean up any existing database from previous runs
    cleanup_database()

    # Determine if the test is running in simulation mode
    print(f"Running in {'simulation' if use_sim else 'deployment'} mode.")

    # Start the SuperLink
    print("Starting SuperLink...")
    superlink_proc = run_superlink()

    # Allow time for SuperLink to start
    time.sleep(1)

    # Start the SuperExec
    print("Starting SuperExec...")
    superexec_proc = run_superexec()

    try:
        # Submit the first run
        print("Starting the first run...")
        run_id1 = flwr_run()

        # Get the PID of the first app process
        while True:
            if pids := get_pids(app_cmd):
                app_pid = pids[0]
                break
            time.sleep(0.1)

        # Submit the second run
        print("Starting the second run...")
        run_id2 = flwr_run()

        # Wait up to 6 seconds for both runs to reach RUNNING status
        tic = time.time()
        is_running = False
        while (time.time() - tic) < 6:
            run_status = flwr_ls()
            if (
                run_status.get(run_id1) == Status.RUNNING
                and run_status.get(run_id2) == Status.RUNNING
            ):
                is_running = True
                break
            time.sleep(1)
        assert is_running, "Run IDs did not start within 6 seconds"
        print("Both runs are running.")

        # Kill SuperLink process first to simulate restart scenario
        # This prevents ServerApp from notifying SuperLink, isolating the heartbeat test
        print("Terminating SuperLink process...")
        kill_process(superlink_proc, "SuperLink")

        # Kill the first ServerApp process
        print("Terminating the first ServerApp process...")
        os.kill(app_pid, signal.SIGKILL)  # SIGKILL to ensure it stops immediately

        # Wait for SQLite to fully release the database lock
        time.sleep(0.5)

        # Restart the SuperLink
        print("Restarting SuperLink...")
        superlink_proc = run_superlink()

        # Allow time for SuperLink to start
        time.sleep(1)

        # Allow time for SuperLink to detect heartbeat failures and update statuses
        tic = time.time()
        is_valid = False
        while (time.time() - tic) < 25:
            run_status = flwr_ls()
            if (
                run_status[run_id1] == f"{Status.FINISHED}:{SubStatus.FAILED}"
                and run_status[run_id2] == f"{Status.FINISHED}:{SubStatus.COMPLETED}"
            ):
                is_valid = True
                break
            time.sleep(1)
        assert is_valid, f"Run statuses are not updated correctly:\n{run_status}"
        print("Run statuses are updated correctly.")

    finally:
        # Clean up processes even if test fails
        print("Cleaning up processes...")
        kill_process(superexec_proc, "SuperExec")
        kill_process(superlink_proc, "SuperLink")
        cleanup_database()


if __name__ == "__main__":
    main()
