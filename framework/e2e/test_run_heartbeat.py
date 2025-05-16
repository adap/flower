"""Test run heartbeat functionality."""

import json
import subprocess
import sys
import time

import tomli
import tomli_w

from flwr.common.constant import Status, SubStatus


def add_e2e_federation() -> None:
    """Add e2e federation to pyproject.toml."""
    # Load the pyproject.toml file
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)

    # Remove e2e federation if exists
    pyproject["tool"]["flwr"]["federations"].pop("e2e", None)

    # Add e2e federation
    pyproject["tool"]["flwr"]["federations"]["e2e"] = {
        "address": "127.0.0.1:9093",
        "insecure": True,
        "options": {"num-supernodes": 0},
    }

    # Write the updated pyproject.toml file
    with open("pyproject.toml", "wb") as f:
        tomli_w.dump(pyproject, f)


def run_superlink(is_simulation: bool) -> subprocess.Popen:
    """Run the SuperLink."""
    executor_arg = "flwr.superexec.deployment:executor"
    if is_simulation:
        executor_arg = "flwr.superexec.simulation:executor"
    return subprocess.Popen(
        [
            "flower-superlink",
            "--insecure",
            "--database",
            "tmp.db",
            "--executor",
            executor_arg,
            "--isolation",
            "process",
        ],
    )


def run_worker_process(is_simulation: bool) -> subprocess.Popen:
    """Run the flwr-serverapp or flwr-simulation process."""
    cmd = "flwr-simulation" if is_simulation else "flwr-serverapp"
    return subprocess.Popen(
        [cmd, "--insecure"],
    )


def flwr_run() -> int:
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


def flwr_ls() -> dict[int, str]:
    """Run `flwr ls` command and return a mapping of run_id to status.

    Returns
    -------
    dict[int, str]
        A dictionary where keys are run IDs and values are their statuses.
    """
    # Run the command
    result = subprocess.run(
        ["flwr", "ls", ".", "e2e", "--format", "json"],
        capture_output=True,
        text=True,
        check=True,
    )

    # Parse JSON output and ensure the command succeeded
    data = json.loads(result.stdout)
    assert data["success"], "flwr ls failed"

    # Return a dictionary mapping run_id to status
    return {entry["run-id"]: entry["status"] for entry in data["runs"]}


def main() -> None:
    """."""
    # Determine if the test is running in simulation mode
    is_simulation = sys.argv[1] == "simulation" if len(sys.argv) > 1 else False
    print(f"Running in {'simulation' if is_simulation else 'deployment'} mode.")

    # Add e2e federation to pyproject.toml
    add_e2e_federation()

    # Start the SuperLink
    print("Starting SuperLink...")
    superlink_proc = run_superlink(is_simulation)

    # Allow time for SuperLink to start
    time.sleep(1)

    # Submit the first run and run the first ServerApp process
    print("Starting the first run and ServerApp process...")
    run_id1 = flwr_run()
    serverapp_proc = run_worker_process(is_simulation)

    # Brief pause to allow the ServerApp process to initialize
    time.sleep(1)

    # Submit the second run and run the second ServerApp process
    print("Starting the second run and ServerApp process...")
    run_id2 = flwr_run()
    serverapp_proc2 = run_worker_process(is_simulation)

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
    superlink_proc.terminate()
    superlink_proc.wait()

    # Kill the first ServerApp process
    # The ServerApp process cannot be terminated gracefully yet,
    # so we need to kill it via SIGKILL.
    print("Terminating the first ServerApp process...")
    serverapp_proc.kill()
    serverapp_proc.wait()

    # Restart the SuperLink
    print("Restarting SuperLink...")
    superlink_proc = run_superlink(is_simulation)

    # Allow time for SuperLink to start
    time.sleep(1)

    # Allow time for SuperLink to detect heartbeat failures and update statuses
    tic = time.time()
    is_valid = False
    while (time.time() - tic) < 20:
        run_status = flwr_ls()
        if (
            run_status[run_id1] == f"{Status.FINISHED}:{SubStatus.FAILED}"
            and run_status[run_id2] == f"{Status.FINISHED}:{SubStatus.COMPLETED}"
        ):
            is_valid = True
            break
        time.sleep(1)
    assert is_valid, "Run statuses are not updated correctly"
    print("Run statuses are updated correctly.")

    # Clean up
    serverapp_proc2.kill()
    serverapp_proc2.wait()
    superlink_proc.terminate()
    superlink_proc.wait()


if __name__ == "__main__":
    main()
