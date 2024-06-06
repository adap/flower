"""Test module for executor."""

import subprocess

from .executor import Executor, Run


class TestExec(Executor):
    """Test executor."""

    def start_run(self, fab_file, ttl=None):
        """Echos success."""
        _ = fab_file
        _ = ttl
        return Run(run_id=10, proc=subprocess.Popen(["echo", "success"]))


exec = TestExec()
