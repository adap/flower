"""Test module for executor."""

import subprocess
from typing import Optional

from .executor import Executor, Run


class TestExec(Executor):
    """Test executor."""

    def start_run(self, fab_file: bytes, ttl: Optional[float] = None) -> Run:
        """Echos success."""
        _ = fab_file
        _ = ttl
        return Run(
            run_id=10,
            proc=subprocess.Popen(
                ["sh", "-c", "for i in {1..50}; do echo $i && fortune | cowsay ; sleep 2; done"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ),
        )


exec = TestExec()
