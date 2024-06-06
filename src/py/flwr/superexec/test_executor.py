from .executor import Executor, Run
import subprocess


class TestExec(Executor):

    def start_run(self, fab_file, ttl=None):
        return Run(run_id=10, proc=subprocess.Popen(["echo", "success"]))
