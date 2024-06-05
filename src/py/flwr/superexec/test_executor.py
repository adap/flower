from .executor import Executor, Run
import subprocess
from flwr.cli.install import install_from_fab


class TestExec(Executor):

    def start_run(self, fab_file, ttl=None):
        install_from_fab(fab_file, None, True)
        return Run(run_id=10, proc=subprocess.Popen(["echo", "what"]))
