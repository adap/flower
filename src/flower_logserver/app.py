# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Simple flask server log server."""

import logging
import json 
from pathlib import Path
from datetime import datetime

from flask import Flask, request

logging.getLogger("werkzeug").setLevel(logging.ERROR)

# pylint: disable=invalid-name
app = Flask(__name__)

logfile = "{:%Y-%m-%d}.log".format(datetime.now())

# Create a flower_logs directory to store the logfiles.
Path("flower_logs").mkdir(exist_ok=True)

def write_to_logfile(line: str) -> None:
    """Write line to logfile."""
    with open(f"flower_logs/{logfile}", "a+") as lfd:
        lfd.write(line + "\n")

@app.route("/log", methods=["POST"])
def index() -> str:
    """Handle logs."""
    data = json.dumps(request.form)

    print(data)
    write_to_logfile(str(data))

    return ""
