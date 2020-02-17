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
"""Provides a logserver."""


import argparse
import ast
import json
import logging
import re
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Event, Thread
from typing import Dict, List, Optional, Tuple, Union

import boto3
import numpy as np

from flwr_experimental.baseline.plot import plot

LOGDIR = "flower_logs"
LOGFILE = "{logdir}/flower.log".format(logdir=LOGDIR)
LOGFILE_UPLOAD_INTERVAL = 60
SERVER_TIMEOUT = 1200

CONFIG: Dict[str, Optional[str]] = {"s3_bucket": None, "s3_key": None}

Accuracies = List[Tuple[int, float]]


def write_to_logfile(line: str) -> None:
    """Write line to logfile."""
    with open(f"{LOGFILE}", "a+") as lfd:
        lfd.write(line + "\n")


def is_credentials_available() -> bool:
    """Return True is credentials are available in CONFIG."""
    return all([v is not None for v in CONFIG.values()])


def upload_file(local_filepath: str, s3_key: Optional[str]) -> None:
    """Upload logfile to S3."""
    if not is_credentials_available():
        logging.info(
            "Skipping S3 logfile upload as s3_bucket or s3_key was not provided."
        )
    elif not Path(LOGFILE).is_file():
        logging.info("No logfile found.")
    elif s3_key is not None:
        try:
            logging.info("Uploading logfile to S3.")
            boto3.resource("s3").meta.client.upload_file(
                Filename=local_filepath,
                Bucket=CONFIG["s3_bucket"],
                Key=s3_key,
                ExtraArgs={
                    "ContentType": "application/pdf"
                    if s3_key.endswith(".pdf")
                    else "text/plain"
                },
            )
        # pylint: disable=broad-except
        except Exception as err:
            logging.error(err)


def continous_logfile_upload(stop_condition: Event, interval: int) -> None:
    """Call upload_logfile function regularly until stop_condition Event is
    set."""
    while True:
        upload_file(LOGFILE, CONFIG["s3_key"])

        if stop_condition.is_set():
            break

        time.sleep(interval)


def on_record(record: Dict[str, str]) -> None:
    """Call on each new line."""

    # Print record as JSON and write it to a logfile
    line = str(json.dumps(record))
    print(line)
    write_to_logfile(line)

    # Analyze record and if possible extract a plot_type and data from it
    plot_type, data = parse_plot_message(record["message"])

    if plot_type == "accuracies" and data is not None:
        plot_accuracies(data)


def parse_plot_message(
    message: str,
) -> Tuple[Optional[str], Optional[Union[Accuracies]]]:
    """Parse message and return its type and the data if possible.

    If the message does not contain plotable data return None.
    """
    accuracies_str = "app_fit: accuracies_centralized "

    if accuracies_str in message:
        values_str = re.sub(accuracies_str, "", message)
        values: Accuracies = ast.literal_eval(values_str)
        return "accuracies", values

    return None, None


def plot_accuracies(values: Accuracies) -> str:
    """Plot accuracies."""
    filename = f'{CONFIG["s3_key"]}.accuracies'

    line = [val * 100 for _, val in values]

    local_path = plot.line_chart(
        lines=[np.array(line)],
        labels=["Train"],
        x_label="Rounds",
        y_label="Accuracy",
        filename=filename,
    )
    upload_file(local_path, filename + ".pdf")
    return local_path


class RequestHandler(BaseHTTPRequestHandler):
    """Provide custom POST handler."""

    def _set_response(self) -> None:
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    def do_POST(self) -> None:  # pylint: disable=invalid-name
        """Handle POST request."""
        content_length = int(self.headers["Content-Length"])
        post_qs = self.rfile.read(content_length).decode("utf-8")
        record: Dict[str, str] = {
            "client_address": f"{self.client_address[0]}:{self.client_address[1]}"
        }

        for key, val in urllib.parse.parse_qs(post_qs).items():
            record[key] = str(val[0]) if len(val) == 1 else str(val)

        self._set_response()
        self.wfile.write("POST request for {}".format(self.path).encode("utf-8"))

        thread = Thread(target=on_record, args=(record,))
        thread.start()


class LogServer(HTTPServer):
    """Log server with timeout."""

    timeout = SERVER_TIMEOUT

    def handle_timeout(self) -> None:
        """Cleanup and upload logfile to S3."""
        self.server_close()
        raise TimeoutError()


def main() -> None:
    """Start log server."""
    # Create a flower_logs directory to store the logfiles.
    Path(LOGDIR).mkdir(exist_ok=True)
    Path(LOGFILE).touch()

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Flower LogServer")
    parser.add_argument(
        "--s3_bucket",
        type=str,
        help="S3 bucket where the logfile should be uploaded to.",
    )
    parser.add_argument(
        "--s3_key",
        type=str,
        help="S3 key under which the logfile should be uploaded.",
    )
    args = parser.parse_args()

    CONFIG["s3_bucket"] = args.s3_bucket
    CONFIG["s3_key"] = args.s3_key

    server = LogServer(("", 8081), RequestHandler)
    logging.info("Starting logging server...\n")

    # Start file upload loop
    sync_loop_stop_condition = Event()
    sync_loop = Thread(
        target=continous_logfile_upload,
        args=(sync_loop_stop_condition, LOGFILE_UPLOAD_INTERVAL),
    )
    sync_loop.start()

    try:
        while True:
            server.handle_request()
    except TimeoutError:
        print(
            f"TimeoutError raised as no request was received for {SERVER_TIMEOUT} seconds."
        )
        sync_loop_stop_condition.set()
        sync_loop.join()

    # Final upload
    upload_file(LOGFILE, CONFIG["s3_key"])

    logging.info("Stopping logging server...\n")


if __name__ == "__main__":
    main()
