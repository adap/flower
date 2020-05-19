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
"""Start log server.

The server will shutdown automatically if it does not receive a message for more
than an hour.
"""
import argparse
import json
import logging
import time
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Event, Thread

import boto3

LOGDIR = "flower_logs"
LOGFILE = "{logdir}/{:%Y-%m-%d}.log".format(datetime.now(), logdir=LOGDIR)
LOGFILE_UPLOAD_INTERVAL = 60
SERVER_TIMEOUT = 3600

CONFIG = {"s3_bucket": None, "s3_key": None}

# Create a flower_logs directory to store the logfiles.
Path(LOGDIR).mkdir(exist_ok=True)


def write_to_logfile(line: str) -> None:
    """Write line to logfile."""
    with open(f"{LOGFILE}", "a+") as lfd:
        lfd.write(line + "\n")


def upload_logfile() -> None:
    """Upload logfile to S3."""
    if any([v is None for v in CONFIG.values()]):
        logging.info(
            "Skipping S3 logfile upload as s3_bucket or s3_key was not provided."
        )
    elif not Path(LOGFILE).is_file():
        logging.info("No logfile found")
    else:
        try:
            logging.info("Uploading logfile to S3.")
            boto3.resource("s3").meta.client.upload_file(
                Filename=LOGFILE,
                Bucket=CONFIG["s3_bucket"],
                Key=CONFIG["s3_key"],
                ExtraArgs={"ContentType": "text/plain"},
            )
        # pylint: disable=broad-except
        except Exception as err:
            logging.error(err)


def continous_logfile_upload(stop_condition: Event, interval: int) -> None:
    """Call upload_logfile function regularly until stop_condition Event is set."""
    while True:
        upload_logfile()

        if stop_condition.is_set():
            break

        time.sleep(interval)


class RequestHandler(BaseHTTPRequestHandler):
    """Provide custom POST handler."""

    def _set_response(self) -> None:
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    def do_POST(self):  # pylint: disable=invalid-name
        """Handle POST request."""
        content_length = int(self.headers["Content-Length"])
        post_qs = self.rfile.read(content_length).decode("utf-8")
        post_data = {}

        for key, val in urllib.parse.parse_qs(post_qs).items():
            post_data[key] = val[0] if len(val) == 1 else val

        self._set_response()
        self.wfile.write("POST request for {}".format(self.path).encode("utf-8"))

        line = str(json.dumps(post_data))
        print(line)
        write_to_logfile(line)


class LogServer(HTTPServer):
    """Log server with timeout."""

    timeout = SERVER_TIMEOUT

    def handle_timeout(self):
        """Cleanup and upload logfile to S3."""
        self.server_close()
        raise TimeoutError()


def main() -> None:
    """Start log server."""
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Flower LogServer")
    parser.add_argument(
        "--s3_bucket",
        type=str,
        help="S3 bucket where the logfile should be uploaded to.",
    )
    parser.add_argument(
        "--s3_key", type=str, help="S3 key under which the logfile should be uploaded.",
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
    upload_logfile()

    logging.info("Stopping logging server...\n")


if __name__ == "__main__":
    main()
