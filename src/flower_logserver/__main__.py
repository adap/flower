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
import logging
import time
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import boto3

LOGDIR = "flower_logs"
LOGFILE = "{logdir}/{:%Y-%m-%d}.log".format(datetime.now(), logdir=LOGDIR)
LOGFILE_UPLOAD_INTERVAL = 15
SERVER_TIMEOUT = 60

CONFIG = {"s3_bucket": None, "s3_key": None}

# Create a flower_logs directory to store the logfiles.
Path(LOGDIR).mkdir(exist_ok=True)


def upload_logfile() -> None:
    """Upload logfile to S3."""
    if any([v is None for v in CONFIG.values()]):
        logging.info(
            "Skipping S3 logfile upload as s3_bucket or s3_key was not provided."
        )
    else:
        logging.warn("Uploading logfile to S3.")
        boto3.resource("s3").meta.client.upload_file(
            LOGFILE, CONFIG["s3_bucket"], CONFIG["s3_key"]
        )


def write_to_logfile(line: str) -> None:
    """Write line to logfile."""
    with open(f"{LOGFILE}", "a+") as lfd:
        lfd.write(line + "\n")


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

        print(post_data)
        write_to_logfile(str(post_data))


class LogServer(HTTPServer):
    """Log server with timeout."""

    timeout = SERVER_TIMEOUT

    def handle_timeout(self):
        """Cleanup and upload logfile to S3."""
        upload_logfile()
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

    last_logfile_upload = int(time.time())

    try:
        while True:
            server.handle_request()

            time_elapsed = int(time.time()) - last_logfile_upload

            print(time_elapsed)

            if time_elapsed > LOGFILE_UPLOAD_INTERVAL:
                upload_logfile()
                last_logfile_upload = int(time.time())
    except TimeoutError:
        pass

    logging.info("Stopping logging server...\n")


if __name__ == "__main__":
    main()
