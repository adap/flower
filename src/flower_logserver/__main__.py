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
"""Start log server."""
import argparse

from .app import app


def main() -> None:
    """Start server."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--s3_bucket", type=str, help=f"S3 bucket for logfile",
    )
    parser.add_argument(
        "--s3_key", type=str, help=f"S3 key for logfile",
    )
    args = parser.parse_args()

    app.config["S3_BUCKET"] = args.s3_bucket if args.s3_bucket else None
    app.config["S3_KEY"] = args.s3_key if args.s3_key else None


    app.run(host="0.0.0.0", port=8081, debug=True)


main()
