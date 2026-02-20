# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Flask-based profiling UI that queries the Control API directly."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any, Iterator

from flask import Flask, Response, jsonify, request, send_from_directory
import grpc
from werkzeug.exceptions import BadRequest

from flwr.cli.config_utils import (
    exit_if_no_address,
    load_and_validate,
    process_loaded_project_config,
    validate_federation_in_project_config,
)
from flwr.cli.utils import init_channel, load_cli_auth_plugin
from flwr.proto.control_pb2 import GetRunProfileRequest, StreamRunProfileRequest
from flwr.proto.control_pb2_grpc import ControlStub


def _load_federation_config(
    app_path: Path, federation: str | None, address: str | None, insecure: bool
) -> tuple[str, dict[str, Any]]:
    pyproject_path = app_path / "pyproject.toml"
    config, errors, warnings = load_and_validate(pyproject_path, check_module=False)
    config = process_loaded_project_config(config, errors, warnings)
    federation, federation_config = validate_federation_in_project_config(
        federation, config
    )
    if address:
        federation_config["address"] = address
    if insecure:
        federation_config["insecure"] = True
        federation_config.pop("root-certificates", None)
    exit_if_no_address(federation_config, "profile-ui")
    return federation, federation_config


def create_app(
    app_path: Path, federation: str | None, address: str | None, insecure: bool
) -> Flask:
    ui_dir = Path(__file__).parent
    flask_app = Flask(
        __name__,
        static_folder=str(ui_dir),
        static_url_path="",
    )

    federation_name, federation_config = _load_federation_config(
        app_path, federation, address, insecure
    )
    auth_plugin = load_cli_auth_plugin(app_path, federation_name, federation_config)

    def _fetch_profile(run_id: int) -> dict[str, Any]:
        channel = init_channel(app_path, federation_config, auth_plugin)
        try:
            stub = ControlStub(channel)
            res = stub.GetRunProfile(GetRunProfileRequest(run_id=run_id))
            if not res.summary_json:
                return {"run_id": run_id, "entries": [], "events": []}
            return json.loads(res.summary_json)
        finally:
            channel.close()

    def _stream_profile(run_id: int) -> Iterator[str]:
        channel = init_channel(app_path, federation_config, auth_plugin)
        stub = ControlStub(channel)
        try:
            req = StreamRunProfileRequest(run_id=run_id)
            for res in stub.StreamRunProfile(req):
                if not res.summary_json:
                    continue
                yield f"data: {res.summary_json}\n\n"
        finally:
            channel.close()

    @flask_app.route("/")
    def index() -> Response:
        return send_from_directory(ui_dir, "index.html")

    @flask_app.route("/api/profile")
    def profile() -> Response:
        run_id_raw = request.args.get("run_id")
        live = request.args.get("live", "0") in {"1", "true", "True"}
        if not run_id_raw:
            raise BadRequest("Missing run_id")
        try:
            run_id = int(run_id_raw)
        except ValueError as exc:
            raise BadRequest("run_id must be an integer") from exc

        try:
            if live:
                return Response(_stream_profile(run_id), mimetype="text/event-stream")
            return jsonify(_fetch_profile(run_id))
        except grpc.RpcError as exc:
            details = exc.details()
            code = exc.code().name if exc.code() else "UNKNOWN"
            message = details or str(exc) or "gRPC request failed"
            flask_app.logger.exception(
                "Profile request failed: %s (%s)", message, code
            )
            return (
                jsonify({"error": message, "code": code}),
                502,
            )
        except Exception as exc:  # pragma: no cover - safety net
            flask_app.logger.exception("Profile request failed")
            return jsonify({"error": str(exc)}), 500

    return flask_app


def main() -> None:
    parser = ArgumentParser(description="Flower profiling UI")
    parser.add_argument("--app", type=Path, default=Path("."), help="Flower app path")
    parser.add_argument(
        "--federation", type=str, default=None, help="Federation name"
    )
    parser.add_argument(
        "--address",
        type=str,
        default=None,
        help="Control API address override",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS (overrides config)",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    flask_app = create_app(args.app, args.federation, args.address, args.insecure)
    flask_app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
