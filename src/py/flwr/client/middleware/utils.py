# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
"""Utility functions for middleware layers."""

from typing import List

from flwr.client.typing import Bwd, Fwd

from .typing import App, Layer


def make_app(app: App, middleware_layers: List[Layer]) -> App:
    """."""

    def wrap_app(_app: App, _layer: Layer) -> App:
        def new_app(fwd: Fwd) -> Bwd:
            return _layer(fwd, _app)

        return new_app

    for layer in reversed(middleware_layers):
        app = wrap_app(app, layer)

    return app
