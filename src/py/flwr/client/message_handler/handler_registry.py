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
"""Handler registry for gRPC-rere."""


from typing import Dict

__reg = type("", (), {"functions": {}, "type2handlers": None})()


def register_handler(message_type: str):
    def register(func):
        __reg.functions.setdefault(message_type, []).append(func)
        __reg.type2handlers = None
        return func

    return register


def get_handlers(cls_type: type) -> Dict[str, str]:
    if type(cls_type) != type:
        cls_type = type(cls_type)
    if __reg.type2handlers is None:
        type2handlers: Dict[type, set] = {}
        for msg_type, func_lst in __reg.functions.items():
            for func in func_lst:
                cls = func.__globals__[func.__qualname__.split(".")[0]]
                type2handlers.setdefault(cls, set()).add((msg_type, func.__name__))
        __reg.type2handlers = type2handlers
    type2handlers = __reg.type2handlers
    handlers = set()
    for _type, _set in type2handlers.items():
        if issubclass(cls_type, _type):
            handlers.update(_set)
    ret = {}
    for msg_type, func_name in handlers:
        if msg_type in ret and ret[msg_type] != func_name:
            raise ValueError(
                f'A handler for the "{msg_type}" message type is already registered.'
            )
        ret[msg_type] = func_name
    return ret
