# Copyright 2021 Adap GmbH. All Rights Reserved.
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

from flwr.server.criterion import Criterion
from flwr.server.client_proxy import ClientProxy
from flwr.common.typing import PropertiesIns


class KeywordCriterion(Criterion):
    def __init__(self, keyword="priority"):
        self.keyword = keyword

    def select(self, client: ClientProxy) -> bool:
        ins = PropertiesIns(config={})
        properties = client.get_properties(ins=ins).properties
        return self.keyword in properties
