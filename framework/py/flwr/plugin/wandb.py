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
"""This module contains functions related to Wandb."""

import wandb
from flwr.common import Message, MetricRecord, ArrayRecord
from typing import Iterable, Any

def plugin_wandb(strategy: Any, key: str, project: str, config: dict, name: str = None) -> Any:
    """
    Any Strategy object can be dynamically injected with WandB logging functionality.
    """
    # 1. WandB 초기화
    wandb.login(key=key)
    wandb.init(project=project, config=config, name=name)

    # 2. 전달받은 strategy 객체의 '원래 클래스'를 상속받는 동적 클래스 생성
    class WandbInjectedStrategy(strategy.__class__):
        def aggregate_train(self, server_round, replies):
            arrays, metrics = super().aggregate_train(server_round, replies)
            
            if metrics:
                wandb.log(metrics, step=server_round, commit=False)
                
            return arrays, metrics

        def aggregate_evaluate(self, server_round, replies):
            metrics = super().aggregate_evaluate(server_round, replies)
            
            if metrics:
                wandb.log(metrics, step=server_round, commit=True)
                
            return metrics

    # 3. 파이썬 매직: 객체의 클래스를 방금 만든 동적 클래스로 바꿔치기
    strategy.__class__ = WandbInjectedStrategy
    
    return strategy