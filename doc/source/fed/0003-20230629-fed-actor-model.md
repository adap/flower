---
fed-number: 0000
title: Client
authors: ["@panh99", "@danieljanes"]
creation-date: 2023-06-29
last-updated: 2023-06-30
status: provisional
---

# FED Template

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Summary](#summary)
- [Goals](#goals)
- [Proposal](#proposal)
  - [Task](#task)
  - [Client](#client)
  - [Router](#router)
  - [Example](#example)

## Summary

The future version of the flower, built on top of Driver APIs, shall support multiple strategies running concurrently over possibly overlapping sets of clients, which necessitates a client-side mechanism that creates and manages one isolated client instance for each running strategy on the same device.

Thanks to [Daniel](https://github.com/danieljanes)'s idea, the solution to this problem leverages the concept of actor models, as illustrated by the follwing ~~napkin~~ image. Specifically, a Lead Client, e.g., `C1` in the image, is responsible for creating Client per workload, e.g., `C1W1` and `C1W2`. The Lead Client serves as a router that fowards server messages to corresponding clients via matching the `workload_id` and uploads client responses to the server. `D1`, `D2`, and `D3` in the image denote 3 different running `driver.py` files.

![672bd1454270b082a1dd0f6a96a62cc7.png](https://imgtr.ee/images/2023/06/30/672bd1454270b082a1dd0f6a96a62cc7.png)


## Goals

1. Clients can be created by Lead Client in the runtime.
2. Lead Client should call `PullTaskIns` periodically, e.g., once per sec.
3. Lead Client should foward `TaskIns` to the correct Client as soon as it arrives.
4. Lead Client should wrap the client responses in `TaskRes` and send them back to the server as soon as they are produced.
5. Clients managed by the same Lead Client should be able to run concurrently, and their message-handling functions should not block one another.
6. As efficient as possible.

## Proposal

#### Task

As an illustration, define `Task` as follows. For the sake of simplicity, all communications are based on `Task` instead of real `TaskIns` and `TaskRes`.
``` python
from dataclasses import dataclass

@dataclass
class Task:
    workload_id: int
    message_type: str
    content: str
```

#### Client

The Client class, which represents the per-workload client instance, is dfined in the following code. As this FED doc focuses on the As this FED doc focuses on the actor-model-based design on the client side, it is assumed that the `workload_id` should be agreed upon among parties by a different mechanism.

Each client receives and sends `Task` messages in its coroutine. Once received a `Task` message, the client will start a separate thread to execute the corresponding handler function, which is the `example_handler` function in this example. `example_handler` prints the message it receives, sleeps 2 seconds, and returns the original text it receives concatenated with " handled". The `outbox` and `pool` (Thread Pool) variables are set by the external `Router` instance (see the next subsection). All client instances use the same `outbox` and `pool` and have their own respective `inbox`. Both `inbox` and `outbox` are `asyncio.Queue`.

``` python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Dict, List

class Client:
    def __init__(self, workload_id):
        self.workload_id = workload_id
        self.inbox = asyncio.Queue()
        self.outbox: asyncio.Queue = None
        self.pool: ThreadPoolExecutor = None

    def set(self, pool: ThreadPoolExecutor, outbox: asyncio.Queue):
        self.pool, self.outbox = pool, outbox

    async def put(self, task: Task):
        await self.inbox.put(task)

    def example_handler(self, task: Task) -> Task:
        msg = task.content
        print(f"WL{self.workload_id} Start: {msg} ({self.inbox.qsize()} remaining)")
        time.sleep(2)
        print(f"WL{self.workload_id} End: {msg}")
        return Task(
            workload_id=self.workload_id,
            message_type="",
            content=msg + " handled"
        )

    async def run(self):
        while True:
            task: Task = await self.inbox.get()
            loop = asyncio.get_running_loop()
            try:
                assert task.workload_id == self.workload_id
                res = await loop.run_in_executor(self.pool, self.example_handler, task)
                await self.outbox.put(res)
            except Exception as e:
                print(e)
```

#### Router

`Router` is the Lead Client which orchestrates the communication among clients and the server, as shown in the following code. It receives and sends `Task` messages in coroutines. Once a sec it checks its `inbox` and forwards the `Task` messages to the corresponding client, which simulates the process of calling `PullTaskIns`. Besides, it will send the client responses to the server as soon as they are put into the `outbox`.

```python	
import asyncio
from typing import Dict, List

class Router:
    def __init__(self):
        self.inbox: List[Task] = []
        self.outbox = asyncio.Queue()
        self.pool = ThreadPoolExecutor()
        # map workload_id to client
        self.client_dict: Dict[int, Client] = {}
        self.start_clients: List[Client] = []

    def add_client(self, c: Client):
        self.start_clients.append(c)

    async def run(self):
        send_task = asyncio.create_task(self._send())
        recv_task = asyncio.create_task(self._receive())
        await asyncio.gather(send_task, recv_task)

    async def _send(self):
        while True:
            res: Task = await self.outbox.get()
            print(f"Send from WL{res.workload_id}: {res.content}")

    async def _receive(self):
        while True:
            while any(self.start_clients):
                c: Client = self.start_clients.pop(0)
                self.client_dict[c.workload_id] = c
                c.set(self.pool, self.outbox)
                asyncio.create_task(c.run())
            while any(self.inbox):
                task: Task = self.inbox.pop(0)
                print(f"Route to WL{task.workload_id}: {task.content}")
                await self.client_dict[task.workload_id].put(task)
            await asyncio.sleep(1)
```

#### Example

The following example shows 100 clients running concurrently. 400 `Task` messages are generated and allocated randomly to clients.

```python
import asyncio
import random

async def main():
    num_clients = 100
    gen_content = lambda msg_idx, workload_id: f"Msg {msg_idx}"

    m = Router()
    for c in [Client(i) for i in range(num_clients)]:
        m.add_client(c)

    coro_task = asyncio.create_task(m.run())

    for i in range(400):
        wl_id = random.randint(0, num_clients - 1)
        task = Task(
            workload_id=wl_id,
            message_type="",
            content=gen_content(i, wl_id)
        )
        m.inbox.append(task)
        await asyncio.sleep(0.04)
    await coro_task

asyncio.run(main())
```

