import asyncio
import json
import logging
import os
import socket
import uuid
from datetime import datetime
from typing import Optional

import requests
import uvicorn
from fastapi import FastAPI
from pydantic.main import BaseModel

handlers_list = [logging.StreamHandler()]
if os.environ.get("MONITORING") == "1":
    handlers_list.append(logging.FileHandler("./fedops/client_manager.log"))

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)8.8s] %(message)s",
    handlers=handlers_list,
)
logger = logging.getLogger(__name__)
app = FastAPI()


today = datetime.today()
today_str = today.strftime("%Y-%m-%d")


def get_mac_address():
    mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
    return ":".join([mac[i : i + 2] for i in range(0, 12, 2)])


def get_hostname():
    return socket.gethostname()


class FLTask(BaseModel):
    FL_task_ID: Optional[str] = None
    Device_mac: Optional[str] = None
    Device_hostname: Optional[str] = None
    Device_online: Optional[bool] = None
    Device_training: Optional[bool] = None


class ManagerStatus(BaseModel):
    FL_client: str = "localhost:8003"
    server_ST: str = "ccl.gachon.ac.kr:40019"
    server: str = "ccl.gachon.ac.kr"
    S3_bucket: str = "fl-gl-model"
    s3_ready: bool = False
    GL_Model_V: int = 0
    FL_ready: bool = False

    client_online: bool = False
    client_training: bool = False

    task_id: str = ""
    task_status: FLTask = None

    client_mac: str = get_mac_address()
    client_name: str = get_hostname()


manager = ManagerStatus()


def inform_se_url(path: str) -> str:
    return f"http://{manager.server_ST}/FLSe/{path}"


@app.on_event("startup")
def startup():
    loop = asyncio.get_event_loop()
    loop.set_debug(True)
    loop.create_task(check_flclient_online())
    loop.create_task(health_check())
    loop.create_task(start_training())


def fl_server_closed():
    try:
        requests.put(
            inform_se_url(f"FLSeClosed/{manager.task_id}"),
            params={"FLSeReady": "false"},
        )
        logging.info("server status FLSeReady => False")
    except Exception as e:
        logging.error(f"fl_server_closed error: {e}")


@app.get("/trainFin")
def fin_train():
    logging.info("fin")
    manager.client_training = False
    manager.FL_ready = False
    fl_server_closed()
    return manager


@app.get("/trainFail")
def fail_train():
    logging.info("Fail")
    manager.client_training = False
    manager.FL_ready = False
    fl_server_closed()
    return manager


@app.get("/info")
def get_manager_info():
    return manager


@app.get("/flclient_out")
def flclient_out():
    manager.client_online = False
    manager.client_training = False
    return manager


def async_dec(awaitable_func):
    async def keeping_state():
        while True:
            try:
                await awaitable_func()
            except Exception as e:
                logging.error("[E]%s: %s", awaitable_func.__name__, e)
            await asyncio.sleep(0.5)

    return keeping_state


@async_dec
async def health_check():
    health_check_result = {
        "client_training": manager.client_training,
        "client_online": manager.client_online,
        "FL_ready": manager.FL_ready,
    }
    logging.info("health_check - %s", json.dumps(health_check_result))

    if not manager.FL_ready:
        manager.client_training = False

    if (not manager.client_training) and manager.client_online:
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(
            None,
            requests.get,
            f"http://{manager.server_ST}/FLSe/info/{manager.task_id}/{get_mac_address()}",
        )
        if (res.status_code == 200) and (res.json()["Server_Status"]["FLSeReady"]):
            manager.FL_ready = res.json()["Server_Status"]["FLSeReady"]
            manager.GL_Model_V = res.json()["Server_Status"]["GL_Model_V"]
            task_status_data = res.json()["Server_Status"]["Task_status"]
            logging.info("task_status_data - %s", task_status_data)
            manager.task_status = FLTask(**task_status_data) if task_status_data else None
        elif res.status_code != 200:
            logging.error("FLSe/info: %s FL_server_ST offline", res.status_code)

    await asyncio.sleep(10)
    return manager


@async_dec
async def check_flclient_online():
    logging.info("Check client online info")
    if not manager.client_training:
        try:
            loop = asyncio.get_event_loop()
            res_on = await loop.run_in_executor(
                None,
                requests.get,
                f"http://{manager.FL_client}/online",
            )
            if (res_on.status_code == 200) and (res_on.json()["client_online"]):
                manager.client_online = res_on.json()["client_online"]
                manager.client_training = res_on.json()["client_start"]
                manager.task_id = res_on.json()["task_id"]
                logging.info("client_online")
            else:
                logging.info("client offline")
        except requests.exceptions.ConnectionError:
            logging.info("client offline")

        res_task = requests.put(
            inform_se_url("RegisterFLTask"),
            data=json.dumps(
                {
                    "FL_task_ID": manager.task_id,
                    "Device_mac": manager.client_mac,
                    "Device_hostname": manager.client_name,
                    "Device_online": manager.client_online,
                    "Device_training": manager.client_training,
                }
            ),
        )

        if res_task.status_code != 200:
            logging.error("FLSe/RegisterFLTask: server_ST offline")

    await asyncio.sleep(6)
    return manager


def post_request(url, json_data):
    return requests.post(url, json=json_data)


@async_dec
async def start_training():
    if manager.task_status:
        if manager.client_online and (not manager.client_training) and manager.FL_ready:
            logging.info("start training")
            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(
                None,
                post_request,
                f"http://{manager.FL_client}/start",
                {"server_ip": manager.server, "client_mac": manager.client_mac},
            )

            manager.client_training = True
            logging.info("client_start code: %s", res.status_code)
            if (res.status_code == 200) and (res.json()["FL_client_start"]):
                logging.info("flclient learning")
            elif res.status_code != 200:
                manager.client_online = False
                logging.info("flclient offline")
    else:
        logging.info("FL_task_status is None")

    await asyncio.sleep(8)
    return manager


if __name__ == "__main__":
    uvicorn.run(
        "fedopsmnist.client_manager_main:app",
        host="0.0.0.0",
        port=8004,
        reload=False,
        loop="asyncio",
    )
