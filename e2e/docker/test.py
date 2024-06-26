"""Tests for Flower Docker containers."""

import re
import time
from typing import Callable, Union

import pytest
from testcontainers.compose import ContainerIsNotRunning, DockerCompose

COMPOSE_MANIFEST = "compose.yaml"
RUN_COMPLETE_STR = "Run finished"


def test_compose_and_teardown():
    """Test if Flower containers can successfully start and teardown."""
    # create the services
    services = DockerCompose(context=".", compose_file_name=COMPOSE_MANIFEST)

    try:
        # first the containers do not exist
        containers = services.get_containers(include_all=True)
        assert len(containers) == 0

        # then we create them and they exists
        services.start()
        containers = services.get_containers(include_all=True)
        assert len(containers) == 4
        containers = services.get_containers()
        assert len(containers) == 4

        # all containers should be running
        assert containers[0].State == "running"
        assert containers[1].State == "running"
        assert containers[2].State == "running"
        assert containers[3].State == "running"

        # each container has a specific name with the order
        # determined by the `depends_on` list of services
        assert containers[0].Service == "serverapp"
        assert containers[1].Service == "superlink"
        assert containers[2].Service == "supernode"
        assert containers[3].Service == "supernode"

        # stop the services but don't remove containers,
        # networks, volumes, and images
        services.stop(down=False)

        # Check the state for ServerApp and SuperLink.
        # TODO: Add test for SuperNode services  # pylint: disable=W0511
        with pytest.raises(ContainerIsNotRunning):
            assert services.get_container("serverapp") is None

        with pytest.raises(ContainerIsNotRunning):
            assert services.get_container("superlink") is None

        # check that ServerApp service has exited
        serverapp = services.get_container("serverapp", include_all=True)
        assert serverapp.State == "exited"

        # check that SuperLink service has exited
        superlink = services.get_container("superlink", include_all=True)
        assert superlink.State == "exited"
    finally:
        services.stop()


def test_compose_logs():
    """Test if Flower containers can successfully finish training."""
    # create the services
    services = DockerCompose(context=".", compose_file_name=COMPOSE_MANIFEST)

    with services:
        wait_for_logs(services, RUN_COMPLETE_STR, 120)


def wait_for_logs(
    services: "DockerCompose",
    predicate: Union[Callable, str],
    timeout: float = 60,
    interval: float = 1,
) -> float:
    """Wait for the services to emit logs satisfying the predicate.

    Args:
        services: Services whose logs to wait for.
        predicate: Predicate that should be satisfied by the logs. If a string,
            then it is used as the pattern for a multiline regular expression search.
        timeout: Number of seconds to wait for the predicate to be satisfied.
            Defaults to wait indefinitely.
        interval: Interval at which to poll the logs.

    Returns
    -------
        duration: Number of seconds until the predicate was satisfied.
    """
    if isinstance(predicate, str):
        predicate = re.compile(predicate, re.MULTILINE).search
    start = time.time()
    while True:
        duration = time.time() - start
        stdout, stderr = services.get_logs()
        if predicate(stdout) or predicate(stderr):
            return duration
        if duration > timeout:
            raise TimeoutError(
                f"Services did not emit logs satisfying predicate in {timeout:.3f} "
                "seconds"
            )
        time.sleep(interval)
