"""Tests for Flower Docker containers."""

import re
from time import sleep
from typing import Callable, Union

import pytest
from testcontainers.compose import ContainerIsNotRunning, DockerCompose

COMPOSE_MANIFEST = "compose.yaml"
RUN_COMPLETE_STR = "Run finished 3 round(s) in"


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
        sleep(45)  # generate some logs
        stdout, stderr = services.get_logs()

    assert not stderr
    assert stdout

    predicate: Union[Callable, str] = RUN_COMPLETE_STR
    predicate = re.compile(predicate, re.MULTILINE).search

    assert predicate(
        stdout
    ), "Containers did not finish training in the allocated time."
