from testcontainers.compose import DockerCompose
import pytest


def test_compose():

    # Create a tc-DockerCompose object
    services = DockerCompose(context=".", compose_file_name="compose.yaml")

    # containers = services.get_containers(include_all=True)
    # assert len(containers) == 4

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

        print(containers)
        assert containers[0].State == "running"
        assert containers[1].State == "running"
        assert containers[2].State == "running"
        assert containers[3].State == "running"

        assert containers[0].Service == "serverapp"
        assert containers[1].Service == "superlink"
        assert containers[2].Service == "supernode"
        assert containers[3].Service == "supernode"
        # # test that get_container returns the same object, value assertions, etc
        # from_all = containers[0]
        # assert from_all.State == "running"
        # assert from_all.Service == "alpine"

        # by_name = services.get_container("alpine")

        # assert by_name.Name == from_all.Name
        # assert by_name.Service == from_all.Service
        # assert by_name.State == from_all.State
        # assert by_name.ID == from_all.ID

        # assert by_name.ExitCode == 0

        # # what if you want to get logs after it crashes:
        # services.stop(down=False)

        # with pytest.raises(ContainerIsNotRunning):
        #     assert services.get_container("serverapp") is None

        # # what it looks like after it exits
        # stopped = services.get_container("alpine", include_all=True)
        # assert stopped.State == "exited"
    finally:
        services.stop()
    
    services.stop()

# @pytest.mark.parametrize(
#         "container_state,desired_state",
#         [zip(container.State, "running") for container in containers]
# )
# def test_running_containers(container_state, desired_state):
#     print("lal")
#     assert container_state == desired_state
