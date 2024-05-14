from testcontainers.compose import DockerCompose


def test_compose():

    # Create a tc-DockerCompose object
    services = DockerCompose(context=".", compose_file_name="compose.yaml")

    containers = services.get_containers(include_all=True)
    assert len(containers) == 4
