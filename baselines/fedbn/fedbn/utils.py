"""fedbn: A Flower Baseline."""


def extract_weights(net, algorithm_name):
    """Extract model parameters as numpy arrays from state_dict."""
    if algorithm_name == "FedAvg":
        return [val.cpu().numpy() for _, val in net.state_dict().items()]
    return [
        val.cpu().numpy()
        for name, val in net.state_dict().items()
        if "bn" not in name
    ]
