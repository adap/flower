"""Base sampler class."""

from collections.abc import Generator

from torch.nn import Module


class BaseSampler:
    """Base class implementing p-value sampling per layer."""

    def __init__(self, model: Module, with_layer: bool = False) -> None:
        """Initialise sampler.

        :param model: OD model
        :param with_layer: whether to return layer upon call.
        """
        self.model = model
        self.with_layer = with_layer
        self.prepare_sampler()
        self.width_samples = self.width_sampler()
        self.layer_samples = self.layer_sampler()

    def prepare_sampler(self) -> None:
        """Prepare sampler."""
        self.num_od_layers = 0
        self.widths = []
        self.od_layers = []
        for m in self.model.modules():
            if hasattr(m, "is_od") and m.is_od:
                self.num_od_layers += 1
                self.widths.append(m.width)
                self.od_layers.append(m)

    def width_sampler(self) -> Generator:  # pylint: disable=no-self-use
        """Sample width."""
        while True:
            yield None

    def layer_sampler(self) -> Module:
        """Sample layer."""
        while True:
            for m in self.od_layers:
                yield m

    def __call__(self):
        """Call sampler."""
        if self.with_layer:
            return next(self.width_samples), next(self.layer_samples)

        return next(self.width_samples)
