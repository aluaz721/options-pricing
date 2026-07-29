from abc import ABC, abstractmethod

import numpy as np


class Payoff(ABC):
    """A payoff is a function of a simulated price path, not just a terminal price.

    This is what lets vanilla and path-dependent payoffs share one interface: vanilla
    payoffs simply ignore everything but the last column of `paths`.
    """

    # Not consulted by any engine's supports() check yet (those dispatch on
    # concrete payoff type instead), but every path-dependent payoff sets it —
    # reserved for engines that want to reject "anything path-dependent" in
    # one check rather than enumerating every payoff subclass.
    is_path_dependent: bool = False

    @abstractmethod
    def __call__(self, paths: np.ndarray) -> np.ndarray:
        """paths: (n_paths, n_steps + 1) array, path[:, 0] is spot, path[:, -1] is terminal.

        Returns an (n_paths,) array of undiscounted payoff values.
        """
        raise NotImplementedError
