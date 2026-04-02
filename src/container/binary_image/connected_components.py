import numpy as np
from dataclasses import dataclass

@dataclass
class ConnectedComponents:
    """
    A class to represent connected components.

    Attributes
    ----------
    n_labels: int
        The number of connected components.
    labels: np.ndarray
        The labels of the connected components.
    stats: np.ndarray
        The statistics of the connected components.
    centroids: np.ndarray
        The centroids of the connected components.
    """
    n_labels: int
    labels: np.ndarray
    stats: np.ndarray
    centroids: np.ndarray