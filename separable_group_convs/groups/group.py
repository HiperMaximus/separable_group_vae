"""Module for defining an abstract Lie group class.

This module provides an abstract base class, `Group`, which serves as an
interface for implementing various Lie groups. Any specific Lie group
implementation (e.g., SO(2), SE(2), Sim(2)) should inherit from this class
and implement its abstract methods. This ensures a consistent API for
group operations used in equivariant neural networks.
"""

import torch
from torch import nn


class Group(nn.Module):
    """Abstract base class for a Lie group.

    This class defines the essential operations and properties of a Lie group
    for use in a geometric deep learning context. It is designed to be subclassed
    to implement specific groups. The methods handle group-theoretic operations,
    mappings to and from the Lie algebra, and actions on vector spaces.
    """

    def __init__(self, dimension: int, identity: list[float]) -> None:
        """Initialize the Group.

        Args:
            dimension (int): The dimensionality of the Lie algebra, which is the
                number of parameters needed to specify a group element locally.
            identity (list[float]): A list of floats representing the identity
                element of the group.

        Raises:
            ValueError: If the dimension is not a positive integer.
            ValueError: If the length of the identity list does not match the dimension.

        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        if dimension <= 0:
            msg = "Group dimension must be a positive integer."
            raise ValueError(msg)
        if len(identity) != dimension:
            msg = f"Identity element length {len(identity)} must match group dimension {dimension}."
            raise ValueError(msg)

        self.dimension: int = dimension
        self.register_buffer("identity", torch.tensor(identity, dtype=torch.float32))

    def product(self, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        """Compute the group product of two group elements.

        This method should be implemented to compute `g1 * g2`.

        Args:
            g1 (torch.Tensor): A tensor representing the first group element(s).
                               Shape: (..., dimension).
            g2 (torch.Tensor): A tensor representing the second group element(s).
                               Shape: (..., dimension).

        Returns:
            torch.Tensor: The result of the group product. Shape: (..., dimension).

        """
        raise NotImplementedError

    def inverse(self, g: torch.Tensor) -> torch.Tensor:
        """Compute the inverse of a group element.

        Args:
            g (torch.Tensor): A tensor representing the group element(s) to be inverted.
                              Shape: (..., dimension).

        Returns:
            torch.Tensor: The inverse of the group element(s). Shape: (..., dimension).

        """
        raise NotImplementedError

    def exponential_map(self, h: torch.Tensor) -> torch.Tensor:
        """Map an element from the Lie algebra to the Lie group (exp).

        Args:
            h (torch.Tensor): An element in the Lie algebra. Shape: (..., dimension).

        Returns:
            torch.Tensor: The corresponding element in the Lie group. Shape: (..., dimension).

        """
        raise NotImplementedError

    def logarithmic_map(self, g: torch.Tensor) -> torch.Tensor:
        """Map an element from the Lie group to the Lie algebra (log) when possible.

        Args:
            g (torch.Tensor): An element in the Lie group. Shape: (..., dimension).

        Returns:
            torch.Tensor: The corresponding element in the Lie algebra. Shape: (..., dimension).

        """
        raise NotImplementedError

    def left_action_on_Rd(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:  # noqa: N802
        """Compute the left action of the group on a vector in R^d.

        For example, for SO(2), this would be the rotation of a 2D vector.

        Args:
            g (torch.Tensor): The group element(s) acting on the vector(s).
                              Shape: (num_elements, dimension).
            x (torch.Tensor): The vector(s) in R^d to be transformed.
                              Shape: (d, ...).

        Returns:
            torch.Tensor: The transformed vector(s).

        """
        raise NotImplementedError

    def left_action_on_H(self, g: torch.Tensor, h: torch.Tensor) -> torch.Tensor:  # noqa: N802
        """Compute the left action of the group on itself (the regular representation).

        This is typically equivalent to the group product.

        Args:
            g (torch.Tensor): The group element(s) acting. Shape: (..., dimension).
            h (torch.Tensor): The group element(s) being acted upon. Shape: (..., dimension).

        Returns:
            torch.Tensor: The transformed group element(s). Shape: (..., dimension).

        """
        raise NotImplementedError

    def representation(self, g: torch.Tensor) -> torch.Tensor:
        """Return a matrix representation of a group element.

        This is often used to implement the group action on R^d.

        Args:
            g (torch.Tensor): The group element(s). Shape: (..., dimension).

        Returns:
            torch.Tensor: The matrix representation(s).

        """
        raise NotImplementedError

    def determinant(self, g: torch.Tensor) -> torch.Tensor:
        """Calculate the determinant of the representation of a group element.

        This is used for normalization in the group convolution integral to account
        for the change of volume induced by the group action.

        Args:
            g (torch.Tensor): The group element(s). Shape: (..., dimension).

        Returns:
            torch.Tensor: The determinant(s). Shape: (...).

        """
        raise NotImplementedError

    def normalize(self, g: torch.Tensor) -> torch.Tensor:
        """Normalize group element coordinates to a canonical range (e.g., [-1, 1]).

        This is useful when feeding group coordinates into a neural network like a SIREN.

        Args:
            g (torch.Tensor): A tensor of group elements or algebra elements.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        raise NotImplementedError

    def sample(self, num_elements: int, method: str = "discretise") -> torch.Tensor:
        """Sample elements from the group.

        This is used to create the grid over which group convolutions are performed.

        Args:
            num_elements (int): The number of elements to sample.
            method (str): The sampling method, e.g., 'discretise' for a uniform
                grid or 'uniform' for random sampling.

        Returns:
            torch.Tensor: A tensor of sampled group elements. Shape: (num_elements, dimension).

        """
        raise NotImplementedError
