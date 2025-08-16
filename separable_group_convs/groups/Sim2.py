"""Define the Sim(2) Lie group and its constituent subgroups, SO(2) and R+.

This module provides concrete implementations for the 2D rotation group (SO(2)),
the positive scaling group (R+), and the 2D similarity group (Sim(2)), which
is the direct product of SO(2) and R+.

These classes inherit from the abstract `Group` class and provide the necessary
mathematical operations for use in equivariant neural networks.
"""

import math

import torch

from .group import Group

# --- Constituent Group Implementations ---


class SO2(Group):
    """Implement the Special Orthogonal group SO(2) for 2D rotations."""

    def __init__(self) -> None:
        """Initialize the SO(2) group. Group elements are represented by angles in radians."""
        super().__init__(dimension=1, identity=[0.0])

    def product(self, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:  # noqa: PLR6301
        """Compute the product of two rotations (addition of angles).

        Args:
            g1 (torch.Tensor): A tensor of rotation angles.
            g2 (torch.Tensor): A tensor of rotation angles.

        Returns:
            torch.Tensor: The resulting rotation angles, wrapped to [0, 2*pi).

        """
        return torch.remainder(g1 + g2, 2 * math.pi)

    def inverse(self, g: torch.Tensor) -> torch.Tensor:  # noqa: PLR6301
        """Compute the inverse of a rotation (negation of the angle).

        Args:
            g (torch.Tensor): A tensor of rotation angles.

        Returns:
            torch.Tensor: The inverse rotation angles, wrapped to [0, 2*pi).

        """
        return torch.remainder(-g, 2 * math.pi)

    def exponential_map(self, h: torch.Tensor) -> torch.Tensor:  # noqa: PLR6301
        """Map from the Lie algebra to the group via the exponential map.

        For SO(2), this is the identity, mapping algebra elements (angles) to
        group elements (angles).

        Args:
            h (torch.Tensor): A tensor of angles from the Lie algebra.

        Returns:
            torch.Tensor: The corresponding angles in the group, wrapped to [0, 2*pi).

        """
        return torch.remainder(h, 2 * math.pi)

    def logarithmic_map(self, g: torch.Tensor) -> torch.Tensor:  # noqa: PLR6301
        """Map from the Lie group to the Lie algebra via the logarithmic map.

        For SO(2), this is the identity.

        Args:
            g (torch.Tensor): A tensor of angles from the Lie group.

        Returns:
            torch.Tensor: The corresponding angles in the Lie algebra.

        """
        return g

    def representation(self, g: torch.Tensor) -> torch.Tensor:  # noqa: PLR6301
        """Return the 2x2 rotation matrix for a given angle.

        Args:
            g (torch.Tensor): A tensor of rotation angles. Shape (...,).

        Returns:
            torch.Tensor: The corresponding rotation matrices. Shape (..., 2, 2).

        """
        cos_g = torch.cos(g)
        sin_g = torch.sin(g)
        # Tensors are created on the same device as the input `g`.
        return torch.stack([
            torch.stack([cos_g, -sin_g], dim=-1),
            torch.stack([sin_g, cos_g], dim=-1),
        ], dim=-2)

    def left_action_on_Rd(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:  # noqa: N802
        """Rotate a vector x in R^2 by the group element g.

        Args:
            g (torch.Tensor): A batch of rotation angles. Shape (B,).
            x (torch.Tensor): A batch of 2D vectors. Shape (B, 2, ...).

        Returns:
            torch.Tensor: The rotated vectors. Shape (B, 2, ...).

        """
        rot_matrix = self.representation(g)
        return torch.einsum("bij,j...->bi...", rot_matrix, x)

    def determinant(self, g: torch.Tensor) -> torch.Tensor:
        """Return the determinant of an SO(2) rotation matrix, which is always 1.

        Args:
            g (torch.Tensor): A tensor of rotation angles.

        Returns:
            torch.Tensor: A tensor of ones with the same shape and device as g.

        """
        return torch.ones_like(g)

    def normalize(self, g: torch.Tensor) -> torch.Tensor:
        """Normalize angles from [0, 2*pi] to [-1, 1] for network input.

        Args:
            g (torch.Tensor): A tensor of angles in radians.

        Returns:
            torch.Tensor: The normalized angles in the range [-1, 1].

        """
        return (g / math.pi) - 1.0

    def sample(self, num_elements: int, method: str = "discretise") -> torch.Tensor:
        """Sample angles from the group.

        Args:
            num_elements (int): The number of angles to sample.
            method (str): 'discretise' for a uniform grid or 'uniform' for random sampling.

        Returns:
            torch.Tensor: A tensor of sampled angles. Shape (num_elements,).

        Raises:
            ValueError: If the sampling method is unknown.

        """
        if method == "discretise":
            return torch.linspace(0, 2 * math.pi, num_elements + 1)[:-1]
        if method == "uniform":
            return torch.rand(num_elements) * 2 * math.pi
        msg = f"Unknown sampling method: {method}"
        raise ValueError(msg)


class Rplus(Group):
    """Implement the positive real numbers under multiplication (R+), for scaling."""

    def __init__(self, max_scale: float = 3.0) -> None:
        """Initialize the R+ group.

        Args:
            max_scale (float): The maximum scale factor to consider. The scale
                range will be [1/max_scale, max_scale].

        Raises:
            ValueError: If max_scale is not positive.

        """
        super().__init__(dimension=1, identity=[1.0])
        if max_scale <= 0:
            msg = "max_scale must be positive."
            raise ValueError(msg)
        self.max_scale = max_scale
        self.min_scale = 1.0 / max_scale

    def product(self, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        """Compute the product of two scales (multiplication).

        Args:
            g1 (torch.Tensor): A tensor of scale factors.
            g2 (torch.Tensor): A tensor of scale factors.

        Returns:
            torch.Tensor: The product of the scale factors.

        """
        return g1 * g2

    def inverse(self, g: torch.Tensor) -> torch.Tensor:
        """Compute the inverse of a scale (reciprocal).

        Args:
            g (torch.Tensor): A tensor of scale factors.

        Returns:
            torch.Tensor: The inverse scale factors.

        """
        return 1.0 / g

    def exponential_map(self, h: torch.Tensor) -> torch.Tensor:
        """Map from the algebra (log-scale) to the group (scale) via exp.

        Args:
            h (torch.Tensor): A tensor of log-scale values from the Lie algebra.

        Returns:
            torch.Tensor: The corresponding scale factors in the group.

        """
        return torch.exp(h)

    def logarithmic_map(self, g: torch.Tensor) -> torch.Tensor:
        """Map from the group (scale) to the algebra (log-scale) via log.

        Args:
            g (torch.Tensor): A tensor of scale factors from the Lie group.

        Returns:
            torch.Tensor: The corresponding log-scale values in the Lie algebra.

        """
        return torch.log(g)

    def representation(self, g: torch.Tensor) -> torch.Tensor:
        """Return the representation, which is the scale factor itself for isotropic scaling.

        Args:
            g (torch.Tensor): A tensor of scale factors.

        Returns:
            torch.Tensor: The input tensor of scale factors.

        """
        return g

    def left_action_on_Rd(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Scale a vector x in R^d by the group element g.

        Args:
            g (torch.Tensor): A batch of scale factors. Shape (B,).
            x (torch.Tensor): A batch of d-dimensional vectors. Shape (B, d, ...).

        Returns:
            torch.Tensor: The scaled vectors.

        """
        scale = g.view(-1, *([1] * (x.dim() - 1)))
        return scale * x

    def determinant(self, g: torch.Tensor) -> torch.Tensor:
        """Return the determinant of the Jacobian for a scaling `s` in R^d, which is `s^d`.

        Args:
            g (torch.Tensor): A tensor of scale factors.

        Returns:
            torch.Tensor: The determinants (g^2 for R^2).

        """
        return g ** 2

    def normalize(self, g: torch.Tensor) -> torch.Tensor:
        """Normalize log-scale values to [-1, 1] for network input.

        Args:
            g (torch.Tensor): A tensor of log-scale values from the Lie algebra.

        Returns:
            torch.Tensor: The normalized log-scale values.

        """
        log_max = math.log(self.max_scale)
        return g / log_max

    def sample(self, num_elements: int, method: str = "discretise") -> torch.Tensor:
        """Sample scales from the group, typically uniformly in the log domain.

        Args:
            num_elements (int): The number of scales to sample.
            method (str): 'discretise' for a log-uniform grid or 'uniform' for random log-uniform.

        Returns:
            torch.Tensor: A tensor of sampled scale factors. Shape (num_elements,).

        Raises:
            ValueError: If the sampling method is unknown.

        """
        log_max = math.log(self.max_scale)
        log_min = math.log(self.min_scale)

        if method == "discretise":
            log_scales = torch.linspace(log_min, log_max, num_elements)
        elif method == "uniform":
            log_scales = torch.rand(num_elements) * (log_max - log_min) + log_min
        else:
            msg = f"Unknown sampling method: {method}"
            raise ValueError(msg)

        return self.exponential_map(log_scales)

# --- Main Sim(2) Group Implementation ---


class Sim2(Group):
    """Implement the Sim(2) group, the direct product of SO(2) and R+."""

    def __init__(self, max_scale: float = 3.0) -> None:
        """Initialize the Sim(2) group.

        Args:
            max_scale (float): The maximum scale factor to consider. The scale
                range will be [1/max_scale, max_scale].

        """
        super().__init__(dimension=2, identity=[0.0, 1.0])
        self.so2 = SO2()
        self.rplus = Rplus(max_scale=max_scale)

    def product(self, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        """Compute the component-wise product for the direct product group.

        Args:
            g1 (torch.Tensor): A tensor of Sim(2) elements. Shape (..., 2).
            g2 (torch.Tensor): A tensor of Sim(2) elements. Shape (..., 2).

        Returns:
            torch.Tensor: The resulting Sim(2) elements. Shape (..., 2).

        """
        rotations = self.so2.product(g1[..., 0], g2[..., 0])
        scales = self.rplus.product(g1[..., 1], g2[..., 1])
        return torch.stack([rotations, scales], dim=-1)

    def inverse(self, g: torch.Tensor) -> torch.Tensor:
        """Compute the component-wise inverse.

        Args:
            g (torch.Tensor): A tensor of Sim(2) elements. Shape (..., 2).

        Returns:
            torch.Tensor: The inverse Sim(2) elements. Shape (..., 2).

        """
        inv_rotations = self.so2.inverse(g[..., 0])
        inv_scales = self.rplus.inverse(g[..., 1])
        return torch.stack([inv_rotations, inv_scales], dim=-1)

    def exponential_map(self, h: torch.Tensor) -> torch.Tensor:
        """Compute the component-wise exponential map.

        Args:
            h (torch.Tensor): A tensor from the Sim(2) Lie algebra. Shape (..., 2).

        Returns:
            torch.Tensor: The corresponding elements in the Sim(2) group. Shape (..., 2).

        """
        rotations = self.so2.exponential_map(h[..., 0])
        scales = self.rplus.exponential_map(h[..., 1])
        return torch.stack([rotations, scales], dim=-1)

    def logarithmic_map(self, g: torch.Tensor) -> torch.Tensor:
        """Compute the component-wise logarithmic map.

        Args:
            g (torch.Tensor): A tensor of Sim(2) elements. Shape (..., 2).

        Returns:
            torch.Tensor: The corresponding elements in the Lie algebra. Shape (..., 2).

        """
        log_rotations = self.so2.logarithmic_map(g[..., 0])
        log_scales = self.rplus.logarithmic_map(g[..., 1])
        return torch.stack([log_rotations, log_scales], dim=-1)

    def representation(self, g: torch.Tensor) -> torch.Tensor:
        """Return the 2x2 matrix representation: s * R.

        Args:
            g (torch.Tensor): A tensor of Sim(2) elements. Shape (..., 2).

        Returns:
            torch.Tensor: The corresponding 2x2 matrices. Shape (..., 2, 2).

        """
        rot_matrices = self.so2.representation(g[..., 0])
        scales = g[..., 1]
        return scales.unsqueeze(-1).unsqueeze(-1) * rot_matrices

    def left_action_on_Rd(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:  # noqa: N802
        """Apply rotation and scaling to a vector x in R^2.

        Args:
            g (torch.Tensor): A batch of Sim(2) elements. Shape (B, 2).
            x (torch.Tensor): A batch of 2D vectors. Shape (2, ...).

        Returns:
            torch.Tensor: The transformed vectors. Shape (B, 2, ...).

        """
        transform_matrix = self.representation(g)
        return torch.einsum("bij,j...->bi...", transform_matrix, x)

    def determinant(self, g: torch.Tensor) -> torch.Tensor:
        """Return the determinant of the transformation matrix, which is s^2.

        Args:
            g (torch.Tensor): A tensor of Sim(2) elements. Shape (..., 2).

        Returns:
            torch.Tensor: The determinants of the transformations.

        """
        scales = g[..., 1]
        return self.rplus.determinant(scales)

    def normalize(self, g: torch.Tensor) -> torch.Tensor:
        """Normalize Lie algebra elements component-wise to [-1, 1].

        Args:
            g (torch.Tensor): A tensor from the Sim(2) Lie algebra. Shape (..., 2).

        Returns:
            torch.Tensor: The normalized tensor. Shape (..., 2).

        """
        norm_rot = self.so2.normalize(g[..., 0])
        norm_scale = self.rplus.normalize(g[..., 1])
        return torch.stack([norm_rot, norm_scale], dim=-1)

    def sample(
        self,
        num_elements: int,
        method: str = "discretise",
    ) -> torch.Tensor:
        """Sample elements from the Sim(2) group.

        Args:
            num_elements (int): Samples a square grid of
                sqrt(num_elements) x sqrt(num_elements).
            method (str): The sampling method for the SO(2) part, either
                'discretise' or 'uniform'. R+ is always discretised.
            separable (bool): If True, returns separate tensors for rotation and
                scale samples. Useful for separable convolutions.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: If separable is False,
                a tensor of shape (n_rot * n_scale, 2). If True, a tuple of
                (rot_samples, scale_samples).

        """
        rot_samples = self.so2.sample(num_elements, method=method)
        scale_samples = self.rplus.sample(num_elements, method="discretise")

        # Create a grid from the samples, ensuring it's on the correct device.
        # torch.meshgrid will create new tensors.
        grid_components = torch.meshgrid(rot_samples, scale_samples, indexing="ij")

        grid = torch.stack(grid_components, dim=-1)
        return grid.reshape(-1, 2)
