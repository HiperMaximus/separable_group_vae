"""Module to implement the SIREN (Sinusoidal Representation Networks) architecture.

It includes the SineLayer class, which uses a sine activation function for periodic
representation learning, as described in the SIREN paper. Aswell as the SIREN MLP.
"""

import math

import torch
from torch import nn


class SineLayer(nn.Module):
    """Sine Layer with a periodic activation function.

    This layer implements the sine activation function: `sin(omega_0 * (Wx + b))`.
    It is a core component of Sinusoidal Representation Networks (SIRENs) and
    can be used as the first layer or a subsequent hidden layer. The weight
    initialization strategy is adjusted based on whether it's the first layer,
    as described in the SIREN paper.
    """

    def __init__(self, in_features: int, out_features: int, omega_0: float = 20.0, *, is_first: bool = False) -> None:
        """Initialize the SineLayer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            omega_0 (float): The frequency parameter for the sine activation.
                             A value of 30.0 is recommended for the first layer.
            is_first (bool): If True, initializes weights for the first layer
                             according to the SIREN paper. Default is False.

        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.is_first: bool = is_first
        self.omega_0: float = omega_0

        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize the weights of the linear layer."""
        with torch.no_grad():
            if self.is_first:
                # Initialization for the first layer (Sec 3.2 and Appendix 1.5)
                # Weights uniformly distributed in [-1/in_features, 1/in_features]
                bound: float = 1.0 / self.in_features
                self.linear.weight.uniform_(from_=-bound, to=bound)
            else:
                # Initialization for subsequent layers (Theorem 1.8, Appendix 1.3 & 1.5)
                # Weights uniformly distributed in [-sqrt(6/in_features)/omega_0, sqrt(6/in_features)/omega_0]
                # This corresponds to W_hat when W = W_hat * omega_0
                bound: float = math.sqrt(6.0 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

            # Bias initialization (often to zero or small uniform)
            # The SIREN paper is less explicit or varies in examples.
            # For hypernetwork target SIREN (Appendix 9.1), bias was U[-1/n, 1/n]
            # added omega_0 to follow initialization observations.
            bound: float = 1 / (self.in_features * self.omega_0)
            self.linear.bias.uniform_(-bound, bound)  # Or self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the sine layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ..., in_features).
                              The input should be normalized to [-1, 1] for best results.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, ..., out_features).
        The output is computed as:
            Output = sin(omega_0 * (Wx + b)), where W is the weight matrix and b is the bias.
            The sine activation introduces non-linearity to the output.
        Output = sin(omega_0 * (Wx + b)).

        """
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """Sinusoidal Representation Networks (SIREN).

    This class implements a multi-layer perceptron (MLP) that uses sine
    activation functions (`SineLayer`) for all hidden layers. This architecture
    is particularly well-suited for representing complex, continuous signals
    and their derivatives, as detailed in the paper "Implicit Neural
    Representations with Periodic Activation Functions".
    """

    def __init__(
        self,
        input_feature_dimension: int,
        output_feature_dimension: int,
        list_hidden_layer_dimensions: list[int],
        omega_0: float = 30.0,
    ) -> None:
        """Initialize the SIREN.

        Args:
            input_feature_dimension (int): Dimension of input features.
            output_feature_dimension (int): Dimension of output features.
            list_hidden_layer_dimensions (list[int]): List of integers representing
                the number of features for each hidden layer. For example,
                `[256, 256]` creates two hidden layers with 256 features each.
            omega_0 (float): The frequency parameter for the sine activation.
                A value of 30.0 is recommended.

        Raises:
            ValueError: If `input_feature_dimension` or `output_feature_dimension`
                        are not positive integers.
            ValueError: If `list_hidden_layer_dimensions` is empty.

        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        # Param validation
        if input_feature_dimension <= 0:
            msg = "Input features must be a positive integer."
            raise ValueError(msg)
        if output_feature_dimension <= 0:
            msg = "Output features must be a positive integer."
            raise ValueError(msg)
        if not list_hidden_layer_dimensions:
            msg = "List of hidden features cannot be empty (min 2 linear layers)."
            raise ValueError(msg)

        # Build the list of layers for the sequential model
        layers: list[nn.Module] = []
        # First input layer
        layers.append(SineLayer(
                in_features=input_feature_dimension,
                out_features=list_hidden_layer_dimensions[0],
                omega_0=omega_0,
                is_first=True,
            ))

        # Hidden SineLayers
        layers.extend(SineLayer(
                in_features=list_hidden_layer_dimensions[i],
                out_features=list_hidden_layer_dimensions[i + 1],
                omega_0=omega_0,
                is_first=False,
            ) for i in range(len(list_hidden_layer_dimensions) - 1))

        # Final linear layer
        layers.append(nn.Linear(
            in_features=list_hidden_layer_dimensions[-1],
            out_features=output_feature_dimension,
            bias=True,
        ))

        self.net: nn.Sequential = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SIREN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ..., in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, ..., out_features).

        """
        return self.net(x)
