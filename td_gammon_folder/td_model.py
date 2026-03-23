"""
Neural network code for the TD-Gammon-style implementation.

This file only contains the value function approximator.

The model predicts how good a position is from the CURRENT PLAYER'S
perspective. That perspective idea is very important in zero-sum games.

IMPORTANT:
This version uses NumPy instead of PyTorch so it can run on systems
where torch is not installed.
"""

from typing import Dict, Tuple

import numpy as np


class ValueNet:
    """
    A multilayer perceptron that predicts the value of a position.

    Output range: [-1, +1] because of tanh.
      +1 means: very good for the current player
      -1 means: very bad for the current player

    Why this matters:
    Full backgammon has a huge state space, so we do not use a table.
    Instead, we approximate the value function with a neural network.

    This matches the main TD-Gammon idea:
    use a neural network to estimate board value during self-play.
    """

    def __init__(self, input_dim: int, hidden_size: int):
        """
        Build the network.

        Parameters
        ----------
        input_dim:
            Number of input features in the board encoding.
        hidden_size:
            Width of the hidden layers.
        """
        rng = np.random.default_rng()

        # Layer 1
        self.W1 = rng.normal(0.0, 0.1, size=(input_dim, hidden_size))
        self.b1 = np.zeros(hidden_size, dtype=np.float64)

        # Layer 2
        self.W2 = rng.normal(0.0, 0.1, size=(hidden_size, hidden_size))
        self.b2 = np.zeros(hidden_size, dtype=np.float64)

        # Output layer
        self.W3 = rng.normal(0.0, 0.1, size=(hidden_size, 1))
        self.b3 = np.zeros(1, dtype=np.float64)

    def forward(self, x: np.ndarray) -> float:
        """
        Predict V(s) for one encoded board state.

        Parameters
        ----------
        x:
            1D NumPy array of input features.

        Returns
        -------
        float
            Scalar value estimate in [-1, +1].
        """
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(0.0, z1)  # ReLU

        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0.0, z2)  # ReLU

        z3 = a2 @ self.W3 + self.b3
        y = np.tanh(z3[0])

        return float(y)

    def forward_with_cache(self, x: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Forward pass that also stores intermediate values needed for backpropagation.

        This is used during TD(lambda) updates so we can manually compute gradients
        without needing PyTorch.
        """
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(0.0, z1)

        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0.0, z2)

        z3 = a2 @ self.W3 + self.b3
        y = np.tanh(z3[0])

        cache = {
            "x": x,
            "z1": z1,
            "a1": a1,
            "z2": z2,
            "a2": a2,
            "z3": z3,
            "y": np.array([y], dtype=np.float64),
        }

        return float(y), cache

    def gradients(self, cache: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute gradients of V(s) with respect to all network parameters.

        This is the manual NumPy replacement for autograd.

        Returns
        -------
        dict
            Gradients with keys:
            - W1, b1
            - W2, b2
            - W3, b3
        """
        x = cache["x"]
        z1 = cache["z1"]
        a1 = cache["a1"]
        z2 = cache["z2"]
        a2 = cache["a2"]
        y = cache["y"][0]

        # Output derivative for tanh
        dz3 = 1.0 - y * y  # scalar derivative of tanh

        # Gradients for W3, b3
        dW3 = np.outer(a2, np.array([dz3], dtype=np.float64))
        db3 = np.array([dz3], dtype=np.float64)

        # Backprop into second hidden layer
        da2 = self.W3[:, 0] * dz3
        dz2 = da2 * (z2 > 0.0).astype(np.float64)

        dW2 = np.outer(a1, dz2)
        db2 = dz2

        # Backprop into first hidden layer
        da1 = self.W2 @ dz2
        dz1 = da1 * (z1 > 0.0).astype(np.float64)

        dW1 = np.outer(x, dz1)
        db1 = dz1

        return {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
            "W3": dW3,
            "b3": db3,
        }

    def zero_traces(self) -> Dict[str, np.ndarray]:
        """
        Create zero-filled eligibility traces matching all network parameters.
        """
        return {
            "W1": np.zeros_like(self.W1),
            "b1": np.zeros_like(self.b1),
            "W2": np.zeros_like(self.W2),
            "b2": np.zeros_like(self.b2),
            "W3": np.zeros_like(self.W3),
            "b3": np.zeros_like(self.b3),
        }

    def apply_td_update(
        self,
        traces: Dict[str, np.ndarray],
        grads: Dict[str, np.ndarray],
        delta: float,
        alpha: float,
        gamma: float,
        lam: float,
    ) -> None:
        """
        Apply one semi-gradient TD(lambda) update using manual eligibility traces.

        trace <- gamma * lambda * trace + grad(V(S_t))
        param <- param + alpha * delta * trace
        """
        for name in traces.keys():
            traces[name] = gamma * lam * traces[name] + grads[name]

        self.W1 += alpha * delta * traces["W1"]
        self.b1 += alpha * delta * traces["b1"]
        self.W2 += alpha * delta * traces["W2"]
        self.b2 += alpha * delta * traces["b2"]
        self.W3 += alpha * delta * traces["W3"]
        self.b3 += alpha * delta * traces["b3"]