from __future__ import annotations

from typing import Union

import numpy as np

from .base import BaseModel

ArrayLike = Union[np.ndarray, list]


class LogisticRegression(BaseModel):
    """Binary logistic regression trained with batch gradient descent.

    Parameters
    ----------
    lr : float, default=0.01
        Learning rate (step size) for gradient descent.
    n_iters : int, default=1000
        Number of gradient descent iterations.
    fit_intercept : bool, default=True
        Whether to add an intercept term (bias) to the model.
    tol : float | None, default=None
        Optional tolerance for early stopping based on the absolute change in
        the loss function. If *None*, run for exactly *n_iters*.
    """

    def __init__(
        self,
        lr: float = 0.01,
        n_iters: int = 1000,
        fit_intercept: bool = True,
        tol: float | None = None,
    ) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.fit_intercept = fit_intercept
        self.tol = tol

        # Learned parameters -----------------------------
        self._w: np.ndarray | None = None  # We will initialize this in fit()

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Append a column of 1s to *X* for the bias term."""
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack((ones, X))

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically–stable sigmoid (logistic) function."""
        # Guard against overflow for large negative z
        positive = z >= 0
        negative = ~positive
        out = np.empty_like(z)
        out[positive] = 1 / (1 + np.exp(-z[positive]))
        exp_z = np.exp(z[negative])
        out[negative] = exp_z / (1 + exp_z)
        return out

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LogisticRegression":
        """Fit the model to features *X* and binary targets *y* (0/1)."""
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()

        if self.fit_intercept:
            X_arr = self._add_intercept(X_arr)

        n_samples, n_features = X_arr.shape
        self._w = np.zeros(n_features, dtype=float)

        prev_loss: float | None = None
        for _ in range(self.n_iters):
            linear = X_arr @ self._w  # shape (n_samples,)
            y_pred = self._sigmoid(linear)

            # Gradient of negative log-likelihood
            grad = (X_arr.T @ (y_pred - y_arr)) / n_samples
            self._w -= self.lr * grad

            if self.tol is not None:
                # Compute loss for early stopping
                loss = -(
                    y_arr * np.log(y_pred + 1e-15)
                    + (1 - y_arr) * np.log(1 - y_pred + 1e-15)
                ).mean()
                if prev_loss is not None and abs(prev_loss - loss) < self.tol:
                    break
                prev_loss = loss
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return probabilities for the positive class (shape = (n_samples,))."""
        if self._w is None:
            raise RuntimeError("Model must be fitted before calling predict_proba().")

        X_arr = np.asarray(X, dtype=float)
        if self.fit_intercept:
            X_arr = self._add_intercept(X_arr)
        return self._sigmoid(X_arr @ self._w)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return class labels (0 or 1) for *X*."""
        return (self.predict_proba(X) >= 0.5).astype(int)

    def __repr__(self) -> str:  # pragma: no cover – cosmetic only
        params = (
            f"lr={self.lr}, n_iters={self.n_iters}, "
            f"fit_intercept={self.fit_intercept}, tol={self.tol}"
        )
        return f"LogisticRegression({params})"
