import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from sklearn.utils.validation import validate_data

class GeneralizedRidge(BaseEstimator, RegressorMixin):
    def __init__(self, alphas=1.0, solver="auto", fit_intercept=True):
        """
        Generalized Ridge Regression with per-feature penalties.

        There is some prior work on this (for theoretical justification), e.g.
        https://academic.oup.com/biostatistics/article/22/2/348/5584177
        https://www.tandfonline.com/doi/full/10.1080/10618600.2021.1904962#d1e151
        https://www.tandfonline.com/doi/full/10.1080/10618600.2019.1624294#d1e423

        Parameters
        ----------
        alphas : array-like of shape (n_features,) or float
            Per-feature penalties (diagonal of Λ). If a scalar, the same penalty
            is applied to all features.
        solver : {"auto", "cholesky", "svd"}, default="auto"
            Numerical solver to use.
        fit_intercept : bool, default=True
            Whether to calculate the intercept for this model.
        """

        self.alphas = alphas
        self.solver = solver
        self.fit_intercept = fit_intercept

    def set_alphas(self, alphas):
        self.alphas = alphas

    def fit(self, X, y):
        # Validate inputs
        X, y = validate_data(self, X=X, y=y)
        n_samples, n_features = X.shape

        # Validate alphas
        if np.ndim(self.alphas) == 0:
            alphas = np.full(n_features, float(self.alphas))
        else:
            alphas = np.asarray(self.alphas, dtype=float)
            if alphas.shape[0] != n_features:
                raise ValueError(
                    f"alphas must have length {n_features}, got {alphas.shape[0]}"
                )
        self.alphas_ = alphas

        # Center data if fit_intercept
        if self.fit_intercept:
            self.X_mean_ = X.mean(axis=0)
            self.y_mean_ = y.mean()
            Xc = X - self.X_mean_
            yc = y - self.y_mean_
        else:
            self.X_mean_ = np.zeros(n_features)
            self.y_mean_ = 0.0
            Xc, yc = X, y

        # Construct matrices
        A = Xc.T @ Xc + np.diag(self.alphas_)
        b = Xc.T @ yc

        solver = self.solver
        if solver == "auto":
            solver = "cholesky" if n_features < 5000 else "svd"

        if solver == "cholesky":
            c, low = linalg.cho_factor(A)
            coef = linalg.cho_solve((c, low), b)
        elif solver == "svd":
            raise NotImplementedError("svd solver with varying alphas not implemented")
        else:
            raise ValueError(f"Unknown solver: {solver}")

        self.coef_ = coef

        # beta coefficients:
        self.beta_coef_ = self.coef_ * Xc.std(axis=0) / yc.std()

        # Compute intercept_
        if self.fit_intercept:
            self.intercept_ = self.y_mean_ - self.X_mean_ @ self.coef_
        else:
            self.intercept_ = 0.0

        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        X = check_array(X)
        return X @ self.coef_ + self.intercept_

    def get_params(self, deep=True):
        return {"alphas": self.alphas, "solver": self.solver, "fit_intercept": self.fit_intercept}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
