import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator
from util.generalized_ridge import GeneralizedRidge
from sklearn.linear_model import Ridge, LinearRegression


def test_fit_predict_simple():
    rng = np.random.RandomState(0)
    X = rng.randn(50, 3)
    coef_true = np.array([2.0, -1.0, 0.5])
    y = X @ coef_true + rng.randn(50) * 0.1

    model = GeneralizedRidge(alphas=[0.1, 0.1, 0.1], solver="cholesky")
    model.fit(X, y)
    y_pred = model.predict(X)

    # Predictions should correlate strongly with y
    corr = np.corrcoef(y, y_pred)[0, 1]
    assert corr > 0.95

def test_different_alphas_change_solution():
    rng = np.random.RandomState(1)
    X = rng.randn(100, 4)
    y = X @ np.array([1, 2, 3, 4]) + rng.randn(100)

    model_low_penalty = GeneralizedRidge(alphas=[0.0, 0.0, 0.0, 0.0])
    model_high_penalty = GeneralizedRidge(alphas=[10.0, 10.0, 10.0, 10.0])

    model_low_penalty.fit(X, y)
    model_high_penalty.fit(X, y)

    # High penalty should shrink coefficients more
    assert np.linalg.norm(model_high_penalty.coef_) < np.linalg.norm(model_low_penalty.coef_)

def test_pipeline_integration():
    rng = np.random.RandomState(2)
    X = rng.randn(30, 2)
    y = X @ np.array([1.0, -1.0]) + rng.randn(30) * 0.1

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", GeneralizedRidge(alphas=[0.1, 0.1], solver="auto"))
    ])
    pipe.fit(X, y)
    y_pred = pipe.predict(X)

    # Predictions should be close
    mse = np.mean((y - y_pred)**2)
    assert mse < 0.5

def test_invalid_alphas_length():
    X = np.random.randn(20, 3)
    y = np.random.randn(20)

    model = GeneralizedRidge(alphas=[0.1, 0.2])  # wrong length
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_consistency_auto_vs_cholesky():
    rng = np.random.RandomState(3)
    X = rng.randn(40, 3)
    y = X @ np.array([1.5, -2.0, 0.5]) + rng.randn(40) * 0.1
    alphas = [0.5, 0.5, 0.5]

    model_auto = GeneralizedRidge(alphas=alphas, solver="auto").fit(X, y)
    model_chol = GeneralizedRidge(alphas=alphas, solver="cholesky").fit(X, y)

    # Coefficients should be very close
    assert np.allclose(model_auto.coef_, model_chol.coef_, atol=1e-6)

def test_sklearn_estimator_compatibility():
    # Run sklearn's estimator checks (slow but thorough)
    check_estimator(GeneralizedRidge(alphas=0.1))

def test_equivalence_to_linear_regression_with_zero_alpha():
    rng = np.random.RandomState(6)
    X = rng.randn(100, 5)
    y = X @ np.array([1.0, -2.0, 0.5, 0.0, 3.0]) + rng.randn(100) * 0.1

    # Generalized Ridge with zero alphas
    model_ridge = GeneralizedRidge(alphas=[0.0] * X.shape[1], solver="cholesky")
    model_ridge.fit(X, y)

    # Ordinary Least Squares
    model_ols = LinearRegression()
    model_ols.fit(X, y)

    # Coefficients should match
    assert np.allclose(model_ridge.coef_, model_ols.coef_, atol=1e-2)

def test_equivalence_to_sklearn_ridge_when_alphas_equal():
    rng = np.random.RandomState(5)
    X = rng.randn(80, 4)
    y = X @ np.array([1.0, 0.5, -1.0, 2.0]) + rng.randn(80) * 0.2

    alpha_value = 2.5
    alphas = [alpha_value] * X.shape[1]

    model_gen = GeneralizedRidge(alphas=alphas, solver="cholesky").fit(X, y)
    model_skl = Ridge(alpha=alpha_value, fit_intercept=False, solver="cholesky").fit(X, y)

    assert np.allclose(model_gen.coef_, model_skl.coef_, atol=1e-2)

def test_fit_intercept_effect():
    rng = np.random.RandomState(7)
    X = rng.randn(50, 2)
    y = 3 + X @ np.array([1.0, -2.0]) + rng.randn(50) * 0.1

    model_with_intercept = GeneralizedRidge(alphas=[0.1, 0.1], fit_intercept=True)
    model_no_intercept = GeneralizedRidge(alphas=[0.1, 0.1], fit_intercept=False)

    model_with_intercept.fit(X, y)
    model_no_intercept.fit(X, y)

    # With intercept should be closer to true intercept (≈ 3)
    assert abs(model_with_intercept.intercept_ - 3) < 0.5
    # Without intercept should be far
    assert abs(model_no_intercept.intercept_ - 3) > 1.0

def test_predict_shape():
    rng = np.random.RandomState(8)
    X = rng.randn(10, 3)
    y = rng.randn(10)

    model = GeneralizedRidge(alphas=[0.1, 0.1, 0.1])
    model.fit(X, y)

    y_pred = model.predict(X)
    assert y_pred.shape == (10,)

def test_invalid_solver():
    X = np.random.randn(20, 3)
    y = np.random.randn(20)

    model = GeneralizedRidge(alphas=1.0, solver="nonsense")
    with pytest.raises(ValueError, match="Unknown solver"):
        model.fit(X, y)

def test_single_feature():
    rng = np.random.RandomState(9)
    X = rng.randn(30, 1)
    y = 2 * X[:, 0] + rng.randn(30) * 0.1

    model = GeneralizedRidge(alphas=[0.1])
    model.fit(X, y)
    y_pred = model.predict(X)

    corr = np.corrcoef(y, y_pred)[0, 1]
    assert corr > 0.95

def test_large_alpha_shrinks_coeffs():
    rng = np.random.RandomState(10)
    X = rng.randn(40, 3)
    y = X @ np.array([1.0, 2.0, -1.0]) + rng.randn(40)

    model = GeneralizedRidge(alphas=[1e6, 1e6, 1e6])
    model.fit(X, y)

    assert np.allclose(model.coef_, 0.0, atol=1e-3)

def test_zero_variance_feature():
    rng = np.random.RandomState(11)
    X = rng.randn(30, 2)
    X = np.column_stack([X, np.ones(30)])  # third feature is constant
    y = X[:, 0] * 2.0 + rng.randn(30)

    model = GeneralizedRidge(alphas=[0.1, 0.1, 0.1])
    model.fit(X, y)

    # Constant feature should get a small coefficient
    assert abs(model.coef_[2]) < 0.1


#
#
# test_fit_predict_simple()
# test_different_alphas_change_solution()
# test_pipeline_integration()
# test_invalid_alphas_length()
# test_consistency_auto_vs_cholesky()
# test_sklearn_estimator_compatibility()
# test_equivalence_to_linear_regression_with_zero_alpha()
# test_equivalence_to_sklearn_ridge_when_alphas_equal()
# test_fit_intercept_effect()
# test_predict_shape()
# test_invalid_solver()
# test_single_feature()
# test_large_alpha_shrinks_coeffs()
# test_zero_variance_feature()
