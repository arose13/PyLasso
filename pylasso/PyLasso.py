import numpy as np
import numpy.linalg as la
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from lightning.regression import FistaRegressor


def _get_lam_max_min(x, y, eps):
    # Compute lambda max
    n, p = x.shape
    dots = np.zeros(p)
    for j in range(p):
        dots[j] = x[:, j].T @ y
    lam_max = np.abs(dots).max() / n

    # Compute lambda_min
    assert eps < 1, 'eps must be less than 1'
    lam_min = eps * lam_max

    return lam_max, lam_min


class _LassoProjection:
    def __init__(self, w):
        self.w = w

    def projection(self, m, alpha, L):
        return np.sign(m) * np.maximum(abs(m) - ((self.w * alpha) / L), 0)

    def regularization(self, m):
        return la.norm(self.w * m, 1)


class PyLassoRegression(BaseEstimator, RegressorMixin):
    """
    The objective function that is solved is...
    J(m) = 1/(2*n) sum((y_i - x_i m)^2) + lam * sum(|w_j * m_j|)
    or
    J(m) = 1/(2*n) ||y - Xm||^2_2 + lam||w.m||_1

    :param w: the multiplier for each parameter where 0 is not regularized and 1 is lam
    :param lam: If this is set then this is the only lambda solved for.
    :param eps: the factor lambda_min is smaller than lambda_max
    :param n_lam: number of lambdas in the lambda path
    :param n_jobs:
    :param metric: metric to judge the best model. Must be a SMALLER IS BETTER metric.
    """
    def __init__(self, w=None, lam=None, eps=1e-3, n_lam=30, cv=5, n_jobs=-1, metric=mean_squared_error):
        """
        The objective function that is solved is...
        J(m) = 1/(2*n) sum((y_i - x_i m)^2) + lam * sum(|w_j * m_j|)
        or
        J(m) = 1/(2*n) ||y - Xm||^2_2 + lam||w.m||_1

        :param w: the multiplier for each parameter where 0 is not regularized and 1 is lam
        :param lam: If this is set then this is the only lambda solved for.
        :param eps: the factor lambda_min is smaller than lambda_max
        :param n_lam: number of lambdas in the lambda path
        :param n_jobs:
        :param metric: metric to judge the best model. Must be a SMALLER IS BETTER metric.
        """
        self.eps = eps
        self.lam = lam
        self.w = w
        self.n_lam = n_lam
        self.cv = cv
        self.n_jobs = n_jobs
        self.metric = metric

        # Solved attributes
        self.lambda_path_ = None
        self.coef_path_ = None
        self.model_path_ = None
        self.score_path_ = None
        self.best_index = None

    def fit(self, X, y):
        if self.w is None:
            self.w = np.ones(X.shape[1])

        if self.lam is None:
            lam_max, lam_min = _get_lam_max_min(X, y, self.eps)
            self.lambda_path_ = np.logspace(np.log10(lam_max), np.log10(lam_min), self.n_lam)
        else:
            self.lambda_path_ = [self.lam]

        scorer = make_scorer(self.metric)
        self.coef_path_, self.model_path_, self.score_path_ = [], [], []
        for lam_i in self.lambda_path_:
            # Setup model
            per_model_n = len(y) * ((self.cv-1) / self.cv)
            model_i = FistaRegressor(
                C=1/per_model_n,
                penalty=_LassoProjection(self.w),
                alpha=lam_i
            )

            # Get fit data
            scores_i = cross_val_score(
                model_i,
                X, y,
                scoring=scorer,
                cv=self.cv,
                n_jobs=self.n_jobs
            )

            # Fit model
            model_i.fit(X, y)
            self.coef_path_.append(model_i.coef_)
            self.score_path_.append(scores_i.mean())
            self.model_path_.append(model_i)

        self.coef_path_ = np.vstack(self.coef_path_)
        self.best_index = np.argmin(self.score_path_)

    def predict(self, X):
        # TODO (2/11/2019) add the ability to get a prediction at a particular lambda
        return self.model_path_[np.argmin(self.score_path_)].predict(X)

    def plot_coef_path(self, figsize=(10, 5)):
        import matplotlib.pyplot as graph

        graph.figure(figsize=figsize)
        graph.plot(self.lambda_path_, self.coef_path_, linewidth=2)
        graph.xscale('log')
        graph.ylabel(r'$\beta$')
        graph.xlabel(r'$\lambda$')

    def plot_score_path(self, figsize=(10, 5)):
        import matplotlib.pyplot as graph

        graph.figure(figsize=figsize)
        graph.plot(self.lambda_path_, self.score_path_)
        graph.xscale('log')
        graph.ylabel('Metric')
        graph.xlabel(r'$\lambda$')
