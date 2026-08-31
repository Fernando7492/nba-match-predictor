import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class HomeCourtBaseline:
    def __init__(self):
        self.win_rate: float = 0.5

    def fit(self, x: np.ndarray, y: np.ndarray) -> "HomeCourtBaseline":
        self.win_rate = float(np.mean(y))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        n_samples = len(x)
        p = np.full((n_samples, 2), [1.0 - self.win_rate, self.win_rate])
        return p

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)

class LogisticRegressionBaseline:
    def __init__(self, c: float = 1.0, random_state: int = 42):
        self.model = LogisticRegression(C=c, random_state=random_state, max_iter=1000)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticRegressionBaseline":
        self.model.fit(x, y.ravel())
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

class RandomForestBaseline:
    def __init__(self, n_estimators: int = 150, max_depth: int = 8, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomForestBaseline":
        self.model.fit(x, y.ravel())
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)
