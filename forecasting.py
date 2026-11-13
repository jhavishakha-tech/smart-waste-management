"""24-hour-ahead bin fill-accumulation forecasting from sensor history."""
import numpy as np
import joblib, os, time
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import config as C
from simulator import BinField

WINDOWS = [1, 3, 6, 12, 24, 48]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "forecaster.joblib")


def make_features(ghat, t, field):
    """Features for every bin at time t using only ghat[:t+1] (the sensor-derived hourly increments)."""
    n = ghat.shape[1]
    csum = np.vstack([np.zeros((1, n)), np.cumsum(ghat[:t + 1], axis=0)])     # csum[k] = sum ghat[:k]
    feats = [csum[t + 1] - csum[max(t + 1 - w, 0)] for w in WINDOWS]
    hrs = (np.arange(t + 1, t + 25)) // 24
    weekend_frac = np.mean((hrs % 7) >= 5)
    h = t % 24
    kind_oh = np.eye(len(C.KINDS))[field.kind_idx]
    tf = np.tile([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), weekend_frac], (n, 1))
    return np.column_stack(feats + [tf, kind_oh])


def build_dataset(seeds, n_bins=100, days=21, noise=C.SENSOR_NOISE_SD, stride=3):
    X, y = [], []
    for s in seeds:
        f = BinField(n_bins, seed=1000 + s)
        gen = f.generate(days * 24)
        rng = np.random.default_rng(s)
        cum = np.cumsum(gen, axis=0)
        obs = cum + rng.normal(0, noise, cum.shape)                         # uncapped, noisy "sensor"
        ghat = np.diff(np.vstack([np.zeros((1, n_bins)), obs]), axis=0)
        for t in range(48, days * 24 - 25, stride):
            X.append(make_features(ghat, t, f)); y.append(gen[t + 1:t + 25].sum(0))
    return np.vstack(X), np.concatenate(y)


def get_model(name):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=0)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=0))
    raise ValueError(name)


class Forecaster:
    def __init__(self, name="hgb"):
        self.name = name; self.model = get_model(name)

    def fit(self, X, y):
        self.model.fit(X, y); return self

    def predict(self, ghat, t, field):
        return np.clip(self.model.predict(make_features(ghat, t, field)), 0, None)


class PersistenceForecaster:
    """Baseline: next 24 h accumulation = last 24 h accumulation."""
    name = "persistence"

    def predict(self, ghat, t, field):
        return np.clip(make_features(ghat, t, field)[:, WINDOWS.index(24)], 0, None)


def metrics(y, p):
    e = y - p
    return dict(MAE=float(np.mean(np.abs(e))) * 100, RMSE=float(np.sqrt(np.mean(e ** 2))) * 100,
                R2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)))   # MAE/RMSE in % of bin capacity


def compare_models(n_train=6, n_test=2):
    Xtr, ytr = build_dataset(range(n_train)); Xte, yte = build_dataset(range(100, 100 + n_test))
    rows = []
    p = Xte[:, WINDOWS.index(24)]
    rows.append(dict(model="Persistence (last 24 h)", train_s=0.0, **metrics(yte, p)))
    for nm, lab in [("ridge", "Ridge regression"), ("mlp", "MLP (64-32)"), ("hgb", "Gradient Boosting")]:
        t0 = time.time(); m = get_model(nm).fit(Xtr, ytr); dt = time.time() - t0
        rows.append(dict(model=lab, train_s=dt, **metrics(yte, m.predict(Xte))))
    return rows, (Xte, yte)


def train_default(save=True):
    X, y = build_dataset(range(8))
    f = Forecaster("hgb").fit(X, y)
    if save:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True); joblib.dump(f, MODEL_PATH)
    return f


def load_or_train():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return train_default()
