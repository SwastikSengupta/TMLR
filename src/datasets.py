"""
Real-dataset suite. Fourteen UCI classification datasets, downloaded as
distributed by the archive and parsed here. No synthetic data is used anywhere
in this study.

Multi-class targets are binarised as largest-class-versus-rest so that AUC is
defined identically for every dataset, matching the metric used in the original
study being reproduced.
"""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def _read_arff(path):
    lines = Path(path).read_text(errors="ignore").splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().lower().startswith("@data"))
    body = "\n".join(l for l in lines[start + 1:] if l.strip() and not l.startswith("%"))
    return pd.read_csv(io.StringIO(body), header=None)


def _finish(df, target_col, drop=None, name=""):
    if drop:
        df = df.drop(columns=drop)
    y_raw = df[target_col]
    X = df.drop(columns=[target_col])
    X = X.apply(pd.to_numeric, errors="coerce")
    # drop columns that are entirely non-numeric, then impute nothing here
    X = X.loc[:, X.notna().any()]
    keep = X.notna().all(axis=1)
    X, y_raw = X[keep], y_raw[keep]
    vals, counts = np.unique(y_raw.astype(str), return_counts=True)
    major = vals[np.argmax(counts)]
    y = (y_raw.astype(str) != major).astype(int).to_numpy()
    return name, X.to_numpy(dtype=float), y


def _wine():
    df = pd.read_csv(DATA / "d109/wine.data", header=None)
    return _finish(df, 0, name="wine")


def _mammographic():
    df = pd.read_csv(DATA / "d161/mammographic_masses.data", header=None,
                     na_values="?")
    return _finish(df, 5, name="mammographic")


def _parkinsons():
    df = pd.read_csv(DATA / "d174/parkinsons.data")
    return _finish(df, "status", drop=["name"], name="parkinsons")


def _transfusion():
    df = pd.read_csv(DATA / "d176/transfusion.data")
    return _finish(df, df.columns[-1], name="transfusion")


def _seeds():
    df = pd.read_csv(DATA / "d236/seeds_dataset.txt", sep=r"\s+", header=None)
    return _finish(df, 7, name="seeds")


def _biodeg():
    df = pd.read_csv(DATA / "d254/biodeg.csv", sep=";", header=None)
    return _finish(df, df.columns[-1], name="qsar_biodeg")


def _banknote():
    df = pd.read_csv(DATA / "d267/data_banknote_authentication.txt", header=None)
    return _finish(df, 4, name="banknote")


def _dermatology():
    df = pd.read_csv(DATA / "d33/dermatology.data", header=None, na_values="?")
    return _finish(df, 34, name="dermatology")


def _htru2():
    df = pd.read_csv(DATA / "d372/HTRU_2.csv", header=None)
    return _finish(df, 8, name="htru2_pulsar")


def _glass():
    df = pd.read_csv(DATA / "d42/glass.data", header=None)
    return _finish(df, 10, drop=[0], name="glass")


def _ionosphere():
    df = pd.read_csv(DATA / "d52/ionosphere.data", header=None)
    return _finish(df, 34, name="ionosphere")


def _spambase():
    df = pd.read_csv(DATA / "d94/spambase.data", header=None)
    return _finish(df, 57, name="spambase")


def _vertebral():
    df = _read_arff(DATA / "d212/column_2C_weka.arff")
    return _finish(df, df.columns[-1], name="vertebral")


def _retinopathy():
    df = _read_arff(DATA / "d329/messidor_features.arff")
    return _finish(df, df.columns[-1], name="retinopathy")


def _raisin():
    inner = DATA / "d850/Raisin_Dataset.zip"
    with zipfile.ZipFile(inner) as z:
        nm = [n for n in z.namelist() if n.endswith((".xlsx", ".arff", ".csv"))][0]
        with z.open(nm) as f:
            if nm.endswith(".xlsx"):
                df = pd.read_excel(f)
            elif nm.endswith(".arff"):
                txt = f.read().decode(errors="ignore").splitlines()
                s = next(i for i, l in enumerate(txt) if l.lower().startswith("@data"))
                df = pd.read_csv(io.StringIO("\n".join(txt[s + 1:])), header=None)
            else:
                df = pd.read_csv(f)
    return _finish(df, df.columns[-1], name="raisin")


LOADERS = [_wine, _mammographic, _parkinsons, _transfusion, _seeds, _biodeg,
           _banknote, _dermatology, _htru2, _glass, _ionosphere, _spambase,
           _vertebral, _retinopathy, _raisin]


def load_all(min_n=100, max_n=20000, verbose=True):
    out = []
    for fn in LOADERS:
        try:
            name, X, y = fn()
        except Exception as e:
            if verbose:
                print(f"  skip {fn.__name__}: {str(e)[:60]}")
            continue
        if len(y) < min_n or len(np.unique(y)) < 2:
            continue
        if len(y) > max_n:                       # cap for runtime
            rng = np.random.default_rng(0)
            idx = rng.choice(len(y), max_n, replace=False)
            X, y = X[idx], y[idx]
        out.append((name, X, y))
        if verbose:
            print(f"  {name:16s} n={len(y):6d}  p={X.shape[1]:4d}  "
                  f"minority={min(y.mean(), 1-y.mean()):.3f}")
    return out


if __name__ == "__main__":
    print("Real UCI datasets:")
    ds = load_all()
    print(f"\n{len(ds)} datasets loaded")
