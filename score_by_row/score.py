exec(open(__file__.replace("score.py", "train.py")).read())

from sklearn.metrics import roc_auc_score
import numpy as np

# In some commits the fitted model is `final_model`; in others it's `model`.
if "final_model" in dir():
    model = final_model

test = pd.read_csv(f"{data_dir}/2006-slice2-1m.csv")
test_100k = test.head(100_000)

subsets = [
    ("full 1m rows",    test),
    ("first 100k rows", test_100k),
]

for label, test_subset in subsets:
    t0 = time.time()
    X_test, y_test = prepare(test_subset)
    y_prob = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)
    print(f"{label}: test time {time.time() - t0:.3f}s, test AUC {test_auc:.4f}")


# prepare() costs milliseconds per call whatever the frame size, so scoring the
# rows one at a time is dominated by that fixed cost. With the function left as
# it is, the only lever is calling it in parallel.
test_10k = test.head(10_000)

import multiprocessing as mp

n_workers = 4
per_worker = -(-len(test_10k) // n_workers)
bounds = [(i, min(i + per_worker, len(test_10k))) for i in range(0, len(test_10k), per_worker)]

def prepare_slice(lo_hi):
    lo, hi = lo_hi
    out = [prepare(test_10k.iloc[[i]]) for i in range(lo, hi)]
    return pd.concat([X for X, _ in out]), np.concatenate([y for _, y in out])

label = f"first 10k rows, per-row prepare in {n_workers} processes + batch predict"

t0 = time.time()
# fork so the workers inherit prepare() and its cat_levels as they stand
with mp.get_context("fork").Pool(n_workers) as pool:
    out = pool.map(prepare_slice, bounds)
X_test = pd.concat([X for X, _ in out])
y_test = np.concatenate([y for _, y in out])
y_prob = model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_prob)
print(f"{label}: test time {time.time() - t0:.3f}s, test AUC {test_auc:.4f}")

assert np.array_equal(y_prob, model.predict_proba(prepare(test_10k)[0])[:, 1])
