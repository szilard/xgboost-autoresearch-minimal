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


# first 1k rows, scored one row at a time (prepare + predict per row)
test_1k = test.head(1_000)

for thread_label, nthread in [("all cores", -1), ("1 thread", 1)]:
    # n_jobs is baked into the booster at fit time, so predict uses it too
    model.set_params(n_jobs=nthread)
    model.get_booster().set_param({"nthread": nthread})

    label = f"first 1k rows, row by row, {thread_label}"
    y_prob = np.empty(len(test_1k))
    y_test = np.empty(len(test_1k), dtype=int)

    t0 = time.time()
    for i in range(len(test_1k)):
        X_row, y_row = prepare(test_1k.iloc[[i]])
        y_prob[i] = model.predict_proba(X_row)[0, 1]
        y_test[i] = y_row[0]
    test_auc = roc_auc_score(y_test, y_prob)
    print(f"{label}: test time {time.time() - t0:.3f}s, test AUC {test_auc:.4f}")


# first 1k rows: prepare() per row, then one batched predict on the assembled frame
model.set_params(n_jobs=-1)
model.get_booster().set_param({"nthread": -1})

label = "first 1k rows, per-row prepare + batch predict"

t0 = time.time()
prepared = [prepare(test_1k.iloc[[i]]) for i in range(len(test_1k))]
X_test = pd.concat([X_row for X_row, _ in prepared])
y_test = np.concatenate([y_row for _, y_row in prepared])
y_prob = model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_prob)
print(f"{label}: test time {time.time() - t0:.3f}s, test AUC {test_auc:.4f}")


# prepare() costs ~2.5ms per call whatever the frame size, so with the function
# left as it is the only lever is calling it in parallel.
import multiprocessing as mp

n_workers = 8
per_worker = -(-len(test_1k) // n_workers)
bounds = [(i, min(i + per_worker, len(test_1k))) for i in range(0, len(test_1k), per_worker)]

def prepare_slice(lo_hi):
    lo, hi = lo_hi
    out = [prepare(test_1k.iloc[[i]]) for i in range(lo, hi)]
    return pd.concat([X for X, _ in out]), np.concatenate([y for _, y in out])

label = f"first 1k rows, per-row prepare in {n_workers} processes + batch predict"

t0 = time.time()
# fork so the workers inherit prepare() and its cat_levels as they stand
with mp.get_context("fork").Pool(n_workers) as pool:
    out = pool.map(prepare_slice, bounds)
X_test = pd.concat([X for X, _ in out])
y_test = np.concatenate([y for _, y in out])
y_prob = model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_prob)
print(f"{label}: test time {time.time() - t0:.3f}s, test AUC {test_auc:.4f}")

assert np.array_equal(y_prob, model.predict_proba(prepare(test_1k)[0])[:, 1])
