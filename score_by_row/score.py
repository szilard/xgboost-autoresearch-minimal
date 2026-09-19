exec(open(__file__.replace("score.py", "train.py")).read())

from sklearn.metrics import roc_auc_score
import numpy as np
import multiprocessing as mp

# In some commits the fitted model is `final_model`; in others it's `model`.
if "final_model" in dir():
    model = final_model

n_workers = 4


def score_batch(label, df):
    """prepare() the whole frame in one call, then predict."""
    t0 = time.time()
    X_test, y_test = prepare(df)
    y_prob = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)
    print(f"{label}: test time {time.time() - t0:.3f}s, test AUC {test_auc:.8f}")
    return y_prob


def prepare_rows(df):
    """prepare() each row on its own; runs in a worker process."""
    out = [prepare(df.iloc[[i]]) for i in range(len(df))]
    return pd.concat([X for X, _ in out]), np.concatenate([y for _, y in out])


def score_by_row(label, df, workers=n_workers):
    """prepare() row by row across processes, then predict in one batch.

    prepare() costs milliseconds per call whatever the frame size, so scoring
    the rows one at a time is dominated by that fixed cost. With the function
    left as it is, the only lever is calling it in parallel.
    """
    t0 = time.time()
    per_worker = -(-len(df) // workers)
    chunks = [df.iloc[i:i + per_worker] for i in range(0, len(df), per_worker)]
    # fork so the workers inherit prepare() and its cat_levels as they stand
    with mp.get_context("fork").Pool(workers) as pool:
        out = pool.map(prepare_rows, chunks)
    X_test = pd.concat([X for X, _ in out])
    y_test = np.concatenate([y for _, y in out])
    y_prob = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)
    print(f"{label}: test time {time.time() - t0:.3f}s, test AUC {test_auc:.8f}")
    return y_prob


test = pd.read_csv(f"{data_dir}/2006-slice2-1m.csv")
test_100k = test.head(100_000)

score_batch("full 1m rows", test)
batch_prob = score_batch("first 100k rows", test_100k)
by_row_prob = score_by_row(f"first 100k rows, per-row prepare in {n_workers} processes", test_100k)

assert np.array_equal(batch_prob, by_row_prob)
