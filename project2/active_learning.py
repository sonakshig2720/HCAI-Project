import numpy as np
from sklearn.linear_model import LogisticRegression
from .representation import build_vectorizer

def random_sampling(probas):
    """Pick an unlabeled example uniformly at random."""
    return np.random.randint(probas.shape[0])

def uncertainty_sampling(probas):
    """Pick the example whose top‐class probability is smallest."""
    return np.argmin(np.max(probas, axis=1))

def margin_sampling(probas):
    """
    Pick the example with the smallest margin between its two highest class probabilities.
    """
    part = np.partition(-probas, 1, axis=1)
    top1 = -part[:, 0]
    top2 = -part[:, 1]
    margins = top1 - top2
    return np.argmin(margins)

def entropy_sampling(probas):
    """Pick the example with the highest entropy."""
    entropy = -np.sum(probas * np.log(probas + 1e-12), axis=1)
    return np.argmax(entropy)


def run_pool_al(pool_texts, pool_labels, X_test, y_test,
                init_size, budget, strategy_fn):
    """
    Single‐query pool‐based active learning.
    Returns a list of test‐set accuracies, one per query.
    """
    # 1) Fit TF–IDF once
    vectorizer = build_vectorizer()
    vectorizer.fit(pool_texts)
    X_pool = vectorizer.transform(pool_texts)
    X_test_vec = vectorizer.transform(X_test)

    all_idxs = set(range(X_pool.shape[0]))
    init_idxs = np.random.choice(list(all_idxs), size=init_size, replace=False)
    labeled = set(init_idxs)
    unlabeled = all_idxs - labeled

    learning_curve = []
    for _ in range(budget):
        # Train on labeled
        clf = LogisticRegression(max_iter=500)
        idxs = list(labeled)
        clf.fit(X_pool[idxs], [pool_labels[i] for i in idxs])

        # Evaluate
        acc = clf.score(X_test_vec, y_test)
        learning_curve.append(acc)

        # Query one new point
        unl_idxs = list(unlabeled)
        probas = clf.predict_proba(X_pool[unl_idxs])
        choice = strategy_fn(probas)
        chosen = unl_idxs[choice]
        labeled.add(chosen)
        unlabeled.remove(chosen)

    return learning_curve


def run_batch_pool_al(pool_texts, pool_labels, X_test, y_test,
                      init_size, budget, strategy_fn, batch_size):
    """
    Batch‐mode pool‐based active learning.
    At each iteration, pick up to batch_size new points, then retrain once.
    Returns two lists:
      - xs: cumulative # labeled after each batch
      - ys: test accuracies at those points
    """
    # 1) Fit TF–IDF once
    vectorizer = build_vectorizer()
    vectorizer.fit(pool_texts)
    X_pool = vectorizer.transform(pool_texts)
    X_test_vec = vectorizer.transform(X_test)

    all_idxs = set(range(X_pool.shape[0]))
    init_idxs = np.random.choice(list(all_idxs), size=init_size, replace=False)
    labeled = set(init_idxs)
    unlabeled = all_idxs - labeled

    xs, ys = [], []
    queries_made = 0
    # iterate until budget is exhausted
    while queries_made < budget and unlabeled:
        # Train on current labeled set
        clf = LogisticRegression(max_iter=500)
        idxs = list(labeled)
        clf.fit(X_pool[idxs], [pool_labels[i] for i in idxs])

        # Evaluate and record
        acc = clf.score(X_test_vec, y_test)
        xs.append(len(labeled))
        ys.append(acc)

        # Compute probabilities on unlabeled
        unl_idxs = list(unlabeled)
        probas = clf.predict_proba(X_pool[unl_idxs])

        # Pick batch_size new points
        picks = []
        for _ in range(min(batch_size, budget - queries_made)):
            choice = strategy_fn(probas)
            real_idx = unl_idxs[choice]
            picks.append((choice, real_idx))
            # remove so we don’t pick the same twice
            unlabeled.remove(real_idx)
            unl_idxs.pop(choice)
            probas = np.delete(probas, choice, axis=0)
            queries_made += 1

        # Add all picks to labeled
        for _, real_idx in picks:
            labeled.add(real_idx)

    return xs, ys
