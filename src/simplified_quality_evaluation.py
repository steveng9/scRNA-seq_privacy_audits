import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# ====== Utility Metrics Reimplemented from Evaluation Code ======

def mmd_rbf(X, Y, gamma=1.0):
    """Simple RBF MMD for dense matrices."""
    from sklearn.metrics.pairwise import rbf_kernel
    XX = rbf_kernel(X, X, gamma=gamma)
    YY = rbf_kernel(Y, Y, gamma=gamma)
    XY = rbf_kernel(X, Y, gamma=gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def distance_to_closest_neighbor(X_real, X_syn):
    """Average Euclidean distance from each synthetic cell to nearest real cell."""
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(X_syn, X_real)
    return np.min(D, axis=1).mean()

def discriminative_score(X_real, X_syn, seed=0):
    """Real vs Synthetic classifier F1."""
    X = np.vstack([X_real, X_syn])
    y = np.array([1]*len(X_real) + [0]*len(X_syn))
    X, y = shuffle(X, y, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return f1_score(y_test, y_pred)

def feature_overlap(feats_a, feats_b):
    """Count overlapping top features (fake reimplementation)."""
    a = set(feats_a)
    b = set(feats_b)
    overlap = len(a & b)
    prop = overlap / max(len(a), 1)
    return overlap, prop

def top_features(model, top_n=10):
    """Extract top features from a One-vs-Rest logistic model."""
    feats = []
    for est in model.estimators_:
        coef = est.coef_[0]
        idx = coef.argsort()[-top_n:]
        feats.extend(idx.tolist())
    return feats

# ========= SIMPLE EVALUATION PIPELINE =============

def evaluate(real_file, syn_file):
    print("Loading AnnData...")
    real = sc.read_h5ad(real_file)
    syn = sc.read_h5ad(syn_file)

    # Ensure dense arrays
    X_real = real.X.toarray() if hasattr(real.X, "toarray") else real.X
    X_syn = syn.X.toarray() if hasattr(syn.X, "toarray") else syn.X

    # ---------- Basic shape alignment ----------
    common_genes = real.var_names.intersection(syn.var_names)
    real = real[:, common_genes]
    syn = syn[:, common_genes]
    X_real = real.X.toarray()
    X_syn = syn.X.toarray()

    # ---------- Train classifier on synthetic → test on real ----------
    print("Running classification metrics...")
    y_real = real.obs.iloc[:, 0].astype(str).values
    y_syn = syn.obs.iloc[:, 0].astype(str).values

    encoder = LabelEncoder()
    encoder.fit(np.concatenate([y_real, y_syn]))

    y_real_enc = encoder.transform(y_real)
    y_syn_enc = encoder.transform(y_syn)

    model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    model.fit(X_syn, y_syn_enc)

    y_pred = model.predict(X_real)
    y_proba = model.predict_proba(X_real)

    acc_syn = accuracy_score(y_real_enc, y_pred)
    avgpr_syn = average_precision_score(y_real_enc, y_proba, average='macro')

    # ---------- Train on real → test on real (baseline) ----------
    model2 = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    model2.fit(X_real, y_real_enc)
    y_pred2 = model2.predict(X_real)
    y_proba2 = model2.predict_proba(X_real)
    acc_real = accuracy_score(y_real_enc, y_pred2)
    avgpr_real = average_precision_score(y_real_enc, y_proba2, average='macro')

    feats_syn = top_features(model)
    feats_real = top_features(model2)
    overlap_cnt, overlap_prop = feature_overlap(feats_syn, feats_real)

    # ---------- Statistical metrics ----------
    mmd_score = mmd_rbf(X_real, X_syn)
    dist_syn = distance_to_closest_neighbor(X_real, X_syn)
    dist_base = distance_to_closest_neighbor(X_real, X_real)

    disc = discriminative_score(X_real, X_syn)

    return {
        "accuracy_synthetic": acc_syn,
        "avgpr_synthetic": avgpr_syn,
        "accuracy_real": acc_real,
        "avgpr_real": avgpr_real,
        "feature_overlap_count": overlap_cnt,
        "feature_overlap_prop": overlap_prop,
        "mmd": mmd_score,
        "distance_to_closest": dist_syn,
        "distance_to_closest_base": dist_base,
        "discriminative_score": disc,
    }

# ============================================================
# =============== COMMAND-LINE INTERFACE =====================
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple synthetic dataset evaluator")
    parser.add_argument("--real", required=True, help="Real train .h5ad dataset")
    parser.add_argument("--synthetic", required=True, help="Synthetic .h5ad dataset")
    parser.add_argument("--out", required=True, help="Output CSV file")

    args = parser.parse_args()

    results = evaluate(args.real, args.synthetic)
    df = pd.DataFrame([results])
    df.to_csv(args.out, index=False)
    print("\nDONE! Saved results to:", args.out)
