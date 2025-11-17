
import numpy as np, pandas as pd, joblib, pathlib
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

OUT = pathlib.Path("artifacts"); OUT.mkdir(exist_ok=True, parents=True)

X, y = make_classification(
    n_samples=2000, n_features=10, n_informative=6, class_sep=1.2, random_state=42
)
cols = [f"f{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=cols); df["label"] = y

X_train, X_test, y_train, y_test = train_test_split(df[cols], df["label"], test_size=0.25, random_state=42)
clf = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, proba)

joblib.dump(clf, OUT/"model.pkl")
df.iloc[:5].to_csv(OUT/"sample.csv", index=False)

with open(OUT/"metrics.json","w") as f:
    f.write(f'{{"roc_auc": {auc:.4f}}}')
print(f"Trained. AUC={auc:.3f}. Artifacts saved to {OUT}/")
