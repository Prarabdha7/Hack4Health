import joblib, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from config import TABULAR_PATH, TABULAR_FEATURES, STRESS_CLASSES, TABULAR_MODEL_PATH

df=pd.read_csv(TABULAR_PATH); X=df[TABULAR_FEATURES]; y=df["Mental_Health_Status"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
models={
"Logistic Regression":Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=3000,class_weight="balanced"))]),
"Random Forest":RandomForestClassifier(n_estimators=500,class_weight="balanced_subsample",random_state=42,n_jobs=-1),
"Extra Trees":ExtraTreesClassifier(n_estimators=500,class_weight="balanced",random_state=42,n_jobs=-1),
"HistGradientBoosting":HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,max_leaf_nodes=31,l2_regularization=1.0,random_state=42),
}
results={}
cv=StratifiedKFold(5,shuffle=True,random_state=42)
for name,m in models.items():
    m.fit(Xtr,ytr); p=m.predict(Xte); macro=f1_score(yte,p,average="macro")
    results[name]=macro
    print(f"\n{name}: accuracy={accuracy_score(yte,p):.4f} macro_f1={macro:.4f} weighted_f1={f1_score(yte,p,average='weighted'):.4f}")
    print(classification_report(yte,p,zero_division=0)); print(confusion_matrix(yte,p,labels=STRESS_CLASSES))
    try: print(f"5-fold CV macro-F1: {cross_val_score(m,Xtr,ytr,cv=cv,scoring='f1_macro',n_jobs=-1).mean():.4f}")
    except Exception: pass
best=models[max(results,key=results.get)]; joblib.dump(best,TABULAR_MODEL_PATH); print(f"\nSaved best tabular classifier: {max(results,key=results.get)}")
