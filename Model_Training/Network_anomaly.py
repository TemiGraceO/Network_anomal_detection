import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load
df = pd.read_csv('networkanomalydataset.csv')
X = df[['Inbound Rate(bit/s)', 'Outbound Rate(bit/s)', 'Inbound Bandwidth Utilization(%)', 'Outbound Bandwidth Utilization(%)']]
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} - Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))


# retrain best model on full data
best_model = RandomForestClassifier(n_estimators=100, random_state=42)
best_model.fit(X, y)
joblib.dump(best_model, '../web/congestion_model.pkl')