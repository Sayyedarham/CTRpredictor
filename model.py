import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

data = pd.read_csv('/mnt/data/Marketing campaign dataset.csv')

data['CTR'] = data['clicks'] / data['impressions']
data['click'] = (data['CTR'] > 0).astype(int)  # Binary target: 1 if clicked, 0 otherwise

data.fillna(data.median(), inplace=True)

label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

data = data.drop(['campaign_item_id', 'time', 'CTR'], axis=1)

scaler = StandardScaler()
numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_features] = scaler.fit_transform(data[numerical_features])

X = data.drop('click', axis=1)
y = data['click']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)[:, 1]

logloss = log_loss(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f'Log Loss: {logloss}')
print(f'ROC AUC: {roc_auc}')

predictions = pd.DataFrame({'Actual': y_test[:5], 'Predicted Probability': y_pred_proba[:5]})
print("\Predictions:")
print(predictions)

