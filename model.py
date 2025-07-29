import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

# Generate random sample data
np.random.seed(42)
num_samples = 1000
data = pd.DataFrame({
    'impressions': np.random.randint(1, 1000, num_samples),
    'clicks': np.random.randint(0, 1000, num_samples),
    'campaign_type': np.random.choice(['email', 'social', 'display'], num_samples),
    'budget': np.random.uniform(100, 10000, num_samples),
    'region': np.random.choice(['north', 'south', 'east', 'west'], num_samples),
    'campaign_item_id': np.random.randint(10000, 20000, num_samples),
    'time': pd.date_range(start='2022-01-01', periods=num_samples, freq='H')
})

# Derived features
data['CTR'] = data['clicks'] / data['impressions']
data['click'] = (data['CTR'] > 0).astype(int)

# Encode categorical features
for column in data.select_dtypes(include=['object']).columns:
    data[column] = LabelEncoder().fit_transform(data[column])

# Drop unused columns
data.drop(['campaign_item_id', 'time', 'CTR'], axis=1, inplace=True)

# Scale numeric features except target
scaler = StandardScaler()
features_to_scale = data.drop(columns=['click']).columns
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Train-test split
X = data.drop('click', axis=1)
y = data['click']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# Confusion matrix display
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues, colorbar=False)
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()
