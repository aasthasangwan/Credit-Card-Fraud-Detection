import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print("📦 Loading dataset...")
df = pd.read_csv("creditcard.csv")
print("✅ Dataset loaded.")

X = df.drop("Class", axis=1)
y = df["Class"]

print("⚙️ Normalizing data...")
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])
X['Time'] = scaler.fit_transform(X[['Time']])

print("🔀 Splitting data...")
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("🌲 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, "model_rf.pkl")
print("✅ Random Forest model saved as model_rf.pkl")

print("🧠 Building Neural Network...")
model = Sequential()
model.add(Dense(32, input_dim=X.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("🚀 Training Neural Network...")
model.fit(X_train, y_train, epochs=5, batch_size=64)
model.save("model_nn.h5")
print("✅ Neural Network saved as model_nn.h5")