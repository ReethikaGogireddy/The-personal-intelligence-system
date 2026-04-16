import pickle
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 200

# Features: sleep_hours, mood, energy, work_hours, exercise_minutes, stress
sleep = np.random.uniform(4, 9, n)
mood = np.random.randint(1, 6, n)
energy = np.random.randint(1, 6, n)
work_hours = np.random.uniform(4, 12, n)
exercise = np.random.uniform(0, 90, n)
stress = np.random.randint(1, 6, n)

X = np.column_stack([sleep, mood, energy, work_hours, exercise, stress])

# Productivity score (1-10): more sleep/mood/energy = higher, more stress = lower
productivity = (
    sleep * 0.5 + mood * 0.8 + energy * 0.9
    - stress * 0.7 + exercise * 0.02
    - work_hours * 0.1 + np.random.normal(0, 0.5, n)
)
productivity = np.clip(productivity, 1, 10)

# Burnout: high stress + low sleep + low mood = burnout
burnout_score = stress * 1.5 - sleep * 0.5 - mood * 0.4 - energy * 0.3
burnout = np.where(burnout_score > 4, 2, np.where(burnout_score > 2, 1, 0))
# 0 = Low, 1 = Medium, 2 = High

# Train productivity model
scaler_p = StandardScaler()
X_scaled = scaler_p.fit_transform(X)
prod_model = LinearRegression()
prod_model.fit(X_scaled, productivity)

# Train burnout model
scaler_b = StandardScaler()
X_scaled_b = scaler_b.fit_transform(X)
burnout_model = LogisticRegression(max_iter=200)
burnout_model.fit(X_scaled_b, burnout)

# Save all 4 pkl files
with open("productivity_model.pkl", "wb") as f:
    pickle.dump(prod_model, f)
with open("productivity_scaler.pkl", "wb") as f:
    pickle.dump(scaler_p, f)
with open("burnout_model.pkl", "wb") as f:
    pickle.dump(burnout_model, f)
with open("burnout_scaler.pkl", "wb") as f:
    pickle.dump(scaler_b, f)

print("✅ All 4 pkl files saved!")