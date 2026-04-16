import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_and_save_models():
    np.random.seed(42)
    n = 200

    data = pd.DataFrame({
        'sleep_hours':      np.random.uniform(4, 10, n),
        'mood':             np.random.randint(1, 6, n),
        'energy':           np.random.randint(1, 6, n),
        'work_hours':       np.random.uniform(2, 12, n),
        'exercise_minutes': np.random.uniform(0, 90, n),
        'stress':           np.random.randint(1, 6, n),
    })

    data['productivity'] = (
        data['sleep_hours'] * 0.5 +
        data['mood'] * 0.8 +
        data['energy'] * 0.6 -
        data['stress'] * 0.4 +
        data['exercise_minutes'] * 0.03
    ).clip(1, 10)

    data['burnout'] = (
        (data['stress'] >= 4) &
        (data['sleep_hours'] < 6) &
        (data['work_hours'] > 9)
    ).map({True: 'High', False: 'Low'})

    features = ['sleep_hours','mood','energy','work_hours','exercise_minutes','stress']
    X = data[features]

    prod_model = LinearRegression()
    prod_model.fit(X, data['productivity'])

    burn_model = RandomForestClassifier(n_estimators=50, random_state=42)
    burn_model.fit(X, data['burnout'])

    pickle.dump(prod_model, open('productivity_model.pkl', 'wb'))
    pickle.dump(burn_model, open('burnout_model.pkl', 'wb'))
    print("Models trained and saved!")

if __name__ == '__main__':
    train_and_save_models()