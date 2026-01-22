import numpy as np

FEATURES = 10

def generate_normal_data(samples=1000):
    return np.random.normal(0.0, 1.0, size=(samples, FEATURES))

def generate_anomaly_data(samples=50):
    return np.random.normal(5.0, 1.5, size=(samples, FEATURES))
