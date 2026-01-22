import numpy as np

FEATURES = 10

def generate_normal_data(samples=1000):
    """
    Simulate normal sensor behavior
    """
    return np.random.normal(
        loc=0.0,
        scale=1.0,
        size=(samples, FEATURES)
    )

def generate_anomaly_data(samples=50):
    """
    Simulate abnormal sensor behavior
    """
    return np.random.normal(
        loc=5.0,
        scale=1.5,
        size=(samples, FEATURES)
    )
