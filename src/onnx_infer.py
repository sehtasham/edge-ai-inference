import numpy as np
import onnxruntime as ort
from data.sensor_data import generate_normal_data, generate_anomaly_data

MODEL_PATH = "models/autoencoder.onnx"

def anomaly_score(session, x):
    outputs = session.run(
        None,
        {"sensor_input": x.astype(np.float32)}
    )
    recon = outputs[0]
    return np.mean((x - recon) ** 2)

def main():
    session = ort.InferenceSession(MODEL_PATH)

    normal = generate_normal_data(1)
    anomaly = generate_anomaly_data(1)

    print("ONNX Normal score :", anomaly_score(session, normal))
    print("ONNX Anomaly score:", anomaly_score(session, anomaly))

if __name__ == "__main__":
    main()
