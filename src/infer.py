import torch

from src.model import AutoEncoder
from data.sensor_data import generate_normal_data, generate_anomaly_data

def anomaly_score(model, x):
    with torch.no_grad():
        recon = model(x)
        return torch.mean((x - recon) ** 2).item()

def main():
    model = AutoEncoder(input_dim=10)
    model.load_state_dict(torch.load("models/autoencoder.pth"))
    model.eval()

    normal = torch.tensor(generate_normal_data(1), dtype=torch.float32)
    anomaly = torch.tensor(generate_anomaly_data(1), dtype=torch.float32)

    print("Normal score :", anomaly_score(model, normal))
    print("Anomaly score:", anomaly_score(model, anomaly))

if __name__ == "__main__":
    main()
