import torch
import torch.nn as nn
import torch.optim as optim

from src.model import AutoEncoder
from data.sensor_data import generate_normal_data

def main():
    data_np = generate_normal_data()
    data = torch.tensor(data_np, dtype=torch.float32)

    model = AutoEncoder(input_dim=data.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), "models/autoencoder.pth")
    print("Model saved to models/autoencoder.pth")

if __name__ == "__main__":
    main()





