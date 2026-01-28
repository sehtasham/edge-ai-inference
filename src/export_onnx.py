import torch
from src.model import AutoEncoder

MODEL_PATH = "models/autoencoder.pth"
ONNX_PATH = "models/autoencoder.onnx"

def main():
    model = AutoEncoder(input_dim=10)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    dummy_input = torch.randn(1, 10)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=11,
        input_names=["sensor_input"],
        output_names=["reconstruction"],
        dynamic_axes={
            "sensor_input": {0: "batch_size"},
            "reconstruction": {0: "batch_size"},
        }
    )

    print(f"ONNX model saved to {ONNX_PATH}")

if __name__ == "__main__":
    main()
