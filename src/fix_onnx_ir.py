import onnx

model = onnx.load("models/autoencoder.onnx")

print("Before IR:", model.ir_version)

# Force downgrade IR version
model.ir_version = 9

onnx.save(model, "models/autoencoder.onnx")

print("After IR:", model.ir_version)
