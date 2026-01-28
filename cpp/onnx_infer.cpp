#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>

// Mean Squared Error for anomaly score
float anomaly_score(const std::vector<float>& input,
                    const std::vector<float>& recon) {
    float sum = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        float diff = input[i] - recon[i];
        sum += diff * diff;
    }
    return sum / input.size();
}

int main() {
    // 1. Initialize runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "edge-ai");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    Ort::Session session(env, "models/autoencoder.onnx", session_options);

    // 2. Allocator
    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name_alloc  = session.GetInputNameAllocated(0, allocator);
    auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);

    const char* input_name  = input_name_alloc.get();
    const char* output_name = output_name_alloc.get();

    // 3. Example sensor input (10 features)
    std::vector<float> input_data = {
        0.1f, -0.2f, 0.05f, 0.3f, -0.1f,
        0.2f, 0.0f, -0.05f, 0.15f, -0.2f
    };

    std::vector<int64_t> input_shape = {1, 10};

    // 4. Create tensor
    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        input_data.data(),
        input_data.size(),
        input_shape.data(),
        input_shape.size()
    );

    // 5. Run inference
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        &input_name,
        &input_tensor,
        1,
        &output_name,
        1
    );

    // 6. Read output
    float* output_data =
        output_tensors[0].GetTensorMutableData<float>();

    std::vector<float> reconstructed(
        output_data, output_data + input_data.size()
    );

    // 7. Compute anomaly score
    float score = anomaly_score(input_data, reconstructed);
    std::cout << "C++ Anomaly Score: " << score << std::endl;

    return 0;
}
