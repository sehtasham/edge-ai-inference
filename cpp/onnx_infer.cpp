#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <unistd.h>
#include <sstream>
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

// Get memory usage in MB
float get_memory_usage_mb() {
    std::stringstream path;
    path << "/proc/" << getpid() << "/status";
    std::ifstream file(path.str());
    std::string line;
    float vmrss = 0.0f;

    while (std::getline(file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::stringstream ss(line);
            std::string key, value, unit;
            ss >> key >> value >> unit;
            vmrss = std::stof(value) / 1024.0f; // KB -> MB
            break;
        }
    }
    return vmrss;
}

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "edge-ai");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(env, "models/autoencoder.onnx", session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name_alloc  = session.GetInputNameAllocated(0, allocator);
    auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);
    const char* input_name  = input_name_alloc.get();
    const char* output_name = output_name_alloc.get();

    std::vector<float> input_data = {0.1f, -0.2f, 0.05f, 0.3f, -0.1f,
                                     0.2f, 0.0f, -0.05f, 0.15f, -0.2f};
    std::vector<int64_t> input_shape = {1, 10};

    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        input_data.data(),
        input_data.size(),
        input_shape.data(),
        input_shape.size()
    );

    // Measure inference time
    auto start = std::chrono::high_resolution_clock::now();
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        &input_name,
        &input_tensor,
        1,
        &output_name,
        1
    );
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> inference_time = end - start;

    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    std::vector<float> reconstructed(output_data, output_data + input_data.size());

    float score = anomaly_score(input_data, reconstructed);

    float mem_usage = get_memory_usage_mb();

    std::cout << "C++ Anomaly Score: " << score << std::endl;
    std::cout << "Inference Time (ms): " << inference_time.count() << std::endl;
    std::cout << "Memory Usage (MB): " << mem_usage << std::endl;

    return 0;
}
