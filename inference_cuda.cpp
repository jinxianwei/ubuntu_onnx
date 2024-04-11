#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
using namespace std;

class InferenceOnnx
{
private:
    std::vector<const char *> input_names = {"images"};
    std::vector<const char *> output_names = {"prob", "features"};

    cv::Mat image;

    size_t input_size;
    size_t prob_size;
    size_t features_size;
    std::vector<int64_t> input_dims;
    std::vector<int64_t> prob_dims;
    std::vector<int64_t> features_dims;

    Ort::Env m_env;
    Ort::MemoryInfo m_memoryInfo;
    Ort::SessionOptions m_options;

    Ort::Session *m_session = nullptr;

public:
    InferenceOnnx(const std::string &modelPath);
    ~InferenceOnnx();

    int inference(const std::string imgPath);
};

InferenceOnnx::InferenceOnnx(const std::string &modelPath) : m_env(ORT_LOGGING_LEVEL_WARNING, "test"),
                                                             m_memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) // 初始化 memoryinfo
{
    m_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    OrtCUDAProviderOptions cuda_options;
    m_options.AppendExecutionProvider_CUDA(cuda_options);
    m_session = new Ort::Session(m_env, modelPath.c_str(), m_options); // (+_+)?

    input_dims = m_session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    prob_dims = m_session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    features_dims = m_session->GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();

    input_size = input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3];
    prob_size = prob_dims[0] * prob_dims[1];
    features_size = features_dims[0] * features_dims[1];
}

InferenceOnnx::~InferenceOnnx()
{
}

int InferenceOnnx::inference(const std::string imgPath)
{
    auto image = imread(imgPath, cv::IMREAD_COLOR);
    cv::resize(image, image, cv::Size(512, 512), cv::INTER_LINEAR);

    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0);

    std::vector<Ort::Value> inputTensors;
    std::vector<float> input_data(input_size, 0.0f);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(m_memoryInfo, blob.ptr<float>(), blob.total(), input_dims.data(), input_dims.size());
    inputTensors.push_back(std::move(input_tensor));

    std::vector<Ort::Value> outputTensors;
    std::vector<float> prob_data(prob_size, 0.0f);
    Ort::Value output_prob_tensor = Ort::Value::CreateTensor<float>(m_memoryInfo, prob_data.data(), prob_size, prob_dims.data(), prob_dims.size());
    outputTensors.push_back(std::move(output_prob_tensor));

    std::vector<float> feature_data(features_size, 0.0f);
    Ort::Value output_feature_tensor = Ort::Value::CreateTensor<float>(m_memoryInfo, feature_data.data(), features_size, features_dims.data(), features_dims.size());
    outputTensors.push_back(std::move(output_feature_tensor));

    m_session->Run(Ort::RunOptions{nullptr}, input_names.data(), inputTensors.data(), 1, output_names.data(), outputTensors.data(), 2);

    float *prob_result = outputTensors[0].GetTensorMutableData<float>();
    float *feature_result = outputTensors[1].GetTensorMutableData<float>();

    int result_index = 0;
    float max_prob = 0;
    cout << "预测的概率为：";
    for (int i = 0; i < prob_dims[1]; i++)
    {

        cout << prob_result[i] << ",";
        if (max_prob < prob_result[i])
        {
            max_prob = prob_result[i];
            result_index = i;
        }
    }
    cout << endl
         << "预测的类别下标为:" << result_index << endl;

    return result_index;
}

int main()
{
    std::string modelPath = "shufflenet_v2.onnx";
    std::string image_path = "ISIC_0024306.jpg";
    InferenceOnnx inferenceonnx(modelPath);
    int result = inferenceonnx.inference(image_path);
    cout << result << endl;
}
