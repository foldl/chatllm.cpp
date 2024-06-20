//
// Created by RobinQu on 2024/6/19.
//

#include <iostream>
#include <thread>

#include "chat.h"

int main() {
    using namespace chatllm;
    ModelLoader model_loader {"/Users/robinqu/Workspace/github/robinqu/instinct.cpp/build/modules/instinct-transformer/test/_assets/model_bins/bge-reranker-v2-m3.bin"};

    ModelFactory::Result result;
    ModelFactory::load(model_loader, result, {});
    std::vector<int> ids;
    result.tokenizer->encode_qa("hello", R"(Create an Endpoint\n\nAfter your first login, you will be directed to the [Endpoint creation page](https://ui.endpoints.huggingface.co/new). As an example, this guide will go through the steps to deploy [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) for text classification. \n\n## 1. Enter the Hugging Face Repository ID and your desired endpoint name:\n\n<img src=\"https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/1_repository.png\" alt=\"select repository\" />",
      "## 2. Select your Cloud Provider and region. Initially, only AWS will be available as a Cloud Provider with the `us-east-1` and `eu-west-1` regions. We will add Azure soon, and if you need to test Endpoints with other Cloud Providers or regions, please let us know.\n\n<img src=\"https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/1_region.png\" alt=\"select region\" />\n\n## 3. Define the [Security Level](security) for the Endpoint:\n\n<img src=\"https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/1_security.png\" alt=\"define security\" />",
      "## 4. Create your Endpoint by clicking **Create Endpoint**. By default, your Endpoint is created with a medium CPU (2 x 4GB vCPUs with Intel Xeon Ice Lake) The cost estimate assumes the Endpoint will be up for an entire month, and does not take autoscaling into account.\n\n<img src=\"https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/1_create_cost.png\" alt=\"create endpoint\" />\n\n## 5. Wait for the Endpoint to build, initialize and run which can take between 1 to 5 minutes.\n\n<img src=\"https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/overview.png\" alt=\"overview\" />\n\n## 6. Test your Endpoint in the overview with the Inference widget \ud83c\udfc1 \ud83c\udf89!)", ids);

    {
        GenerationConfig config;
        config.num_threads = std::thread::hardware_concurrency();
        std::cout << result.model->qa_rank(config, ids) << std::endl;
    }

    {
        GenerationConfig config;
        config.num_threads = 1;
        std::cout << result.model->qa_rank(config, ids) << std::endl;
    }


}
