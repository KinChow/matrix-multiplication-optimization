/*
 * @Author: Zhou Zijian
 * @Date: 2024-02-21 01:06:38
 * @Last Modified by: Zhou Zijian
 * @Last Modified time: 2024-02-27 00:44:13
 */

#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>
#include "config.h"
#include "log.h"
#include "gemm.h"

static const char *helpStr = "\n " PROJECT_NAME " [OPTIONS]"
                             "\n"
                             "\n OPTIONS:"
                             "\n  --test n                    run test n"
                             "\n  --all-tests                 run all above tests [default]"
                             "\n  --size size                 size of data"
                             "\n  --check                     check result"
                             "\n  -v, --version               display version"
                             "\n  -h, --help                  display help message"
                             "\n";

int main(int argc, char *argv[])
{
    bool allTests = true;
    std::vector<std::pair<std::function<void(Matrix &, Matrix &, Matrix &)>, bool>> tests{
        {GeMM::Optimize1, false},
        {GeMM::Optimize2, false},
        {GeMM::Optimize3, false},
        {GeMM::Optimize4, false},
        {GeMM::Optimize5, false},
        {GeMM::Optimize6, false},
        {GeMM::Optimize7, false},
        {GeMM::Optimize8, false},
        {GeMM::Optimize9, false},
        {GeMM::Optimize10, false},
        {GeMM::Optimize11, false},
        {GeMM::Optimize12, false},
        {GeMM::Optimize13, false},
        {GeMM::Optimize14, false},
        {GeMM::Optimize15, false},
        {GeMM::Optimize16, false},
    };
    int size = 1024;
    bool check = false;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "--help") == 0)) {
            LOGI("%s", helpStr);
            exit(0);
        } else if ((strcmp(argv[i], "-v") == 0) || (strcmp(argv[i], "--version") == 0)) {
            LOGI(PROJECT_NAME " version: %d.%d.%d.%d", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH, VERSION_TWEAK);
            exit(0);
        } else if (strcmp(argv[i], "--test") == 0) {
            allTests = false;
            if (i + 1 < argc) {
                int idx = atoi(argv[i + 1]);
                if (idx > 0 && idx <= tests.size()) {
                    tests[idx - 1].second = true;
                } else {
                    LOGE("Invalid test index: %d", idx);
                    exit(-1);
                }
                i++;
            }
        } else if (strcmp(argv[i], "--all-tests") == 0) {
            allTests = false;
            for (auto &test : tests) {
                test.second = true;
            }
        } else if (strcmp(argv[i], "--size") == 0) {
            if (i + 1 < argc) {
                size = atoi(argv[i + 1]);
                i++;
            }
        } else if (strcmp(argv[i], "--check") == 0) {
            check = true;
        } else {
            LOGE("Invalid option: %s", argv[i]);
            LOGI("%s", helpStr);
            exit(-1);
        }
    }
    if (allTests) {
        for (auto &test : tests) {
            test.second = true;
        }
    }

    std::vector<float> input1Data(size * size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            input1Data[i * size + j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }
    Matrix input1{input1Data, size, size};
    std::vector<float> input2Data(size * size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            input2Data[i * size + j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }
    Matrix input2{input2Data, size, size};

    if (check) {
        for (int i = 0; i < tests.size(); i++) {
            if (!tests[i].second) {
                continue;
            }
            std::vector<float> output1Data(size * size);
            Matrix output1{output1Data, size, size};
            std::vector<float> output2Data(size * size);
            Matrix output2{output2Data, size, size};
            GeMM::Origin(input1, input2, output1);
            tests[i].first(input1, input2, output2);
            if (GeMM::CheckResult(output1, output2)) {
                LOGI("Optimize%d passed!", i + 1);
            } else {
                LOGE("Optimize%d failed!", i + 1);
            }
        }
    } else {
        for (int i = 0; i < tests.size(); i++) {
            if (!tests[i].second) {
                continue;
            }
            std::vector<float> outputData(size * size);
            Matrix output{outputData, size, size};
            tests[i].first(input1, input2, output);
        }
    }
    return 0;
}