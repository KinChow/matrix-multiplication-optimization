/*
 * @Author: Zhou Zijian 
 * @Date: 2023-12-16 01:37:29 
 * @Last Modified by: Zhou Zijian
 * @Last Modified time: 2024-01-19 00:44:44
 */

#ifndef LOG_H
#define LOG_H

#include <cstdio>
#include <ctime>
#include "config.h"

#ifndef PROJECT_NAME
#define PROJECT_NAME ""
#endif  // PROJECT_NAME

#ifndef LOGD
#define LOGD(format, ...)                                                                                         \
    printf("[DEBUG] [%s] [%-24.24s] [%-4d] [" format "]\n", PROJECT_NAME, __FUNCTION__, __LINE__, ##__VA_ARGS__); \
    fflush(stdout)
#endif  // LOGD

#ifndef LOGI
#define LOGI(format, ...)                                                                                        \
    printf("[INFO] [%s] [%-24.24s] [%-4d] [" format "]\n", PROJECT_NAME, __FUNCTION__, __LINE__, ##__VA_ARGS__); \
    fflush(stdout)
#endif  // LOGI

#ifndef LOGW
#define LOGW(format, ...)                                                                                        \
    printf("[WARN] [%s] [%-24.24s] [%-4d] [" format "]\n", PROJECT_NAME, __FUNCTION__, __LINE__, ##__VA_ARGS__); \
    fflush(stdout)
#endif  // LOGW

#ifndef LOGE
#define LOGE(format, ...)                                                                                         \
    printf("[ERROR] [%s] [%-24.24s] [%-4d] [" format "]\n", PROJECT_NAME, __FUNCTION__, __LINE__, ##__VA_ARGS__); \
    fflush(stdout)
#endif  // LOGE

#endif  // LOG_H