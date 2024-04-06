/*
 * @Author: Zhou Zijian 
 * @Date: 2023-02-23 22:57:37 
 * @Last Modified by: Zhou Zijian
 * @Last Modified time: 2024-02-21 02:15:01
 */

#ifndef TIME_PERF_H
#define TIME_PERF_H

#include <string>
#include <chrono>
#include "log.h"

#ifndef TIME_PERF_ON
#define TIME_PERF_ON 0
#endif

class TimePerf {
public:
    TimePerf(const char *name) : m_name(name), m_startTime(std::chrono::high_resolution_clock::now())
    {}
    ~TimePerf()
    {
#if TIME_PERF_ON
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> tm = endTime - m_startTime;
        LOGD("%s cost : %f ms", m_name.c_str(), tm.count());
#endif
    }

private:
    std::string m_name;
    std::chrono::high_resolution_clock::time_point m_startTime;
};

#ifndef TIMEPERF
#define TIMEPERF(x) TimePerf TIME_PERF_##x(#x);
#endif

#ifndef TIMEPERT_BEGIN
#define TIMEPERT_BEGIN(x) TimePerf *TIME_PERF_##x = new TimePerf(#x);
#endif

#ifndef TIMEPERT_END
#define TIMEPERT_END(x) delete TIME_PERF_##x;
#endif

#endif  // TIME_PERF_H
