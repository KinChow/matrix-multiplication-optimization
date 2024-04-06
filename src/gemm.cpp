/*
 * @Author: Zhou Zijian 
 * @Date: 2024-02-21 01:47:27 
 * @Last Modified by: Zhou Zijian
 * @Last Modified time: 2024-02-27 00:38:27
 */

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include "TimePerf.h"
#include "log.h"
#include "gemm.h"

constexpr float EPSILON = 1e-5;

bool GeMM::CheckResult(Matrix &a, Matrix &b)
{
    if (a.w != b.w) {
        LOGE("Matrix A's width(%d) is not equal to Matrix B's width(%d)", a.w, b.w);
        return false;
    }
    if (a.h != b.h) {
        LOGE("Matrix A's height(%d) is not equal to Matrix B's height(%d)", a.h, b.h);
        return false;
    }
    if (!a.data || !b.data) {
        LOGE("Matrix A(%p), B(%p) is null", a.data, b.data);
        return false;
    }
    for (int i = 0; i < a.h; i++) {
        for (int j = 0; j < a.w; j++) {
            if (std::abs(a.data[i * a.w + j] - b.data[i * a.w + j]) > EPSILON) {
                LOGE("Matrix A[%d][%d]=%f is not equal to Matrix B[%d][%d]=%f", i, j, a.data[i * a.w + j], i, j,
                    b.data[i * a.w + j]);
                return false;
            }
        }
    }
    return true;
}

/**
 * matrix multiplication
 * i for c height, j for c width, k for a width
 * for loop ijk
 *
 * @param a The first input matrix
 * @param b The second input matrix
 * @param c The output matrix to store the result
 *
 * @return void
 *
 * @throws None
 */
void GeMM::Origin(Matrix &a, Matrix &b, Matrix &c)
{
    if (!CheckParam(a, b, c)) {
        return;
    }
    float *pA = a.data;
    float *pB = b.data;
    float *pC = c.data;
    {
        TIMEPERF(Origin);
        for (int i = 0; i < a.h; i++) {
            for (int j = 0; j < b.w; j++) {
                for (int k = 0; k < a.w; k++) {
                    pC[i * c.w + j] += pA[i * a.w + k] * pB[k * b.w + j];
                }
            }
        }
    }
}

/**
 * matrix multiplication
 * i for c height, j for c width, k for a width
 * for loop ikj
 *
 * @param a The first input matrix
 * @param b The second input matrix
 * @param c The output matrix to store the result
 *
 * @return void
 *
 * @throws None
 */
void GeMM::Optimize1(Matrix &a, Matrix &b, Matrix &c)
{
    if (!CheckParam(a, b, c)) {
        return;
    }
    float *pA = a.data;
    float *pB = b.data;
    float *pC = c.data;
    {
        TIMEPERF(Optimize1);
        for (int i = 0; i < a.h; i++) {
            for (int k = 0; k < a.w; k++) {
                float a0 = pA[i * a.w + k];
                for (int j = 0; j < b.w; j++) {
                    pC[i * c.w + j] += a0 * pB[k * b.w + j];
                }
            }
        }
    }
}

/**
 * matrix multiplication
 * i for c height, j for c width, k for a width
 * for loop kij
 *
 * @param a The first input matrix
 * @param b The second input matrix
 * @param c The output matrix to store the result
 *
 * @return void
 *
 * @throws None
 */
void GeMM::Optimize2(Matrix &a, Matrix &b, Matrix &c)
{
    if (!CheckParam(a, b, c)) {
        return;
    }
    float *pA = a.data;
    float *pB = b.data;
    float *pC = c.data;
    {
        TIMEPERF(Optimize2);
        for (int k = 0; k < a.w; k++) {
            for (int i = 0; i < a.h; i++) {
                float a0 = pA[i * a.w + k];
                for (int j = 0; j < b.w; j++) {
                    pC[i * c.w + j] += a0 * pB[k * b.w + j];
                }
            }
        }
    }
}

/**
 * matrix multiplication
 * i for c height, j for c width, k for a width
 * for loop ikj
 * unroll j
 *
 * @param a The first input matrix
 * @param b The second input matrix
 * @param c The output matrix to store the result
 *
 * @return void
 *
 * @throws None
 */
void GeMM::Optimize3(Matrix &a, Matrix &b, Matrix &c)
{
    if (!CheckParam(a, b, c)) {
        return;
    }
    float *pA = a.data;
    float *pB = b.data;
    float *pC = c.data;
    {
        TIMEPERF(Optimize3);
        for (int i = 0; i < a.h; i++) {
            for (int k = 0; k < a.w; k++) {
                float a0 = pA[i * a.w + k];
                int j = 0;
                for (; j < (b.w & ~3); j += 4) {
                    pC[i * c.w + j] += a0 * pB[k * b.w + j];
                    pC[i * c.w + j + 1] += a0 * pB[k * b.w + j + 1];
                    pC[i * c.w + j + 2] += a0 * pB[k * b.w + j + 2];
                    pC[i * c.w + j + 3] += a0 * pB[k * b.w + j + 3];
                }
                for (; j < b.w; j++) {
                    pC[i * c.w + j] += a0 * pB[k * b.w + j];
                }
            }
        }
    }
}

/**
 * matrix multiplication
 * i for c height, j for c width, k for a width
 * for loop kij
 * unroll j
 *
 * @param a The first input matrix
 * @param b The second input matrix
 * @param c The output matrix to store the result
 *
 * @return void
 *
 * @throws None
 */
void GeMM::Optimize4(Matrix &a, Matrix &b, Matrix &c)
{
    if (!CheckParam(a, b, c)) {
        return;
    }
    float *pA = a.data;
    float *pB = b.data;
    float *pC = c.data;
    {
        TIMEPERF(Optimize4);
        for (int k = 0; k < a.w; k++) {
            for (int i = 0; i < a.h; i++) {
                float a0 = pA[i * a.w + k];
                int j = 0;
                for (; j < (b.w & ~3); j += 4) {
                    pC[i * c.w + j] += a0 * pB[k * b.w + j];
                    pC[i * c.w + j + 1] += a0 * pB[k * b.w + j + 1];
                    pC[i * c.w + j + 2] += a0 * pB[k * b.w + j + 2];
                    pC[i * c.w + j + 3] += a0 * pB[k * b.w + j + 3];
                }
                for (; j < b.w; j++) {
                    pC[i * c.w + j] += a0 * pB[k * b.w + j];
                }
            }
        }
    }
}

/**
 * matrix multiplication
 * i for c height, j for c width, k for a width
 * for loop ikj
 * simd j
 *
 * @param a The first input matrix
 * @param b The second input matrix
 * @param c The output matrix to store the result
 *
 * @return void
 *
 * @throws None
 */
void GeMM::Optimize5(Matrix &a, Matrix &b, Matrix &c)
{
    if (!CheckParam(a, b, c)) {
        return;
    }
    float *pA = a.data;
    float *pB = b.data;
    float *pC = c.data;
    {
        TIMEPERF(Optimize5);
        for (int i = 0; i < a.h; i++) {
            for (int k = 0; k < a.w; k++) {
                float a0 = pA[i * a.w + k];
#ifdef __ARM_NEON
                float32x4_t vA0 = vdupq_n_f32(a0);
#endif
                int j = 0;
#ifdef __ARM_NEON
                for (; j < (b.w & ~3); j += 4) {
                    float32x4_t vB = vld1q_f32(pB + k * b.w + j);
                    float32x4_t vC = vld1q_f32(pC + i * c.w + j);
                    vC = vfmaq_f32(vC, vA0, vB);
                    vst1q_f32(pC + i * c.w + j, vC);
                }
#endif
                for (; j < b.w; j++) {
                    pC[i * c.w + j] += a0 * pB[k * b.w + j];
                }
            }
        }
    }
}

/**
 * matrix multiplication
 * i for c height, j for c width, k for a width
 * for loop kij
 * simd j
 *
 * @param a The first input matrix
 * @param b The second input matrix
 * @param c The output matrix to store the result
 *
 * @return void
 *
 * @throws None
 */
void GeMM::Optimize6(Matrix &a, Matrix &b, Matrix &c)
{
    if (!CheckParam(a, b, c)) {
        return;
    }
    float *pA = a.data;
    float *pB = b.data;
    float *pC = c.data;
    {
        TIMEPERF(Optimize6);
        for (int k = 0; k < a.w; k++) {
            for (int i = 0; i < a.h; i++) {
                float a0 = pA[i * a.w + k];
#ifdef __ARM_NEON
                float32x4_t vA0 = vdupq_n_f32(a0);
#endif
                int j = 0;
#ifdef __ARM_NEON
                for (; j < (b.w & ~3); j += 4) {
                    float32x4_t vB = vld1q_f32(pB + k * b.w + j);
                    float32x4_t vC = vld1q_f32(pC + i * c.w + j);
                    vC = vfmaq_f32(vC, vA0, vB);
                    vst1q_f32(pC + i * c.w + j, vC);
                }
#endif
                for (; j < b.w; j++) {
                    pC[i * c.w + j] += a0 * pB[k * b.w + j];
                }
            }
        }
    }
}

/**
 * matrix multiplication
 * i for c height, j for c width, k for a width
 * for loop ikj
 * unroll kj
 *
 * @param a The first input matrix
 * @param b The second input matrix
 * @param c The output matrix to store the result
 *
 * @return void
 *
 * @throws None
 */
void GeMM::Optimize7(Matrix &a, Matrix &b, Matrix &c)
{
    if (!CheckParam(a, b, c)) {
        return;
    }
    float *pA = a.data;
    float *pB = b.data;
    float *pC = c.data;
    {
        TIMEPERF(Optimize7);
        for (int i = 0; i < a.h; i++) {
            int k = 0;
            for (; k < (a.w & ~3); k += 4) {
                float a0 = pA[i * a.w + k];
                float a1 = pA[i * a.w + k + 1];
                float a2 = pA[i * a.w + k + 2];
                float a3 = pA[i * a.w + k + 3];
                int j = 0;
                for (; j < (b.w & ~3); j += 4) {
                    pC[i * c.w + j] += a0 * pB[k * b.w + j];
                    pC[i * c.w + j] += a1 * pB[(k + 1) * b.w + j];
                    pC[i * c.w + j] += a2 * pB[(k + 2) * b.w + j];
                    pC[i * c.w + j] += a3 * pB[(k + 3) * b.w + j];

                    pC[i * c.w + j + 1] += a0 * pB[k * b.w + j + 1];
                    pC[i * c.w + j + 1] += a1 * pB[(k + 1) * b.w + j + 1];
                    pC[i * c.w + j + 1] += a2 * pB[(k + 2) * b.w + j + 1];
                    pC[i * c.w + j + 1] += a3 * pB[(k + 3) * b.w + j + 1];

                    pC[i * c.w + j + 2] += a0 * pB[k * b.w + j + 2];
                    pC[i * c.w + j + 2] += a1 * pB[(k + 1) * b.w + j + 2];
                    pC[i * c.w + j + 2] += a2 * pB[(k + 2) * b.w + j + 2];
                    pC[i * c.w + j + 2] += a3 * pB[(k + 3) * b.w + j + 2];

                    pC[i * c.w + j + 3] += a0 * pB[k * b.w + j + 3];
                    pC[i * c.w + j + 3] += a1 * pB[(k + 1) * b.w + j + 3];
                    pC[i * c.w + j + 3] += a2 * pB[(k + 2) * b.w + j + 3];
                    pC[i * c.w + j + 3] += a3 * pB[(k + 3) * b.w + j + 3];
                }
                for (; j < b.w; j++) {
                    pC[i * c.w + j] += a0 * pB[k * b.w + j];
                    pC[i * c.w + j] += a1 * pB[(k + 1) * b.w + j];
                    pC[i * c.w + j] += a2 * pB[(k + 2) * b.w + j];
                    pC[i * c.w + j] += a3 * pB[(k + 3) * b.w + j];
                }
            }
            for (; k < a.w; k++) {
                float a0 = pA[i * a.w + k];
                int j = 0;
                for (; j < (b.w & ~3); j += 4) {
                    pC[i * c.w + j] += a0 * pB[k * b.w + j];
                    pC[i * c.w + j + 1] += a0 * pB[k * b.w + j + 1];
                    pC[i * c.w + j + 2] += a0 * pB[k * b.w + j + 2];
                    pC[i * c.w + j + 3] += a0 * pB[k * b.w + j + 3];
                }
                for (; j < b.w; j++) {
                    pC[i * c.w + j] += a0 * pB[k * b.w + j];
                }
            }
        }
    }
}

/**
 * matrix multiplication
 * i for c height, j for c width, k for a width
 * for loop kij
 * unroll kj
 *
 * @param a The first input matrix
 * @param b The second input matrix
 * @param c The output matrix to store the result
 *
 * @return void
 *
 * @throws None
 */
void GeMM::Optimize8(Matrix &a, Matrix &b, Matrix &c)
{
    if (!CheckParam(a, b, c)) {
        return;
    }
    float *pA = a.data;
    float *pB = b.data;
    float *pC = c.data;
    {
        TIMEPERF(Optimize8);
        int k = 0;
        for (; k < (a.w & ~3); k += 4) {
            for (int i = 0; i < a.h; i++) {
                float a0 = pA[i * a.w + k];
                float a1 = pA[i * a.w + k + 1];
                float a2 = pA[i * a.w + k + 2];
                float a3 = pA[i * a.w + k + 3];
                int j = 0;
                for (; j < (b.w & ~3); j += 4) {
                    pC[i * c.w + j] += a0 * pB[k * b.w + j];
                    pC[i * c.w + j] += a1 * pB[(k + 1) * b.w + j];
                    pC[i * c.w + j] += a2 * pB[(k + 2) * b.w + j];
                    pC[i * c.w + j] += a3 * pB[(k + 3) * b.w + j];

                    pC[i * c.w + j + 1] += a0 * pB[k * b.w + j + 1];
                    pC[i * c.w + j + 1] += a1 * pB[(k + 1) * b.w + j + 1];
                    pC[i * c.w + j + 1] += a2 * pB[(k + 2) * b.w + j + 1];
                    pC[i * c.w + j + 1] += a3 * pB[(k + 3) * b.w + j + 1];

                    pC[i * c.w + j + 2] += a0 * pB[k * b.w + j + 2];
                    pC[i * c.w + j + 2] += a1 * pB[(k + 1) * b.w + j + 2];
                    pC[i * c.w + j + 2] += a2 * pB[(k + 2) * b.w + j + 2];
                    pC[i * c.w + j + 2] += a3 * pB[(k + 3) * b.w + j + 2];

                    pC[i * c.w + j + 3] += a0 * pB[k * b.w + j + 3];
                    pC[i * c.w + j + 3] += a1 * pB[(k + 1) * b.w + j + 3];
                    pC[i * c.w + j + 3] += a2 * pB[(k + 2) * b.w + j + 3];
                    pC[i * c.w + j + 3] += a3 * pB[(k + 3) * b.w + j + 3];
                }
                for (; j < b.w; j++) {
                    pC[i * c.w + j] += a0 * pB[k * b.w + j];
                    pC[i * c.w + j] += a1 * pB[(k + 1) * b.w + j];
                    pC[i * c.w + j] += a2 * pB[(k + 2) * b.w + j];
                    pC[i * c.w + j] += a3 * pB[(k + 3) * b.w + j];
                }
            }
        }
        for (; k < a.w; k++) {
            for (int i = 0; i < a.h; i++) {
                float a0 = pA[i * a.w + k];
                int j = 0;
                for (; j < (b.w & ~3); j += 4) {
                    pC[i * c.w + j] += a0 * pB[k * b.w + j];
                    pC[i * c.w + j + 1] += a0 * pB[k * b.w + j + 1];
                    pC[i * c.w + j + 2] += a0 * pB[k * b.w + j + 2];
                    pC[i * c.w + j + 3] += a0 * pB[k * b.w + j + 3];
                }
                for (; j < b.w; j++) {
                    pC[i * c.w + j] += a0 * pB[k * b.w + j];
                }
            }
        }
    }
}

/**
 * matrix multiplication
 * i for c height, j for c width, k for a width
 * for loop ikj
 * simd kj
 *
 * @param a The first input matrix
 * @param b The second input matrix
 * @param c The output matrix to store the result
 *
 * @return void
 *
 * @throws None
 */
void GeMM::Optimize9(Matrix &a, Matrix &b, Matrix &c)
{
    if (!CheckParam(a, b, c)) {
        return;
    }
    float *pA = a.data;
    float *pB = b.data;
    float *pC = c.data;
    {
        TIMEPERF(Optimize9);
        for (int i = 0; i < a.h; i++) {
            int k = 0;
            for (; k < (a.w & ~3); k += 4) {
                float a0 = pA[i * a.w + k];
                float a1 = pA[i * a.w + k + 1];
                float a2 = pA[i * a.w + k + 2];
                float a3 = pA[i * a.w + k + 3];
#ifdef __ARM_NEON
                float32x4_t vA0 = vdupq_n_f32(a0);
                float32x4_t vA1 = vdupq_n_f32(a1);
                float32x4_t vA2 = vdupq_n_f32(a2);
                float32x4_t vA3 = vdupq_n_f32(a3);
#endif
                int j = 0;
#ifdef __ARM_NEON
                for (; j < (b.w & ~3); j += 4) {
                    float32x4_t vB0 = vld1q_f32(pB + k * b.w + j);
                    float32x4_t vB1 = vld1q_f32(pB + (k + 1) * b.w + j);
                    float32x4_t vB2 = vld1q_f32(pB + (k + 2) * b.w + j);
                    float32x4_t vB3 = vld1q_f32(pB + (k + 3) * b.w + j);

                    float32x4_t vC = vld1q_f32(pC + i * c.w + j);
                    vC = vfmaq_f32(vC, vA0, vB0);
                    vC = vfmaq_f32(vC, vA1, vB1);
                    vC = vfmaq_f32(vC, vA2, vB2);
                    vC = vfmaq_f32(vC, vA3, vB3);
                    vst1q_f32(pC + i * c.w + j, vC);
                }
#endif
                for (; j < b.w; j++) {
                    pC[i * c.w + j] += a0 * pB[k * b.w + j];
                    pC[i * c.w + j] += a1 * pB[(k + 1) * b.w + j];
                    pC[i * c.w + j] += a2 * pB[(k + 2) * b.w + j];
                    pC[i * c.w + j] += a3 * pB[(k + 3) * b.w + j];
                }
            }
            for (; k < a.w; k++) {
                float a0 = pA[i * a.w + k];
#ifdef __ARM_NEON
                float32x4_t vA0 = vdupq_n_f32(a0);
#endif
                int j = 0;
#ifdef __ARM_NEON
                for (; j < (b.w & ~3); j += 4) {
                    float32x4_t vB = vld1q_f32(pB + k * b.w + j);
                    float32x4_t vC = vld1q_f32(pC + i * c.w + j);
                    vC = vfmaq_f32(vC, vA0, vB);
                    vst1q_f32(pC + i * c.w + j, vC);
                }
#endif
                for (; j < b.w; j++) {
                    pC[i * c.w + j] += a0 * pB[k * b.w + j];
                }
            }
        }
    }
}

/**
 * matrix multiplication
 * i for c height, j for c width, k for a width
 * for loop kij
 * simd kj
 *
 * @param a The first input matrix
 * @param b The second input matrix
 * @param c The output matrix to store the result
 *
 * @return void
 *
 * @throws None
 */
void GeMM::Optimize10(Matrix &a, Matrix &b, Matrix &c)
{
    if (!CheckParam(a, b, c)) {
        return;
    }
    float *pA = a.data;
    float *pB = b.data;
    float *pC = c.data;
    {
        TIMEPERF(Optimize10);
        int k = 0;
        for (; k < (a.w & ~3); k += 4) {
            for (int i = 0; i < a.h; i++) {
                float a0 = pA[i * a.w + k];
                float a1 = pA[i * a.w + k + 1];
                float a2 = pA[i * a.w + k + 2];
                float a3 = pA[i * a.w + k + 3];
#ifdef __ARM_NEON
                float32x4_t vA0 = vdupq_n_f32(a0);
                float32x4_t vA1 = vdupq_n_f32(a1);
                float32x4_t vA2 = vdupq_n_f32(a2);
                float32x4_t vA3 = vdupq_n_f32(a3);
#endif
                int j = 0;
#ifdef __ARM_NEON
                for (; j < (b.w & ~3); j += 4) {
                    float32x4_t vB0 = vld1q_f32(pB + k * b.w + j);
                    float32x4_t vB1 = vld1q_f32(pB + (k + 1) * b.w + j);
                    float32x4_t vB2 = vld1q_f32(pB + (k + 2) * b.w + j);
                    float32x4_t vB3 = vld1q_f32(pB + (k + 3) * b.w + j);

                    float32x4_t vC = vld1q_f32(pC + i * c.w + j);
                    vC = vfmaq_f32(vC, vA0, vB0);
                    vC = vfmaq_f32(vC, vA1, vB1);
                    vC = vfmaq_f32(vC, vA2, vB2);
                    vC = vfmaq_f32(vC, vA3, vB3);
                    vst1q_f32(pC + i * c.w + j, vC);
                }
#endif
                for (; j < b.w; j++) {
                    pC[i * c.w + j] += a0 * pB[k * b.w + j];
                    pC[i * c.w + j + 1] += a1 * pB[(k + 1) * b.w + j];
                    pC[i * c.w + j + 2] += a2 * pB[(k + 2) * b.w + j];
                    pC[i * c.w + j + 3] += a3 * pB[(k + 3) * b.w + j];
                }
            }
        }
        for (; k < a.w; k++) {
            for (int i = 0; i < a.h; i++) {
                float a0 = pA[i * a.w + k];
#ifdef __ARM_NEON
                float32x4_t vA = vdupq_n_f32(a0);
#endif
                int j = 0;
#ifdef __ARM_NEON
                for (; j < (b.w & ~3); j += 4) {
                    float32x4_t vB = vld1q_f32(pB + k * b.w + j);
                    float32x4_t vC = vld1q_f32(pC + i * c.w + j);
                    vC = vfmaq_f32(vC, vA, vB);
                    vst1q_f32(pC + i * c.w + j, vC);
                }
#endif
                for (; j < b.w; j++) {
                    pC[i * c.w + j] += a0 * pB[k * b.w + j];
                }
            }
        }
    }
}

/**
 * matrix multiplication
 * i for c height, j for c width, k for a width
 * for loop ikj
 * simd kj
 * align 4 kj
 *
 * @param a The first input matrix
 * @param b The second input matrix
 * @param c The output matrix to store the result
 *
 * @return void
 *
 * @throws None
 */
void GeMM::Optimize11(Matrix &a, Matrix &b, Matrix &c)
{
    if (!CheckParam(a, b, c)) {
        return;
    }
    float *pA = a.data;
    float *pB = b.data;
    float *pC = c.data;
    {
        TIMEPERF(Optimize11);
#ifdef __ARM_NEON
        for (int i = 0; i < a.h; i++) {
            int k = 0;
            for (; k < (a.w & ~3); k += 4) {
                float32x4_t vA = vld1q_f32(pA + i * a.w + k);
                float32x4_t vA0 = vdupq_laneq_f32(vA, 0);
                float32x4_t vA1 = vdupq_laneq_f32(vA, 1);
                float32x4_t vA2 = vdupq_laneq_f32(vA, 2);
                float32x4_t vA3 = vdupq_laneq_f32(vA, 3);
                int j = 0;
                for (; j < (b.w & ~3); j += 4) {
                    float32x4_t vB0 = vld1q_f32(pB + k * b.w + j);
                    float32x4_t vB1 = vld1q_f32(pB + (k + 1) * b.w + j);
                    float32x4_t vB2 = vld1q_f32(pB + (k + 2) * b.w + j);
                    float32x4_t vB3 = vld1q_f32(pB + (k + 3) * b.w + j);
                    float32x4_t vC = vld1q_f32(pC + i * c.w + j);
                    vC = vfmaq_f32(vC, vA0, vB0);
                    vC = vfmaq_f32(vC, vA1, vB1);
                    vC = vfmaq_f32(vC, vA2, vB2);
                    vC = vfmaq_f32(vC, vA3, vB3);
                    vst1q_f32(pC + i * c.w + j, vC);
                }
            }
        }
#endif
    }
}

/**
 * matrix multiplication
 * i for c height, j for c width, k for a width
 * for loop kij
 * simd kj
 * align 4 kj
 *
 * @param a The first input matrix
 * @param b The second input matrix
 * @param c The output matrix to store the result
 *
 * @return void
 *
 * @throws None
 */
void GeMM::Optimize12(Matrix &a, Matrix &b, Matrix &c)
{
    if (!CheckParam(a, b, c)) {
        return;
    }
    float *pA = a.data;
    float *pB = b.data;
    float *pC = c.data;
    {
        TIMEPERF(Optimize12);
#ifdef __ARM_NEON
        int k = 0;
        for (; k < (a.w & ~3); k += 4) {
            for (int i = 0; i < a.h; i++) {
                float32x4_t vA = vld1q_f32(pA + i * a.w + k);
                float32x4_t vA0 = vdupq_laneq_f32(vA, 0);
                float32x4_t vA1 = vdupq_laneq_f32(vA, 1);
                float32x4_t vA2 = vdupq_laneq_f32(vA, 2);
                float32x4_t vA3 = vdupq_laneq_f32(vA, 3);
                int j = 0;
                for (; j < (b.w & ~3); j += 4) {
                    float32x4_t vB0 = vld1q_f32(pB + k * b.w + j);
                    float32x4_t vB1 = vld1q_f32(pB + (k + 1) * b.w + j);
                    float32x4_t vB2 = vld1q_f32(pB + (k + 2) * b.w + j);
                    float32x4_t vB3 = vld1q_f32(pB + (k + 3) * b.w + j);
                    float32x4_t vC = vld1q_f32(pC + i * c.w + j);
                    vC = vfmaq_f32(vC, vA0, vB0);
                    vC = vfmaq_f32(vC, vA1, vB1);
                    vC = vfmaq_f32(vC, vA2, vB2);
                    vC = vfmaq_f32(vC, vA3, vB3);
                    vst1q_f32(pC + i * c.w + j, vC);
                }
            }
        }
#endif
    }
}

/**
 * matrix multiplication
 * i for c height, j for c width, k for a width
 * for loop ikj
 * simd kj
 * align 4 ikj
 *
 * @param a The first input matrix
 * @param b The second input matrix
 * @param c The output matrix to store the result
 *
 * @return void
 *
 * @throws None
 */
void GeMM::Optimize13(Matrix &a, Matrix &b, Matrix &c)
{
    if (!CheckParam(a, b, c)) {
        return;
    }
    float *pA = a.data;
    float *pB = b.data;
    float *pC = c.data;
    {
        TIMEPERF(Optimize13);
#ifdef __ARM_NEON
        int aIdx;
        int bIdx;
        int cIdx;
        float32x4_t vA0;
        float32x4_t vA1;
        float32x4_t vA2;
        float32x4_t vA3;
        float32x4_t vB0;
        float32x4_t vB1;
        float32x4_t vB2;
        float32x4_t vB3;
        float32x4_t vC0;
        float32x4_t vC1;
        float32x4_t vC2;
        float32x4_t vC3;
        for (int i = 0; i < (a.h & ~3); i += 4) {
            for (int k = 0; k < (a.w & ~3); k += 4) {
                aIdx = i * a.w + k;
                vA0 = vld1q_f32(pA + aIdx);
                vA1 = vld1q_f32(pA + aIdx + a.w);
                vA2 = vld1q_f32(pA + aIdx + a.w * 2);
                vA3 = vld1q_f32(pA + aIdx + a.w * 3);
                for (int j = 0; j < (b.w & ~3); j += 4) {
                    bIdx = k * b.w + j;
                    vB0 = vld1q_f32(pB + bIdx);
                    vB1 = vld1q_f32(pB + bIdx + b.w);
                    vB2 = vld1q_f32(pB + bIdx + b.w * 2);
                    vB3 = vld1q_f32(pB + bIdx + b.w * 3);
                    cIdx = i * c.w + j;
                    vC0 = vld1q_f32(pC + cIdx);
                    vC0 = vfmaq_laneq_f32(vC0, vB0, vA0, 0);
                    vC0 = vfmaq_laneq_f32(vC0, vB1, vA0, 1);
                    vC0 = vfmaq_laneq_f32(vC0, vB2, vA0, 2);
                    vC0 = vfmaq_laneq_f32(vC0, vB3, vA0, 3);
                    vst1q_f32(pC + cIdx, vC0);
                    vC1 = vld1q_f32(pC + cIdx + c.w);
                    vC1 = vfmaq_laneq_f32(vC1, vB0, vA1, 0);
                    vC1 = vfmaq_laneq_f32(vC1, vB1, vA1, 1);
                    vC1 = vfmaq_laneq_f32(vC1, vB2, vA1, 2);
                    vC1 = vfmaq_laneq_f32(vC1, vB3, vA1, 3);
                    vst1q_f32(pC + cIdx + c.w, vC1);
                    vC2 = vld1q_f32(pC + cIdx + c.w * 2);
                    vC2 = vfmaq_laneq_f32(vC2, vB0, vA2, 0);
                    vC2 = vfmaq_laneq_f32(vC2, vB1, vA2, 1);
                    vC2 = vfmaq_laneq_f32(vC2, vB2, vA2, 2);
                    vC2 = vfmaq_laneq_f32(vC2, vB3, vA2, 3);
                    vst1q_f32(pC + cIdx + c.w * 2, vC2);
                    vC3 = vld1q_f32(pC + cIdx + c.w * 3);
                    vC3 = vfmaq_laneq_f32(vC3, vB0, vA3, 0);
                    vC3 = vfmaq_laneq_f32(vC3, vB1, vA3, 1);
                    vC3 = vfmaq_laneq_f32(vC3, vB2, vA3, 2);
                    vC3 = vfmaq_laneq_f32(vC3, vB3, vA3, 3);
                    vst1q_f32(pC + cIdx + c.w * 3, vC3);
                }
            }
        }
#endif
    }
}

/**
 * matrix multiplication
 * i for c height, j for c width, k for a width
 * for loop kij
 * simd kj
 * align 4 kij
 *
 * @param a The first input matrix
 * @param b The second input matrix
 * @param c The output matrix to store the result
 *
 * @return void
 *
 * @throws None
 */
void GeMM::Optimize14(Matrix &a, Matrix &b, Matrix &c)
{
    if (!CheckParam(a, b, c)) {
        return;
    }
    float *pA = a.data;
    float *pB = b.data;
    float *pC = c.data;
    {
        TIMEPERF(Optimize14);
#ifdef __ARM_NEON
        int aIdx;
        int bIdx;
        int cIdx;
        float32x4_t vA0;
        float32x4_t vA1;
        float32x4_t vA2;
        float32x4_t vA3;
        float32x4_t vB0;
        float32x4_t vB1;
        float32x4_t vB2;
        float32x4_t vB3;
        float32x4_t vC0;
        float32x4_t vC1;
        float32x4_t vC2;
        float32x4_t vC3;
        for (int k = 0; k < (a.w & ~3); k += 4) {
            for (int i = 0; i < (a.h & ~3); i += 4) {
                aIdx = i * a.w + k;
                vA0 = vld1q_f32(pA + aIdx);
                vA1 = vld1q_f32(pA + aIdx + a.w);
                vA2 = vld1q_f32(pA + aIdx + a.w * 2);
                vA3 = vld1q_f32(pA + aIdx + a.w * 3);
                for (int j = 0; j < (b.w & ~3); j += 4) {
                    bIdx = k * b.w + j;
                    vB0 = vld1q_f32(pB + bIdx);
                    vB1 = vld1q_f32(pB + bIdx + b.w);
                    vB2 = vld1q_f32(pB + bIdx + b.w * 2);
                    vB3 = vld1q_f32(pB + bIdx + b.w * 3);
                    cIdx = i * c.w + j;
                    vC0 = vld1q_f32(pC + cIdx);
                    vC0 = vfmaq_laneq_f32(vC0, vB0, vA0, 0);
                    vC0 = vfmaq_laneq_f32(vC0, vB1, vA0, 1);
                    vC0 = vfmaq_laneq_f32(vC0, vB2, vA0, 2);
                    vC0 = vfmaq_laneq_f32(vC0, vB3, vA0, 3);
                    vst1q_f32(pC + cIdx, vC0);
                    vC1 = vld1q_f32(pC + cIdx + c.w);
                    vC1 = vfmaq_laneq_f32(vC1, vB0, vA1, 0);
                    vC1 = vfmaq_laneq_f32(vC1, vB1, vA1, 1);
                    vC1 = vfmaq_laneq_f32(vC1, vB2, vA1, 2);
                    vC1 = vfmaq_laneq_f32(vC1, vB3, vA1, 3);
                    vst1q_f32(pC + cIdx + c.w, vC1);
                    vC2 = vld1q_f32(pC + cIdx + c.w * 2);
                    vC2 = vfmaq_laneq_f32(vC2, vB0, vA2, 0);
                    vC2 = vfmaq_laneq_f32(vC2, vB1, vA2, 1);
                    vC2 = vfmaq_laneq_f32(vC2, vB2, vA2, 2);
                    vC2 = vfmaq_laneq_f32(vC2, vB3, vA2, 3);
                    vst1q_f32(pC + cIdx + c.w * 2, vC2);
                    vC3 = vld1q_f32(pC + cIdx + c.w * 3);
                    vC3 = vfmaq_laneq_f32(vC3, vB0, vA3, 0);
                    vC3 = vfmaq_laneq_f32(vC3, vB1, vA3, 1);
                    vC3 = vfmaq_laneq_f32(vC3, vB2, vA3, 2);
                    vC3 = vfmaq_laneq_f32(vC3, vB3, vA3, 3);
                    vst1q_f32(pC + cIdx + c.w * 3, vC3);
                }
            }
        }
#endif
    }
}

/**
 * matrix multiplication
 * i for c height, j for c width, k for a width
 * for loop ikj
 * simd kj
 * align 4 ikj
 *
 * @param a The first input matrix
 * @param b The second input matrix
 * @param c The output matrix to store the result
 *
 * @return void
 *
 * @throws None
 */
void GeMM::Optimize15(Matrix &a, Matrix &b, Matrix &c)
{
    if (!CheckParam(a, b, c)) {
        return;
    }
    float *pA = a.data;
    float *pB = b.data;
    float *pC = c.data;
    {
        TIMEPERF(Optimize15);
#ifdef __ARM_NEON
        int aIdx;
        int bIdx;
        int cIdx;
        float32x4_t vA0;
        float32x4_t vA1;
        float32x4_t vA2;
        float32x4_t vA3;
        float32x4_t vB0;
        float32x4_t vB1;
        float32x4_t vB2;
        float32x4_t vB3;
        float32x4_t vC0;
        float32x4_t vC1;
        float32x4_t vC2;
        float32x4_t vC3;
        int aHAlign = a.h & ~3;
        int aWAlign = a.w & ~3;
        int bWAlign = b.w & ~3;
        for (int i = 0; i < aHAlign; i += 4) {
            int aIdxBase = i * a.w;
            int cIdxBase = i * c.w;
            for (int k = 0; k < aWAlign; k += 4) {
                aIdx = aIdxBase + k;
                vA0 = vld1q_f32(pA + aIdx);
                vA1 = vld1q_f32(pA + aIdx + a.w);
                vA2 = vld1q_f32(pA + aIdx + a.w * 2);
                vA3 = vld1q_f32(pA + aIdx + a.w * 3);
                int bIdxBase = k * b.w;
                ;
                for (int j = 0; j < bWAlign; j += 4) {
                    bIdx = bIdxBase + j;
                    vB0 = vld1q_f32(pB + bIdx);
                    vB1 = vld1q_f32(pB + bIdx + b.w);
                    vB2 = vld1q_f32(pB + bIdx + b.w * 2);
                    vB3 = vld1q_f32(pB + bIdx + b.w * 3);
                    cIdx = cIdxBase + j;
                    vC0 = vld1q_f32(pC + cIdx);
                    vC0 = vfmaq_laneq_f32(vC0, vB0, vA0, 0);
                    vC0 = vfmaq_laneq_f32(vC0, vB1, vA0, 1);
                    vC0 = vfmaq_laneq_f32(vC0, vB2, vA0, 2);
                    vC0 = vfmaq_laneq_f32(vC0, vB3, vA0, 3);
                    vst1q_f32(pC + cIdx, vC0);
                    vC1 = vld1q_f32(pC + cIdx + c.w);
                    vC1 = vfmaq_laneq_f32(vC1, vB0, vA1, 0);
                    vC1 = vfmaq_laneq_f32(vC1, vB1, vA1, 1);
                    vC1 = vfmaq_laneq_f32(vC1, vB2, vA1, 2);
                    vC1 = vfmaq_laneq_f32(vC1, vB3, vA1, 3);
                    vst1q_f32(pC + cIdx + c.w, vC1);
                    vC2 = vld1q_f32(pC + cIdx + c.w * 2);
                    vC2 = vfmaq_laneq_f32(vC2, vB0, vA2, 0);
                    vC2 = vfmaq_laneq_f32(vC2, vB1, vA2, 1);
                    vC2 = vfmaq_laneq_f32(vC2, vB2, vA2, 2);
                    vC2 = vfmaq_laneq_f32(vC2, vB3, vA2, 3);
                    vst1q_f32(pC + cIdx + c.w * 2, vC2);
                    vC3 = vld1q_f32(pC + cIdx + c.w * 3);
                    vC3 = vfmaq_laneq_f32(vC3, vB0, vA3, 0);
                    vC3 = vfmaq_laneq_f32(vC3, vB1, vA3, 1);
                    vC3 = vfmaq_laneq_f32(vC3, vB2, vA3, 2);
                    vC3 = vfmaq_laneq_f32(vC3, vB3, vA3, 3);
                    vst1q_f32(pC + cIdx + c.w * 3, vC3);
                }
            }
        }
#endif
    }
}

/**
 * matrix multiplication
 * i for c height, j for c width, k for a width
 * for loop kij
 * simd kj
 * align 4 kij
 *
 * @param a The first input matrix
 * @param b The second input matrix
 * @param c The output matrix to store the result
 *
 * @return void
 *
 * @throws None
 */
void GeMM::Optimize16(Matrix &a, Matrix &b, Matrix &c)
{
    if (!CheckParam(a, b, c)) {
        return;
    }
    float *pA = a.data;
    float *pB = b.data;
    float *pC = c.data;
    {
        TIMEPERF(Optimize16);
#ifdef __ARM_NEON
        int aIdx;
        int bIdx;
        int cIdx;
        float32x4_t vA0;
        float32x4_t vA1;
        float32x4_t vA2;
        float32x4_t vA3;
        float32x4_t vB0;
        float32x4_t vB1;
        float32x4_t vB2;
        float32x4_t vB3;
        float32x4_t vC0;
        float32x4_t vC1;
        float32x4_t vC2;
        float32x4_t vC3;
        int aHAlign = a.h & ~3;
        int aWAlign = a.w & ~3;
        int bWAlign = b.w & ~3;
        for (int k = 0; k < aWAlign; k += 4) {
            int bIdxBase = k * b.w;
            for (int i = 0; i < aHAlign; i += 4) {
                int aIdxBase = i * a.w;
                int cIdxBase = i * c.w;
                aIdx = aIdxBase + k;
                vA0 = vld1q_f32(pA + aIdx);
                vA1 = vld1q_f32(pA + aIdx + a.w);
                vA2 = vld1q_f32(pA + aIdx + a.w * 2);
                vA3 = vld1q_f32(pA + aIdx + a.w * 3);
                for (int j = 0; j < bWAlign; j += 4) {
                    bIdx = bIdxBase + j;
                    vB0 = vld1q_f32(pB + bIdx);
                    vB1 = vld1q_f32(pB + bIdx + b.w);
                    vB2 = vld1q_f32(pB + bIdx + b.w * 2);
                    vB3 = vld1q_f32(pB + bIdx + b.w * 3);
                    cIdx = cIdxBase + j;
                    vC0 = vld1q_f32(pC + cIdx);
                    vC0 = vfmaq_laneq_f32(vC0, vB0, vA0, 0);
                    vC0 = vfmaq_laneq_f32(vC0, vB1, vA0, 1);
                    vC0 = vfmaq_laneq_f32(vC0, vB2, vA0, 2);
                    vC0 = vfmaq_laneq_f32(vC0, vB3, vA0, 3);
                    vst1q_f32(pC + cIdx, vC0);
                    vC1 = vld1q_f32(pC + cIdx + c.w);
                    vC1 = vfmaq_laneq_f32(vC1, vB0, vA1, 0);
                    vC1 = vfmaq_laneq_f32(vC1, vB1, vA1, 1);
                    vC1 = vfmaq_laneq_f32(vC1, vB2, vA1, 2);
                    vC1 = vfmaq_laneq_f32(vC1, vB3, vA1, 3);
                    vst1q_f32(pC + cIdx + c.w, vC1);
                    vC2 = vld1q_f32(pC + cIdx + c.w * 2);
                    vC2 = vfmaq_laneq_f32(vC2, vB0, vA2, 0);
                    vC2 = vfmaq_laneq_f32(vC2, vB1, vA2, 1);
                    vC2 = vfmaq_laneq_f32(vC2, vB2, vA2, 2);
                    vC2 = vfmaq_laneq_f32(vC2, vB3, vA2, 3);
                    vst1q_f32(pC + cIdx + c.w * 2, vC2);
                    vC3 = vld1q_f32(pC + cIdx + c.w * 3);
                    vC3 = vfmaq_laneq_f32(vC3, vB0, vA3, 0);
                    vC3 = vfmaq_laneq_f32(vC3, vB1, vA3, 1);
                    vC3 = vfmaq_laneq_f32(vC3, vB2, vA3, 2);
                    vC3 = vfmaq_laneq_f32(vC3, vB3, vA3, 3);
                    vst1q_f32(pC + cIdx + c.w * 3, vC3);
                }
            }
        }
#endif
    }
}

bool GeMM::CheckParam(Matrix &a, Matrix &b, Matrix &c)
{
    if (a.w != b.h) {
        LOGE("Matrix A's width(%d) is not equal to Matrix B's height(%d)", a.w, b.h);
        return false;
    }
    if (a.h != c.h) {
        LOGE("Matrix C's height(%d) is not equal to Matrix A's height(%d)", c.h, a.h);
        return false;
    }
    if (b.w != c.w) {
        LOGE("Matrix C's width(%d) is not equal to Matrix B's width(%d)", c.w, b.w);
        return false;
    }
    if (!a.data || !b.data || !c.data) {
        LOGE("Matrix A(%p), B(%p), C(%p) is null", a.data, b.data, c.data);
        return false;
    }
    return true;
}
