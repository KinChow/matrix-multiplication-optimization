/*
 * @Author: Zhou Zijian 
 * @Date: 2024-02-21 01:45:49 
 * @Last Modified by: Zhou Zijian
 * @Last Modified time: 2024-02-27 00:44:05
 */

#ifndef GEMMH
#define GEMMH

#include <vector>

class Matrix {
public:
    /**
     * @brief Construct a new Matrix object
     * 
     * @param data The data for the matrix
     * @param h The height of the matrix
     * @param w The width of the matrix
     */
    Matrix(std::vector<float> &data, int h, int w) : data(data.data()), h(h), w(w) {}

public:
    float *data; /**< Pointer to the data for the matrix */
    int h;       /**< The height of the matrix */
    int w;       /**< The width of the matrix */
};

class GeMM {
public:
    static bool CheckResult(Matrix &a, Matrix &b);
    static void Origin(Matrix &a, Matrix &b, Matrix &c);
    static void Optimize1(Matrix &a, Matrix &b, Matrix &c);
    static void Optimize2(Matrix &a, Matrix &b, Matrix &c);
    static void Optimize3(Matrix &a, Matrix &b, Matrix &c);
    static void Optimize4(Matrix &a, Matrix &b, Matrix &c);
    static void Optimize5(Matrix &a, Matrix &b, Matrix &c);
    static void Optimize6(Matrix &a, Matrix &b, Matrix &c);
    static void Optimize7(Matrix &a, Matrix &b, Matrix &c);
    static void Optimize8(Matrix &a, Matrix &b, Matrix &c);
    static void Optimize9(Matrix &a, Matrix &b, Matrix &c);
    static void Optimize10(Matrix &a, Matrix &b, Matrix &c);
    static void Optimize11(Matrix &a, Matrix &b, Matrix &c);
    static void Optimize12(Matrix &a, Matrix &b, Matrix &c);
    static void Optimize13(Matrix &a, Matrix &b, Matrix &c);
    static void Optimize14(Matrix &a, Matrix &b, Matrix &c);
    static void Optimize15(Matrix &a, Matrix &b, Matrix &c);
    static void Optimize16(Matrix &a, Matrix &b, Matrix &c);

private:
    static bool CheckParam(Matrix &a, Matrix &b, Matrix &c);
};

#endif  // GEMMH