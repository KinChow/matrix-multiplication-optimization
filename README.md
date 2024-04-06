# MatrixMultiplicationOptimization

#### 介绍
矩阵乘优化



测试用例

* 循环交换
  * ijk
  * ikj
  * kij
* unroll
  * 1x4
    * ikj
    * kij
  * 4x4
    * ikj
    * kij
* simd
  * 1x4
    * ikj
    * kij
  * 4x4
    * ikj
    * kij
  * 4x4 (align 4)
    * ikj
    * kij
* unroll + simd
  * 4x4 (align 4)
    * ikj
    * kji




#### 安装教程

```shell
python build.py
python build.py --clean
```

运行build.py安装程序，选项如下：

* clean：清空构建目录和安装目录



#### 使用说明

```shell
# 通过simpleperf查看测试用例性能指标
python run.py --size=16 --debug

# 验证测试用例正确性
python run.py --size=16 --check
```





#### 参考

gemm

* https://zhenhuaw.me/blog/2019/gemm-optimization.html
* https://developer.arm.com/documentation/102467/0201/Example---matrix-multiplication
* https://github.com/flame/how-to-optimize-gemm



perf

* https://zhuanlan.zhihu.com/p/445260558
* http://zhengheng.me/2015/11/12/perf-stat/

