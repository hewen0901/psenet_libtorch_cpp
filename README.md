## `psenet和crnn网络pytorch模型的c++调用测试代码`
### 功能介绍
- 字符检测网络psenet和字符识别网络crnn的pytorch模型的c++测试代码
### 环境依赖
- pytorch 1.3.1
- opencv3.X
### 编译说明
```bash
> mkdir build
> cd build
> cmake ..
> make
```
### 代码参考
- https://zhuanlan.zhihu.com/p/91019893?from_voters_page=true 借鉴了作者发出来的后处理代码
- https://blog.csdn.net/xiamentingtao/article/details/98673967 借鉴了pytorch模型转pt文件的python代码以及将tensor转换为mat类型的c++代码
