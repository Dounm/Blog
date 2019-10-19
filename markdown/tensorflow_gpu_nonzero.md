# Tensorflow代码详解（一）：如何在GPU上实现Nonzero

[TOC]

## 1. 背景

[numpy.nonzero(a)](<https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html> )，行为是返回输入a中非零元素的坐标所构成的array。



在深度学习网络中，`nonzero`操作常出现在RCNN系列的网络中。

而在主流深度学习框架中，PyTorch的`torch.nonzero()`和numpy的接口基本一致，其所返回的tensor由输入tensor中所有非零元素的坐标。

但对于TensorFlow而言，并没有`tf.non_zero`，有的只是`tf.where`，其参数为`tf.where(condition, x=None, y=None, name=None)`。

当仅传入一个参数`condition`时，`tf.where()`返回的就是`condition`中true elements的坐标。



## 2. TF的Python端where的相关代码

`tf.where`对应的Python段代码位于 *tensorflow/python/ops/array_ops.py*。

其主要逻辑代码如下：

``` python
  if x is None and y is None:
    return gen_array_ops.where(condition=condition, name=name)
  elif x is not None and y is not None:
    return gen_math_ops.select(condition=condition, x=x, y=y, name=name)
  else:
    raise ValueError()
```

可以看到，当仅传入`condition`时，最终调用的是`gen_array_ops.where()`。

`gen_xxx_ops`是TensorFlow使用的Swig库对C++后端暴露的Op的封装接口。具体到`gen_array_ops.where()`，其实际上对应的是C++端的array_ops里的`Where`。



## 3. TF的C++端where的相关代码

### 3.1 WhereOp的注册

C++端的`Where`的Op定义于 *tensorflow/core/ops/array_ops.cc*，其声明代码如下

```c++
REGISTER_OP("Where")
    .Input("input: T")
    .Attr("T: {numbertype, bool} = DT_BOOL")
    .Output("index: int64")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Matrix(c->UnknownDim(), c->Rank(c->input(0))));
      return Status::OK();
    });
```

这段代码的意思是：

注册了一个名为`Where`的Op，其输入名为`input`，是`numbertype`（数值类型）或`bool`类型的tensor；输出名为`index`，是`int64`类型的tensor。

`WhereOp`的output tensor shape的推导函数为最后一个lambda函数，大致含义是：

第0个output（即`index` tensor）的shape的第0维未知（因为是动态的，kerne执行时才知道），第1维是第0个input的rank值。

即若`input` tensor是3维的，那么`index` tensor的第1维就是数字`3`，用来存储表示`input` tensor的3维坐标。



### 3.2 WhereOpKernel的实现

C++端的`WhereOpKernel`定义于 *tensorflow/core/kernels/where_op.cc*，其主要包括如下两个类

``` c++
template <typename T>
class WhereCPUOp : public OpKernel {
    void Compute(OpKernelContext* context) override {
        ...
    }
};

template <typename T>
class WhereGPUOp : public AsyncOpKernel {
 public:
    void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
        ...
    }
};
```

顾名思义，`WhereCPUOp`是CPU上的kernel实现，`WhereGPUOp`是GPU上的kernel实现。

`WhereGPUKernel`继承自`AsyncOpKernel`，而`AsyncOpKernel`则继承自`OpKernel`，其相比于基类`OpKernel`来说，只是`Compute()`变成了`ComputeAsync()`,多了一个`DoneCallback done`。



#### 3.2.1 WhereGPUOp的相关代码

具体来说，我们详细分析下`WhereGPUOp::ComputeAsync()`函数。

整个函数的步骤是

1. 在gpu上分配nnz(number of nonzero)
2. 调用`functor::NumTrue<GPUDevice, T, Tindex>::Compute(...)`计算nnz的值
3. 把nnz的值从gpu拷贝到host cpu上
4. 基于host上nnz的值分配output tensor的内存（作为shape的第0维）
5. 调用`functor::Where<GPUDevice, NDIM, T, Tindex>::Compute(...)`计算output tensor



需要注意的是，`functor::Where`相比于`functor::NumTrue`多了一个模板参数`NDIM`，其对应的核心代码如下

``` c++
#define HANDLE_DIM(NDIM)                                                \
  case NDIM: {                                                          \
    Status s = functor::Where<GPUDevice, NDIM, T, Tindex>::Compute(...);\
  } break;

      switch (input_dims) {
        HANDLE_DIM(1);
        HANDLE_DIM(2);
        HANDLE_DIM(3);
        HANDLE_DIM(4);
        HANDLE_DIM(5);
        HANDLE_DIM(6);
        HANDLE_DIM(7);
        HANDLE_DIM(8);
        default: ...
      }
#undef HANDLE_DIM
```

即根据input tensor的shape的维度不同，来去偏特化模板类`functor::Where`。

之所以要把`NDIM`作为模板参数，而非普通的函数参数，是为了CUDA Kernel的优化，这个下文会详细解释。



#### 3.2.2 functor的代码路径

`functor`是 *tensorflow/core/kernels/where_op_gpu.cu.h* 中的namespace，其中主要包含`NumTrue`和`Where`两个struct用于分别计算num_nonzero和output tensor。

而 *where_op_gpu.cu.h* 作为头文件，仅被 *where_op_gpu_impl_1.cu.cc* 到 *where_op_gpu_impl_8.cu.cc* 共8个.cu源文件所include。其与 *where_op.cc* 之间仅是在编译的linking阶段通过符号表的解析来匹配上。



#### 3.2.3 functor::NumTrue

对于`struct NumTrue<T, TIndex>`，其只有一个函数`Compute(...)`，该函数的输入是`TTypes<T>::ConstFlat input`，即`T`类型的const且flat的tensor；其输出是`TTypes<TIndex>::Scalar num_true`，即`TIndex`类型的scalar的tensor。

函数体内部主要逻辑如下

``` c++
auto reducer = CubDeviceReduceCount<T, TIndex>();
auto first_success = reducer(/*temp_storage*/ nullptr, 
							temp_storage_bytes, input_data, num_true_data, input.size(), cu_stream);

... // 基于第一次调用reducer的结果，去分配temp_storage tensor的内存

auto second_success = reducer(/*temp_storage*/ temp_storage.flat<int8>().data(), temp_storage_bytes, 
							temp_storage_bytes, input_data, num_true_data, input.size(), cu_stream);
```

可以看到，`NumTrue<T, TIndex>::Compute(...)`的主要逻辑就是调用了两次`CubDeviceReduceCount`的对象`reducer`。

`CubDeviceReduceCount`内部使用了`cub::DeviceReduce::Sum()`，所以按照使用`cub`库的通用习惯，需要分两次调用：

1. 第一次调用传入的temp_storage是`nullptr`，目的是计算出`cub::DeviceReduce::Sum()`需要的temp_storage的大小
2. 第二次调用传入第一次时计算出的指定大小的temp_storage，计算出真正的结果



再来细看下`CubDeviceReduceCount<T, TIndex>`的实现

``` c++
template <typename T>
struct IsNonzero {
  EIGEN_DEVICE_FUNC IsNonzero() : zero(T(0)) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const T& x) const {
    return (x != zero);
  }
  const T zero;
};

template <typename T, typename TIndex>
struct CubDeviceReduceCount {
  gpuError_t operator()(void* d_temp_storage, size_t& temp_storage_bytes,
                        const T* d_in, TIndex* d_out, int num_items,
                        gpuStream_t stream = 0,
                        bool debug_synchronous = false) {
    IsNonzero<T> is_nonzero;
    // gpuprim is alias of cub
    gpuprim::TransformInputIterator<bool, IsNonzero<T>, const T*>
        is_nonzero_iter(d_in, is_nonzero);
    return gpuprim::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                                      is_nonzero_iter, d_out, num_items, stream,
                                      debug_synchronous);
  }
};

template <typename TIndex>
struct CubDeviceReduceCount<bool, TIndex> {
  gpuError_t operator()(void* d_temp_storage, size_t& temp_storage_bytes,
                        const bool* d_in, TIndex* d_out, int num_items,
                        gpuStream_t stream = 0,
                        bool debug_synchronous = false) {
    return gpuprim::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in,
                                      d_out, num_items, stream,
                                      debug_synchronous);
  }
};
```

可以看到，当`T`不是`bool`时，需要调用`cub::TransformInputIterator`构造出一个`is_nonzer_iter`来替代`Const T*`类型的raw ptr `d_in`；而当`T`是`bool`时，直接传入`d_in`。

这是因为`cub::DeviceReduce::Sum`需要作为input的iterator返回`bool`值，所以对`non-bool T`来说，就必须将其变换成`bool`才能正常使用。

常见的作法是分配一块与`d_in`同size但类型为`bool`的内存，然后提前把`d_in`的所有元素都cast为`bool`。但此处我们可以借助`cub::TransformInputIterator`来传入一个`IsNonzero`的functor，来让`cub::DeviceReduce::Sum()`在每次迭代raw ptr时都自行调用`IsNonzero()`来将其cast为`bool`，这样相比之前的做法来说就省去了一块额外的内存。



#### 3.2.4 functor::Where

`functor::Where<NDIM,T, TIndex>::Compute(...)`的具体实现与`functor::NumTrue<T,  TIndex>::Compute(...)`较为相似，主要逻辑如下

``` C++
TIndex* found_true_device = found_true_t.scalar<TIndex>().data();
WhereOutputIterator<NDIM> output_iterator(output.data(), /* max_row */ output.dimension(0));

CubDeviceSelectFlaggedCounter<T, TIndex, decltype(output_iterator) /*OutputIterator*/,
        std::is_convertible<DT, bool>::value /*IsConvertibleToBool*/> counter;
auto first_success = counter(/*temp_storage*/ nullptr, temp_storage_bytes, 
                             /*d_flags*/ input.data(),
                             /*d_out*/ output_iterator,
                             /*d_num_selected_out*/ found_true_device, input.size(), cu_stream);
... // 基于第一次调用counter的结果，去分配temp_storage tensor的内存
auto second_success = counter(/*temp_storage*/ nullptr, temp_storage_bytes, 
                             /*d_flags*/ input.data(),
                             /*d_out*/ output_iterator,
                             /*d_num_selected_out*/ found_true_device, input.size(), cu_stream);

TF_CHECK_OK(GpuLaunchKernel(PropagateWhereIndicesKernel<NDIM, TIndex>, /*other paras*/...);
```

`CubDeviceSelectFlaggedCounter`内部调用的是`cub::DeviceSelect::Flagged()`，作用是从input iterator中选取对应的flags为`true`的元素，并输出到output iterator中。

在`CubDeviceSelectFlaggedCounter`的实现中，input iterator是从0开始自增、每步加1直到`input.size()`的`cub::CountingInputIterator`，flags则是通过`cub::TransformInputIterator`转换为`bool`的input tensor。

因为输入是从0自增、每步加1的iterator，所以`cub::DeviceSelect::Flagged()`输入相当于把input tensor打平（flatten）后的index，其根据flags所选出的输出自然也是flatten index。

举例而言，input tensor中某个nonzero元素的坐标是二维的`(2,3)`，但在`CoutingInputIterator`中，其对应的坐标是flatten后一维的`(2*input_shape.At(0) + 3)`，因此最后输出到`output_iterator`中对应的值也是`2*input_shape.At(0) + 3`。

但我们真正需要的是`(2,3)`，所以最后需要一步`PropagateWhereIndicesKernel<NDIM, TIndex>`来把flatten index还原为真正的index。

正常来说，我们需要一块额外的内存来存储flatten output（`shape={num_nonzero, 1}`），然后把flatten output转变为真正的output tensor（`shape={num_nonzero, NDIM}`）。

但实际上，我们可以把flatten output存放在output tensor中，只不过对于output iterator来说，每一步就不是`iter + 1`，而是`iter + NDIM`了，而`WhereOutputIterator`类正是封装了每步`+ NDIM`的iterator。



具体到`PropagateWhereIndicesKernel<NDIM, TIndex>`的实现来说，

``` C++
template <int NDIM, typename TIndex>
__global__ void PropagateWhereIndicesKernel(
    const TIndex output_rows, const typename Eigen::array<TIndex, NDIM> strides,               
    int64* __restrict__ output) {         
  GPU_1D_KERNEL_LOOP(i, output_rows) {                                                         
    TIndex index_value = ldg(output + NDIM * i);                                               
#pragma unroll                                                                                 
    for (int c = 0; c < NDIM; ++c) {
      *(output + NDIM * i + c) = index_value / strides[c];                                     
      index_value %= strides[c];
    } 
  }   
}
```

其有如下两个值得注意的优化

- 使用`ldg()`访问只读的gpu mem，相比于普通的访问加快了速度
- `#pragma unroll`展开了`NDIM`的循环，消除了循环，加快了执行速度。



## 4. 设计总结及改进

### 4.2 NDIM作为模板参数的必要性

NDIM之所以作为模板参数，而非普通的函数参数，我觉着主要目的是考虑`PropagateWhereIndicesKernel()`的方便程度和性能。

首先就是，`PropagateWhereIndicesKernel()`的其中一个参数是`strides`，该参数是一个数组，数组元素的个数等于`NDIM - 1`。所以如果`NDIM`不是模板参数，那么如何处理strides就是个麻烦。

其次就是前面提到的，把`NDIM`变成模板参数，可以支持`#pragma unroll`的展开，从而加快CUDA kernel的执行速度。（具体原理见[CUDA程序调优指南（二）：性能调优](<https://zhuanlan.zhihu.com/p/84510732> )-3.5）

### 4.2 NumTrue的必要性

`functor::NumTrue`中使用`cub::DeviceReduce::Sum()`来获得`num_nonzero`的值，进而根据此值构造output tensor。

但实际上`functor::Where`中使用的`cub::DeviceSelect::Flagged()`的输出除了被选中的indices之外，还有被选中的indices的个数（即代码中的`/*d_num_selected_out*/ found_true_device`）。

从某种角度来说，似乎没必要使用`cub::DeviceReduce::Sum()`。但因为需要先确定output tensor的形状，才能计算出output tensor的内容，所以TF只能这么做，导致浪费一些计算。

但假如我们换种思路，先分配一个最大shape的output tensor（`shape={input_shape.elem_cnt, input_shape.dims}`）出来，那么后续再根据输出的`d_num_selected_out`对output tensor进行裁剪，也未必不是一种实现思路。

### 4.3 可改进的地方

- 当`NDIM==1`时，没必要调用`PropagateWhereIndicesKernel()`了。

