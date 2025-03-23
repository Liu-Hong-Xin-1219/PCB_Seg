
本文档介绍了大图slice为若干块，对每一块进行分割，最后将分割结果融合的过程和相关代码
本过程中最难以转为CPP代码的部分是分割结果融合.
# 1.整体流程
在`segapi.py`的 `class SEGAPI` 当中的 `def infer`完成了 a)大图slice, b)小图元件分割, c)分割结果融合的pipeline.以下为各个步骤的相关代码. 需要您方着重帮忙的是c)部分. 

```python
# a)大图slice为小图
slice_image_result = slice_image(
            image=image,
            output_file_name=None,
            output_dir=None,
            slice_height=None,
            slice_width=None,
            overlap_height_ratio=None,
            overlap_width_ratio=None,
            auto_slice_resolution=True,
        )
```


```python
# b)小图元件分割
# 首先对slice出来的每一张小图(一般为1024*1024)进行元件分割,然后对整张大图(一般为3072*4096)进行分割. 小图分割是为了有更高的精度,并且能处理比较小的元件; 大图分割是为了跨小图元件的掩码融合做准备. 
for group_ind in range(num_group):
    #...
     prediction_result = self.get_prediction(
                images=image_list,
                shift_amount=shift_amount_list,
                full_shape=[
                    slice_image_result.original_image_height,
                    slice_image_result.original_image_width,
                ],
            )
    #...
# perform standard prediction
if num_slices > 1 and perform_standard_pred:
    prediction_result = self.get_prediction(
        images=[image],
        shift_amount=[0, 0],
        full_shape=[
            slice_image_result.original_image_height,
            slice_image_result.original_image_width,
        ],
        postprocess=None,
    )
    object_prediction_list.extend(prediction_result[0].object_prediction_list)
```


```python
# c)分割结果融合初始化
postprocess_constructor = POSTPROCESS_NAME_TO_CLASS[postprocess_type]
        postprocess = postprocess_constructor(
            match_threshold=postprocess_match_threshold,
            match_metric=postprocess_match_metric,
            class_agnostic=postprocess_class_agnostic,
        )

        #.....
# merge matching predictions 融合分割结果, 这个分割结果包括了各个小图的分割结果和大图的分割结果. 
        if len(object_prediction_list) > 1:
            object_prediction_list = postprocess(object_prediction_list)
```
接下来我们直接分析c)部分的代码

# 2. 后处理, 融合分割结果.
以下部分是python转cpp的难点. 
上述代码中postprocess(object_prediction_list)
直接调用了 `combine.py`中 `class GreedyNMMPostprocess(PostprocessPredictions)`的 `def __call__`方法, 下面将详细分析此方法的步骤:

### 2.1 序列化
通过下面的代码,将分割的单个物体的数据形式从python class改为torch.tensor, 从而方便下一步操作
```python
object_prediction_list = ObjectPredictionList(object_predictions)
```

### 2.2 判断是否需要融合
通过下面的代码, 在**同类**的物体之间进行判断, 判断是否需要融合从而保证跨小图的物体不会被认为是两个不同元件而是同一个元件.
```python
keep_to_merge_list = batched_greedy_nmm(...)
```

### 2.3 融合
通过下面的代码,根据2.2步骤的结果,进行掩码等等信息的融合. 
```python
object_prediction_list[keep_ind] = merge_object_prediction_pair(
                        object_prediction_list[keep_ind].tolist(), object_prediction_list[merge_ind].tolist()
                    )
```

元件的融合`utils.py`的`def merge_object_prediction_pair()`实现, 代码如下:
```python
#其他信息的融合
merged_bbox: BoundingBox = get_merged_bbox(pred1, pred2)
merged_score: float = get_merged_score(pred1, pred2)
merged_category: Category = get_merged_category(pred1, pred2)

# mask的融合
if pred1.mask and pred2.mask:
    merged_mask: Mask = get_merged_mask(pred1, pred2)
```

`get_merged_mask()`实现了mask的融合