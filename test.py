# # # import numpy as np

# # # # 假设 x 的形状为 (b, n, c)
# # # x = np.random.rand(2, 3, 6)  # 示例数据
# # # b,n,c=x.shape
# # # # 假设 nm 和 conf_threshold 的值
# # # nm = 1
# # # conf_threshold = 0.5

# # # # 计算条件
# # # max_values = np.amax(x[..., 4:-nm], axis=-1)  # 形状为 (b, n)
# # # condition = max_values > conf_threshold  # 形状为 (b, n)

# # # # 扩展布尔数组的形状
# # # condition_expanded = np.expand_dims(condition, axis=-1)  # 形状为 (b, n, 1)
# # # condition_expanded = np.repeat(condition_expanded, c, axis=-1)  # 形状为 (b, n, c)

# # # # 应用布尔索引
# # # x_filtered = x[condition_expanded]

# # # print(x_filtered.shape)
# # # import cv2
# # # # image=cv2.imread("/root/autodl-tmp/sahi/build/simple_test_result.jpeg")
# # # image=cv2.imread("/root/autodl-tmp/sahi/build/the mask.png",0)
# # # image2=0
# # # print("i am here")


# # def estimate_tile_size(h, w):
# #         # 找出大图的窄边
# #         narrow_side = min(h, w)
# #         # 初始化最大得分和对应的边长
# #         max_score = -1
# #         best_size = 0
# #         # 从 narrow_side // 4 开始，以 32 为步长进行循环
# #         start_size = max(4, narrow_side // 4-((narrow_side // 32)%32))
# #         for size in range(start_size, narrow_side + 1, 32):
# #             for num_parts in range(2, 5):
# #                 if size * num_parts > narrow_side:
# #                     # 新的得分计算方式，考虑份数与 3 的接近程度
# #                     score = 100 - abs(num_parts - 3) * 10 - abs(size * num_parts - narrow_side)
# #                     if score > max_score:
# #                         max_score = score
# #                         best_size = size
# #         return best_size, best_size
# # h = 3072
# # w = 4096
# # tile_h, tile_w = estimate_tile_size(h, w)
# # print(f"小图的高: {tile_h}, 小图的宽: {tile_w}")


def estimate_tile_size(H, W):
    """
    自动估计大图切成小图的合适边长。
    
    :param H: 大图高度
    :param W: 大图宽度
    :return: 计算得到的合适小图边长
    """
    min_side = min(H, W)  # 找到较短的一边
    max_side = max(H, W)

    # 选择合理的份数（2~4）
    for n in range(4, 1, -1):  # 优先尝试 4 份，如果不合适就尝试 3 或 2 份
        tile_size = min_side / n
        # 让 tile_size 变成 4 的倍数
        tile_size = round(tile_size / 4) * 4  # 取最近的4的倍数
        if tile_size > 0:
            break

    # 确保 tile_size 不会过大（不超过 max_side 的 1/2）
    tile_size = min(tile_size, max_side // 2)

    return tile_size

# 示例：
H, W = 3072, 4096  # 输入大图尺寸
tile_size = estimate_tile_size(H, W)
print(f"推荐的小图边长: {tile_size}")
# import cv2
# import numpy as np
# from shapely.geometry import Polygon
# from shapely.validation import explain_validity

# def visualize_polygon(point_list):
#     # 创建多边形对象
#     polygon = Polygon(point_list)
#     is_valid = polygon.is_valid
#     reason = explain_validity(polygon)

#     # 归一化处理
#     min_x = min([point[0] for point in point_list])
#     max_x = max([point[0] for point in point_list])
#     min_y = min([point[1] for point in point_list])
#     max_y = max([point[1] for point in point_list])

#     width = max_x - min_x
#     height = max_y - min_y

#     normalized_points = []
#     for x, y in point_list:
#         norm_x = (x - min_x) / width if width != 0 else 0
#         norm_y = (y - min_y) / height if height != 0 else 0
#         normalized_points.append((norm_x, norm_y))

#     # 新的图像大小
#     new_width = 500
#     new_height = 500

#     scaled_points = []
#     for norm_x, norm_y in normalized_points:
#         x = int(norm_x * new_width)
#         y = int(norm_y * new_height)
#         scaled_points.append((x, y))

#     image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

#     points = np.array(scaled_points, np.int32)
#     points = points.reshape((-1, 1, 2))

#     # 计算合适的点半径
#     num_points = len(point_list)
#     if num_points < 10:
#         point_radius = 5
#     elif num_points < 50:
#         point_radius = 3
#     else:
#         point_radius = 1

#     # 绘制多边形的点
#     for point in scaled_points:
#         cv2.circle(image, point, point_radius, (0, 0, 255), -1)

#     # 绘制多边形的边
#     cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

#     # 添加文本标注多边形是否有效
#     text = f"Is Valid: {is_valid}"
#     if not is_valid:
#         text += f", Reason: {reason}"
#     cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#     return image







# # 定义点列表
# # point_list = [(390, 584), (390, 601), (399, 601), (400, 600), (400, 590), (401, 589), (401, 587), (404, 584), (405, 584)]
# point_list=[(430, 586), (429, 587), (426, 587), (426, 588), (428, 590), (428, 600), (425, 603), (415, 603), (414, 602), (409, 602), (408, 603), (407, 603), (407, 604), (431, 604), (432, 605), (435, 605), (436, 606), (437, 606), (437, 605), (438, 604), (438, 594), (439, 593), (439, 587), (436, 587), (435, 586)]
# # 创建多边形对象
# polygon = Polygon(point_list)

# # 判断多边形是否为简单多边形
# is_simple = polygon.is_simple
# print(f"该多边形是否为简单多边形: {is_simple}")

# # 确定图像的大小
# max_x = max([point[0] for point in point_list])
# max_y = max([point[1] for point in point_list])
# image = np.zeros((max_y + 10, max_x + 10, 3), dtype=np.uint8)

# # 将点列表转换为适合 OpenCV 的格式
# points = np.array(point_list, np.int32)
# points = points.reshape((-1, 1, 2))

# # 在图像上绘制多边形
# cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

# # 保存图像
# cv2.imwrite('polygon_image.png', image)
# reason = explain_validity(polygon)
# print(f"多边形无效的原因: {reason}")

# vis_poly_img=visualize_polygon(point_list)
# cv2.imwrite("part1.png",vis_poly_img)


# 显示图像（可选）
# cv2.imshow('Polygon Image', image)


# cv2.waitKey(0)
# cv2.destroyAllWindows()