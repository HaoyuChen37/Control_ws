import numpy as np
import cv2

# 假设的相机内参矩阵和畸变系数
cameraMatrix = np.array([[436.04379372 ,  0.    ,     322.2056478 ],
 [  0.    ,     408.81252158 , 231.98256365],
 [  0.    ,       0.         ,  1.        ]])
distCoeffs = np.array( [[-0.09917725 , 0.1034774  , 0.00054878,  0.0001342 , -0.01694831]])

# 假设的外参矩阵
# 相机相对于世界坐标系的齐次变换矩阵
extrinsic_matrix = np.array([
    [0.72119598, 0.10725118, 0.68437822, 0.04008849],
    [-0.69155041, 0.05381167, 0.720321, 0.05381581],
    [0.04042774, -0.99277464, 0.11297836, -0.10454239],
    [0.0, 0.0, 0.0, 1.0]
])

# 已知的3D点（世界坐标系）
object_points = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0]
], dtype=np.float32)

# 已知的2D点（图像坐标系）
image_points = np.array([
    [100, 150],
    [200, 150],
    [100, 250],
    [200, 250]
], dtype=np.float32)

# 提取外参矩阵中的旋转矩阵和平移向量
rvec, _ = cv2.Rodrigues(extrinsic_matrix[:3, :3])  # 旋转矩阵转为旋转向量
tvec = extrinsic_matrix[:3, 3]

# 使用投影函数将3D点投影到图像平面
projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, cameraMatrix, distCoeffs)

# 将投影点的形状调整为与图像点一致
projected_points = projected_points.reshape(-1, 2)

# 计算投影点与实际图像点之间的误差
errors = np.linalg.norm(projected_points - image_points, axis=1)
mean_error = np.mean(errors)
std_error = np.std(errors)

print(f"Mean reprojection error: {mean_error:.2f} pixels")
print(f"Standard deviation of reprojection error: {std_error:.2f} pixels")

# 可视化结果
import matplotlib.pyplot as plt

# 绘制图像点
plt.scatter(image_points[:, 0], image_points[:, 1], color='blue', label='Image Points')

# 绘制投影点
plt.scatter(projected_points[:, 0], projected_points[:, 1], color='red', label='Projected Points')

# 添加连线
for i in range(len(image_points)):
    plt.plot([image_points[i, 0], projected_points[i, 0]], [image_points[i, 1], projected_points[i, 1]], color='gray', linestyle='--')

plt.legend()
plt.gca().invert_yaxis()  # 翻转Y轴，符合图像坐标系
plt.title("Projection Validation")
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.show()