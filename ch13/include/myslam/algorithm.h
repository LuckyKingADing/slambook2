//
// Created by gaoxiang on 19-5-4.
//

#ifndef MYSLAM_ALGORITHM_H
#define MYSLAM_ALGORITHM_H

// algorithms used in myslam
#include "myslam/common_include.h"

namespace myslam {

/**
 * linear triangulation with SVD
 * @param poses     poses,
 * @param points    points in normalized plane
 * @param pt_world  triangulated point in the world
 * @return true if success
 */
inline bool triangulation(const std::vector<SE3> &poses,
                   const std::vector<Vec3> points, Vec3 &pt_world) {
    // 1. 构造线性方程
    // poses是包含左右相机位姿的std::vector向量，poses.size()为2
    MatXX A(2 * poses.size(), 4); // 设poses.size()为N，则A矩阵的大小为2N行4列
    VecX b(2 * poses.size());     // b是一个2N维的向量，初始化为零
    b.setZero();

    // 2. 填充A矩阵
        // 遍历每个相机的位姿，构造线性方程 A⋅X=b。
    for (size_t i = 0; i < poses.size(); ++i) {
        Mat34 m = poses[i].matrix3x4(); // 将SE3类型的位姿(包含旋转和平移)转换为3x4的矩阵形式，包含旋转R和平移t信息
        /* points[i][0] 是点的 x 坐标
        *  points[i][1] 是点的 y 坐标
        *  m.row(2) 是位姿矩阵的第三行
        *  m.row(0) 和 m.row(1) 分别是第一行和第二行。
        *  A.block<1, 4>(2 * i, 0) 表示将矩阵 A 的第 2*i 行的前 4 列赋值。 
        *  A.block<1, 4>(2 * i + 1, 0) 表示将矩阵 A 的第 2*i+1 行的前 4 列赋值。 
        *  有左右两个相机，因此应该有四行线性方程，分别对应于左相机和右相机的投影关系。对于每个相机，构造两行方程来描述点在该相机坐标系下的投影关系。
           */
        A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0);
        A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1);
    }

    // 3. SVD分解求解线性方程：SVD（奇异值分解）是一种矩阵分解方法，可以将一个矩阵分解为三个矩阵的乘积：A = U * Σ * V^T，其中 U 和 V 是正交矩阵，Σ 是一个对角矩阵，包含了 A 的奇异值。通过 SVD 分解，我们可以得到 A 的奇异值和对应的奇异向量，从而求解线性方程 A⋅X=b 的最小二乘解。具体来说，SVD 分解可以帮助我们找到 A 的零空间（null space），从而得到一个非零解 X，使得 A⋅X 接近于零，即满足线性方程的近似解。
        // 在这里，我们使用 Eigen 库中的 bdcSvd 函数来进行 SVD 分解，并指定 ComputeThinU 和 ComputeThinV 选项来计算 U 和 V 矩阵的薄版本。通过 svd.matrixV().col(3) 可以获取 V 矩阵的第 4 列（索引从 0 开始），该列对应于 A 的零空间中的一个向量。将该向量除以其最后一个元素（svd.matrixV()(3, 3)）可以得到归一化的解，然后取前 3 个元素作为最终的三维点坐标 pt_world。
    auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);

    pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();

    // 4. 判断解的质量：通过比较奇异值来判断解的质量。如果第四个奇异值与第三个奇异值的比值小于一个阈值（如1e-2），则认为解的质量好，返回true
    if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2) {
       
        return true;
    }
    return false;  // 解质量不好，放弃
}

// converters
inline Vec2 toVec2(const cv::Point2f p) { return Vec2(p.x, p.y); }

}  // namespace myslam

#endif  // MYSLAM_ALGORITHM_H
