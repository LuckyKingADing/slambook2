#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h> // 块求解器（处理稀疏矩阵分块）
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>  // 稀疏线性求解器（CSparse）
#include <g2o/core/robust_kernel_impl.h>// 鲁棒核（处理外点）
#include <iostream>

#include "common.h"
#include "sophus/se3.hpp" // Sophus库处理李群/李代数

using namespace Sophus;
using namespace Eigen;
using namespace std;

/// 姿态和内参的结构
struct PoseAndIntrinsics {
    PoseAndIntrinsics() {}

     // 从内存读取参数初始化
    explicit PoseAndIntrinsics(double *data_addr) {
        rotation = SO3d::exp(Vector3d(data_addr[0], data_addr[1], data_addr[2])); // 李代数转李群
        translation = Vector3d(data_addr[3], data_addr[4], data_addr[5]);
        focal = data_addr[6];
        k1 = data_addr[7]; k2 = data_addr[8];
    }

    // 将优化后的参数写回内存；也会优化内参和畸变参数？
    void set_to(double *data_addr) {
        auto r = rotation.log(); // 李群转李代数（3维向量）
        for (int i = 0; i < 3; ++i) data_addr[i] = r[i]; // 旋转李代数
        for (int i = 0; i < 3; ++i) data_addr[i + 3] = translation[i]; // 平移
        data_addr[6] = focal; data_addr[7] = k1; data_addr[8] = k2; // 内参+畸变
    }

    SO3d rotation; // 旋转（SO3李群，避免欧拉角奇异性）
    Vector3d translation = Vector3d::Zero(); // 平移（t）
    double focal = 0; // 焦距（f）
    double k1 = 0, k2 = 0; // 径向畸变参数
};

// 顶点定义：相机顶点（9 维）+ 路标点顶点（3 维）
/// 位姿加相机内参的顶点，9维，前三维为so3，接下去为t, f, k1, k2

// （1）相机顶点：VertexPoseAndIntrinsics
class VertexPoseAndIntrinsics : public g2o::BaseVertex<9, PoseAndIntrinsics> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoseAndIntrinsics() {}

    // 重置顶点（初始化）
    virtual void setToOriginImpl() override {
        _estimate = PoseAndIntrinsics();
    }

    // 增量更新（核心！李群的加法）
    virtual void oplusImpl(const double *update) override {
        // 旋转增量：李代数exp后左乘原旋转（左扰动模型）
        _estimate.rotation = SO3d::exp(Vector3d(update[0], update[1], update[2])) * _estimate.rotation;
        // 平移增量：直接加法
        _estimate.translation += Vector3d(update[3], update[4], update[5]);
        // 内参/畸变增量：直接加法
        _estimate.focal += update[6];
        _estimate.k1 += update[7];
        _estimate.k2 += update[8];
    }

    // 投影函数：3D路标点 → 2D像素（带畸变）
    /// 根据估计值投影一个点
    Vector2d project(const Vector3d &point) {
        // 步骤1：世界坐标 → 相机坐标（Pc = R*Pw + t）
        Vector3d pc = _estimate.rotation * point + _estimate.translation;
        // 步骤2：相机坐标 → 归一化平面（z轴取负，BAL数据集约定）
        pc = -pc / pc[2];
        // 步骤3：径向畸变计算
        double r2 = pc.squaredNorm(); // r² = X²+Y²（归一化平面）
        double distortion = 1.0 + r2 * (_estimate.k1 + _estimate.k2 * r2); // 畸变因子

        // 步骤4：归一化坐标 → 像素坐标（u=f*distortion*X/Z, v=f*distortion*Y/Z）
        return Vector2d(_estimate.focal * distortion * pc[0],
                        _estimate.focal * distortion * pc[1]);
    }

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}
};

// （2）路标点顶点：VertexPoint
class VertexPoint : public g2o::BaseVertex<3, Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoint() {}

    // 初始化 0 0 0 
    virtual void setToOriginImpl() override {
        _estimate = Vector3d(0, 0, 0);
    }

    // 路标点增量更新：直接加法（3D坐标是向量，无李群约束）
    virtual void oplusImpl(const double *update) override {
        _estimate += Vector3d(update[0], update[1], update[2]);
    }

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}
};

//  边定义：EdgeProjection（重投影误差边）；二元边
class EdgeProjection :
    public g2o::BaseBinaryEdge<2, Vector2d, VertexPoseAndIntrinsics, VertexPoint> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    // 计算误差：重投影误差 = 预测像素 - 观测像素
    virtual void computeError() override {
        // 获取连接的两个顶点：相机顶点（v0）、路标点顶点（v1）
            // (VertexPoseAndIntrinsics *)：把基类指针 Vertex* 强制转换为我们自定义的VertexPoseAndIntrinsics*（相机顶点指针）；
        // _vertices[0] = 边连接的第一个顶点（相机顶点），_vertices[1] = 边连接的第二个顶点（路标点顶点）。
        auto v0 = (VertexPoseAndIntrinsics *) _vertices[0];
        auto v1 = (VertexPoint *) _vertices[1];

        // 预测像素：用相机参数投影路标点
            // v1->estimate()：调用路标点顶点的estimate()方法，返回值是Vector3d（3D 路标点坐标）；
        auto proj = v0->project(v1->estimate());

        // 误差计算（g2o会自动最小化这个误差的平方和）：预测减去观测
        _error = proj - _measurement;
    }

    // use numeric derivatives
    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}


    /* - **核心属性**：
        - 二元边：连接“相机顶点”和“路标点顶点”；
        - 边维度：2维（像素坐标u/v）；
        - `computeError`：计算重投影误差，是BA的核心代价函数；
        - 未手动实现雅可比：代码中用了g2o的**数值微分**（自动计算雅可比），简化开发（缺点是速度略慢，工业级会手动写解析雅可比）。 */

};

void SolveBA(BALProblem &bal_problem);

int main(int argc, char **argv) {

    if (argc != 2) {
        cout << "usage: bundle_adjustment_g2o bal_data.txt" << endl;
        return 1;
    }

    BALProblem bal_problem(argv[1]); // 加载BAL数据
    bal_problem.Normalize();         // 数据归一化（提升数值稳定性）
    bal_problem.Perturb(0.1, 0.5, 0.5); // 添加扰动
    bal_problem.WriteToPLYFile("initial.ply"); // 保存优化前的点云
    SolveBA(bal_problem);            // 核心：求解BA
    bal_problem.WriteToPLYFile("final.ply"); // 保存优化后的点云
    return 0;
}

void SolveBA(BALProblem &bal_problem) {
    // 模块 1：数据准备（提取 BAL 数据指针）
    // 1. 提取BAL数据的基础信息和指针
    const int point_block_size = bal_problem.point_block_size(); // 路标点块大小（固定为3，对应X/Y/Z）
    const int camera_block_size = bal_problem.camera_block_size(); // 相机块大小（固定为9，3旋转+3平移+1焦距+2畸变）
    double *points = bal_problem.mutable_points(); // 路标点数据指针（指向原始数组，可修改）
    double *cameras = bal_problem.mutable_cameras(); // 相机数据指针（指向原始数组，可修改）

    // 模块 2：配置 g2o 求解器（核心！决定优化效率和稳定性）
    // 2. 配置g2o的求解器链（LM + 块求解器 + 稀疏线性求解器）
    // 2.1 定义块求解器：相机顶点9维，路标点顶点3维（必须和自定义顶点维度一致） 
    // pose dimension 9, landmark is 3
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>> BlockSolverType;
    // 2.2 定义线性求解器：CSparse（处理稀疏矩阵，适配BA的稀疏性）
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    // 2.3 配置LM优化算法（列文伯格-马夸尔特，兼顾速度和稳定性）
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    // 2.4 初始化稀疏优化器（g2o的核心对象，管理所有顶点/边）
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver); // 绑定求解器
    optimizer.setVerbose(true); // 打印优化过程（迭代次数、误差变化等）


    // 模块 3：构建图优化的「顶点」（相机 + 路标点）
    /// build g2o problem
    
    
    // vertex
    // 顶点 ID 必须唯一：这里相机 ID 从 0 开始，路标点 ID 从num_cameras()开始（避免冲突）

    // 子模块 3.1：添加相机顶点（9 维，位姿 + 内参）
    vector<VertexPoseAndIntrinsics *> vertex_pose_intrinsics;// 存储相机顶点指针的数组（方便后续访问）
    vector<VertexPoint *> vertex_points;
    for (int i = 0; i < bal_problem.num_cameras(); ++i) { // 
        // 创建自定义相机顶点对象
        VertexPoseAndIntrinsics *v = new VertexPoseAndIntrinsics();
        // 找到第i个相机的参数指针（内存偏移：camera_block_size * i）
        double *camera = cameras + camera_block_size * i;
        // 设置顶点ID（唯一，避免冲突，这里直接用相机索引i）
        v->setId(i);
        // 设置初始估计值（从BAL数据中读取）,可能带扰动？是优化的起点
        v->setEstimate(PoseAndIntrinsics(camera));
        // 将顶点添加到优化器
        optimizer.addVertex(v);
        // 保存指针到数组，供后续构建边使用
        vertex_pose_intrinsics.push_back(v);
    }

    // 子模块 3.2：添加路标点顶点（3 维，3D 坐标）
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        // 创建自定义路标点顶点对象
        VertexPoint *v = new VertexPoint();
        // 找到第i个路标点的参数指针（内存偏移：point_block_size * i）
        double *point = points + point_block_size * i;
        // 设置顶点ID（相机ID用完后，从num_cameras()开始）
        v->setId(i + bal_problem.num_cameras());
        // 设置初始估计值（从BAL数据中读取）
        v->setEstimate(Vector3d(point[0], point[1], point[2]));
        // ！核心！设置路标点为「待边缘化顶点」（利用舒尔补加速）
        v->setMarginalized(true);
        /* setMarginalized(true)：
            作用：告诉 g2o，优化时先通过舒尔补（Schur Complement）消去路标点变量，只求解相机变量，大幅降低计算量（对应你之前学的 H 矩阵稀疏性）；
            为什么只边缘化路标点：路标点数量远多于相机（比如 100 个相机 + 10000 个路标点），消去路标点后，求解规模从 30900 维降到 900 维，效率提升一个量级。 */

        // 将顶点添加到优化器
        optimizer.addVertex(v);
        // 保存指针到数组
        vertex_points.push_back(v);
    }

    // 模块 4：构建图优化的「边」（重投影误差边）
    // edge
    // 提取观测数据指针（每个观测是2个值：u/v）
    const double *observations = bal_problem.observations();
    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        // 创建自定义重投影误差边
        EdgeProjection *edge = new EdgeProjection;
        // 绑定边的第一个顶点：第i个观测对应的相机顶点
        edge->setVertex(0, vertex_pose_intrinsics[bal_problem.camera_index()[i]]);
        // 绑定边的第二个顶点：第i个观测对应的路标点顶点
        edge->setVertex(1, vertex_points[bal_problem.point_index()[i]]);
        // 设置观测值（真实像素坐标，从BAL数据读取）
        edge->setMeasurement(Vector2d(observations[2 * i + 0], observations[2 * i + 1])); // 这是怎么设置观测值的？索引是怎么设置的？
        // 设置信息矩阵（权重，单位矩阵表示u/v等权重）信息矩阵Ω是误差的权重，单位矩阵表示 u/v 方向的误差同等重要；如果某方向噪声大，可减小对应权重（比如Matrix2d::Diagonal(1, 0.5)）；
        edge->setInformation(Matrix2d::Identity());
        // 设置鲁棒核（Huber核，处理外点）
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        // 将边添加到优化器
        optimizer.addEdge(edge);
    }

    // 模块 5：执行优化（核心求解步骤）
   // 初始化优化器（构建H矩阵、雅可比矩阵等）
    optimizer.initializeOptimization();
    // 执行优化，迭代40次（足够收敛）
    optimizer.optimize(40);
    /* 优化过程的本质：
        初始化：g2o 遍历所有边，计算初始重投影误差，构建初始 H 矩阵和 g 向量；
        迭代 40 次：每次迭代做 3 件事：
            计算所有边的误差和雅可比（数值微分）；
            构建稀疏线性方程组 HΔx=g，用 CSparse 求解增量Δx；
            更新顶点估计值（相机 / 路标点），判断是否收敛（误差变化 < 阈值）；
        停止：迭代 40 次后，输出优化后的顶点参数。
        （或者达到阈值收敛阶段，提前退出迭代）
        */

    // 模块 6：优化结果回写（将结果写回 BAL 数据）
    // // 提取观测数据指针（每个观测是2个值：u/v）
    // set to bal problem
    // // 6.1 写回相机参数
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        double *camera = cameras + camera_block_size * i;
        auto vertex = vertex_pose_intrinsics[i];
        auto estimate = vertex->estimate(); // 获得优化估计值
        estimate.set_to(camera);  // 将优化后的相机参数（李群→李代数）写回内存
    }
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        double *point = points + point_block_size * i;
        auto vertex = vertex_points[i];
        // 将优化后的3D坐标写回内存
        for (int k = 0; k < 3; ++k) point[k] = vertex->estimate()[k];
    }
}
