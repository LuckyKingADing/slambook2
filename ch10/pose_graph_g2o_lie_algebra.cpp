#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Core>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include <sophus/se3.hpp>

using namespace std;
using namespace Eigen;
using Sophus::SE3d;
using Sophus::SO3d;

/************************************************
 * 本程序演示如何用g2o solver进行位姿图优化
 * sphere.g2o是人工生成的一个Pose graph，我们来优化它。
 * 尽管可以直接通过load函数读取整个图，但我们还是自己来实现读取代码，以期获得更深刻的理解
 * 本节使用李代数表达位姿图，节点和边的方式为自定义
 * **********************************************/

typedef Matrix<double, 6, 6> Matrix6d;

// 给定误差求J_R^{-1}的近似
Matrix6d JRInv(const SE3d &e) {
    Matrix6d J;
    J.block(0, 0, 3, 3) = SO3d::hat(e.so3().log()); // hat表示反对称矩阵，log表示李代数
    J.block(0, 3, 3, 3) = SO3d::hat(e.translation());
    J.block(3, 0, 3, 3) = Matrix3d::Zero(3, 3);
    J.block(3, 3, 3, 3) = SO3d::hat(e.so3().log());
    // J = J * 0.5 + Matrix6d::Identity();
    J = Matrix6d::Identity();    // try Identity if you want；用一阶近似，但对大角度/大误差不够精确，可以自己设置成上面的式子试试效果
    return J;

    /*
        // 解释（中文注释）:
        // 对于一个 SE3 误差量 e：
        //  - e.so3()            : 提取旋转分量 R（类型为 Sophus::SO3d）
        //  - e.so3().log()      : 将旋转 R 映射到李代数 so(3)，得到旋转向量 phi (3x1)，
        //                         该向量的方向为旋转轴，范数约为旋转角。
        //  - SO3d::hat(v)       : ``hat`` 运算，把 3x1 向量 v 映成 3x3 反对称矩阵 v^,
        //                         满足 v^ * w = v x w（叉乘矩阵），用于表示小角近似下的交叉项。
        //  - e.translation()    : 提取平移分量 t (3x1)。
        // 
        // 在书中给出的 J_r^{-1}(e) 的近似块结构，直观上包含了旋转和平移之间的耦合项：
        //  J = [ hat(phi)    hat(t) ]
        //      [   0         hat(phi)]
        //  其中 phi = e.so3().log(), t = e.translation()
        //  - 左上块 hat(phi) 表示旋转误差在线性化时对平移/旋转的影响（旋转与旋转的交叉项）
        //  - 右上块 hat(t)   表示平移与旋转耦合（例如 t x phi 项）
        //  - 右下块 hat(phi) 表示旋转对自身的影响
        //  - 左下块为零（平移对旋转的一阶影响被此处近似为 0）

        // 代码实现（可替换为更精确的 J_r^{-1} 表达式）：
        J.block(0, 0, 3, 3) = SO3d::hat(e.so3().log());       // hat(phi)
        J.block(0, 3, 3, 3) = SO3d::hat(e.translation());     // hat(t)
        J.block(3, 0, 3, 3) = Matrix3d::Zero(3, 3);           // 0
        J.block(3, 3, 3, 3) = SO3d::hat(e.so3().log());       // hat(phi)

        // 注：下面这一行把 J 近似为单位矩阵，这相当于采用 J_r^{-1} ≈ I 的一阶近似，
        // 在误差很小（小角、小位移）的情况下通常成立；若误差较大，应注释掉并使用上面的真实 J。
        // J = J * 0.5 + Matrix6d::Identity();
        J = Matrix6d::Identity();    // 若需更精确，可改为上面构造的 J 或书中更完整表达
    */
        
}

// 李代数顶点
typedef Matrix<double, 6, 1> Vector6d;

class VertexSE3LieAlgebra : public g2o::BaseVertex<6, SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    /* read / write：读取 7 个值（tx,ty,tz, qx,qy,qz,qw），写出时用四元数。注意读写四元数分量顺序与规范（read 中构造 Quaterniond(data[6], data[3], data[4], data[5])）。*/

    virtual bool read(istream &is) override {
        // 从流中读取一个顶点的 7 个数值：tx ty tz qx qy qz qw
        // 其中前 3 个是平移分量，后 4 个为四元数（注意顺序和构造方式）
        // 这里使用 Quaterniond(w, x, y, z) 的构造顺序，因此把 data[6] 放到第一个参数
        double data[7];
        for (int i = 0; i < 7; i++)
            is >> data[i];
        // 将读取的平移和四元数封装为 Sophus::SE3d 作为顶点的初始估计值
        setEstimate(SE3d(
            Quaterniond(data[6], data[3], data[4], data[5]),
            Vector3d(data[0], data[1], data[2])
        ));
        return true;
    }

    virtual bool write(ostream &os) const override {
        // 按 g2o 标准格式输出顶点信息：id tx ty tz qx qy qz qw
        // 这里用 unit_quaternion() 获取单位四元数表示，q.coeffs() 返回 (x,y,z,w)
        // 注意写出次序和读取时的构造次序要匹配
        os << id() << " ";
        Quaterniond q = _estimate.unit_quaternion();
        os << _estimate.translation().transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << endl;
        return true;
    }

    // 初始化
    virtual void setToOriginImpl() override {
        _estimate = SE3d(); // 单位元
    }

    // 左乘更新
    virtual void oplusImpl(const double *update) override {
        Vector6d upd;
        upd << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = SE3d::exp(upd) * _estimate;
    }
};

// 两个李代数节点之边：二元边
class EdgeSE3LieAlgebra : public g2o::BaseBinaryEdge<6, SE3d, VertexSE3LieAlgebra, VertexSE3LieAlgebra> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /* read / write：读取测量（7 值）并填信息矩阵（对称）；写出时按 VERTEX_SE3:QUAT / EDGE_SE3:QUAT 格式输出。 */
    virtual bool read(istream &is) override {
        // 读取一条边的测量：tx ty tz qx qy qz qw
        // 并读取该边的对称信息矩阵的上三角部分（随后对称填充）
        double data[7];
        for (int i = 0; i < 7; i++)
            is >> data[i];
        Quaterniond q(data[6], data[3], data[4], data[5]);
        q.normalize();
        // 将测量值封装为 SE3d（用于 computeError 中与顶点估计比较）
        setMeasurement(SE3d(q, Vector3d(data[0], data[1], data[2]))); // 设置测量值为SE3d
        // 读取信息矩阵的上三角并填充对称项；信息矩阵表示测量噪声的逆协方差
        for (int i = 0; i < information().rows() && is.good(); i++)
            for (int j = i; j < information().cols() && is.good(); j++) {
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j); // 对称填充，上三角到下三角
            }
        return true;
    }

    virtual bool write(ostream &os) const override {
        // 输出边的数据，格式为：idx1 idx2 tx ty tz qx qy qz qw [information upper-triangular]
        // 该格式与 g2o 的 EDGE_SE3:QUAT 格式兼容，便于用 g2o_viewer 或其他工具查看结果
        VertexSE3LieAlgebra *v1 = static_cast<VertexSE3LieAlgebra *> (_vertices[0]);
        VertexSE3LieAlgebra *v2 = static_cast<VertexSE3LieAlgebra *> (_vertices[1]);
        os << v1->id() << " " << v2->id() << " ";
        SE3d m = _measurement; 
        Eigen::Quaterniond q = m.unit_quaternion();
        os << m.translation().transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << " ";

        // 按上三角顺序输出信息矩阵元素（文件中通常只存储上三角以节省空间）
        for (int i = 0; i < information().rows(); i++)
            for (int j = i; j < information().cols(); j++) {
                os << information()(i, j) << " ";
            }
        os << endl;
        return true;
    }

    // 误差计算与书中推导一致
    virtual void computeError() override {
        SE3d v1 = (static_cast<VertexSE3LieAlgebra *> (_vertices[0]))->estimate();
        SE3d v2 = (static_cast<VertexSE3LieAlgebra *> (_vertices[1]))->estimate();
        _error = (_measurement.inverse() * v1.inverse() * v2).log(); // log表示映射到李代数
    }

    // 雅可比计算：关键实现
    virtual void linearizeOplus() override {
        // v1 和 v2 是连接的两个顶点的估计值
        SE3d v1 = (static_cast<VertexSE3LieAlgebra *> (_vertices[0]))->estimate();
        SE3d v2 = (static_cast<VertexSE3LieAlgebra *> (_vertices[1]))->estimate();
        // J表示J_R^{-1}；这里把 _error（在李代数中）exp 回群再传 JRInv，最终 J 近似为 J_r^{-1}(e)（实现细节有小差别但目的相同）。
        Matrix6d J = JRInv(SE3d::exp(_error));
        // 尝试把J近似为I？
        _jacobianOplusXi = -J * v2.inverse().Adj(); // Adj表示伴随矩阵,Ad因子？
        _jacobianOplusXj = J * v2.inverse().Adj(); 
    }
};

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: pose_graph_g2o_SE3_lie sphere.g2o" << endl;
        return 1;
    }
    ifstream fin(argv[1]);
    if (!fin) {
        cout << "file " << argv[1] << " does not exist." << endl;
        return 1;
    }

    // 设定g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;     // 图模型
    optimizer.setAlgorithm(solver);   // 设置求解器
    optimizer.setVerbose(true);       // 打开调试输出

    int vertexCnt = 0, edgeCnt = 0; // 顶点和边的数量

    vector<VertexSE3LieAlgebra *> vectices;
    vector<EdgeSE3LieAlgebra *> edges;
    while (!fin.eof()) {
        string name;
        fin >> name;
        if (name == "VERTEX_SE3:QUAT") {
            // 顶点
            VertexSE3LieAlgebra *v = new VertexSE3LieAlgebra();
            int index = 0;
            fin >> index;
            v->setId(index);
            v->read(fin);
            optimizer.addVertex(v);
            vertexCnt++;
            vectices.push_back(v);
            if (index == 0)
                v->setFixed(true);
        } else if (name == "EDGE_SE3:QUAT") {
            // SE3-SE3 边
            EdgeSE3LieAlgebra *e = new EdgeSE3LieAlgebra();
            int idx1, idx2;     // 关联的两个顶点
            fin >> idx1 >> idx2;
            e->setId(edgeCnt++);
            e->setVertex(0, optimizer.vertices()[idx1]);
            e->setVertex(1, optimizer.vertices()[idx2]);
            e->read(fin);
            optimizer.addEdge(e);
            edges.push_back(e);
        }
        if (!fin.good()) break;
    }

    cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges." << endl;

    cout << "optimizing ..." << endl;
    optimizer.initializeOptimization();
    optimizer.optimize(30);

    cout << "saving optimization results ..." << endl;

    // 因为用了自定义顶点且没有向g2o注册，这里保存自己来实现
    // 伪装成 SE3 顶点和边，让 g2o_viewer 可以认出
    ofstream fout("result_lie.g2o");
    for (VertexSE3LieAlgebra *v:vectices) {
        fout << "VERTEX_SE3:QUAT ";
        v->write(fout);
    }
    for (EdgeSE3LieAlgebra *e:edges) {
        fout << "EDGE_SE3:QUAT ";
        e->write(fout);
    }
    fout.close();
    return 0;
}
