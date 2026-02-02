#include <iostream>
#include <fstream>
#include <string>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

using namespace std;

/************************************************
 * 本程序演示如何用g2o solver进行位姿图优化
 * sphere.g2o是人工生成的一个Pose graph，我们来优化它。
 * 尽管可以直接通过load函数读取整个图，但我们还是自己来实现读取代码，以期获得更深刻的理解
 * 这里使用g2o/types/slam3d/中的SE3表示位姿，它实质上是**四元数**而非李代数.
 * **********************************************/

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: pose_graph_g2o_SE3 sphere.g2o" << endl;
        return 1;
    }
    ifstream fin(argv[1]);
    if (!fin) {
        cout << "file " << argv[1] << " does not exist." << endl;
        return 1;
    }

    // 初始化求解器与优化器
    // 设定g2o
        // 参数块
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
        // 线性求解器
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
        // 设置求解器
    auto solver = new g2o::OptimizationAlgorithmLevenberg( // LM-列文伯格算法
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
        // 解算器，使用稀疏非线性最小二乘配置
    g2o::SparseOptimizer optimizer;     // 图模型
    optimizer.setAlgorithm(solver);   // 设置求解器
    optimizer.setVerbose(true);       // 打开调试输出

    int vertexCnt = 0, edgeCnt = 0; // 顶点和边的数量

    // 读取文件并构建图
    while (!fin.eof()) {
        string name;
        fin >> name; // 读取标签，name可能是VERTEX_SE3:QUAT或EDGE_SE3:QUAT
        if (name == "VERTEX_SE3:QUAT") {
            // SE3 顶点
            g2o::VertexSE3 *v = new g2o::VertexSE3();
            int index = 0;
            fin >> index; // 读取顶点id
            v->setId(index); // 设置id
            v->read(fin); // 从流读位姿（四元数形式）
            optimizer.addVertex(v); // 加入图中
            vertexCnt++; // 顶点数+1
            if (index == 0) // 第0个节点固定；把 id 为 0 的顶点固定（v->setFixed(true)）以锚定全局自由度。
                v->setFixed(true);
        } else if (name == "EDGE_SE3:QUAT") {
            // SE3-SE3 边
            g2o::EdgeSE3 *e = new g2o::EdgeSE3();
            int idx1, idx2;     // 关联的两个顶点
            fin >> idx1 >> idx2; // 读取两个顶点的id
            e->setId(edgeCnt++); // 设置边的id
            e->setVertex(0, optimizer.vertices()[idx1]); // 设置连接的顶点
            e->setVertex(1, optimizer.vertices()[idx2]); // 设置连接的顶点
            e->read(fin); // 从流读测量值（相对变换）与信息矩阵
            optimizer.addEdge(e); // 加入图中
        }
        if (!fin.good()) break;
    }

    cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges." << endl; // 输出读入的顶点和边数

    cout << "optimizing ..." << endl; // 打印优化开始
    
    // 优化
    optimizer.initializeOptimization(); // 初始化因子图；构建内部稀疏矩阵结构和因子索引。
    optimizer.optimize(30);             // 执行优化，最多迭代30次

    cout << "saving optimization results ..." << endl; // 打印保存结果开始
    optimizer.save("result.g2o"); // 保存优化结果到文件result.g2o

    return 0;
}