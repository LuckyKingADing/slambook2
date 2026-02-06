#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

#include <boost/timer.hpp>

// for sophus
#include <sophus/se3.hpp>

using Sophus::SE3d;

// for eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

/**********************************************
* 本程序演示了单目相机在已知轨迹下的稠密深度估计
* 使用极线搜索 + NCC 匹配的方式，与书本的 12.2 节对应
* 请注意本程序并不完美，你完全可以改进它——我其实在故意暴露一些问题(这是借口)。
***********************************************/

// ------------------------------------------------------------------
// parameters
const int boarder = 20;         // 边缘宽度
const int width = 640;          // 图像宽度
const int height = 480;         // 图像高度
const double fx = 481.2f;       // 相机内参
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;
const int ncc_window_size = 3;    // NCC 取的窗口半宽度
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1); // NCC窗口面积
const double min_cov = 0.1;     // 收敛判定：最小方差
const double max_cov = 10;      // 发散判定：最大方差

// ------------------------------------------------------------------
// 重要的函数
/// 从 REMODE 数据集读取数据
bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    vector<SE3d> &poses,
    cv::Mat &ref_depth
);

/**
 * 根据新的图像更新深度估计
 * @param ref           参考图像
 * @param curr          当前图像
 * @param T_C_R         参考图像到当前图像的位姿
 * @param depth         深度
 * @param depth_cov     深度方差
 * @return              是否成功
 */
bool update(
    const Mat &ref,
    const Mat &curr,
    const SE3d &T_C_R,
    Mat &depth,
    Mat &depth_cov2
);

/**
 * 极线搜索
 * @param ref           参考图像
 * @param curr          当前图像
 * @param T_C_R         位姿
 * @param pt_ref        参考图像中点的位置
 * @param depth_mu      深度均值
 * @param depth_cov     深度方差
 * @param pt_curr       当前点
 * @param epipolar_direction  极线方向
 * @return              是否成功
 */
bool epipolarSearch(
    const Mat &ref,
    const Mat &curr,
    const SE3d &T_C_R,
    const Vector2d &pt_ref,
    const double &depth_mu,
    const double &depth_cov,
    Vector2d &pt_curr,
    Vector2d &epipolar_direction
);

/**
 * 更新深度滤波器
 * @param pt_ref    参考图像点
 * @param pt_curr   当前图像点
 * @param T_C_R     位姿
 * @param epipolar_direction 极线方向
 * @param depth     深度均值
 * @param depth_cov2    深度方向
 * @return          是否成功
 */
bool updateDepthFilter(
    const Vector2d &pt_ref,
    const Vector2d &pt_curr,
    const SE3d &T_C_R,
    const Vector2d &epipolar_direction,
    Mat &depth,
    Mat &depth_cov2
);

/**
 * 计算 NCC 评分
 * @param ref       参考图像
 * @param curr      当前图像
 * @param pt_ref    参考点
 * @param pt_curr   当前点
 * @return          NCC评分
 */
double NCC(const Mat &ref, const Mat &curr, const Vector2d &pt_ref, const Vector2d &pt_curr);

// 双线性灰度插值
inline double getBilinearInterpolatedValue(const Mat &img, const Vector2d &pt) {
    uchar *d = &img.data[int(pt(1, 0)) * img.step + int(pt(0, 0))];
    double xx = pt(0, 0) - floor(pt(0, 0));
    double yy = pt(1, 0) - floor(pt(1, 0));
    return ((1 - xx) * (1 - yy) * double(d[0]) +
            xx * (1 - yy) * double(d[1]) +
            (1 - xx) * yy * double(d[img.step]) +
            xx * yy * double(d[img.step + 1])) / 255.0;
}

// ------------------------------------------------------------------
// 一些小工具
// 显示估计的深度图
void plotDepth(const Mat &depth_truth, const Mat &depth_estimate);

// 像素到相机坐标系
inline Vector3d px2cam(const Vector2d px) {
    return Vector3d(
        (px(0, 0) - cx) / fx,
        (px(1, 0) - cy) / fy,
        1
    );
}

// 相机坐标系到像素
inline Vector2d cam2px(const Vector3d p_cam) {
    return Vector2d(
        p_cam(0, 0) * fx / p_cam(2, 0) + cx,
        p_cam(1, 0) * fy / p_cam(2, 0) + cy
    );
}

// 检测一个点是否在图像边框内
inline bool inside(const Vector2d &pt) {
    return pt(0, 0) >= boarder && pt(1, 0) >= boarder
           && pt(0, 0) + boarder < width && pt(1, 0) + boarder <= height;
}

// 显示极线匹配
void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_curr);

// 显示极线
void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_min_curr,
                      const Vector2d &px_max_curr);

/// 评测深度估计
void evaludateDepth(const Mat &depth_truth, const Mat &depth_estimate);
// ------------------------------------------------------------------


int main(int argc, char **argv) {
    // 程序入口：期望传入一个数据集目录
    // 用法： dense_mapping path_to_test_dataset
    if (argc != 2) {
        cout << "Usage: dense_mapping path_to_test_dataset" << endl;
        return -1;
    }

    // ------------------------------------------------------------------
    // 1) 从数据集读取图像列表、每帧位姿以及参考深度（ground-truth）
    //    readDatasetFiles 按本示例约定的文件格式解析并填充：
    //      - color_image_files: 每帧图像的路径（用于读取灰度图像）
    //      - poses_TWC: 每帧在世界坐标系下的相机位姿（T_W_C）
    //      - ref_depth: 参考帧的真实深度图（用于评估和可视化）
    //    若读入失败则退出程序。
    vector<string> color_image_files;
    vector<SE3d> poses_TWC;
    Mat ref_depth;
    bool ret = readDatasetFiles(argv[1], color_image_files, poses_TWC, ref_depth);
    if (ret == false) {
        cout << "Reading image files failed!" << endl;
        return -1;
    }
    cout << "read total " << color_image_files.size() << " files." << endl;

    // ------------------------------------------------------------------
    // 2) 初始化：把第一帧作为参考帧（reference frame）
    //    depth / depth_cov2 为对参考帧每个像素的深度后验（均值与方差），
    //    程序先用常数初始化（init_depth, init_cov2），随后逐帧融合观测进行更新。
    Mat ref = imread(color_image_files[0], 0);                // 以灰度方式读取第一帧
    SE3d pose_ref_TWC = poses_TWC[0];                         // 参考帧的世界位姿
    double init_depth = 3.0;    // 初始深度（单位：米，示例值）
    double init_cov2 = 3.0;     // 初始方差（不确定度较大）
    Mat depth(height, width, CV_64F, init_depth);             // 每像素深度均值（double）
    Mat depth_cov2(height, width, CV_64F, init_cov2);         // 每像素深度方差（double）

    // ------------------------------------------------------------------
    // 3) 主循环：对每个后续帧进行一次观测更新
    //    处理步骤（每帧）：
    //      a) 读取当前帧图像 curr（灰度）
    //      b) 计算参考帧到当前帧的相对变换 T_C_R，供投影与极线搜索使用
    //      c) 调用 update()：遍历参考帧每个像素，在极线上搜索匹配并融合深度观测
    //      d) 用 ref_depth（ground-truth）评估并用 plotDepth 可视化
    for (int index = 1; index < color_image_files.size(); index++) {
        cout << "*** loop " << index << " ***" << endl;
        Mat curr = imread(color_image_files[index], 0); // 当前帧灰度图
        if (curr.data == nullptr) continue;            // 读取失败则跳过

        // pose_curr_TWC 是当前帧在世界坐标系下的位姿（T_W_C）
        SE3d pose_curr_TWC = poses_TWC[index];

        // 计算参考帧 R 到当前帧 C 的变换 T_C_R = T_C_W * T_W_R = inverse(T_W_C_curr) * T_W_C_ref
        // 该变换用于把参考帧中某像素按某深度投影到当前帧上（极线/投影依赖此）
        SE3d pose_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC;

        // 用当前帧更新参考帧的深度后验：对参考帧所有像素进行极线搜索 + NCC 匹配 + 三角化 + 高斯融合
        update(ref, curr, pose_T_C_R, depth, depth_cov2);

        // 每帧评估与可视化：与真值对比并显示误差图、估计图等
        evaludateDepth(ref_depth, depth);
        plotDepth(ref_depth, depth);
        imshow("image", curr);
        waitKey(1);
    }

    // ------------------------------------------------------------------
    // 4) 所有帧处理完成后保存结果
    //    注意：depth 为 CV_64F，直接用 imwrite 保存为 PNG 会将数据转换并可能丢失精度。
    //    若需要精确保存深度数值，建议使用 FileStorage 保存为 YAML/EXR 或先归一化后保存为 16-bit 图像。
    cout << "estimation returns, saving depth map ..." << endl;
    imwrite("depth.png", depth);
    cout << "done." << endl;

    return 0;
}

bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    std::vector<SE3d> &poses,
    cv::Mat &ref_depth) {
    // 打开数据列表文件：每行包含一张图像文件名和对应的位姿（tx ty tz qx qy qz qw）
    // 说明：位姿按 T_W_C（相机在世界坐标系下的位姿）格式给出，而非 T_C_W
    ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
    if (!fin) return false; // 文件打开失败则返回 false

    // 逐行读取直到文件结尾。每行第一个字段是图像文件名，随后 7 个 double 表示位姿。
    // 注意：文件中四元数的顺序在本数据集中是 qx qy qz qw（书中/库中构造 Quaterniond 需要 qw 在第一个参数）
    while (!fin.eof()) {
        // 数据格式：图像文件名 tx ty tz qx qy qz qw ，注意是 T_W_C 而非 T_C_W
        string image;
        fin >> image; // 读取图像名
        double data[7];
        for (double &d : data) fin >> d; // 读取 7 个数值（tx ty tz qx qy qz qw）

        // 拼接得到图像完整路径并加入列表（假定 images 目录位于 path 下）
        color_image_files.push_back(path + string("/images/") + image);

        // 将读取的位姿数据转换为 Sophus::SE3d：注意构造顺序 Quaterniond(w, x, y, z)
        // data 布局： [tx, ty, tz, qx, qy, qz, qw]
        poses.push_back(
            SE3d(Quaterniond(data[6], data[3], data[4], data[5]),
                 Vector3d(data[0], data[1], data[2]))
        );

        // 如果读取过程中发生错误（例如文件末尾或格式异常），跳出循环
        if (!fin.good()) break;
    }
    fin.close();

    // 读取参考深度文件。该数据集中的深度文件格式为每个像素一个深度值（单位为厘米），
    // 因此需要除以 100 转换为米并存入 CV_64F 的 cv::Mat 中。
    fin.open(path + "/depthmaps/scene_000.depth");
    ref_depth = cv::Mat(height, width, CV_64F);
    if (!fin) return false; // 深度文件打开失败则返回 false

    // 深度文件按行按列顺序存储像素深度（通常为整数或浮点数，单位为厘米）
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            double depth = 0;
            fin >> depth; // 读取深度（单位：厘米）
            ref_depth.ptr<double>(y)[x] = depth / 100.0; // 转为米并写入 Mat（行 y，列 x）
        }

    return true; // 成功读取所有必要数据
}

// 对整个深度图进行更新
bool update(const Mat &ref, const Mat &curr, const SE3d &T_C_R, Mat &depth, Mat &depth_cov2) {
    // 注意：外层是 x，内层是 y（列主顺序），访问像素用 depth.ptr<double>(y)[x]
    for (int x = boarder; x < width - boarder; x++)
        for (int y = boarder; y < height - boarder; y++) {
            // 遍历图像中每个非边界像素

            // 取出当前像素的深度方差（在 depth_cov2 中存储）并判断是否需要继续更新：
            //  - 若方差小于 min_cov，认为已经收敛（不再更新）
            //  - 若方差大于 max_cov，认为发散/不可靠（也跳过）
            if (depth_cov2.ptr<double>(y)[x] < min_cov || depth_cov2.ptr<double>(y)[x] > max_cov) // 深度已收敛或发散
                continue;

            // 在极线上搜索该像素的匹配点：需要传入参考帧像素坐标、当前深度分布的均值与标准差
            // 准备输出变量：匹配点坐标 pt_curr，以及估计的极线方向 epipolar_direction
            Vector2d pt_curr;
            Vector2d epipolar_direction;

            // 调用 epipolarSearch：
            //  - ref: 参考图像（灰度）
            //  - curr: 当前图像（灰度）
            //  - T_C_R: 参考帧到当前帧的变换（用于把参考像素按不同深度投影到当前帧）
            //  - Vector2d(x,y): 参考像素位置（列 x，行 y）
            //  - depth.ptr<double>(y)[x]: 当前像素深度均值（double）
            //  - sqrt(depth_cov2.ptr<double>(y)[x]): 当前像素深度标准差（方差开根号），epipolarSearch 用来限定搜索区间
            //  - pt_curr, epipolar_direction: 输出结果
            bool ret = epipolarSearch(
                ref,
                curr,
                T_C_R,
                Vector2d(x, y),
                depth.ptr<double>(y)[x],
                sqrt(depth_cov2.ptr<double>(y)[x]),
                pt_curr,
                epipolar_direction
            );

            // 若在极线上没有找到可靠匹配（NCC 分数低或不可达），跳过该像素
            if (ret == false) // 匹配失败
                continue;

            // 可视化匹配（调试用）：若需要在运行时看到匹配，解除下一行注释
            // showEpipolarMatch(ref, curr, Vector2d(x, y), pt_curr);

            // 匹配成功：根据当前匹配（pt_curr）和极线方向，使用三角化得到的观测
            // 更新该像素的高斯深度后验（在 updateDepthFilter 中实现融合逻辑）
            updateDepthFilter(Vector2d(x, y), pt_curr, T_C_R, epipolar_direction, depth, depth_cov2);
        }
}

// 极线搜索
// 方法见书 12.2 12.3 两节
bool epipolarSearch(
    const Mat &ref, const Mat &curr,
    const SE3d &T_C_R, const Vector2d &pt_ref,
    const double &depth_mu, const double &depth_cov,
    Vector2d &pt_curr, Vector2d &epipolar_direction) {
    // 将参考帧像素坐标转换到归一化相机坐标系的方向向量 f_ref
    // 注意：px2cam 返回的 Vector3d 为方向（z=1），需归一化以表示单位视线方向
    Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();

    // 以深度均值构建参考帧上的 3D 点 P_ref = depth_mu * f_ref
    Vector3d P_ref = f_ref * depth_mu;    // 参考帧的 P 向量

    // 将该 3D 点投影到当前帧，得到按深度均值投影的像素位置
    // T_C_R * P_ref 表示将参考帧坐标系下的点变换到当前帧，再用 cam2px 得到像素坐标
    Vector2d px_mean_curr = cam2px(T_C_R * P_ref); // 按深度均值投影的像素

    // 根据均值和方差构造一个搜索深度区间 [d_min, d_max]（使用 3*sigma 作为区间半宽）
    double d_min = depth_mu - 3 * depth_cov, d_max = depth_mu + 3 * depth_cov;
    if (d_min < 0.1) d_min = 0.1; // 避免深度为负或过小导致数值不稳定

    // 将区间的两端投影到当前帧上，得到极线段的端点（对应最小/最大深度）
    Vector2d px_min_curr = cam2px(T_C_R * (f_ref * d_min));    // 按最小深度投影的像素
    Vector2d px_max_curr = cam2px(T_C_R * (f_ref * d_max));    // 按最大深度投影的像素

    // 极线在图像上可以近似为上述两个端点连成的线段
    Vector2d epipolar_line = px_max_curr - px_min_curr;    // 极线（线段形式）

    // epipolar_direction 表示沿极线的单位方向，用于在极线上采样
    epipolar_direction = epipolar_line;        // 极线方向
    epipolar_direction.normalize();

    // 搜索区间的半长度（以像素为单位），从中点向两侧扩展 half_length
    double half_length = 0.5 * epipolar_line.norm();    // 极线线段的半长度
    if (half_length > 100) half_length = 100;   // 限制最大搜索长度，避免过大计算量/错误匹配

    // 可选可视化：显示极线段用于调试
    // showEpipolarLine( ref, curr, pt_ref, px_min_curr, px_max_curr );

    // 在极线上以步长采样，寻找与参考 patch NCC 最高的像素
    // 这里采用从 -half_length 到 +half_length 的均匀采样，步长 0.7 像素（次采样以覆盖子像素位置）
    double best_ncc = -1.0; // 记录最佳 NCC 分数
    Vector2d best_px_curr;  // 记录最佳对应像素
    for (double l = -half_length; l <= half_length; l += 0.7) { // l+=sqrt(2)
        // 待匹配点为均值投影点偏移 l 个像素沿极线方向
        Vector2d px_curr = px_mean_curr + l * epipolar_direction;  // 待匹配点

        // 若像素位置超出图像有效范围则跳过（inside 会考虑边界 boarder）
        if (!inside(px_curr))
            continue;

        // 计算候选点与参考 patch 的 NCC 相似度（返回值范围 -1..1）
        double ncc = NCC(ref, curr, pt_ref, px_curr);

        // 记录最大的 NCC 和对应像素位置
        if (ncc > best_ncc) {
            best_ncc = ncc;
            best_px_curr = px_curr;
        }
    }

    // 仅在 NCC 很高时才接受匹配（阈值 0.85 是经验值，目的是降低错误匹配率）
    if (best_ncc < 0.85f)      // 只相信 NCC 很高的匹配
        return false;

    // 返回最佳匹配点
    pt_curr = best_px_curr;
    return true;
}

double NCC(
    const Mat &ref, const Mat &curr,
    const Vector2d &pt_ref, const Vector2d &pt_curr) {
    // 计算零均值归一化互相关（Zero-mean Normalized Cross-Correlation）
    // 步骤：
    //  1) 在以 pt_ref, pt_curr 为中心，大小为 (2*ncc_window_size+1)^2 的窗口内采样灰度值
    //     - 对参考图像直接索引（整像素），对当前图像使用双线性插值以获得亚像素值
    //  2) 计算两个窗口的均值
    //  3) 计算零均值互相关： numerator / sqrt(den1 * den2)
    // 返回值范围约为 [-1, 1]，越接近 1 表示越相似

    double mean_ref = 0, mean_curr = 0;
    vector<double> values_ref, values_curr; // 存储窗口内每个像素的值，便于后续计算

    // 遍历窗口，注意这里的循环变量含义：x 对应列方向偏移，y 对应行方向偏移
    for (int x = -ncc_window_size; x <= ncc_window_size; x++)
        for (int y = -ncc_window_size; y <= ncc_window_size; y++) {
            // 参考图像使用直接索引（假定 pt_ref 为有效整数像素位置）
            double value_ref = double(ref.ptr<uchar>(int(y + pt_ref(1, 0)))[int(x + pt_ref(0, 0))]) / 255.0;
            mean_ref += value_ref;

            // 当前图像使用双线性插值以支持亚像素匹配
            double value_curr = getBilinearInterpolatedValue(curr, pt_curr + Vector2d(x, y));
            mean_curr += value_curr;

            values_ref.push_back(value_ref);
            values_curr.push_back(value_curr);
        }

    // 计算窗口均值
    mean_ref /= ncc_area;
    mean_curr /= ncc_area;

    // 计算零均值 NCC 的分子与两个分母项
    double numerator = 0, demoniator1 = 0, demoniator2 = 0;
    for (int i = 0; i < (int)values_ref.size(); i++) {
        double a = values_ref[i] - mean_ref;
        double b = values_curr[i] - mean_curr;
        numerator += a * b;
        demoniator1 += a * a; // 参考窗口的能量
        demoniator2 += b * b; // 当前窗口的能量
    }

    // 加上一个很小的常数以避免除零或数值不稳定
    return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
}

bool updateDepthFilter(
    const Vector2d &pt_ref,
    const Vector2d &pt_curr,
    const SE3d &T_C_R,
    const Vector2d &epipolar_direction,
    Mat &depth,
    Mat &depth_cov2) {
    // 使用两帧观测对参考像素进行三角化并用高斯融合更新深度后验。
    // 输入说明：pt_ref（参考帧像素）、pt_curr（当前帧匹配像素）、T_C_R（参考->当前变换）
    // 1) 将参考->当前的变换取逆得到 T_R_C（当前帧到参考帧），便于构造几何方程
    SE3d T_R_C = T_C_R.inverse();

    // 2) 将像素转换为相机坐标系下的单位方向向量并归一化
    Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();
    Vector3d f_curr = px2cam(pt_curr);
    f_curr.normalize();

    // 两视线的几何关系：
    // d_ref * f_ref = d_cur * (R_RC * f_curr) + t_RC
    // 令 f2 = R_RC * f_curr，t = t_RC，则可写成线性方程组求 d_ref, d_cur

    Vector3d t = T_R_C.translation();
    Vector3d f2 = T_R_C.so3() * f_curr;
    // 右端向量 b，元素为 t·f_ref 和 t·f2
    Vector2d b = Vector2d(t.dot(f_ref), t.dot(f2));

    // 构造 2x2 矩阵 A，来自对两边分别做内积得到的正规方程
    Matrix2d A;
    A(0, 0) = f_ref.dot(f_ref);
    A(0, 1) = -f_ref.dot(f2);
    A(1, 0) = -A(0, 1);
    A(1, 1) = -f2.dot(f2);

    // 求解线性系统 A * [d_ref; d_cur] = b
    Vector2d ans = A.inverse() * b;

    // 从两侧视线恢复 3D 点：ref 侧为 ans[0] * f_ref，cur 侧为 t + ans[1] * f2
    Vector3d xm = ans[0] * f_ref;           // ref 侧的三维点（相对于参考相机）
    Vector3d xn = t + ans[1] * f2;          // cur 侧的三维点（变换回参考相机系）

    // 取两者中点作为最终的 3D 位置估计，深度取该点到相机原点的范数
    Vector3d p_esti = (xm + xn) / 2.0;      // P 的位置估计（参考相机系）
    double depth_estimation = p_esti.norm();   // 深度（米）

    // ------------------------------------------------------------------
    // 估计该深度观测的不确定度（近似）：将像素级误差（约 1 像素）通过三角关系传播到深度方向
    // 该推导为几何近似，步骤简述：
    //  - 计算参考视线与基线 t 的夹角 alpha
    //  - 计算当前视线与基线的夹角 beta
    //  - 将像素方向沿 epipolar_direction 偏移后的视线与基线夹角为 beta_prime
    //  - 利用正弦定理估计受像素偏移影响的深度 p_prime，差值即为深度的不确定量 d_cov
    Vector3d p = f_ref * depth_estimation;
    Vector3d a = p - t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos(f_ref.dot(t) / t_norm);
    double beta = acos(-a.dot(t) / (a_norm * t_norm));
    Vector3d f_curr_prime = px2cam(pt_curr + epipolar_direction); // 当前像素沿极线方向偏移一个像素
    f_curr_prime.normalize();
    double beta_prime = acos(f_curr_prime.dot(-t) / t_norm);
    double gamma = M_PI - alpha - beta_prime;
    double p_prime = t_norm * sin(beta_prime) / sin(gamma);
    double d_cov = p_prime - depth_estimation; // 近似的深度偏差
    double d_cov2 = d_cov * d_cov; // 观测方差

    // ------------------------------------------------------------------
    // 高斯融合：将先验 (mu, sigma2) 与观测 (depth_estimation, d_cov2) 做融合
    // mu_fuse = (d_cov2 * mu + sigma2 * z) / (sigma2 + d_cov2)
    // sigma_fuse2 = (sigma2 * d_cov2) / (sigma2 + d_cov2)
    double mu = depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];
    double sigma2 = depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];

    double mu_fuse = (d_cov2 * mu + sigma2 * depth_estimation) / (sigma2 + d_cov2);
    double sigma_fuse2 = (sigma2 * d_cov2) / (sigma2 + d_cov2);

    // 将融合后的后验写回对应像素位置
    depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = mu_fuse;
    depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = sigma_fuse2;

    return true;
}

// 后面这些太简单我就不注释了（其实是因为懒）
void plotDepth(const Mat &depth_truth, const Mat &depth_estimate) {
    imshow("depth_truth", depth_truth * 0.4);
    imshow("depth_estimate", depth_estimate * 0.4);
    imshow("depth_error", depth_truth - depth_estimate);
    waitKey(1);
}

void evaludateDepth(const Mat &depth_truth, const Mat &depth_estimate) {
    // 评估参考深度与估计深度之间的误差：计算平均误差与均方误差（在图像边界内统计）
    double ave_depth_error = 0;      // 累计的深度误差之和
    double ave_depth_error_sq = 0;   // 累计的深度平方误差之和
    int cnt_depth_data = 0;          // 有效像素计数

    // 遍历除去边界的像素区域（边界由 boarder 指定），以避免访问无效邻域或显示区域的干扰
    for (int y = boarder; y < depth_truth.rows - boarder; y++)
        for (int x = boarder; x < depth_truth.cols - boarder; x++) {
            // 逐像素计算误差：真实值 - 估计值
            double error = depth_truth.ptr<double>(y)[x] - depth_estimate.ptr<double>(y)[x];
            ave_depth_error += error;            // 累加误差
            ave_depth_error_sq += error * error; // 累加平方误差
            cnt_depth_data++;
        }

    // 为防止极端情况下 cnt_depth_data 为 0（理论上不应发生），先检查再做除法
    if (cnt_depth_data > 0) {
        ave_depth_error /= cnt_depth_data;
        ave_depth_error_sq /= cnt_depth_data;
    } else {
        ave_depth_error = 0;
        ave_depth_error_sq = 0;
    }

    // 输出评估结果：均方误差与平均误差
    cout << "Average squared error = " << ave_depth_error_sq << ", average error: " << ave_depth_error << endl;
}

void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_curr) {
    Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, CV_GRAY2BGR);
    cv::cvtColor(curr, curr_show, CV_GRAY2BGR);

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 0, 250), 2);
    cv::circle(curr_show, cv::Point2f(px_curr(0, 0), px_curr(1, 0)), 5, cv::Scalar(0, 0, 250), 2);

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    waitKey(1);
}

void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_min_curr,
                      const Vector2d &px_max_curr) {

    Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, CV_GRAY2BGR);
    cv::cvtColor(curr, curr_show, CV_GRAY2BGR);

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_max_curr(0, 0), px_max_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::line(curr_show, Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), Point2f(px_max_curr(0, 0), px_max_curr(1, 0)),
             Scalar(0, 255, 0), 1);

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    waitKey(1);
}
