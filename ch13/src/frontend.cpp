//
// Created by gaoxiang on 19-5-2.
//

#include <opencv2/opencv.hpp>

#include "myslam/algorithm.h"
#include "myslam/backend.h"
#include "myslam/config.h"
#include "myslam/feature.h"
#include "myslam/frontend.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/viewer.h"

namespace myslam {

Frontend::Frontend() {
    /*  特征检测器初始化：使用 OpenCV 的 GFTTDetector（Good Features to Track Detector）创建特征检测器
        Config::Get<int>("num_features")：从配置文件中读取特征点数量，例如150
        0.01：特征点的质量水平（最小可接受的特征点质量）。
        20：特征点之间的最小欧几里得距离。 */
    gftt_ = cv::GFTTDetector::create(Config::Get<int>("num_features"), 0.01, 20);

    // 初始化阶段所需的最小特征点数量。，从配置文件中读取对应值，例如50
    num_features_init_ = Config::Get<int>("num_features_init");

    // 正常跟踪阶段的特征点数量，例如50
    num_features_ = Config::Get<int>("num_features");
}

bool Frontend::AddFrame(myslam::Frame::Ptr frame) {
    // 输入帧设为当前帧
    current_frame_ = frame;

    switch (status_) {
        case FrontendStatus::INITING:
            StereoInit(); // 前端初始化
            break;
        case FrontendStatus::TRACKING_GOOD:
        case FrontendStatus::TRACKING_BAD:
            Track(); // GOOD 和 BAD 状态下调用 Track()，进行正常的跟踪处理
            break;
        case FrontendStatus::LOST:
            Reset(); // LOST 状态下调用 Reset()，尝试重置前端，在此代码中尚未实现
            break;
    }

    // 更新上一帧
    last_frame_ = current_frame_;
    return true;
}

bool Frontend::Track() {
    if (last_frame_) {
        current_frame_->SetPose(relative_motion_ * last_frame_->Pose());
    }

    int num_track_last = TrackLastFrame();
    tracking_inliers_ = EstimateCurrentPose();

    if (tracking_inliers_ > num_features_tracking_) {
        // tracking good
        status_ = FrontendStatus::TRACKING_GOOD;
    } else if (tracking_inliers_ > num_features_tracking_bad_) {
        // tracking bad
        status_ = FrontendStatus::TRACKING_BAD;
    } else {
        // lost
        status_ = FrontendStatus::LOST;
    }

    InsertKeyframe();
    relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();

    if (viewer_) viewer_->AddCurrentFrame(current_frame_);
    return true;
}

bool Frontend::InsertKeyframe() {
    if (tracking_inliers_ >= num_features_needed_for_keyframe_) {
        // still have enough features, don't insert keyframe
        return false;
    }
    // current frame is a new keyframe
    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(current_frame_);

    LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "
              << current_frame_->keyframe_id_;

    SetObservationsForKeyFrame();
    DetectFeatures();  // detect new features

    // track in right image
    FindFeaturesInRight();
    // triangulate map points
    TriangulateNewPoints();
    // update backend because we have a new keyframe
    backend_->UpdateMap();

    if (viewer_) viewer_->UpdateMap();

    return true;
}

void Frontend::SetObservationsForKeyFrame() {
    for (auto &feat : current_frame_->features_left_) {
        auto mp = feat->map_point_.lock();
        if (mp) mp->AddObservation(feat);
    }
}

int Frontend::TriangulateNewPoints() {
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    SE3 current_pose_Twc = current_frame_->Pose().inverse();
    int cnt_triangulated_pts = 0;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_left_[i]->map_point_.expired() &&
            current_frame_->features_right_[i] != nullptr) {
            // 左图的特征点未关联地图点且存在右图匹配点，尝试三角化
            std::vector<Vec3> points{
                camera_left_->pixel2camera(
                    Vec2(current_frame_->features_left_[i]->position_.pt.x,
                         current_frame_->features_left_[i]->position_.pt.y)),
                camera_right_->pixel2camera(
                    Vec2(current_frame_->features_right_[i]->position_.pt.x,
                         current_frame_->features_right_[i]->position_.pt.y))};
            Vec3 pworld = Vec3::Zero();

            if (triangulation(poses, points, pworld) && pworld[2] > 0) {
                auto new_map_point = MapPoint::CreateNewMappoint();
                pworld = current_pose_Twc * pworld;
                new_map_point->SetPos(pworld);
                new_map_point->AddObservation(
                    current_frame_->features_left_[i]);
                new_map_point->AddObservation(
                    current_frame_->features_right_[i]);

                current_frame_->features_left_[i]->map_point_ = new_map_point;
                current_frame_->features_right_[i]->map_point_ = new_map_point;
                map_->InsertMapPoint(new_map_point);
                cnt_triangulated_pts++;
            }
        }
    }
    LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
    return cnt_triangulated_pts;
}

int Frontend::EstimateCurrentPose() {
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
        LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // vertex
    VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(current_frame_->Pose());
    optimizer.addVertex(vertex_pose);

    // K
    Mat33 K = camera_left_->K();

    // edges
    int index = 1;
    std::vector<EdgeProjectionPoseOnly *> edges;
    std::vector<Feature::Ptr> features;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        auto mp = current_frame_->features_left_[i]->map_point_.lock();
        if (mp) {
            features.push_back(current_frame_->features_left_[i]);
            EdgeProjectionPoseOnly *edge =
                new EdgeProjectionPoseOnly(mp->pos_, K);
            edge->setId(index);
            edge->setVertex(0, vertex_pose);
            edge->setMeasurement(
                toVec2(current_frame_->features_left_[i]->position_.pt));
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber);
            edges.push_back(edge);
            optimizer.addEdge(edge);
            index++;
        }
    }

    // estimate the Pose the determine the outliers
    const double chi2_th = 5.991;
    int cnt_outlier = 0;
    for (int iteration = 0; iteration < 4; ++iteration) {
        vertex_pose->setEstimate(current_frame_->Pose());
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        cnt_outlier = 0;

        // count the outliers
        for (size_t i = 0; i < edges.size(); ++i) {
            auto e = edges[i];
            if (features[i]->is_outlier_) {
                e->computeError();
            }
            if (e->chi2() > chi2_th) {
                features[i]->is_outlier_ = true;
                e->setLevel(1);
                cnt_outlier++;
            } else {
                features[i]->is_outlier_ = false;
                e->setLevel(0);
            };

            if (iteration == 2) {
                e->setRobustKernel(nullptr);
            }
        }
    }

    LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
              << features.size() - cnt_outlier;
    // Set pose and outlier
    current_frame_->SetPose(vertex_pose->estimate());

    LOG(INFO) << "Current Pose = \n" << current_frame_->Pose().matrix();

    for (auto &feat : features) {
        if (feat->is_outlier_) {
            feat->map_point_.reset();
            feat->is_outlier_ = false;  // maybe we can still use it in future
        }
    }
    return features.size() - cnt_outlier;
}

int Frontend::TrackLastFrame() {
    // use LK flow to estimate points in the right image
    std::vector<cv::Point2f> kps_last, kps_current;
    for (auto &kp : last_frame_->features_left_) {
        if (kp->map_point_.lock()) {
            // use project point
            auto mp = kp->map_point_.lock();
            auto px =
                camera_left_->world2pixel(mp->pos_, current_frame_->Pose());
            kps_last.push_back(kp->position_.pt);
            kps_current.push_back(cv::Point2f(px[0], px[1]));
        } else {
            kps_last.push_back(kp->position_.pt);
            kps_current.push_back(kp->position_.pt);
        }
    }

    std::vector<uchar> status;
    Mat error;
    cv::calcOpticalFlowPyrLK(
        last_frame_->left_img_, current_frame_->left_img_, kps_last,
        kps_current, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    int num_good_pts = 0;

    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            cv::KeyPoint kp(kps_current[i], 7);
            Feature::Ptr feature(new Feature(current_frame_, kp));
            feature->map_point_ = last_frame_->features_left_[i]->map_point_;
            current_frame_->features_left_.push_back(feature);
            num_good_pts++;
        }
    }

    LOG(INFO) << "Find " << num_good_pts << " in the last image.";
    return num_good_pts;
}

bool Frontend::StereoInit() {
    int num_features_left = DetectFeatures(); // 在当前帧的左图中检测特征点，返回检测到的特征点数量
    int num_coor_features = FindFeaturesInRight(); // 在当前帧的右图中找到与左图特征点对应的特征点，返回找到的对应特征点数量

     // 如果找到的对应特征点数量小于初始化阶段所需的最小特征点数量，则初始化失败，返回 false
    if (num_coor_features < num_features_init_) {
        return false;
    }

    bool build_map_success = BuildInitMap();  // 构建初始地图，进行三角化等操作，如果成功则返回 true，否则返回 false

     // 如果成功构建初始地图，则将前端状态设置为 TRACKING_GOOD，并更新可视化界面中的当前帧和地图
    if (build_map_success) {
        status_ = FrontendStatus::TRACKING_GOOD;
        if (viewer_) { // 如果可视化界面存在，则将当前帧添加到界面中，并更新地图显示
            viewer_->AddCurrentFrame(current_frame_);
            viewer_->UpdateMap();
        }
        return true;
    }
    return false;
}

// 特征检测，检测并提取左图特征点，返回数量
int Frontend::DetectFeatures() {
    /* 1.创建掩膜（mask）： 
    *  - 掩膜的大小与当前帧左图的大小相同，类型为 CV_8UC1（单通道8位无符号整数），初始值为255（表示所有区域都可用）。
    *  - 遍历当前帧左图中的特征点，对于每个特征点，在掩膜上绘制一个以特征点为中心、边长为20像素的矩形区域，并将该区域的值设置为0（表示该区域不可用）。这样做的目的是为了在后续的特征检测过程中避免在已经存在特征点的附近再次检测到特征点，从而提高特征点的分布均匀性和质量。 
    * - pt 是 cv::KeyPoint 的一个成员变量，类型为 cv::Point2f。它表示特征点的二维坐标，存储为浮点数。例如，pt.x 和 pt.y 分别表示特征点的横坐标和纵坐标。*/

    cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);
    for (auto &feat : current_frame_->features_left_) {
        cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10), 
                      feat->position_.pt + cv::Point2f(10, 10), 0, CV_FILLED); //使用 CV_FILLED 参数将矩形区域填充为 0，表示该区域不可用
    }

    // 2. 特征点检测，使用opencv的GFTTDetector（Good Features to Track Detector）在当前帧的左图中检测特征点，先将检测到的特征点存储在一个 std::vector<cv::KeyPoint> 类型的变量 keypoints 中。检测过程中使用前面创建的掩膜（mask）来限制特征点的检测区域，确保在已经存在特征点的附近不会再次检测到特征点，从而提高特征点的质量和分布均匀性。
    std::vector<cv::KeyPoint> keypoints;
    gftt_->detect(current_frame_->left_img_, keypoints, mask);

    // 3.存储检测到的特征点：遍历检测到的特征点，将每个特征点封装成一个 Feature 对象，并将其添加到当前帧的 features_left_ 成员变量中。同时，统计检测到的特征点数量，并返回该数量。Feature 对象包含了特征点的位置、所属帧等信息，方便后续的跟踪和地图构建等操作。
    int cnt_detected = 0;
    for (auto &kp : keypoints) {
        current_frame_->features_left_.push_back(
            Feature::Ptr(new Feature(current_frame_, kp)));
        cnt_detected++;
    }

    LOG(INFO) << "Detect " << cnt_detected << " new features";
    return cnt_detected;
}

// 通过光流法（LK光流）在右图中找到与左图特征点对应的特征点
int Frontend::FindFeaturesInRight() {
    // use LK flow to estimate points in the right image

    /* 准备特征点数据
    *  - 遍历当前帧左图的特征点，将其坐标存储在 kps_left 向量中。
    *  - 如果特征点关联了地图点，则将地图点投影到右图像中，作为初始猜测，存储在 kps_right 向量中。
    *  - 如果特征点没有关联地图点，则在右图像中使用与左图像相同的像素坐标，存储在 kps_right 向量中。
    * - 扩展：地图点（MapPoint）是三维空间中的一个点，用于表示场景中的某个特征点在世界坐标系下的三维位置；包含 三维坐标：地图点在世界坐标系中的位置（如 Vec3 pos_）、观测信息：哪些图像帧中的哪些特征点观测到了该地图点等等；投影：将地图点的三维坐标通过相机模型投影到图像平面上，得到对应的二维像素坐标，这个过程通常涉及相机内参和当前帧的位姿信息。
    */
    std::vector<cv::Point2f> kps_left, kps_right;
    for (auto &kp : current_frame_->features_left_) {
        kps_left.push_back(kp->position_.pt);
        // 判断左图特征点是否关联了地图点，如果关联了地图点，则将地图点投影到右图像中，作为初始猜测；如果没有关联地图点，则在右图像中使用与左图像相同的像素坐标
        auto mp = kp->map_point_.lock(); // lock() 方法将 std::weak_ptr 转换为 std::shared_ptr，如果地图点存在，则返回一个有效的 std::shared_ptr；否则返回空指针
        if (mp) { // 说明当前特征点已经关联了一个地图点
            // use projected points as initial guess
            /* - mp->pos_：地图点的三维坐标
            *  - current_frame_->Pose()：当前帧的位姿（SE3类型），表示从世界坐标系到当前帧坐标系的变换。即为T_c_w，是左相机世界系 → 双目基线坐标系（左相机系）的变换。
            *  - camera_right_->world2pixel()：将地图点的三维坐标投影到右图像平面上，得到对应的二维像素坐标。这个函数内部会使用当前帧的位姿和相机内参来计算投影结果。
            */
            auto px =
                camera_right_->world2pixel(mp->pos_, current_frame_->Pose()); 
            
            // 将投影结果存储到 kps_right 向量中
            kps_right.push_back(cv::Point2f(px[0], px[1]));
        } else {
            // use same pixel in left image
            kps_right.push_back(kp->position_.pt);
        }
    }

    /* 光流计算 
    *  - cv::calcOpticalFlowPyrLK()：OpenCV函数，用于计算稀疏光流，即在两帧图像之间跟踪特征点的位置变化。参数说明：
    *  - current_frame_->left_img_：当前帧的左图像。
    *  - current_frame_->right_img_：当前帧的右图像。
    *  - kps_left：输入参数，包含当前帧左图像中特征点的二维坐标。
    *  - kps_right：输入/输出参数，初始时包含当前帧左图像中特征点的二维坐标，函数执行后更新为在右图像中对应特征点的二维坐标。   
    *  - status：输出参数，表示每个特征点的匹配状态（1 表示匹配成功，0 表示匹配失败）。
    *  - error：输出参数，表示每个特征点的匹配误差。
    * - cv::Size(11, 11)：搜索窗口的大小，表示在右图像中搜索特征点时考虑的邻域范围。
    * - 3：金字塔层数，表示在计算光流时使用图像金字塔的层数。
    * - cv::TermCriteria(...)：迭代终止条件，表示在计算光流时的迭代停止条件，这里设置为最大迭代次数为30，或者当误差小于0.01时停止迭代；cv::TermCriteria::COUNT + cv::TermCriteria::EPS 表示满足任一条件即可停止迭代。
    * - cv::OPTFLOW_USE_INITIAL_FLOW：标志位，表示在计算光流时使用初始特征点位置作为初始估计，这有助于提高匹配的准确性
    */
    std::vector<uchar> status;
    Mat error;
    cv::calcOpticalFlowPyrLK(
        current_frame_->left_img_, current_frame_->right_img_, kps_left,
        kps_right, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    /* 结果处理
    *  - 遍历 status 向量，检查每个特征点的匹配状态。如果 status[i] 为 1，表示第 i 个特征点在右图像中找到了对应的特征点。
    *  - 对于每个匹配成功的特征点，创建一个新的 Feature 对象，并将其位置设置为在右图像中对应特征点的坐标。
    *  - 将该 Feature 对象添加到当前帧的 features_right_ 成员变量中。
    *  - 统计匹配成功的特征点数量，并返回该数量。
    *  - 如果 status[i] == 0（匹配失败）：在右图特征点列表中添加 nullptr，表示该特征点没有匹配成功。
    */
    int num_good_pts = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            cv::KeyPoint kp(kps_right[i], 7); // kps_right[i] 是特征点的坐标。7 表示特征点的直径（大小），即特征点覆盖的区域大小为 7 像素
            Feature::Ptr feat(new Feature(current_frame_, kp));
            feat->is_on_left_image_ = false; // 标记该特征点位于右图像中
            current_frame_->features_right_.push_back(feat); 
            num_good_pts++;
        } else {
            current_frame_->features_right_.push_back(nullptr);
        }
    }
    LOG(INFO) << "Find " << num_good_pts << " in the right image.";
    return num_good_pts; // 返回在右图像中成功匹配的特征点数量
}

bool Frontend::BuildInitMap() {
    // 存储左相机和右相机的位姿（SE3 类型），用于三角化。
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    // 统计成功三角化的地图点数量
    size_t cnt_init_landmarks = 0;
    // 遍历当前帧左图中的所有特征点
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_right_[i] == nullptr) continue; // 如果对应的右图特征点为空（即没有匹配成功），跳过该特征点 

        // 将camera_left和camera_right的像素坐标转换为相机坐标；
        // create map point from triangulation
        std::vector<Vec3> points{
            // camera_left_->pixel2camera 和 camera_right_->pixel2camera 将像素坐标转换为归一化相机坐标（假设深度为1）
            camera_left_->pixel2camera(
                Vec2(current_frame_->features_left_[i]->position_.pt.x,
                     current_frame_->features_left_[i]->position_.pt.y)),
            camera_right_->pixel2camera(
                Vec2(current_frame_->features_right_[i]->position_.pt.x,
                     current_frame_->features_right_[i]->position_.pt.y))};
        Vec3 pworld = Vec3::Zero();

        // 三角化triangulation()生成地图点
        if (triangulation(poses, points, pworld) && pworld[2] > 0) { // 三角化函数执行成功返回true 并且 pworld[2]>0是在检查三角化生成的点是否位于相机前方（深度值为正）
            // 创建并存储地图点
            auto new_map_point = MapPoint::CreateNewMappoint(); // 创建一个新的地图点对象
            new_map_point->SetPos(pworld); // 设置地图点的三维位置
                // 将左图和右图的特征点与地图点关联：
                    // 地图点记录观测：通过 AddObservation 方法，地图点记录了哪些特征点观测到了它。
                    // 特征点关联地图点：通过设置 map_point_，特征点记录了它对应的地图点。
            new_map_point->AddObservation(current_frame_->features_left_[i]);
            new_map_point->AddObservation(current_frame_->features_right_[i]);
            current_frame_->features_left_[i]->map_point_ = new_map_point;
            current_frame_->features_right_[i]->map_point_ = new_map_point;
            cnt_init_landmarks++; // 统计成功生成的地图点数量+1
                // 将地图点插入到全局地图中
            map_->InsertMapPoint(new_map_point);
        }
    }
    // 设置关键帧并更新地图
    current_frame_->SetKeyFrame(); // 将当前帧标记为关键帧
    map_->InsertKeyFrame(current_frame_);  // 将关键帧插入到全局地图中
    backend_->UpdateMap();  // 通知后端更新地图

    // 输出日志，记录成功生成的初始化地图点的数量
    LOG(INFO) << "Initial map created with " << cnt_init_landmarks
              << " map points";

    return true;
}

bool Frontend::Reset() {
    LOG(INFO) << "Reset is not implemented. "; // 尚未实现重置定位的逻辑
    return true;
}

}  // namespace myslam