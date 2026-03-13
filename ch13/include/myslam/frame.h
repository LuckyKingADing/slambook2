#pragma once

#ifndef MYSLAM_FRAME_H
#define MYSLAM_FRAME_H

#include "myslam/camera.h"
#include "myslam/common_include.h"

namespace myslam {

// forward declare
struct MapPoint;
struct Feature;

/**
 * 帧
 * 每一帧分配独立id，关键帧分配关键帧ID
 */
struct Frame {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW; // Eigen内存对齐
    typedef std::shared_ptr<Frame> Ptr; // Frame的智能指针

    unsigned long id_ = 0;           // id of this frame
    unsigned long keyframe_id_ = 0;  // id of key frame
    bool is_keyframe_ = false;       // 是否为关键帧
    double time_stamp_;              // 时间戳，暂不使用
    SE3 pose_;                       // Tcw 形式Pose
    std::mutex pose_mutex_;          // Pose数据锁
    cv::Mat left_img_, right_img_;   // stereo images

    // extracted features in left image;在左图提取的特征点
    std::vector<std::shared_ptr<Feature>> features_left_;
    // corresponding features in right image, set to nullptr if no corresponding;在右图中找到与左图特征点对应的特征点，如果没有对应特征点则设置为nullptr
    std::vector<std::shared_ptr<Feature>> features_right_;

   public:  // data members
    // 构造函数，默认构造函数
    Frame() {}

    // 构造函数，输入id, 时间戳, 位姿, 左右图像
    Frame(long id, double time_stamp, const SE3 &pose, const Mat &left,
          const Mat &right);

    // 获取位姿，函数内加锁保证线程安全
    SE3 Pose() {    
        std::unique_lock<std::mutex> lck(pose_mutex_);
        return pose_;
    }

    // 设置位姿，函数内加锁保证线程安全
    void SetPose(const SE3 &pose) {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        pose_ = pose;
    }

    /// 设置关键帧并分配关键帧id
    void SetKeyFrame();

    /// 工厂构建模式，分配id 
    static std::shared_ptr<Frame> CreateFrame();
};

}  // namespace myslam

#endif  // MYSLAM_FRAME_H
