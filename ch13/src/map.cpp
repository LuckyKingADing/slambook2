/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "myslam/map.h"
#include "myslam/feature.h"

namespace myslam {

void Map::InsertKeyFrame(Frame::Ptr frame) {
    // 将当前帧设置为当前关键帧
    current_frame_ = frame;

    // 检查当前帧是否已经存在于关键帧集合中
    if (keyframes_.find(frame->keyframe_id_) == keyframes_.end()) {
        // 如果不存在，将其插入到关键帧集合和活跃关键帧集合中
        keyframes_.insert(make_pair(frame->keyframe_id_, frame));
        active_keyframes_.insert(make_pair(frame->keyframe_id_, frame));
    } else {
        // 如果已经存在，更新关键帧集合和活跃关键帧集合中的帧
        keyframes_[frame->keyframe_id_] = frame;
        active_keyframes_[frame->keyframe_id_] = frame;
    }

    // 如果活跃关键帧的数量超过了预设的最大值，移除最旧的关键帧
    if (active_keyframes_.size() > num_active_keyframes_) {
        RemoveOldKeyframe();
    }
}

void Map::InsertMapPoint(MapPoint::Ptr map_point) {
    /*  检查地图点是否已经存在于 landmarks_ 中。
        如果不存在，则将id和map_point作为一对插入到 landmarks_ 和 active_landmarks_ 中。
        如果已经存在，则更新 landmarks_ 和 active_landmarks_ 中的地图点。 */
    if (landmarks_.find(map_point->id_) == landmarks_.end()) {
        landmarks_.insert(make_pair(map_point->id_, map_point));
        active_landmarks_.insert(make_pair(map_point->id_, map_point));
    } else {
        landmarks_[map_point->id_] = map_point;
        active_landmarks_[map_point->id_] = map_point;
    }
}

void Map::RemoveOldKeyframe() {
    // 如果当前帧为空，直接返回
    if (current_frame_ == nullptr) return;

    // 初始化变量，用于寻找与当前帧最近和最远的关键帧
    double max_dis = 0, min_dis = 9999; // 最大距离和最小距离，最大距离从0开始，只会被更大的值更新，因此最终会记录最大的距离，最小距离从一个很大的数开始，只会被更小的值更新，因此最终会记录最小的距离
    double max_kf_id = 0, min_kf_id = 0; // 最大距离和最小距离对应的关键帧ID
    auto Twc = current_frame_->Pose().inverse(); // 当前帧的世界坐标系位姿，求逆得到从当前帧相机坐标系到世界坐标系的变换

    // 遍历活跃关键帧，计算每个关键帧与当前帧的距离
    for (auto& kf : active_keyframes_) { // kf.first是关键帧ID，kf.second是关键帧指针对象
        if (kf.second == current_frame_) continue; // 跳过当前帧

        /*  kf.second->Pose()：获取关键帧的位姿（从世界坐标系到该关键帧坐标系的变换）
        *   Twc：当前帧的世界坐标系位姿（从当前帧相机坐标系到世界坐标系的变换）
        *   kf.second->Pose() * Twc：计算当前关键帧到当前帧的相对位姿变换
        *   .log()：将位姿变换（SE3）转换为李代数（se3）
        *   .norm()：计算李代数的范数，表示位姿之间的距离
        *   通过比较距离，更新最大距离和最小距离以及对应的关键帧ID。 */
        auto dis = (kf.second->Pose() * Twc).log().norm(); // 计算位姿之间的距离
        if (dis > max_dis) {
            max_dis = dis;
            max_kf_id = kf.first;
        }
        if (dis < min_dis) {
            min_dis = dis;
            min_kf_id = kf.first;
        }
    }

    const double min_dis_th = 0.2;  // 最近距离阈值
    Frame::Ptr frame_to_remove = nullptr;
    if (min_dis < min_dis_th) {
        // 如果 min_dis 小于阈值 min_dis_th，说明存在非常接近当前帧的关键帧，优先移除最近的关键帧（min_kf_id）
        frame_to_remove = keyframes_.at(min_kf_id); // at() 是 C++ 标准库中 std::map 和 std::unordered_map 的成员函数，用于根据键（key）访问对应的值（value）。 在这里，keyframes_ 是一个 std::map 或 std::unordered_map，存储了关键帧的 ID 和对应的关键帧对象。通过 keyframes_.at(min_kf_id)，我们可以获取到 ID 为 min_kf_id 的关键帧对象，并将其赋值给 frame_to_remove 变量，以便后续进行移除操作。
    } else {
        // 否则，移除与当前帧距离最远的关键帧（max_kf_id
        frame_to_remove = keyframes_.at(max_kf_id);
    }

    LOG(INFO) << "remove keyframe " << frame_to_remove->keyframe_id_;

    // 从活跃关键帧中移除选定的关键帧，erase()移除
    active_keyframes_.erase(frame_to_remove->keyframe_id_);

    // 移除该关键帧的所有左图特征点的观测
        // 遍历该关键帧的所有左图特征点（features_left_）。如果特征点关联了地图点（map_point_），则调用 RemoveObservation 方法，移除该特征点对地图点的观测
    for (auto feat : frame_to_remove->features_left_) {
        auto mp = feat->map_point_.lock(); // lock()将weak_ptr转换为shared_ptr，如果地图点存在，则返回一个有效的shared_ptr；否则返回空指针
        if (mp) {
            mp->RemoveObservation(feat);
        }
    }

    // 移除该关键帧的所有右图特征点的观测。同理
    for (auto feat : frame_to_remove->features_right_) {
        if (feat == nullptr) continue;
        auto mp = feat->map_point_.lock();
        if (mp) {
            mp->RemoveObservation(feat);
        }
    }

    // 清理地图，移除未被观测的地图点，也就是清除地图中那些没有任何特征点观测到的地图点。这些地图点可能是由于之前移除的关键帧导致的，或者是由于其他原因导致的无效地图点。通过调用 CleanMap() 方法，可以确保地图中只保留那些被至少一个特征点观测到的有效地图点，从而提高地图的质量和效率。
    CleanMap();
}

void Map::CleanMap() {
    // 统计被移除的地图点数量
    int cnt_landmark_removed = 0;

    // 遍历活跃地图点集合
    for (auto iter = active_landmarks_.begin();
         iter != active_landmarks_.end();) {
        // 如果地图点的观测次数为 0，将其从活跃地图点集合中移除
        if (iter->second->observed_times_ == 0) {
            iter = active_landmarks_.erase(iter); // 移除地图点
            cnt_landmark_removed++; // 计数器加 1
        } else {
            ++iter; // 否则继续遍历下一个地图点
        }
    }

    // 打印日志，记录移除的地图点数量
    LOG(INFO) << "Removed " << cnt_landmark_removed << " active landmarks";
}

}  // namespace myslam
