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

#include "myslam/mappoint.h"
#include "myslam/feature.h"

namespace myslam {

MapPoint::MapPoint(long id, Vec3 position) : id_(id), pos_(position) {}

MapPoint::Ptr MapPoint::CreateNewMappoint() {
    /* factory_id 是一个静态变量，用于生成唯一的 ID。
       每次调用该函数时，都会创建一个新的 MapPoint 对象，并将其 ID 设置为当前的 factory_id 值，然后将 factory_id 自增。*/
    static long factory_id = 0; 
    MapPoint::Ptr new_mappoint(new MapPoint);
    new_mappoint->id_ = factory_id++;
    return new_mappoint;
}

void MapPoint::RemoveObservation(std::shared_ptr<Feature> feat) {
    // 加锁以保护数据，防止多线程访问冲突
    std::unique_lock<std::mutex> lck(data_mutex_);

    // 遍历 observations_，找到与给定特征点 feat 对应的观测
    for (auto iter = observations_.begin(); iter != observations_.end();
         iter++) {
        if (iter->lock() == feat) { // 如果找到匹配的观测
            observations_.erase(iter); // 从观测列表中移除该观测
            feat->map_point_.reset(); // 将特征点的地图点关联重置为空：reset()函数的主要操作如下： H = Matrix6d::Zero();b = Vector6d::Zero(); cost = 0;
            observed_times_--; // 观测次数减 1
            break; // 退出循环
        }
    }
}

}  // namespace myslam
