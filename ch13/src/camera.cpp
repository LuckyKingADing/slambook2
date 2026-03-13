#include "myslam/camera.h"

namespace myslam {

Camera::Camera() {
}

/* Camera::world2camera() 函数的实现
*  - 输入参数 p_w 是一个三维向量，表示世界坐标系下的三维点；
*  - 输入参数 T_c_w 是一个 SE3 类型的变换，表示从世界坐标系w系到相机坐标系c系的变换。
*  - 函数的目的是将世界坐标系下的三维点转换为相机坐标系下的三维点。
*  - 函数内部使用相机的外参（pose_）和输入的变换 T_c_w 来进行坐标转换。
*  - 公式：p_c = pose_ * T_c_w * p_w  左相机系点 → 右相机系点（叠加基线平移）
*  - 其中，pose_ 是相机的外参，相当于基线平移，pose_ 是「双目基线坐标系（左相机系）→ 当前相机（左 / 右）坐标系」的变换，左相机调用 world2camera → 返回左相机系点；右相机调用 world2camera → 返回右相机系点。
*  - 具体到代码实现中，先将left图像的世界坐标点p_w通过T_c_w变换到相机坐标系下，然后再通过pose_将立体相机坐标系下的点转换到单目相机坐标系下，最终得到p_c，即右相机坐标系下的三维点。
*/
Vec3 Camera::world2camera(const Vec3 &p_w, const SE3 &T_c_w) {
    return pose_ * T_c_w * p_w; // 得到的是右相机right_camera坐标系下的相机坐标三维点，再通过 camera2pixel 投影为右图像的二维像素坐标（px[0], px[1]）
}

Vec3 Camera::camera2world(const Vec3 &p_c, const SE3 &T_c_w) {
    return T_c_w.inverse() * pose_inv_ * p_c;
}

Vec2 Camera::camera2pixel(const Vec3 &p_c) {
    return Vec2(
            fx_ * p_c(0, 0) / p_c(2, 0) + cx_, // 因为是矩阵，所以用(0,0)(1,0)(2,0)来访问元素，p_c是c系下的x坐标、y坐标、z坐标(深度值)
            fy_ * p_c(1, 0) / p_c(2, 0) + cy_
    );
}

/* Camera::pixel2camera() 函数的实现
*  - 输入参数 p_p 是一个二维向量，表示像素坐标；
*  - 输入参数 depth 是一个标量，表示深度值，默认为1。
*  - 函数的目的是将像素坐标pixel转换为相机坐标系camera下的三维坐标。
*  - 函数内部使用相机的内参（fx, fy, cx, cy）来进行坐标转换。
*  - 公式：
*      X = (u - cx) * Z / fx
*      Y = (v - cy) * Z / fy
*      Z = depth
*/
Vec3 Camera::pixel2camera(const Vec2 &p_p, double depth) {
    return Vec3(
            (p_p(0, 0) - cx_) * depth / fx_,
            (p_p(1, 0) - cy_) * depth / fy_,
            depth
    );
}

/* Camera::world2pixel() 函数的实现
*  - 输入参数 p_w 是一个三维向量，表示世界坐标系下的三维点；
*  - 输入参数 T_c_w 是一个 SE3 类型的变换，表示从世界坐标系w系到相机坐标系c系的变换。
*  - 函数的目的是将世界坐标系下的三维点转换为像素坐标pixel。
*  - 函数内部首先调用 world2camera() 将世界坐标系下的三维点转换为相机坐标系下的三维点，然后调用 camera2pixel() 将相机坐标系下的三维点转换为像素坐标。
*/
Vec2 Camera::world2pixel(const Vec3 &p_w, const SE3 &T_c_w) {
    return camera2pixel(world2camera(p_w, T_c_w));
}

/* Camera::pixel2world() 函数的实现
*  - 输入参数 p_p 是一个二维向量，表示像素坐标；
*  - 输入参数 T_c_w 是一个 SE3 类型的变换，表示从世界坐标系到相机坐标系的变换；
*  - 输入参数 depth 是一个标量，表示深度值，默认为1。
*  - 函数的目的是将像素坐标pixel转换为世界坐标系world下的三维坐标。
*  - 函数内部首先调用 pixel2camera() 将像素坐标转换为相机坐标系下的三维坐标，然后调用 camera2world() 将相机坐标系下的三维坐标转换为世界坐标系下的三维坐标。 
*/
Vec3 Camera::pixel2world(const Vec2 &p_p, const SE3 &T_c_w, double depth) {
    return camera2world(pixel2camera(p_p, depth), T_c_w);
}

}
