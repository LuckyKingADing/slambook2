cmake_minimum_required(VERSION 2.8)
project(myslam)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fopenmp -pthread")

# 核心修复：强制指定依赖路径（解决头文件找不到）
# 1. OpenCV 路径（适配 Ubuntu 系统的 OpenCV 4.x）
set(OpenCV_DIR /usr/lib/cmake/opencv4)
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 2. Sophus 路径（旧版 Sophus 头文件路径）
include_directories(/usr/local/include/sophus)

# 3. 其他依赖基础路径
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
include_directories("/usr/include/eigen3")
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories("/usr/local/include")
include_directories("/usr/local/include/pangolin")

# 输出路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# 链接库（补充完整依赖，解决链接问题）
set(THIRD_PARTY_LIBS
    # OpenCV 核心库
    ${OpenCV_LIBS}
    # Sophus（旧版）
    Sophus
    # Pangolin
    pangolin
    # G2O
    g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension
    # 日志/命令行参数
    glog gflags
    # 系统库
    pthread
    cxsparse
    # 补充 g2o 依赖的库
    cholmod
)

# 编译
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(app)
enable_testing()