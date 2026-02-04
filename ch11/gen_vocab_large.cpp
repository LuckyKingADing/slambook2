#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

/***************************************************
 * 本文件旨在演示如何训练一个更大的字典 (Vocabulary)
 * 
 * 相比于 feature_training.cpp 只用 10 张图，
 * 本程序读取一个数据集（TUM 格式）的所有 RGB 图像，
 * 提取所有图像的特征，并利用这些海量特征训练一个更大、
 * 更通用的 K 叉树字典。
 * 
 * 使用方法: ./gen_vocab_large [dataset_path]
 * 例如: ./gen_vocab_large /home/data/TUM_RGBD/rgbd_dataset_freiburg1_desk
 * 注意: 数据集目录下需要有一个 associate.txt 文件，用于对齐 RGB 和深度图
 * ************************************************/
int main( int argc, char** argv )
{
    // -----------------------------------------------------------------------
    // 1. 读取关联文件 (associate.txt)
    // -----------------------------------------------------------------------
    // 从命令行参数获取数据集路径。如果没有参数，程序会崩溃，实际使用应加判断。
    string dataset_dir = argv[1];
    // associate.txt 是 TUM 数据集经过 associate.py 处理后生成的文件，
    // 每一行包含了对齐后的 RGB 时间戳、RGB 路径、深度图时间戳、深度图路径。
    ifstream fin ( dataset_dir+"/associate.txt" );
    if ( !fin )
    {
        cout<<"please generate the associate file called associate.txt!"<<endl;
        return 1;
    }

    // -----------------------------------------------------------------------
    // 2. 解析文件路径和时间戳
    // -----------------------------------------------------------------------
    vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;
    while ( !fin.eof() )
    {
        string rgb_time, rgb_file, depth_time, depth_file; // rgb_file: RGB 图像文件名 depth_file: 深度图文件名
        // 依次读取一行中的四个数据：time_rgb, file_rgb, time_depth, file_depth
        fin>>rgb_time>>rgb_file>>depth_time>>depth_file;
        
        // 转换并存入 vector
        rgb_times.push_back ( atof ( rgb_time.c_str() ) );
        depth_times.push_back ( atof ( depth_time.c_str() ) );
        rgb_files.push_back ( dataset_dir+"/"+rgb_file );
        depth_files.push_back ( dataset_dir+"/"+depth_file );

        // 检查流状态，如果是最后一行（可能包含空行），则退出
        if ( fin.good() == false )
            break;
    }
    fin.close();
    
    // -----------------------------------------------------------------------
    // 3. 提取所有图像的 ORB 特征
    // -----------------------------------------------------------------------
    cout<<"generating features ... "<<endl;
    // descriptors 存放所有图片的所有描述子，这是一个庞大的集合。
    // 比如 1000 张图，每张 500 个点，那这里就有 50 万个描述子。
    vector<Mat> descriptors;
    Ptr< Feature2D > detector = ORB::create();
    int index = 1;
    for ( string rgb_file:rgb_files )
    {
        Mat image = imread(rgb_file);
        vector<KeyPoint> keypoints; 
        Mat descriptor;
        detector->detectAndCompute( image, Mat(), keypoints, descriptor );
        // 将这一张图的特征描述子加入到大集合中
        descriptors.push_back( descriptor );
        cout<<"extracting features from image " << index++ <<endl;
    }
    // 估算一下提取的总特征数（这行代码假设每张图提取了500个，实际ORB默认就是500）
    cout<<"extract total "<<descriptors.size()*500<<" features."<<endl;
    
    // -----------------------------------------------------------------------
    // 4. 创建并训练字典 (K-means Clustering)
    // -----------------------------------------------------------------------
    cout<<"creating vocabulary, please wait ... "<<endl;
    // 创建一个空的字典对象。
    // 默认参数：k=10 (分支数), d=5 (深度), weighting=TF_IDF, scoring=L1_NORM
    DBoW3::Vocabulary vocab;
    
    // 核心步骤：训练字典。
    // 这一步会非常慢，因为要对几十万个特征向量进行递归 K-means 聚类。
    // 它会构建出 K 叉树的结构，并计算每个叶子节点的 IDF 值。
    vocab.create( descriptors );
    
    cout<<"vocabulary info: "<<vocab<<endl;
    
    // -----------------------------------------------------------------------
    // 5. 保存字典
    // -----------------------------------------------------------------------
    // 将训练好的字典保存为压缩文件，供后续 loop_closure.cpp 或实际 SLAM 系统加载使用。
    vocab.save( "vocab_larger.yml.gz" );
    cout<<"done"<<endl;
    
    return 0;
}