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
 * 本节演示了如何根据data/目录下的十张图训练字典
 * ************************************************/

int main( int argc, char** argv ) {
    // read the image 
    cout<<"reading images... "<<endl;
    vector<Mat> images; 
    for ( int i=0; i<10; i++ )
    {
        string path = "./data/"+to_string(i+1)+".png";
        images.push_back( imread(path) );
    }
    // detect ORB features
    cout<<"detecting ORB features ... "<<endl;
    Ptr< Feature2D > detector = ORB::create(); // 创建 ORB 特征检测器，默认参数
    vector<Mat> descriptors; // 存储所有图像的描述子
    for ( Mat& image:images )
    {
        vector<KeyPoint> keypoints; // 存储关键点
        Mat descriptor; // 存储描述子，存储单个图像的描述子
        detector->detectAndCompute( image, Mat(), keypoints, descriptor ); // 检测关键点并计算描述子
        descriptors.push_back( descriptor ); // 将单个图像的描述子加入到描述子集合中
    }
    
    // create vocabulary 
    cout<<"creating vocabulary ... "<<endl;
    DBoW3::Vocabulary vocab;  // DBoW3 词汇对象，用于存储视觉词典；内部会把这些描述子聚类构建视觉单词（通常通过层次 k-means）。
    vocab.create( descriptors ); // 训练字典，默认参数：10层，5个子节点
    cout<<"vocabulary info: "<<vocab<<endl; 
    vocab.save( "vocabulary.yml.gz" ); // 保存字典文件到vocabulary.yml.gz 保存为压缩 YAML，便于后续加载。
    cout<<"done"<<endl;
    
    return 0;
}