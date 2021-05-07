#include <iostream>
#include <string>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include "undistort.hpp"


void undistort::undistort_an_image (ImageMeta *image_data, ImageMeta *camera_matrix_data, ImageMeta *distortion_coefficients_data, bool remap_flag, bool resize_flag){
    /*
    畸变矫正函数。每拍摄一张图片都需要校正一次。
    Args:
        image:                      待校正图片
        camera_matrix:              相机内参矩阵
        distortion_coefficients:    畸变系数
        remap_flag:                 是否使用remapping的方式来做畸变校正。通常使用默认值即可。
        resize_flag:                是否将校正后的图片归一化到原图大小。通常使用默认值即可。
    Returns:
        undistort_img： 畸变校正后的图片
    */
    cv::Mat image = cv::Mat::zeros(cv::Size(image_data->w,image_data->h),CV_8UC3);
    cv::Mat camera_matrix = cv::Mat::zeros(cv::Size(camera_matrix_data->w,camera_matrix_data->h),CV_32FC1);
    cv::Mat distortion_coefficients = cv::Mat::zeros(cv::Size(distortion_coefficients_data->w,distortion_coefficients_data->h),CV_32FC1);
    
    image.data = image_data->data;
    camera_matrix.data = camera_matrix_data->data;
    distortion_coefficients.data = distortion_coefficients_data->data;

    cv::Mat undistort_img;
    cv::Mat res;
    if (remap_flag) undistort_img = undistort_using_remapping(image, camera_matrix, distortion_coefficients);
    else undistort_img = undistort_direct(image, camera_matrix, distortion_coefficients);
    if (resize_flag) cv::resize(undistort_img, res, cv::Size(image.cols, image.rows));
    memcpy(image_data->data, res.data, res.cols*res.rows);
}

cv::Mat undistort::undistort_direct (cv::Mat img, cv::Mat mtx, cv::Mat dist){
    /*畸变校正方法1，直接调用opencv自带的函数。*/ 
    int h = img.rows;
    int w = img.cols;
    cv::Rect roi;
    // Get optimal new camera matrix
    cv::Mat newcameramtx = cv::getOptimalNewCameraMatrix(mtx, dist, cv::Size(w, h), 1, cv::Size(w, h), &roi);
    cv::Mat dst;
    // undistort
    cv::undistort(img, dst, mtx, dist, newcameramtx);
    // crop the image
    dst = dst(roi);

    return dst;
}

cv::Mat undistort::undistort_using_remapping (cv::Mat img, cv::Mat mtx, cv::Mat dist){
    /*畸变校正方法2， 用remapping的方式，速度比方法1快。*/
    int h = img.rows;
    int w = img.cols;
    cv::Rect roi;
    // Get optimal new camera matrix
    cv::Mat newcameramtx = cv::getOptimalNewCameraMatrix(mtx, dist, cv::Size(w, h), 1, cv::Size(w, h), &roi);
    //std::cout<<"ROI "<<*roi.x<<std::endl;
    cv::Mat mapx, mapy;
    // undistort
    cv::Mat R;
    cv::initUndistortRectifyMap(mtx, dist, R, newcameramtx, cv::Size(w, h), 5, mapx, mapy);
    cv::Mat dst;
    cv::remap(img, dst, mapx, mapy, cv::INTER_LINEAR);
    // crop the image
    dst = dst(roi); 
    return dst;
}


extern "C" {
	undistort und;	 
    void undistort_an_image (ImageMeta *image_data, ImageMeta *camera_matrix_data, ImageMeta *distortion_coefficients_data, bool remap_flag=true, bool resize_flag=true){
        und.undistort_an_image (image_data, camera_matrix_data, distortion_coefficients_data, remap_flag, resize_flag);	
	}
}


// int main(){
//     std::string dataset = "/home/blin/Downloads/atesi项目代码/atesi_camera_calibration-master/images/12004110343219_A1H.jpg";
//     std::string camera_matrix_file = "./camera_mtx.pkl";

//     cv::Mat mtx = (cv::Mat_<float>(3, 3) << 9.69860906e+03, 0.00000000e+00,8.96459030e+02, 0.00000000e+00, 9.43262275e+03, 7.56590679e+02, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00);
//     cv::Mat dist = (cv::Mat_<float>(1, 5) << -5.94787534e+00, 3.61189890e+02,-1.06051936e-02, 4.87760167e-02, -1.11960999e+04);

//     cv::Mat img = cv::imread(dataset);
//     std::cout<<mtx<< dist<<std::endl;

//     undistort uds;
//     time_t first, second;  
//     first=time(NULL);  
//     // clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
//     struct timeval t1,t2;
//     double timeuse;

//     gettimeofday(&t1,NULL);

//     int n = 100;
//     while(n>1){
//         cv::Mat res = uds.undistort_an_image(img, mtx, dist, true, true);
//         cv::imwrite("/home/blin/Downloads/atesi项目代码/atesi_camera_calibration-master/res.jpg", res);
//         n--;
//         //std::cout<<"\n此程序的运行到"<<n<<"次！"<<std::endl;

//         gettimeofday(&t2,NULL);
//         timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
//         std::cout<<"time = "<<timeuse<<std::endl;  //输出时间（单位：ｓ）

//     }   
// }
