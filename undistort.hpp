#ifndef LOAD_DYLIB_TEST_IMGSEGMENTATION_HPP
#define LOAD_DYLIB_TEST_IMGSEGMENTATION_HPP
#endif //LOAD_DYLIB_TEST_IMGSEGMENTATION_HPP
#include <opencv2/opencv.hpp>

typedef struct ImageBase {
    int w;                   //图像的宽
    int h;                   //图像的高
    int c;                   //通道数
    unsigned char *data;     //指针 指向数据
};

typedef ImageBase ImageMeta;

class undistort{
    public:
        void undistort_an_image (ImageMeta *image_data, ImageMeta *camera_matrix_data, ImageMeta *distortion_coefficients_data, bool remap_flag=true, bool resize_flag=true);
    private:
        cv::Mat undistort_direct (cv::Mat img, cv::Mat mtx, cv::Mat dist);
        cv::Mat undistort_using_remapping (cv::Mat img, cv::Mat mtx, cv::Mat dist);
};
