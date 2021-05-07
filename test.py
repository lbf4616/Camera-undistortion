from ctypes import *
from io import BytesIO
import numpy as np
import cv2
 
# 这个写法是看到yolo的darknet.py里看到的，学以致用一下
def c_array(ctype, values):                                      # 把图像的数据转化为内存连续的列表使c++能使用这块内存 
    arr = (ctype * len(values))()
    arr[:] = values
    return arr

def array_to_image(arr):    
    c = arr.shape[2]
    h = arr.shape[0]
    w = arr.shape[1]
    arr = arr.flatten()
    data = c_array(c_uint8, arr)
    im = IMAGE(w, h, c, data)
    return im

def array_to_image_f(arr):    
    c = arr.shape[2]
    h = arr.shape[0]
    w = arr.shape[1]
    arr = arr.flatten()
    data = c_array(c_float, arr)
    im = IMAGE_f(w, h, c, data)
    return im

class IMAGE(Structure):                                           # 这里和ImgSegmentation.hpp里面的结构体保持一致。
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_uint8))]

class IMAGE_f(Structure):                                           # 这里和ImgSegmentation.hpp里面的结构体保持一致。
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]
    
 
img = cv2.imread('/home/blin/Downloads/atesi项目代码/atesi_camera_calibration-master/images/12004110343219_A1H.jpg')
mtx = [9698.60906, 0.0, 896.459030, 0.0, 9432.62275, 756.590679, 0.0, 0.0, 1.0]


dist = [-5.94787534, 361.189890,-0.0106051936, 0.0487760167, -11196.0999]
mtx = np.array(mtx)
mtx = np.resize(mtx, [3,3])

dist = np.array(dist)
dist = np.resize(dist, [1,5])
h, w, c = img.shape[0], img.shape[1], img.shape[2]

dist = dist[ : , :, np.newaxis]
mtx = mtx[ : , :, np.newaxis]
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)*0
# gray = np.reshape(gray, (h, w, 1))                                # 一定要使用(h, w, 1)，最后的1别忘。
input_img = array_to_image(img)
mtx = array_to_image_f(mtx)
dist = array_to_image_f(dist)

lib = CDLL('/home/blin/Desktop/Projects/Cpp/hellow/undistort.so')    # 读取动态库文件
undistort = lib.undistort_an_image                                        # 这个就是我们的函数名
undistort.argtypes = [POINTER(IMAGE), POINTER(IMAGE_f), POINTER(IMAGE_f), c_bool, c_bool]               # 设置函数参数
undistort(input_img, mtx, dist, True, True)                                             # 执行函数，这里直接修改gray_img的内存数据

y = input_img.data                                                 # 获取data
array_length = h*w*c

buffer_from_memory = pythonapi.PyMemoryView_FromMemory            # 这个是python 3的使用方法
buffer_from_memory.restype = py_object
buffer = buffer_from_memory(y, array_length)
img = np.frombuffer(buffer, dtype=np.uint8)
img = np.reshape(img, (h, w, c))

cv2.imshow('test', img)
cv2.waitKey(0)