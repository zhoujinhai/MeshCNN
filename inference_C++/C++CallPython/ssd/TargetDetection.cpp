#include <Python.h> 
#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <numpy/arrayobject.h> 
#include <time.h>

static void help()
{
	std::cout << std::endl;
	std::cout << "This sample demostrates MobileNet-V2-SSDLite detection with tensorflow server inference." << std::endl;
	std::cout << "Call" << std::endl;
}


int main(int argc, char* argv[])
{   
    if (argc != 2) 
	{
		help();
	}
    

    Py_Initialize();                // 初始化 Python 环境
    if (!Py_IsInitialized())
    {
        std::cout << "init faild ..." << std::endl; 
    }
    
    import_array(); 

    // 如果查找函数文件一直是 nullptr 则加上下面两行路径
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('/home/xm/project_ssd/build/')");
    

    PyObject* pModule = nullptr;                               //.py文件  
    pModule = PyImport_ImportModule("inferencePb");            //调用上述路径下的inferencePb.py文件
    if (pModule == nullptr)
	{
        std::cout << "don't find the python file!" << std::endl;
        return -1;
	}
    
    clock_t start, end;
    for (int i=0;i<100;i++)
    {
    start = clock();
    // 这里用视频流替换传入的图像参数
    std::string image = argv[1];
    cv::Mat img = cv::imread(image);
    if (img.empty())
    {
        std::cout << "could not load image ..." << std::endl;
        return -1;
    }

    int m, n;
    n = img.cols *3;
    m = img.rows;


    unsigned char *data = (unsigned char*)malloc(sizeof(unsigned char) * m * n);
    int p = 0;
    for (int i = 0; i < m;i++)
    {
        for (int j = 0; j < n; j++)
        {
            data[p]= img.at<unsigned char>(i, j);
            p++;
        }
    }

    npy_intp Dims[2] = { m,n };  //图像的维度信息

