#include <stdio.h>
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/features2d.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <tesseract\baseapi.h>
#include <leptonica\allheaders.h>
#include <math.h> 
using namespace cv;
using namespace std;

class TwoValue{
private:
	Mat twoValue;
	Mat gray;
	Mat src;
	int weight;
public:
	TwoValue(Mat m){
		weight = m.at<Vec3b>(60, 75)[0] + m.at<Vec3b>(60, 75)[1] + m.at<Vec3b>(60, 75)[2];
		weight /= 3;
		src = m;
		cvtColor(m, gray, CV_BGR2GRAY);
	}
	void HSV(){
		Mat hsv, dst;
		Mat r, r2, g, b; //�U�C�⪺�֭�
		Mat mask = Mat::zeros(src.rows, src.cols, CV_8U); //���F�o����L�C��

		cvtColor(src, hsv, CV_BGR2HSV);//�নhsv����

		inRange(hsv, Scalar(0, 30, 30), Scalar(50, 255, 255), r);
		//�G�ȤơGh�Ȥ���0~10 & s�Ȥ���100~255 & v�Ȥ���120~255
		inRange(hsv, Scalar(130, 30, 30), Scalar(180, 255, 255), r2);
		//�G�ȤơGh�Ȥ���170~180 & s�Ȥ���100~255 & v�Ȥ���120~255
		mask = r + r2 + g + b;//�������G�Ȥƹϲ֥[�_�ӴN�ܦ��B�n

		src.copyTo(dst, mask); //�N��Ϥ��g�ѾB�n�L�o��A�o�쵲�Gdst
		imshow("result", dst);//show���G
	}

	Mat getResult(){
		adaptiveThreshold(gray, twoValue, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 67, 13);
		imshow("twoValue", twoValue);		
		return twoValue;	
	}
};

//http://www.geek-workshop.com/thread-1605-1-1.html
//http://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html#gsc.tab=0