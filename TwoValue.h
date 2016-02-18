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
		Mat r, r2, g, b; //各顏色的閥值
		Mat mask = Mat::zeros(src.rows, src.cols, CV_8U); //為了濾掉其他顏色

		cvtColor(src, hsv, CV_BGR2HSV);//轉成hsv平面

		inRange(hsv, Scalar(0, 30, 30), Scalar(50, 255, 255), r);
		//二值化：h值介於0~10 & s值介於100~255 & v值介於120~255
		inRange(hsv, Scalar(130, 30, 30), Scalar(180, 255, 255), r2);
		//二值化：h值介於170~180 & s值介於100~255 & v值介於120~255
		mask = r + r2 + g + b;//全部的二值化圖累加起來就變成遮罩

		src.copyTo(dst, mask); //將原圖片經由遮罩過濾後，得到結果dst
		imshow("result", dst);//show結果
	}

	Mat getResult(){
		adaptiveThreshold(gray, twoValue, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 67, 13);
		imshow("twoValue", twoValue);		
		return twoValue;	
	}
};

//http://www.geek-workshop.com/thread-1605-1-1.html
//http://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html#gsc.tab=0