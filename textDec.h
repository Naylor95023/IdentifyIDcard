#include "stdafx.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

class textDec{
private:
	vector<Rect> textArea;
public:
	vector<Rect> getTextRec(Mat rgb)
	{
		Mat smallX = rgb;
		// morphological gradient
		Mat grad;
		Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
		morphologyEx(smallX, grad, MORPH_GRADIENT, morphKernel);
		// binarize
		Mat bw;
		threshold(grad, bw, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);
		// connect horizontally oriented regions
		Mat connected;
		morphKernel = getStructuringElement(MORPH_RECT, Size(9, 1));
		morphologyEx(bw, connected, MORPH_OPEN, morphKernel);
		// find contours
		Mat mask = Mat::zeros(bw.size(), CV_8UC1);
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(connected, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		// filter contours
		for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
		{
			Rect rect = boundingRect(contours[idx]);
			Mat maskROI(mask, rect);
			maskROI = Scalar(0, 0, 0);
			// fill the contour
			drawContours(mask, contours, idx, Scalar(255, 255, 255), CV_FILLED);
			// ratio of non-zero pixels in the filled region
			double r = (double)countNonZero(maskROI) / (rect.width*rect.height);

			if (r > .35 &&(rect.height > 15 && rect.width > 30))
			{
				if (rect.width / rect.height >= 3){
					rectangle(rgb, rect, Scalar(0, 255, 0), 0.5);
					textArea.push_back(rect);
				}
			}
		}
		imshow("textArea", rgb);
		return textArea;
	}


	string Trim(const char * c){
		string s;
		for (int i = 0; i < strlen(c); i++)
			if (c[i] != ' ' && c[i] != '\n')
				s += c[i];
		return s;
	}
};