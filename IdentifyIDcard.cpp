#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h> 
#include <cstring>
#include <string>

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

#include "Histogram.h"
#include "HoughLine.h"
#include "AffineTransfrom.h"
#include "textDec.h"
#include "TwoValue.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	string s_scene = "id-41-r10.jpg";
	Mat img_scene = imread(s_scene, CV_LOAD_IMAGE_COLOR);
	if (!img_scene.data)
	{
		std::cout << " --(!) Error reading images " << std::endl;
		return -1;
	}
	//Resize>>
	Mat resize_scene;
	float c = 720.0 / img_scene.cols;
	resize(img_scene, resize_scene, Size(0, 0), c, c, INTER_CUBIC);

	//HoughLine>>
	HoughLine _HL;
	vector<Point> p = _HL.findRec(resize_scene);
	//Transform
	AffineTransfrom _AT(p, resize_scene);
	Mat warp_scene = _AT.warp();

	TwoValue _TV(warp_scene);
	Mat two_value = _TV.getResult();
	
	textDec _TD;
	vector<Rect> textAreas = _TD.getTextRec(two_value);
	//GetTextArea>>
	int maxX = 0;
	int index = 0;
	for (int i = 0; i < textAreas.size(); i++){
		if (textAreas[i].x >= maxX && textAreas[i].y > two_value.rows / 2){
			maxX = textAreas[i].x;
			index = i;
		}
	}
	textAreas[index].x += 3;
	textAreas[index].y += 3;
	textAreas[index].width -= 6;
	textAreas[index].height -= 6;
	Rect tempRect = textAreas[index];
	int exceptTen = 0;
	exceptTen = textAreas[index].width / 10;
	tempRect.width = exceptTen;
	Mat English;
	warp_scene(tempRect).copyTo(English);
	imshow("English", English);
	textAreas[index].x += exceptTen;
	textAreas[index].width -= exceptTen;
	Mat text;
	warp_scene(textAreas[index]).copyTo(text);
	imshow("text", text);

	
	//OCR>>
	tesseract::TessBaseAPI api;
	api.Init("C:\\tessdata", "eng", tesseract::OEM_DEFAULT);
	api.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
	
	api.SetImage((uchar*)English.data, English.size().width, English.size().height,
		English.channels(), English.step1());
	api.Recognize(0);
	const char* eng = api.GetUTF8Text();
	cout << "String:" << _TD.Trim(eng) << endl;
		
	api.SetImage((uchar*)text.data, text.size().width, text.size().height,
		text.channels(), text.step1());
	api.Recognize(0);
	const char* cr = api.GetUTF8Text();
	cout << "String:" << _TD.Trim(cr) << endl;

	api.~TessBaseAPI();
	waitKey(0);
	return 0;	
}
