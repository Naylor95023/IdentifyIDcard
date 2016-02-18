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
#include "Homography.h"
using namespace cv;
using namespace std;

class AffineTransfrom{
private:
	Point2f srcTri[4];
	Point2f dstTri[4];
	Point2f temp[4];
	Mat scene;
public:
	AffineTransfrom(vector<Point> p, Mat m){
		scene = m;
		temp[0] = Point2f(p[0]);
		temp[1] = Point2f(p[1]);
		temp[2] = Point2f(p[2]);
		temp[3] = Point2f(p[3]);
		dstTri[0] = Point2f(3, 3);
		dstTri[1] = Point2f(3, 68);
		dstTri[2] = Point2f(103, 68);
		dstTri[3] = Point2f(103, 3);
	}
	Mat warp (){
		sort();
		Mat warp_mat, warp_scene;
		warp_scene.cols = 720;
		warp_scene.rows = 450;
		warp_mat = getAffineTransform(srcTri, dstTri);
		warpAffine(scene, warp_scene, warp_mat, warp_scene.size());
		imshow("AffineTransform", warp_scene);
		return warp_scene;
	}
		
	
	void sort(){
		int midX = 0, midY = 0;
		for (int i = 0; i < 4; i++){
			midX += temp[i].x;
			midY += temp[i].y;
		}
		Point2f mid = Point2f(midX /= 4, midY /= 4);
		
		int w[4];
		for (int i = 0; i < 4; i++)
			w[i] = weight(temp[i], mid, 0) + weight(mid, temp[i], 0);
		int min = minWeight(w);
		setArray(min);

		/*cout << w[0] << endl;
		cout << w[1] << endl;
		cout << w[2] << endl;
		cout << w[3] << endl;*/
	}

	int weight(Point2f star, Point2f end, int w){
		int x = (star.x + end.x) / 2;
		int y = (star.y + end.y) / 2;
		if (abs(x - star.x) <= 1 && abs(y - star.y)<=1 )return w;
		w += (scene.at<Vec3b>(y, x)[0] / 5 + scene.at<Vec3b>(y, x)[1] + scene.at<Vec3b>(y, x)[2] * 5);
		Point2f eend = Point2f(x, y);
		return weight(star, eend, w);
	}

	int minWeight(int w[4]){
		int min = 0;
		for (int i = 1; i < 4; i++)
			if (w[i] < w[min])min = i;
		return min;
	}
	void setArray(int min){
		srcTri[0] = temp[(min + 0) % 4];
		srcTri[1] = temp[(min + 1) % 4];
		srcTri[2] = temp[(min + 2) % 4];
		srcTri[3] = temp[(min + 3) % 4];
	}
};