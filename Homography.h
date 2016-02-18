

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

class Homography{
private:
	Mat img_object;
	Mat img_scene;
	int minHessian = 400;
	SurfDescriptorExtractor extractor;
	FlannBasedMatcher matcher;
	std::vector<KeyPoint> keypoints_object, keypoints_scene;
	Mat descriptors_object, descriptors_scene;
	std::vector< DMatch > matches;
	double max_dist = 0; double min_dist = 100;
	std::vector< DMatch > good_matches;
	Mat img_matches;
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;
	std::vector<Point2f> scene_corners;
public:
	Homography(Mat obj, Mat scene){
		img_object = obj;
		img_object = scene;
	}
	void DetectkKeypoints(){
		SurfFeatureDetector detector(minHessian);
		detector.detect(img_object, keypoints_object);
		detector.detect(img_scene, keypoints_scene);
	}
	void CalculateDescriptors(){
		extractor.compute(img_object, keypoints_object, descriptors_object);
		extractor.compute(img_scene, keypoints_scene, descriptors_scene);
	}
	void  MatchingDescriptor(){
		matcher.match(descriptors_object, descriptors_scene, matches);
	}
	void CalculationMaxMinDis(){
		for (int i = 0; i < descriptors_object.rows; i++)
		{
			double dist = matches[i].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}
		printf("-- Max dist : %f \n", max_dist);
		printf("-- Min dist : %f \n", min_dist);
	}
	void DrawGoodMatch(){
		for (int i = 0; i < descriptors_object.rows; i++)
		{
			if (matches[i].distance < 3 * min_dist)
			{
				good_matches.push_back(matches[i]);
			}
		}
		drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
			good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
			std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	}
	void FindHomographyRec(){
		for (size_t i = 0; i < good_matches.size(); i++)
		{
			//-- Get the keypoints from the good matches
			obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
			scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
		}

		Mat H = findHomography(obj, scene, CV_RANSAC);

		//-- Get the corners from the image_1 ( the object to be "detected" )
		std::vector<Point2f> obj_corners(4);
		obj_corners[0] = Point(0, 0);
		obj_corners[1] = Point(img_object.cols, 0);
		obj_corners[2] = Point(img_object.cols, img_object.rows);
		obj_corners[3] = Point(0, img_object.rows);

		perspectiveTransform(obj_corners, scene_corners, H);

		cout << "X0 : " << scene_corners[0].x << ", Y0 :" << scene_corners[0].y << endl;
		cout << "X1 : " << scene_corners[1].x << ", Y1 : " << scene_corners[1].y << endl;
		cout << "X2 : " << scene_corners[2].x << ", Y2 :" << scene_corners[2].y << endl;
		cout << "X3 : " << scene_corners[3].x << ", Y3 :" << scene_corners[3].y << endl;
	}
	void DrawLines(){
		Point2f offset((float)img_object.cols, 0);
		line(img_matches, scene_corners[0] + offset, scene_corners[1] + offset, Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[1] + offset, scene_corners[2] + offset, Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[2] + offset, scene_corners[3] + offset, Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[3] + offset, scene_corners[0] + offset, Scalar(0, 255, 0), 4);
		//-- Show detected matches
		imshow("Good Matches & Object detection", img_matches);
	}
};



