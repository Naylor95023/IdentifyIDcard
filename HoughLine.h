#include<iostream>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\core\core.hpp>

using namespace std;
using namespace cv;

class HoughLine{
public:
	double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0) {
		double dx1 = pt1.x - pt0.x;
		double dy1 = pt1.y - pt0.y;
		double dx2 = pt2.x - pt0.x;
		double dy2 = pt2.y - pt0.y;
		return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
	}

	void Draw(Mat &img, vector<vector<Point>> rects)
	{
		for (int i = 0; i < rects.size(); i++)
		{
			vector<Point> rect = rects[i];
			line(img, rect[0], rect[1], Scalar(0, 255, 0), 3);
			line(img, rect[1], rect[2], Scalar(0, 255, 0), 3);
			line(img, rect[2], rect[3], Scalar(0, 255, 0), 3);
			line(img, rect[3], rect[0], Scalar(0, 255, 0), 3);
		}
	}

	void Draw(Mat &img, vector<Point> shap)
	{
		cout << "Size = " << shap.size() << endl;
		for (int i = 0; i < shap.size() - 1; i++)
			line(img, shap[i], shap[i + 1], Scalar(0, 255, 0), 3);

		line(img, shap[shap.size() - 1], shap[0], Scalar(0, 255, 0), 3);
	}

	vector<Point> findRec(cv::Mat src){
		vector<Point> d;
		if (src.empty())return d ;

		//Convert to grayscale
		cv::Mat gray;
		cv::cvtColor(src.clone(), gray, CV_BGR2GRAY);

		// Convert to binary image using Canny
		cv::Mat bw;
		blur(gray, bw, Size(3, 3));
		cv::Canny(bw, bw, 0, 50, 3);

		// Find contours
		std::vector<std::vector<cv::Point> > contours;
		cv::findContours(bw, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

		// The array for storing the approximation curve
		std::vector<cv::Point> approx;

		// We'll put the labels in this destination image
		cv::Mat dst = src;
		vector<Point> tapprox;


		for (int i = 0; i < contours.size(); i++){
			// Approximate contour with accuracy proportional
			// to the contour perimeter
			cv::approxPolyDP(
				cv::Mat(contours[i]),
				approx,
				cv::arcLength(cv::Mat(contours[i]), true) * 0.02,
				true
				);

			// Skip small or non-convex objects 
			if (std::fabs(cv::contourArea(contours[i])) < 1500 || !cv::isContourConvex(approx))
				continue;
			else if (approx.size() >= 4 && approx.size() <= 6)
			{
				// Number of vertices of polygonal curve
				int vtc = approx.size();

				// Get the degree (in cosines) of all corners
				std::vector<double> cos;
				for (int j = 2; j < vtc + 1; j++)
					cos.push_back(angle(approx[j%vtc], approx[j - 2], approx[j - 1]));

				// Sort ascending the corner degree values
				std::sort(cos.begin(), cos.end());

				// Get the lowest and the highest degree
				double mincos = cos.front();
				double maxcos = cos.back();

				// Use the degrees obtained above and the number of vertices
				// to determine the shape of the contour
				
				if (vtc == 4 && mincos >= -0.1 && maxcos <= 0.3){
					// Detect rectangle or square
					cv::Rect r = cv::boundingRect(contours[i]);
					double ratio = std::abs(1 - (double)r.width / r.height);

					if (ratio <= 0.02){
						Draw(dst, contours[i]);
						putText(dst, "SQU", contours[i][0], 2, 2, CV_RGB(255, 255, 255), 2, 8, false);
					}
					else{
						Draw(dst, contours[i]);
						//putText(dst, "RECT", contours[i][0], 2, 2, CV_RGB(255, 255, 255), 2, 8, false);

						approxPolyDP(Mat(contours[i]), tapprox, arcLength(Mat(contours[i]), true)*0.02, true);
						vector<Point> p = tapprox;
						cout << p[0].x << "," << p[0].y << endl;
						cout << p[1].x << "," << p[1].y << endl;
						cout << p[2].x << "," << p[2].y << endl;
						cout << p[3].x << "," << p[3].y << endl;
						cv::imshow("GetRectangle", dst);
						return tapprox;
					}
				}
			}
		} // end of for() loop
		
		return tapprox;
	}
};