#include "opencv2/highgui.hpp"
#include "opencv2/world.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/flann.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/opencv.hpp"
#include "stdio.h"
#include "stdlib.h"
#include "iostream"

using namespace std;
using namespace cv;
/* 垂直边缘检测 */
#define fxf  3
#define ABS(x) (x<0)?((-1)*x):(x)
Mat Straight_Edage_Detect(Mat img)
{
	Mat New;
	New.create(img.rows - fxf + 1, img.cols - fxf + 1, CV_8UC1);
	int height = img.rows;
	int width  = img.cols;
	char filter[fxf][fxf] = {
		{ 1,0,-1 },
		{ 1,0,-1 },
		{ 1,0,-1 }
	};


	for (int i = 0; i < (height - fxf + 1); i++)
	{
		for (int j = 0; j < (width - fxf + 1); j++)
		{
			char pix_val=0,gray;
			gray = (unsigned char)((img.at<Vec3b>(Point(j, i))[0] + img.at<Vec3b>(Point(j, i))[1] + img.at<Vec3b>(Point(j, i))[2])/3);
			pix_val += gray * filter[0][0];
			gray = (unsigned char)((img.at<Vec3b>(Point(j + 1, i))[0] + img.at<Vec3b>(Point(j + 1, i))[1] + img.at<Vec3b>(Point(j + 1, i))[2]) / 3);
			pix_val += gray * filter[0][1];
			gray = (unsigned char)((img.at<Vec3b>(Point(j + 2, i))[0] + img.at<Vec3b>(Point(j + 2, i))[1] + img.at<Vec3b>(Point(j + 2, i))[2]) / 3);
			pix_val += gray * filter[0][2];
			gray = (unsigned char)((img.at<Vec3b>(Point(j, i + 1))[0] + img.at<Vec3b>(Point(j, i + 1))[1] + img.at<Vec3b>(Point(j, i + 1))[2]) / 3);
			pix_val += gray * filter[1][0];
			gray = (unsigned char)((img.at<Vec3b>(Point(j + 1, i + 1))[0] + img.at<Vec3b>(Point(j + 1, i + 1))[1] + img.at<Vec3b>(Point(j + 1, i + 1))[2]) / 3);
			pix_val += gray * filter[1][1];
			gray = (unsigned char)((img.at<Vec3b>(Point(j + 2, i + 1))[0] + img.at<Vec3b>(Point(j + 2, i + 1))[1] + img.at<Vec3b>(Point(j + 2, i + 1))[2]) / 3);
			pix_val += gray * filter[1][2];
			gray = (unsigned char)((img.at<Vec3b>(Point(j, i + 2))[0] + img.at<Vec3b>(Point(j, i + 2))[1] + img.at<Vec3b>(Point(j, i + 2))[2]) / 3);
			pix_val += gray * filter[2][0];
			gray = (unsigned char)((img.at<Vec3b>(Point(j + 1, i + 2))[0] + img.at<Vec3b>(Point(j + 1, i + 2))[1] + img.at<Vec3b>(Point(j + 1 , i + 2))[2]) / 3);
			pix_val += gray * filter[2][1];
			gray = (unsigned char)((img.at<Vec3b>(Point(j + 2, i + 2))[0] + img.at<Vec3b>(Point(j + 2, i + 2))[1] + img.at<Vec3b>(Point(j + 2, i + 2))[2]) / 3);
			pix_val += gray * filter[2][2];
			pix_val = ABS(pix_val);
			New.at<uchar>(Point(j, i)) = pix_val;
		}
	}
	
	return New;
}

/* 水平边缘检测 */
Mat Average_Edage_Detect(Mat img)
{
	Mat New;
	New.create(img.rows - fxf + 1, img.cols - fxf + 1, CV_8UC1);
	int height = img.rows;
	int width = img.cols;
	char filter[fxf][fxf] = {
		{ 1, 1, 1  },
		{ 0, 0 ,0  },
		{ -1,-1,-1 }
	};


	for (int i = 0; i < (height - fxf + 1); i++)
	{
		for (int j = 0; j < (width - fxf + 1); j++)
		{
			char pix_val = 0, gray;
			gray = (unsigned char)((img.at<Vec3b>(Point(j, i))[0] + img.at<Vec3b>(Point(j, i))[1] + img.at<Vec3b>(Point(j, i))[2]) / 3);
			pix_val += gray * filter[0][0];
			gray = (unsigned char)((img.at<Vec3b>(Point(j + 1, i))[0] + img.at<Vec3b>(Point(j + 1, i))[1] + img.at<Vec3b>(Point(j + 1, i))[2]) / 3);
			pix_val += gray * filter[0][1];
			gray = (unsigned char)((img.at<Vec3b>(Point(j + 2, i))[0] + img.at<Vec3b>(Point(j + 2, i))[1] + img.at<Vec3b>(Point(j + 2, i))[2]) / 3);
			pix_val += gray * filter[0][2];
			gray = (unsigned char)((img.at<Vec3b>(Point(j, i + 1))[0] + img.at<Vec3b>(Point(j, i + 1))[1] + img.at<Vec3b>(Point(j, i + 1))[2]) / 3);
			pix_val += gray * filter[1][0];
			gray = (unsigned char)((img.at<Vec3b>(Point(j + 1, i + 1))[0] + img.at<Vec3b>(Point(j + 1, i + 1))[1] + img.at<Vec3b>(Point(j + 1, i + 1))[2]) / 3);
			pix_val += gray * filter[1][1];
			gray = (unsigned char)((img.at<Vec3b>(Point(j + 2, i + 1))[0] + img.at<Vec3b>(Point(j + 2, i + 1))[1] + img.at<Vec3b>(Point(j + 2, i + 1))[2]) / 3);
			pix_val += gray * filter[1][2];
			gray = (unsigned char)((img.at<Vec3b>(Point(j, i + 2))[0] + img.at<Vec3b>(Point(j, i + 2))[1] + img.at<Vec3b>(Point(j, i + 2))[2]) / 3);
			pix_val += gray * filter[2][0];
			gray = (unsigned char)((img.at<Vec3b>(Point(j + 1, i + 2))[0] + img.at<Vec3b>(Point(j + 1, i + 2))[1] + img.at<Vec3b>(Point(j + 1, i + 2))[2]) / 3);
			pix_val += gray * filter[2][1];
			gray = (unsigned char)((img.at<Vec3b>(Point(j + 2, i + 2))[0] + img.at<Vec3b>(Point(j + 2, i + 2))[1] + img.at<Vec3b>(Point(j + 2, i + 2))[2]) / 3);
			pix_val += gray * filter[2][2];
			pix_val = ABS(pix_val);
			New.at<uchar>(Point(j, i)) = pix_val;
		}
	}

	return New;
}
void Gray_Deal(Mat img,Mat &out)
{
	int height = img.rows;
	int width = img.cols;
	out.create(height, width, CV_8UC1);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int gray;
			gray = (img.at<Vec3b>(Point(j, i))[0]
				  + img.at<Vec3b>(Point(j, i))[1]
				  + img.at<Vec3b>(Point(j, i))[2]) / 3;
			out.at<uchar>(Point(j, i)) = gray;
		}
	}
}

Mat Binary_Deal(Mat img)
{
	int height = img.rows;
	int width  = img.cols;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int gray;
			gray = (img.at<Vec3b>(Point(j, i))[0]
				  + img.at<Vec3b>(Point(j, i))[1]
				  + img.at<Vec3b>(Point(j, i))[2]) / 3;
			if (gray > 128)
			{
				img.at<Vec3b>(Point(j, i))[0] = 255;
				img.at<Vec3b>(Point(j, i))[1] = 255;
				img.at<Vec3b>(Point(j, i))[2] = 255;
			}
			else
			{
				img.at<Vec3b>(Point(j, i))[0] = 0;
				img.at<Vec3b>(Point(j, i))[1] = 0;
				img.at<Vec3b>(Point(j, i))[2] = 0;
			}
		}
	}
	return img;
}

int main(int argc, char** argv)
{
	/*
	Mat img;
	Size sz(600,600);
	img = imread("D:\\Baidu-donwload\\opencv\\project\\facedetect\\x64\\Release\\man.jpg");
	resize(img,img,sz,0,0,1);
	if (!img.empty())
		img = Average_Edage_Detect(img);
	imshow("Average_Edage_Detect",img);
	waitKey(30);
	system("pause");
	*/
	Mat img;
	char key = 0;
	int num = 1;
	vector<Rect> face_rect;
	VideoCapture cap;
	CascadeClassifier filter;
	cap.open(0);
	//filter.load("D:\\Baidu-donwload\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml");
	filter.load("./cascade.xml");
	if (!cap.isOpened())
	{
		cout << "cap open failed!\r\n";
		exit(-1);
	}
	while ( (key != 's') && (num <=450) )
	{
		cap.read(img);
		if (!img.empty())
		{
			filter.detectMultiScale(img,face_rect);
			for (size_t i=0; i < face_rect.size(); i++)
			{
				/*
				ellipse(img, Point(face_rect[i].x + face_rect[i].width/2,
					face_rect[i].y + face_rect[i].height/2),
					Size(face_rect[i].width / 2,
						face_rect[i].height / 2),
					0, 0, 360, Scalar(255, 0, 255), 4);
				*/
				Point pt1(face_rect[i].x, face_rect[i].y);
				Point pt2(face_rect[i].x + face_rect[i].width, face_rect[i].y + face_rect[i].height);
				/*
				Mat cut = img(face_rect[i]);
				char cut_img_name[128];
				sprintf_s(cut_img_name,"%s%d%s", "D:\\Baidu-donwload\\opencv\\project\\facedetect\\facedetect\\data\\face\\img\\img_", num++,".jpg");
				imwrite(cut_img_name,cut);
				*/
				rectangle(img, pt1, pt2,
					Scalar(0, 0, 255));
				putText(img,"xiongnuoye",pt2, FONT_HERSHEY_COMPLEX,0.6,Scalar(255,255,0));
			}
			
			char cut_img_name[128];
			//Rect rect(100,100,284,284);
			//Mat cut = img(rect);
			/*
			sprintf_s(cut_img_name, "%s%d%s", "D:\\Baidu-donwload\\opencv\\project\\facedetect\\facedetect\\data\\background\\img\\img_", num++, ".jpg");
			imwrite(cut_img_name, img);
			*/
			imshow("face_detect",img);
			key = waitKey(30);
		}
	}

	return 0;
}

