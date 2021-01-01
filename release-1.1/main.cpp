#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/core.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/video.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/objdetect.hpp"

#include "stdlib.h"
#include "stdio.h"
#include "iostream"

using namespace std;
using namespace cv;

int featnums;

//Parm@ run any nums >= 1         [对象检测]        (any != face|bg)
//Parm@ run face nums             [人脸素材获取]    
//Parm@ run bg nums               [背景素材获取]    
//Parm@ run any nums>=1 own       [训练后的测试]    (any != face|bg)

int main(int argc ,char** argv)
{
    VideoCapture cap;   //摄像头
    CascadeClassifier faceflt;//分类器
    int featnums = 0;
    if( !strcmp(argv[3],"own") )
    {
        if( !faceflt.load(argv[4]) )//装载训练数据
        {
            cout<<"装载训练数据失败，检查xml文件路径！"<<endl;
            return -1;
        }
    }
    else
    {
        if( !faceflt.load("./haarcascade_frontalface_alt.xml") )//装载训练数据
        {
            cout<<"装载训练数据失败，检查xml文件路径！"<<endl;
            return -1;
        }
    }
    cap.open(0);
    if(!cap.isOpened())
    {
        cout<<"摄像头打开失败，检查设备！"<<endl;
        return -1;
    }
    char key = 0;
    while( (key != 'c') && (featnums<atoi(argv[2])) )
    {
        Mat frame;
        cap.read(frame);
        if(!frame.empty())
        {
            //特征识别图像帧
            Mat gray;
            cvtColor(frame,gray,COLOR_BGRA2GRAY);   //转化为灰度图，便于算法执行
            vector<Rect> face;                      //人脸矩阵
            faceflt.detectMultiScale(gray,face);    //人脸检测
            for(size_t i=0;i<face.size();i++)       //监测到的人脸矩阵画框
            {
                rectangle(frame,face[0],Scalar(255,255,0));
                putText(frame,String("My love zhb"),Point(face[0].x+face[0].width,face[0].y+face[0].height),FONT_HERSHEY_COMPLEX,1.0,Scalar(0,255,0));
                //获取特征素材
                if(!strcmp(argv[1],"face")) //对象
                {
                    Mat feature = frame(face[i]);
                    char filenm[64]={0};
                    sprintf(filenm,"%s%d%s","./data/face/img/face_",featnums,".jpg");
                    resize(feature,feature,Size(160,160));
                    imwrite(filenm,feature);
                    featnums++;
                }
            }
            if(!strcmp(argv[1],"bg"))       //背景
            {
                char filenm[64]={0};
                sprintf(filenm,"%s%d%s","./data/bg/img/bg_",featnums,".jpg");
                imwrite(filenm,frame);
                featnums++;
            }
            imshow("capture",frame);
        }
        key = waitKey(1);
    }
    return 0;
}
