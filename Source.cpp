#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#define  WINDOW_NAME "【程序窗口】"
#define  WINDOW_SRC  "【原始图】"
#define  WINDOW_DST  "【结果图】"
using namespace cv;
using namespace std;

Mat srcimage,dstimage,tmpimage,grayimage,image,maskimage;
//wo
 
//-----------------------------resize()
/*  
int main()
{
  srcimage=imread("D:\\img\\1.jpg");
  tmpimage=srcimage;
  imshow("【原始图】",srcimage);
  
  resize(tmpimage,dstimage,Size(tmpimage.cols/2,tmpimage.rows/2),0,0,3);//放大图像（一般）采用线性插值INTER_LINEAR=3
  imshow("【效果图1】",dstimage);
  resize(tmpimage,dstimage,Size(tmpimage.cols*2,tmpimage.rows*2),0,0,1);//缩小图像采用区域插值，INTER_AREA=1
   imshow("【效果图2】",dstimage);
   waitKey(0);
   return 0 ;
}
*/

//---------------------------pyrUP()
/*
int main()
{
	srcimage=imread("D:\\img\\2.jpg");
	tmpimage=srcimage;
	imshow("【原始图】",srcimage);
	pyrUp(tmpimage,dstimage,Size(tmpimage.cols*2,tmpimage.rows*2));
	imshow("【效果图】",dstimage);
	waitKey(0);
	return 0;
}
*/

//-----------------------------pyrDown()
/*
int main()
{
	srcimage=imread("D:\\img\\2.jpg");
	tmpimage=srcimage;
	imshow("【原始图】",srcimage);
	pyrDown(tmpimage,dstimage,Size(tmpimage.cols/2,tmpimage.rows/2));
	imshow("【效果图】",dstimage);
	waitKey(0);
	return 0;
}
*/
/*
int main()
{
	srcimage=imread("D:\\img\\2.jpg");
	if(!srcimage.data)
	{
        printf("读取图片错误");
		return false;
	}
	namedWindow(WINDOW_NAME,WINDOW_AUTOSIZE);
	imshow(WINDOW_NAME,srcimage);
	tmpimage=srcimage;
	dstimage=tmpimage;
	int key=0;
	while (1)
	{
        key=waitKey(9);
		switch (key)
		{
		case 27://按键为“ESC”
			return 0;
			break;
		case 'a':
			pyrUp(tmpimage,dstimage,Size(tmpimage.cols*2,tmpimage.rows*2));
			printf("检测到A被按下，pyrUP图片放大*2\n");
			break;
		case 'w':
			resize(tmpimage,dstimage,Size(tmpimage.cols*2,tmpimage.rows*2));
			printf("检测到W被按下，resize图片放大*2\n");
			break;
		case 'd':
			pyrDown(tmpimage,dstimage,Size(tmpimage.cols/2,tmpimage.rows/2));
			printf("检测到D被按下，pyrDown图片缩小/2\n");
			break;
		case 's':
			resize(tmpimage,dstimage,Size(tmpimage.cols/2,tmpimage.rows/2));
			printf("检测到W被按下，resize图片放大/2\n");
			break;
		}
		imshow(WINDOW_NAME,dstimage);
		tmpimage=dstimage;
	}
	return 0;
}
*/

//-----threshold()阈值化❀

/*
int g_nThresholdValue=100;
int g_nThresholdType=3;
void on_Threshold(int,void*);
int main()
{
   srcimage=imread("D:\\img\\3.png");
   cvtColor(srcimage,tmpimage,COLOR_RGB2GRAY);
   namedWindow(WINDOW_NAME,WINDOW_AUTOSIZE);
   createTrackbar("模式",WINDOW_NAME,&g_nThresholdType,4,on_Threshold);
   createTrackbar("参数值",WINDOW_NAME,&g_nThresholdValue,255,on_Threshold);
   on_Threshold(0,0);
   while (1)
   {
	   int key;
	   key=waitKey(20);
	   if((char)key==27)break;
   }
}

 void on_Threshold(int,void*)
 {
	 threshold(tmpimage,dstimage,g_nThresholdValue,255,g_nThresholdType);
	 imshow(WINDOW_NAME,dstimage);
 }
 */

 //-----canny边缘检测
 /*
 int  main()
{
	srcimage=imread("D:\\img\\4.jfif");
	Mat edge;
	dstimage.create(srcimage.size(),srcimage.type());
	dstimage=Scalar::all(0);
	cvtColor(srcimage,grayimage,COLOR_RGB2GRAY);
	blur(grayimage,edge,Size(6,6));
	Canny(edge,edge,3,6,3);
	srcimage.copyTo(dstimage,edge);//将src拷贝到dst之中，edge作为掩膜
	imshow("【canny效果图】",dstimage);
	waitKey(0);
	return 0;
}
*/
/*
int main()
{
   Mat gray_x,gray_y,abs_gray_x,abs_gray_y;
   srcimage=imread("D:\\img\\6.jpg");
   imshow(WINDOW_NAME,srcimage);
   Sobel(srcimage,gray_x,CV_16S,1,0,3,1,1,BORDER_DEFAULT);
   convertScaleAbs(gray_x,abs_gray_x);
   imshow("【效果图X】",abs_gray_x);
   Sobel(srcimage,gray_y,CV_16S,0,1,3,1,1,BORDER_DEFAULT);
   convertScaleAbs(gray_y,abs_gray_y);
   imshow("【效果图】Y",abs_gray_y);
   addWeighted(abs_gray_x,0.5,abs_gray_y,0.5,0,dstimage);
   imshow("【合成图】",dstimage);
   waitKey(0);
}
*/

//----------------Laplacian算子
/*
int main()
{
	srcimage=imread("D:\\img\\6.jpg");
	imshow("【原始图】",srcimage);
	GaussianBlur(srcimage,srcimage,Size(3,3),0,0,BORDER_DEFAULT);
	cvtColor(srcimage,grayimage,CV_RGB2GRAY);
	Laplacian(grayimage,dstimage,CV_16S,3,1,0,BORDER_DEFAULT);
	convertScaleAbs(dstimage,dstimage);
	imshow(WINDOW_NAME,dstimage);
	waitKey(0);
	return 0;
}
*/

//---------Scharr滤波器
/*
int main()
{

	Mat x_dstimage,y_dstimage;
	srcimage=imread("D:\\img\\7.jfif");
	imshow("【原始图】",srcimage);
	Scharr(srcimage,x_dstimage,CV_16S,1,0,1,0,BORDER_DEFAULT);
	convertScaleAbs(x_dstimage,x_dstimage);
	imshow("【x方向结果图】",x_dstimage);

	Scharr(srcimage,y_dstimage,CV_16S,0,1,1,0,BORDER_DEFAULT);
	convertScaleAbs(y_dstimage,y_dstimage);
	imshow("【y方向结果图】",y_dstimage);

	addWeighted(x_dstimage,0.5,y_dstimage,0.5,0,dstimage);
	imshow(WINDOW_NAME,dstimage);
	waitKey(0);
	return 0;
}
*/

//-------------------边缘检测

/*
Mat cannyDetectedEdges;
 int g_cannyLowThreshold=1;//trcakBar位置参数

 Mat g_sobelGradient_X,g_sobelGradient_Y;
 Mat g_sobelAbsGradient_X,g_sobelAbsGradient_Y;
 int g_sobelkernelSize=1;

 Mat g_scharrGradient_X,g_scharrGradient_Y;
 Mat g_scharrAbxGradient_X,g_scharrAbxGradient_Y;
 
static void on_Canny(int,void*);
static void on_sobel(int,void*);
void scharr();

int main()
{
	system("color 2F");
	srcimage=imread("D:\\img\\1.jpg");
	if (!srcimage.data)
	{
		printf("图片不存在！");
		return false;
	}
	imshow("【原始图】",srcimage);

	dstimage.create(srcimage.size(),dstimage.type());
	cvtColor(srcimage,grayimage,CV_RGB2GRAY);
	namedWindow("【canny边缘检测】",WINDOW_AUTOSIZE);
	namedWindow("【sobel边缘检测】",WINDOW_AUTOSIZE);
	createTrackbar("canny参数","【canny边缘检测】",&g_cannyLowThreshold,120,on_Canny);
	createTrackbar("sobel参数","【sobel边缘检测】",&g_sobelkernelSize,3,on_sobel);

	scharr();
	while ((char(waitKey(1))!='q')){}
	return 0;
}
void on_Canny(int,void*)
 {
	 blur(grayimage,cannyDetectedEdges,Size(3,3));//灰度图片
	 Canny(cannyDetectedEdges,cannyDetectedEdges,g_cannyLowThreshold,g_cannyLowThreshold*3,3);
	 dstimage=Scalar::all(0);
	 srcimage.copyTo(dstimage,cannyDetectedEdges);
	 imshow("【canny边缘检测】",dstimage);
 }

void on_sobel(int,void*)
{
	Sobel(srcimage,g_sobelGradient_X,CV_16S,1,0,(2*g_sobelkernelSize+1),1,1,BORDER_DEFAULT);
	convertScaleAbs(g_sobelGradient_X,g_sobelAbsGradient_X);

	Sobel(srcimage,g_sobelGradient_Y,CV_16S,0,1,(2*g_sobelkernelSize+1),1,1,BORDER_DEFAULT);
	convertScaleAbs(g_sobelGradient_Y,g_sobelAbsGradient_Y);

	addWeighted(g_sobelAbsGradient_Y,0.5,g_sobelAbsGradient_X,0.5,0,dstimage);
    imshow("【sobel边缘检测】",dstimage);
}

void scharr()
{
	Scharr(srcimage,g_scharrGradient_X,CV_16S,1,0,1,0,BORDER_DEFAULT);
	convertScaleAbs(g_scharrGradient_X,g_sobelAbsGradient_X);
	Scharr(srcimage,g_scharrGradient_Y,CV_16S,0,1,1,0,BORDER_DEFAULT);
	convertScaleAbs(g_scharrGradient_Y,g_sobelAbsGradient_Y);
	addWeighted(g_scharrGradient_X,0.5,g_scharrGradient_Y,0.5,0,dstimage);
	imshow("【scharr边缘检测】",dstimage);
}
*/


//-------------Hough线检测
/*
int main()
{
	srcimage=imread("D:\\img\\1.jpg");
// 	cvtColor(srcimage,srcimage,CV_RGB2GRAY);
// 	blur(srcimage,srcimage,Size(3,3));

	Canny(srcimage,tmpimage,50,150,3);
	cvtColor(tmpimage,dstimage,CV_GRAY2BGR);
    vector<Vec2f>lines;
	HoughLines(tmpimage,lines,1,CV_PI/180,30,0,0);

	for (size_t i=0;i<lines.size();i++)
	{
		float rho=lines[i][0],theta=lines[i][1];
		Point pt1,pt2;
		double a=cos(theta),b=sin(theta);
		double x0=a*rho,y0=b*rho;
		pt1.x=cvRound(x0+1000*(-b));
		pt1.y=cvRound(y0+1000*(a));
		pt2.x=cvRound(x0-1000*(-b));
		pt2.y=cvRound(y0-1000*(a));
		line(dstimage,pt1,pt2,Scalar(55,100,195),1,LINE_AA);
		imshow("【原始图】",srcimage);
		imshow("【canny图】",tmpimage);
		imshow("效果图",dstimage);
		waitKey(0);
		return 0;

	}
}
*/
//-----HoughlinesP()
/*
int main()
{ 
	Mat srcimage;
	srcimage=imread("D:\\img\\0.png");
	Canny(srcimage,tmpimage,50,150,3);
	cvtColor(tmpimage,dstimage,CV_GRAY2BGR);
	vector<Vec4i>lines;
	HoughLinesP(tmpimage,lines,1,CV_PI/180,150,50,10);  //累积阈值，最短线长，断线链接
	for (size_t i=0;i<lines.size();i++)
	{
		Vec4i l=lines[i];
		line(dstimage,Point(l[0],l[1]),Point(l[2],l[3]),Scalar(186,88,255),1,LINE_AA);
	}
	imshow("【原始图】",srcimage);
	imshow("【canny图】",tmpimage);
	imshow("【结果图】",dstimage);
	waitKey(0);
	return 0;
}
*/
/*
vector<Vec4i>g_lines;
int g_nthreshold=100;
static void on_HoughLines(int,void*);
int main()
{
	system("color 3F");
	srcimage=imread("D:\\img\\4.jfif");
	imshow("【原始图】",srcimage);
	namedWindow("【效果图】",1);
	createTrackbar("值","【效果图】",&g_nthreshold,200,on_HoughLines);

	Canny(srcimage,tmpimage,50,200,3);
	cvtColor(tmpimage,dstimage,COLOR_GRAY2BGR);

	on_HoughLines(g_nthreshold,0);
	HoughLinesP(tmpimage,g_lines,1,CV_PI/180,80,50,10);
	imshow("【效果图】",dstimage);
	waitKey(0);
	return 0;
}
static void on_HoughLines(int,void*)
{
	Mat g_dstimage=dstimage.clone();
	Mat g_tmpimage=tmpimage.clone();
	vector<Vec4i>mylines;
	HoughLinesP(g_tmpimage,mylines,1,CV_PI/180,g_nthreshold+1,50,10);
    for (size_t i=0;i<mylines.size();i++)
    {
		Vec4i l=mylines[i];
		line(g_dstimage,Point(l[0],l[1]),Point(l[2],l[3]),Scalar(23,180,55),1,LINE_AA);
    }
	imshow("【效果图】",g_dstimage);
}
*/

//-------------HoughCircles()
/*
int main ()
{
	srcimage=imread("D:\\img\\8.jfif");
	imshow("原始图",srcimage);
	cvtColor(srcimage,grayimage,COLOR_RGB2GRAY);
	GaussianBlur(grayimage,grayimage,Size(9,9),2,2);
	vector<Vec3f>circles;
	HoughCircles(grayimage,circles,HOUGH_GRADIENT,1.5,50,200,100,0,0);
	for (size_t i=0;i<circles.size();i++)
	{
		Point center(cvRound(circles[i][0]),cvRound(circles[i][1]));
		int radius=cvRound(circles[i][2]);
		circle(srcimage,center,3,Scalar(0,255,0),-1,8,0);
		circle(srcimage,center,radius,Scalar(155,50,255),3,8,0);
	}
	imshow("效果图",srcimage);
	waitKey(0);
	return 0;
}
*/

//-----------------重映射remap()
/*
int main()
{
	srcimage=imread("D:\\img\\0.png");
	imshow("原始图",srcimage);
	Mat map_x,map_y;
	dstimage.create(srcimage.size(),srcimage.type());
	map_x.create(srcimage.size(),CV_32FC1);
	map_y.create(srcimage.size(),CV_32FC1);
	for (int j=0;j<srcimage.rows;j++)
	{
		for (int i=0;i<srcimage.cols;i++)
		{
			map_x.at<float>(j,i)=static_cast<float>(i);
			map_y.at<float>(j,i)=static_cast<float>(srcimage.rows-j);
		}
	}
	remap(srcimage,dstimage,map_x,map_y,INTER_LINEAR,BORDER_CONSTANT,Scalar(0,0,0));
	imshow("【程序窗口】",dstimage);
	waitKey(0);
	return 0;
}
*/
/*
Mat map_x,map_y;
int update_map(int key);
static void ShowHelpText();
int main()
{
   system("color 2F");
   srcimage=imread("D:\\img\\3.png");
   imshow("【原始图】",srcimage);
   dstimage.create(srcimage.size(),srcimage.type());
   map_x.create(srcimage.size(),CV_32FC1);
   map_y.create(srcimage.size(),CV_32FC1);
   namedWindow(WINDOW_NAME,WINDOW_AUTOSIZE);
   while (1)
   {
	   int key=waitKey(0);
	   if((key&255)==27)
	   {
		   cout<<"程序退出............\n"<<endl;
		   break;
	   }
	   update_map(key);
	   remap(srcimage,dstimage,map_x,map_y,INTER_LINEAR,BORDER_CONSTANT,Scalar(0,0,0));
	   imshow(WINDOW_NAME,dstimage); 
	  
   }
    return 0;
 }
   int  update_map(int key)
   {
	   for (int j=0;j<srcimage.rows;j++)
	   {
		   for (int i=0;i<srcimage.cols;i++)
		   {
			   switch(key)
			   {
			   case '1':
				   map_x.at<float>(j,i)=static_cast<float>(j);
				   map_y.at<float>(j,i)=static_cast<float>(srcimage.rows-i);
				   break;
			   case '2':
				   map_x.at<float>(j,i)=static_cast<float>(srcimage.cols-j);
				   map_y.at<float>(j,i)=static_cast<float>(i);
				   break;
			   case '3':
				   map_x.at<float>(j,i)=static_cast<float>(srcimage.cols-j);
				   map_y.at<float>(j,i)=static_cast<float>(srcimage.rows-i);
				   break;
			   }
		   }
	   }
	   return 1;
   }

   */
   //---------------------仿射变换;
/*
   int main()
{
	system("color 1A");
	srcimage=imread("D:\\img\\7.jfif",1);
	imshow("【原始图】",srcimage);
	//定义两个三角形，得出仿射变换的矩阵;
	Point2f srcTriangle[3];
	Point2f dstTriangle[3];
    Mat rotMat(2,3,CV_32FC1);
	Mat warpMat(2,3,CV_32FC1);
    Mat dstimage_warp,dstimage_rotate;
	dstimage_warp=Mat::zeros(srcimage.rows,srcimage.cols,srcimage.type());

	srcTriangle[0]=Point2f(0,0);
	srcTriangle[1]=Point2f(static_cast<float>(srcimage.cols-1),0);
	srcTriangle[2]=Point2f(0,static_cast<float>(srcimage.rows-1));

	dstTriangle[0]=Point2f(static_cast<float>(srcimage.cols*0.0),static_cast<float>(srcimage.rows*0.33));
	dstTriangle[1]=Point2f(static_cast<float>(srcimage.cols*0.65),static_cast<float>(srcimage.rows*0.35));
	dstTriangle[2]=Point2f(static_cast<float>(srcimage.cols*0.15),static_cast<float>(srcimage.rows*0.6));

	warpMat=getAffineTransform(srcTriangle,dstTriangle);//求得仿射变换

	warpAffine(srcimage,dstimage_warp,warpMat,dstimage_warp.size());

	//对图像进行缩放之后再进行旋转
	Point center=Point(dstimage_warp.cols/2,dstimage_warp.rows/2);
	double angle=-30.0;
	double scale=0.8;
	rotMat=getRotationMatrix2D(center,angle,scale);
	warpAffine(dstimage_warp,dstimage_rotate,rotMat,dstimage_warp.size());
	imshow("仿射变换图",dstimage_warp);
	imshow("变换之后旋转",dstimage_rotate);
	waitKey(0);
	return 0;
}
*/


//--------------直方图均衡化（图像增强）
/*
int main()
{
	srcimage=imread("D:\\img\\9.jpg");
	imshow("原始图",srcimage);
	cvtColor(srcimage,srcimage,COLOR_BGR2GRAY);
	imshow("灰度图",srcimage);
	equalizeHist(srcimage,dstimage);
	imshow("直方图均衡化",dstimage);
	waitKey(0);
	return 0;
}
*/
//---------------寻找轮廓
/*
int main()
   {
	   srcimage=imread("D:\\img\\10.jfif",0);//输入二值图
	   imshow(WINDOW_SRC,srcimage);
	   dstimage=Mat::zeros(srcimage.rows,srcimage.cols,CV_8UC3);
	   srcimage=srcimage>119;
	   imshow("阈值处理后的图",srcimage);
	   vector<vector<Point>>contours;
	   vector<Vec4i>hierarchy;
	   findContours(srcimage,contours,hierarchy,RETR_CCOMP,CHAIN_APPROX_SIMPLE);
	   int index=0;
	   for (;index>=0;index=hierarchy[index][0])
	   {
		   Scalar color(rand()&255,rand()&255,rand()&255);
		   drawContours(dstimage,contours,index,color,FILLED,8,hierarchy);
	   }
	   imshow(WINDOW_DST,dstimage);
	   waitKey(0);
	   return 0;
   }
   */

   /*
   int g_nthresh=80;
   int g_nthreshmax=255;
   RNG g_rng(12345);
   Mat g_cannyMat_output;
   vector<vector<Point>> g_vContours;
   vector<Vec4i> g_vHierarchy;
   void on_ThreshChange(int ,void*);
   int main()
   {
	   system("color 2F");
	   srcimage=imread("D:\\img\\11.jfif");
	   cvtColor(srcimage,grayimage,COLOR_BGR2GRAY);
	   blur(grayimage,grayimage,Size(3,3));
	   imshow(WINDOW_SRC,srcimage);
	   namedWindow("【降噪图】",WINDOW_AUTOSIZE);
	   imshow("【降噪图】",grayimage);
	   createTrackbar("阈值","【降噪图】",&g_nthresh,g_nthreshmax,on_ThreshChange);
	   on_ThreshChange(0,0);
	   waitKey(0);
	   return 0;
   }
   void on_ThreshChange(int ,void*)
   {
	   Canny(grayimage,g_cannyMat_output,g_nthresh,g_nthresh*2,3);
	   findContours(g_cannyMat_output,g_vContours,RETR_TREE,CHAIN_APPROX_SIMPLE,Point(0,0));
	   dstimage=Mat::zeros(g_cannyMat_output.size(),CV_8UC3);
       for (int i=0;i<g_vContours.size();i++)
       {
                Scalar color=Scalar(g_rng.uniform(0,255),g_rng.uniform(0,255),g_rng.uniform(0,255));
				drawContours(dstimage,g_vContours,i,color,2,8,g_vHierarchy,0,Point());
       }
	   imshow(WINDOW_DST,dstimage);

   }
   */
   //------------寻找凸包
   /*
   int main()
{
	Mat image(600,600,CV_8UC3);
	RNG& rng=theRNG();
	while(1)
	{
		char key;
		int count=(unsigned)rng%100+3;
		vector<Point> points;
		for (int i=0;i<count;i++)
		{
			Point point;
			point.x=rng.uniform(image.cols/4,image.rows*3/4);
			point.y=rng.uniform(image.cols/4,image.rows*3/4);
			points.push_back(point);
		}//生成随机点

		image=Scalar::all(0);
		for (int i=0;i<count;i++)
			circle(image,points[i],3,Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255)),FILLED,LINE_AA);
         //绘制

		vector<int>hull;
		convexHull(Mat(points),hull,true);//检测凸包，最外层的点连接起来的凸多边形

		
		int hullcount=hull.size();
		Point Point0=points[hull[hullcount-1]];
		for (int i=0;i<hullcount;i++)
		{
			Point point=points[hull[i]];
			line(image,Point0,point,Scalar(255,255,255),2,LINE_AA);
			Point0=point;
		}
		imshow("效果图",image);
		key=(char)waitKey();
		if(key==27||key=='q'||key=='Q')
			break;
	}
	return 0;
}
*/
//----------------------寻找轮廓，在寻找凸包，并绘出
/*

int g_nthreshold=50;
int g_bthresholdmax=500;
RNG g_rng(12345);
Mat srcimag_copy=srcimage.clone();
Mat g_thresholdimage_output;
vector<vector<Point>>g_vContours;
vector<Vec4i>g_vHierarchy;

void on_ThreshChange(int,void*);

int main()
{
	srcimage=imread("D:\\img\\12.jfif");
	cvtColor(srcimage,grayimage,COLOR_BGR2GRAY);
	blur(grayimage,grayimage,Size(3,3));
	namedWindow(WINDOW_SRC,WINDOW_AUTOSIZE);
	imshow(WINDOW_SRC,grayimage);
	createTrackbar("二值化阈值",WINDOW_SRC,&g_nthreshold,g_bthresholdmax,on_ThreshChange);
	on_ThreshChange(0,0);
	waitKey(0);
	return 0;
}

 void on_ThreshChange(int,void*)
 {
	 threshold(grayimage,g_thresholdimage_output,g_nthreshold,255,THRESH_BINARY);
	 findContours(g_thresholdimage_output,g_vContours,g_vHierarchy,RETR_TREE,CHAIN_APPROX_SIMPLE,Point(0,0));
	 //遍历每个轮廓，寻找其凸包
	 vector<vector<Point>>hull(g_vContours.size());
	 for (int i=0;i<g_vContours.size();i++)
	 {
		 convexHull(Mat(g_vContours[i]),hull[i],false);
	 }
	 Mat drawing=Mat::zeros(g_thresholdimage_output.size(),CV_8UC3);
	 for (int i=0;i<g_vContours.size();i++)
	 {
		 Scalar color=Scalar(g_rng.uniform(0,255),g_rng.uniform(0,255),g_rng.uniform(0,255));
		 drawContours(drawing,g_vContours,i,color,1,8,vector<Vec4i>(),0,Point());
		 drawContours(drawing,hull,i,color,1,8,vector<Vec4i>(),0,Point());
	 }
	 imshow("结果图",drawing);
 }
 */

 //---------寻找最小包围矩形和最小包围圆形
 /*
 int main()
   {
	   Mat image(600,600,CV_8UC3);
	   RNG &rng=theRNG();
	   while (1)
	   {
		   int  count=rng.uniform(3,103);
		   vector<Point>points;
		   for (int i=0;i<count;i++)
		   {
			   Point point;
			   point.x=rng.uniform(image.cols/4,image.cols*3/4);
			   point.y=rng.uniform(image.rows/4,image.cols*3/4);
			   points.push_back(point);
		   }
		   image=Scalar::all(0);
		   for (int i=0;i<count;i++)
			   circle(image,points[i],3,Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255)),FILLED,LINE_AA);
	       //------------------------------------
		   RotatedRect box=minAreaRect(Mat(points));//RotatedRect一种数据结构,存放计算的外接矩形
		   Point2f vertex[4];//4个浮点型的点
		   box.points(vertex);//返回值为矩形四角的点

		   for (int i=0;i<3;i++)
			   line(image,vertex[i],vertex[i+1],Scalar(100,200,150),2,LINE_AA);
		   line(image,vertex[3],vertex[0],Scalar(100,200,150),2,LINE_AA);
	       //------------------------------------
		   Point2f center;
		   float radius=0;
		   minEnclosingCircle(Mat(points),center,radius);
		   //cvRound返回和参数最接近的整数值
		   circle(image,center,cvRound(radius),Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255)),2,LINE_AA);
		   imshow("效果图",image);
		   char key=(char)waitKey();
		   if(key==27||key=='q'||key=='Q')break;
	   }
	   return 0;
   }

*/
//---------使用多边形包围轮廓

/*
int g_nthresh=50;
int g_nMaxthresh=255;
RNG g_rng(12345);
static void on_ContoursChange(int,void*);
int main()
{
	srcimage=imread("D:\\img\\3.png");
	cvtColor(srcimage,grayimage,CV_BGR2GRAY);
	blur(grayimage,grayimage,Size(3,3));
	namedWindow(WINDOW_SRC,WINDOW_AUTOSIZE);
	imshow(WINDOW_SRC,srcimage);
	createTrackbar("阈值",WINDOW_SRC,&g_nthresh,g_nMaxthresh,on_ContoursChange);
	on_ContoursChange(0,0);
	waitKey(0);
	return 0;
}

void on_ContoursChange(int,void*)
{
	Mat thresh_output;
	vector<vector<Point>>contours;
	vector<Vec4i>hierarchy;
	threshold(grayimage,thresh_output,g_nthresh,g_nMaxthresh,THRESH_BINARY);
	findContours(thresh_output,contours,hierarchy,RETR_TREE,CHAIN_APPROX_SIMPLE,Point(0,0));

	vector<vector<Point>> contours_poly(contours.size());
	vector<Rect>boundRect(contours.size());
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());

	for (int i=0;i<contours.size();i++)
	{
		approxPolyDP(Mat(contours[i]),contours_poly[i],3,true);
		boundRect[i]=boundingRect((Mat(contours_poly[i])));
		minEnclosingCircle(contours_poly[i],center[i],radius[i]);
	}

	Mat drawing=Mat::zeros(thresh_output.size(),CV_8UC3);
	for (int i=0;i<contours.size();i++)
	{
		Scalar color=Scalar(g_rng.uniform(0,255),g_rng.uniform(0,255),g_rng.uniform(0,255));
		drawContours(drawing,contours_poly,i,color,1,8,vector<Vec4i>(),0,Point(0,0));
		//fillPoly(drawing,contours_poly,color,LINE_AA,0,Point(0,0));
		rectangle(drawing,boundRect[i].tl(),boundRect[i].br(),color,2,8,0);
		circle(drawing,center[i],(int)radius[i],color,2,8,0);
	}
	namedWindow(WINDOW_DST,WINDOW_AUTOSIZE);
	imshow(WINDOW_DST,drawing);
}

*/

//----------------查找和绘制图像的轮廓矩
/*
int g_nThresh=100;
int g_nMaxThresh=255;
Mat cannyMat;
RNG g_rng(12345);
vector<vector<Point>>contours;
vector<Vec4i>hierarchy;
void on_ThreshChange(int,void*);
int main()
{
	srcimage=imread("D:\\img\\13.jfif",1);
	cvtColor(srcimage,grayimage,COLOR_BGR2GRAY);
	blur(grayimage,grayimage,Size(3,3));
	namedWindow(WINDOW_SRC,WINDOW_AUTOSIZE);
	imshow(WINDOW_SRC,srcimage);
	createTrackbar("二值化阈值",WINDOW_SRC,&g_nThresh,g_nMaxThresh,on_ThreshChange);
	on_ThreshChange(0,0);
	waitKey(0);
	return 0;
}
void on_ThreshChange(int,void*)
{
	Canny(grayimage,cannyMat,g_nThresh,g_nThresh*2,3);
	findContours(cannyMat,contours,hierarchy,RETR_TREE,CHAIN_APPROX_SIMPLE,Point(0,0));

	//计算矩

	vector<Moments> mu(contours.size());
	for (int i=0;i<contours.size();i++)
		mu[i]=moments(contours[i]);


	//计算中心距
	vector<Point2f>mc(contours.size());
	for (int i=0;i<contours.size();i++)
		mc[i]=Point2f(static_cast<float>(mu[i].m10/mu[i].m00),static_cast<float>(mu[i].m01/mu[i].m00));
	//绘制轮廓
		Mat drawing=Mat::zeros(cannyMat.size(),CV_8UC3);
		for (int i=0;i<contours.size();i++)
		{
			Scalar color=Scalar(g_rng.uniform(0,255),g_rng.uniform(0,255),g_rng.uniform(0,255));
			//绘制内层和外层轮廓
			drawContours(drawing,contours,i,color,2,8,hierarchy,0,Point(0,0));
			circle(drawing,mc[i],4,color,-1,8,0);
		}
			namedWindow(WINDOW_DST,WINDOW_AUTOSIZE);
			imshow(WINDOW_DST,drawing);
			printf("\t 输出内容：面积和轮廓长度\n");
		for (int i=0;i<contours.size();i++)
		{
				printf("通过m00计算轮廓 %d 的面积：（M_00）=%.2f \n opencv函数计算出的面积=%.2f,长度：%.2f\n\n",i,mu[i].m00,
					contourArea(contours[i]),arcLength(contours[i],true));

		}
}
*/

//-------------------分水岭算法
/*
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <cstdio>
#include <iostream>
using namespace cv;
using namespace std;
static void help()
{
	cout << "\nThis program demonstrates the famous watershed segmentation algorithm in OpenCV: watershed()\n"
		"Usage:\n"
		"./watershed [image_name -- default is ../data/fruits.jpg]\n" << endl;
	cout << "Hot keys: \n"
		"\tESC - quit the program\n"
		"\tr - restore the original image\n"
		"\tw or SPACE - run watershed segmentation algorithm\n"
		"\t\t(before running it, *roughly* mark the areas to segment on the image)\n"
		"\t  (before that, roughly outline several markers on the image)\n";
}
Mat markerMask, img;
Point prevPt(-1, -1);
static void onMouse( int event, int x, int y, int flags, void* )
{
	if( x < 0 || x >= img.cols || y < 0 || y >= img.rows )
		return;
	if( event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON) )
		prevPt = Point(-1,-1);
	else if( event == EVENT_LBUTTONDOWN )
		prevPt = Point(x,y);
	else if( event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON) )
	{
		Point pt(x, y);
		if( prevPt.x < 0 )
			prevPt = pt;
		line( markerMask, prevPt, pt, Scalar::all(255), 5, 8, 0 );
		line( img, prevPt, pt, Scalar::all(255), 5, 8, 0 );
		prevPt = pt;
		imshow("image", img);
	}
}
int main( int argc, char** argv )
{

	Mat img0 = imread("D:\\img\\14.jfif", 1), imgGray;
	if( img0.empty() )
	{
		cout << "Couldn'g open image " <<endl;
		return 0;
	}
	help();
	namedWindow( "image", 1 );
	img0.copyTo(img);
	cvtColor(img, markerMask, COLOR_BGR2GRAY);
	cvtColor(markerMask, imgGray, COLOR_GRAY2BGR);
	markerMask = Scalar::all(0);
	imshow( "image", img );
	setMouseCallback( "image", onMouse, 0 );
	for(;;)
	{
		int c = waitKey(0);
		if( (char)c == 27 )
			break;
		if( (char)c == 'r' )
		{
			markerMask = Scalar::all(0);
			img0.copyTo(img);
			imshow( "image", img );
		}
		if( (char)c == 'w' || (char)c == ' ' )
		{
			int i, j, compCount = 0;
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			findContours(markerMask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
			if( contours.empty() )
				continue;
			Mat markers(markerMask.size(), CV_32S);
			markers = Scalar::all(0);
			int idx = 0;
			for( ; idx >= 0; idx = hierarchy[idx][0], compCount++ )
				drawContours(markers, contours, idx, Scalar::all(compCount+1), -1, 8, hierarchy, INT_MAX);
			if( compCount == 0 )
				continue;
			vector<Vec3b> colorTab;
			for( i = 0; i < compCount; i++ )
			{
				int b = theRNG().uniform(0, 255);
				int g = theRNG().uniform(0, 255);
				int r = theRNG().uniform(0, 255);
				colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
			}
			double t = (double)getTickCount();
			watershed( img0, markers );
			t = (double)getTickCount() - t;
			printf( "execution time = %gms\n", t*1000./getTickFrequency() );
			Mat wshed(markers.size(), CV_8UC3);
			// paint the watershed image
			for( i = 0; i < markers.rows; i++ )
				for( j = 0; j < markers.cols; j++ )
				{
					int index = markers.at<int>(i,j);
					if( index == -1 )
						wshed.at<Vec3b>(i,j) = Vec3b(255,255,255);
					else if( index <= 0 || index > compCount )
						wshed.at<Vec3b>(i,j) = Vec3b(0,0,0);
					else
						wshed.at<Vec3b>(i,j) = colorTab[index - 1];
				}
				wshed = wshed*0.5 + imgGray*0.5;
				imshow( "watershed transform", wshed );
		}
	}
	return 0;
}

*/

//---------------------图像修补
/*
Point previouspoint(-1,-1);
static void on_Mouse(int event,int x,int y,int flags,void*)
{
	if(event==EVENT_LBUTTONUP||!(flags&EVENT_FLAG_LBUTTON))previouspoint=Point(-1,-1);
	else if(event==EVENT_LBUTTONDOWN)previouspoint=Point(x,y);
	else if(event==EVENT_MOUSEMOVE&&(flags&EVENT_FLAG_LBUTTON))
	{
		Point pt(x,y);
		if(previouspoint.x<0)previouspoint=pt;
		line(maskimage,previouspoint,pt,Scalar::all(255),5,8,0);
		line(tmpimage,previouspoint,pt,Scalar::all(255),5,8,0);
		previouspoint=pt;
		imshow(WINDOW_SRC,tmpimage);
	}
}

int main()
{
	srcimage=imread("D:\\img\\14.jfif");
	tmpimage=srcimage.clone();
	maskimage=Mat::zeros(tmpimage.size(),CV_8U);
	imshow(WINDOW_SRC,tmpimage);
	setMouseCallback(WINDOW_SRC,on_Mouse,0);
	while (1)
	{
		char c=(char)waitKey(0);
		switch (c)
		{
		case 27:
			break;
		case '2':
			maskimage=Scalar::all(0);
			srcimage.copyTo(tmpimage);
			imshow(WINDOW_SRC,tmpimage);
		case '1':
			Mat inpaintimage;
			inpaint(tmpimage,maskimage,inpaintimage,3,INPAINT_TELEA);
			imshow(WINDOW_DST,inpaintimage);
		}
	}
}

*/
//------------------绘制H-S直方图（色调，饱和度）

/*
int main ()
{
	Mat hsvimage;
	srcimage=imread("D:\\img\\1.jpg");
	cvtColor(srcimage,hsvimage,COLOR_BGR2HSV);
	//参数准备
	//将色调量化为30个等级，将饱和度量化为32个等级
	int hueBinNum=30;
	int saturationBinNum=32;
	int histSize[]={hueBinNum,saturationBinNum};

	float hueRanges[]={0,180};
	float saturationRanges[]={0,256};
	const float* ranges[]={hueRanges,saturationRanges};

	MatND dstHist;
	//计算第0通道和第1通道的直方图；
	int channels[]={0,1};

	calcHist(&hsvimage,1,channels,Mat(),dstHist,2,histSize,ranges,true,false);

	//为绘制直方图准备参数
	double maxValue=0;//最大值
	minMaxLoc(dstHist,0,&maxValue,0,0);//在数组中寻找最大值和最小值
	int scale=10;

	Mat histimage=Mat::zeros(saturationBinNum*scale,hueBinNum*10,CV_8UC3);
	//双层循环，进行直方图的绘制
	for(int hue=0;hue<hueBinNum;hue++)
		for (int saturation=0;saturation<saturationBinNum;saturation++)
		{
			float binValue=dstHist.at<float>(hue,saturation);//直方图直条的值

			int intensity=cvRound(binValue*255/maxValue);//强度

			rectangle(histimage,Point(hue*scale,saturation*scale),Point((hue+1)*scale-1,(saturation+1)*scale-1),Scalar::all(intensity),FILLED);

		}
		imshow(WINDOW_SRC,srcimage);
		imshow("H-S直方图",histimage);
		waitKey(0);
}

*/
//------------------------------绘制图像的一维直方图
/*
int main()
{
	srcimage=imread("D:\\img\\9.jpg");
	imshow(WINDOW_SRC,srcimage);

	MatND dstHist;
	int dims=1;
	float hranges[]={0,255};
	const float *ranges[]={hranges};
	int size=256;
	int channels=0;

	calcHist(&srcimage,1,&channels,Mat(),dstHist,dims,&size,ranges);
	int scale=1;
	Mat dstimage(size*scale,size,CV_8U,Scalar(0));

	double minValue=0;
	double maxValue=0;
	minMaxLoc(dstHist,&minValue,&maxValue,0,0);

	//绘制

	int hpt=saturate_cast<int>(0.9*size);

	for (int i=0;i<256;i++)
	{
		float binValue=dstHist.at<float>(i);
		int realValue=saturate_cast<int>(binValue*hpt/maxValue);

		rectangle(dstimage,Point(i*scale,size-1),Point((i+1)*scale-1,size-realValue),Scalar(255));
	}
	imshow("一维直方图",dstimage);
	waitKey(0);
	return 0;

}
*/

//-------------------------绘制RGB三色直方图
/*
int main()
{
	srcimage=imread("D:\\img\\1.jpg");
	imshow(WINDOW_SRC,srcimage);

	int bins=256;
	int hist_size[]={bins};
	float range[]={0,256};
	const float* ranges[]={range};

	MatND redHist,greenHist,blueHist;
	int channels_r[]={0};
	calcHist(&srcimage,1,channels_r,Mat(),redHist,1,hist_size,ranges,true,false);
	int channels_g[]={1};
	calcHist(&srcimage,1,channels_g,Mat(),greenHist,1,hist_size,ranges,true,false);
	int channels_b[]={2};
	calcHist(&srcimage,1,channels_b,Mat(),blueHist,1,hist_size,ranges,true,false);

    double maxValue_red,maxValue_green,maxValue_blue;
	minMaxLoc(redHist,0,&maxValue_red,0,0);
	minMaxLoc(greenHist,0,&maxValue_green,0,0);
	minMaxLoc(blueHist,0,&maxValue_blue,0,0);

	int scale=1;
	int histHeight=256;
	Mat histimage=Mat::zeros(histHeight,bins*3,CV_8UC3);

	for (int i=0;i<bins;i++)
	{
		float binValue_red=redHist.at<float>(i);
		float binValue_green=greenHist.at<float>(i);
		float binValue_blue=blueHist.at<float>(i);

		int intensity_red=cvRound(binValue_red*histHeight/maxValue_red);
		int intensity_blue=cvRound(binValue_blue*histHeight/maxValue_blue);
		int intensity_green=cvRound(binValue_green*histHeight/maxValue_green);

		rectangle(histimage,Point(i*scale,histHeight-1),Point((i+1)*scale-1,histHeight-intensity_red),Scalar(255,0,0));
		rectangle(histimage,Point((i+bins)*scale,histHeight-1),Point((i+bins+1)*scale-1,histHeight-intensity_green),Scalar(0,255,0));
		rectangle(histimage,Point((i+bins*2+1)*scale-1,histHeight-1),Point((i+bins*2+1)*scale-1,histHeight-intensity_blue),Scalar(0,0,255));
	}
	imshow("图像的RGB直方图",histimage);
	waitKey(0);
	return 0;
}
*/


//--------------------------------------模板匹配
/*
Mat g_srcImage,g_templateimage,g_resultImaage;
int g_nMatchMethod;
int g_nMaxTrackBarNum=5;

void on_Matching(int,void*);

int main()
{
	g_srcImage=imread("D:\\img\\3.png");
	g_templateimage=imread("D:\\img\\3.0.png");
	
	namedWindow(WINDOW_SRC,WINDOW_AUTOSIZE);
	namedWindow(WINDOW_DST,WINDOW_AUTOSIZE);

	createTrackbar("方法",WINDOW_SRC,&g_nMatchMethod,g_nMaxTrackBarNum,on_Matching);
	on_Matching(0,0);
	waitKey(0);
	return 0;

}

void on_Matching(int,void*)
{
	Mat srcimage;
	g_srcImage.copyTo(srcimage);

	int resultImage_rows=g_srcImage.rows-g_templateimage.rows+1;
	int resultImage_cols=g_srcImage.cols-g_templateimage.cols+1;
	g_resultImaage.create(resultImage_rows,resultImage_cols,CV_32FC1);

	matchTemplate(g_srcImage,g_templateimage,g_resultImaage,g_nMatchMethod);
	normalize(g_resultImaage,g_resultImaage,0,1,NORM_MINMAX,-1,Mat());

	double minValue;double maxValue;Point minLocation;Point maxLocation;
	Point matchLocation;
	minMaxLoc(g_resultImaage,&minValue,&maxValue,&minLocation,&maxLocation,Mat());

	if(g_nMatchMethod==TM_SQDIFF||g_nMatchMethod==TM_SQDIFF_NORMED)matchLocation=minLocation;
	else matchLocation=maxLocation;

	rectangle(srcimage,matchLocation,Point(matchLocation.x+g_templateimage.cols,matchLocation.y+g_templateimage.rows),Scalar(0,0,255),2,8,0);
    rectangle(g_resultImaage,matchLocation,Point(matchLocation.x+g_templateimage.cols,matchLocation.y+g_templateimage.rows),Scalar(0,0,255),2,8,0);

	imshow(WINDOW_SRC,srcimage);
	imshow(WINDOW_DST,g_resultImaage);
}
*/

//--------------直方图对比
/*
int main()
{
	Mat srcimage_base,hsvimage_base;
	Mat srcimage_test1,hsvimage_test1;
	Mat srcimage_test2,hsvimage_test2;
	Mat hsvimage_halfDown;

	srcimage_base=imread("D:\\img\\15.jpg",1);
	srcimage_test1=imread("D:\\img\\15.1.jpg",1);
	srcimage_test2=imread("D:\\img\\15.2.jpg",1);

	imshow("基准图像",srcimage_base);
	imshow("测试图像1",srcimage_test1);
	imshow("测试图像2",srcimage_test2);

	cvtColor(srcimage_base,hsvimage_base,COLOR_BGR2HSV);
	cvtColor(srcimage_test1,hsvimage_test1,COLOR_BGR2HSV);
	cvtColor(srcimage_test2,hsvimage_test2,COLOR_BGR2HSV);

	hsvimage_halfDown=hsvimage_base(Range(hsvimage_base.rows/2,hsvimage_base.rows-1),Range(0,hsvimage_base.cols-1));

	//hue通道使用30个bin，saturation通道使用32个bin
	int h_bins=50;int s_bins=60;
	int histSize[]={h_bins,s_bins};
	//hue取值范围时0-256，saturation取值范围时0-180
	float h_ranges[]={0,256};
	float s_ranges[]={0,180};

	const float *ranges[]={h_ranges,s_ranges};
	//使用第0和第1通道

	int channels[]={0,1};

	MatND baseHist;
	MatND testHist1;
	MatND testHist2;
	MatND halfDownHist;

	calcHist(&hsvimage_base,1,channels,Mat(),baseHist,2,histSize,ranges,true,false);
	normalize(baseHist,baseHist,0,1,NORM_MINMAX,-1,Mat());
    
	calcHist(&hsvimage_halfDown,1,channels,Mat(),halfDownHist,2,histSize,ranges,true,false);
	normalize(halfDownHist,halfDownHist,0,1,NORM_MINMAX,-1,Mat());

	calcHist(&hsvimage_test1,1,channels,Mat(),testHist1,2,histSize,ranges,true,false);
	normalize(testHist1,testHist1,0,1,NORM_MINMAX,-1,Mat());

	calcHist(&hsvimage_test2,1,channels,Mat(),testHist2,2,histSize,ranges,true,false);
	normalize(testHist2,testHist2,0,1,NORM_MINMAX,-1,Mat());

	for (int i=0;i<4;i++)
	{
		int compare_method=i;
		double base_base=compareHist(baseHist,baseHist,compare_method);
		double base_half=compareHist(baseHist,halfDownHist,compare_method);
		double base_test1=compareHist(baseHist,testHist1,compare_method);
		double base_test2=compareHist(baseHist,testHist2,compare_method);

		printf("方法【%d】直方图匹配结果如下：\n\n 【base-base】：%f\n,【base-半身图】：%f\n,【base-测试图1】：%f\n,【base-测试图2】：%f\n",
			i,base_base,base_half,base_test1,base_test2);
	}
	printf("检测结束！");
	waitKey(0);
	return 0;
}
*/

//---------------反向投影
/*
Mat g_srcimage,g_hsvimage,g_hueimage;
int g_bins=30;
void on_BinChange(int,void*);

int main()
{
    g_srcimage=imread("D:\\img\\16.jfif",1);
	cvtColor(g_srcimage,g_hsvimage,COLOR_BGR2HSV);
	
	//分离Hue色调通道
	g_hueimage.create(g_hsvimage.size(),g_hsvimage.depth());
	int ch[]={0,0};
	mixChannels(&g_hsvimage,1,&g_hueimage,1,ch,1);

	namedWindow(WINDOW_SRC,WINDOW_NORMAL);
	createTrackbar("色调组距",WINDOW_SRC,&g_bins,180,on_BinChange);
	on_BinChange(0,0);

	imshow(WINDOW_SRC,g_srcimage);
	waitKey(0);
	return 0;
}
void on_BinChange(int,void*)
{
	//参数准备
	MatND hist;
	int histSize=MAX(g_bins,2);
	float hue_range[]={0,180};
	const float* ranges={hue_range};
	//计算直方图并归一化
	calcHist(&g_hueimage,1,0,Mat(),hist,1,&histSize,&ranges,true,false);
	normalize(hist,hist,0,255,NORM_MINMAX,-1,Mat());

	MatND backproj;
	calcBackProject(&g_hueimage,1,0,hist,backproj,&ranges,1,true);

	namedWindow("反向投影图",WINDOW_NORMAL);
	imshow("反向投影图",backproj);
	//绘制直方图
	int w=400;
	int h=400;
	int bin_w=cvRound((double)w/histSize);
	Mat histImg=Mat::zeros(w,h,CV_8UC3);

	for (int i=0;i<g_bins;i++)
    rectangle(histImg,Point(i*bin_w,h),Point((i+1)*bin_w,h-cvRound(hist.at<float>(i)*h/255.0)),Scalar(100,123,255),-1);

	imshow("直方图",histImg);

}
*/

//----------------harris角点检测
/*
int main()
{
	srcimage=imread("D:\\img\\17.jpg",0);
	imshow(WINDOW_SRC,srcimage);

	Mat cornerStrength;
	cornerHarris(srcimage,cornerStrength,2,3,0.01);
	Mat harrisCorner;
	threshold(cornerStrength,harrisCorner,0.00001,255,THRESH_BINARY);
	imshow("角点检测之后的二值图",harrisCorner);
	waitKey(0);
	return 0;
}
*/
/*
Mat g_srcimage,g_srcimage1;
int thresh=80;//阈值太低，处理速度会很慢
int max_thresh=175;

void on_CornerHarris(int,void*);
int main()
{
	g_srcimage=imread("D:\\img\\17.jpg",1);
	imshow("yuanshi",g_srcimage);
	g_srcimage1=g_srcimage.clone();
	cvtColor(g_srcimage1,grayimage,COLOR_BGR2GRAY);
	namedWindow(WINDOW_SRC,WINDOW_AUTOSIZE);
	createTrackbar("阈值",WINDOW_SRC,&thresh,max_thresh,on_CornerHarris);
	on_CornerHarris(0,0);
	waitKey(0);
	return 0;
}

void on_CornerHarris(int,void*)
{
	Mat normimage;//归一化后的图像
	Mat scaledImage;//线性变换后的八位无符号整形图

	dstimage=Mat::zeros(g_srcimage.size(),CV_32FC1);
	g_srcimage1=g_srcimage.clone();

	cornerHarris(grayimage,dstimage,2,3,0.04,BORDER_DEFAULT);
	normalize(dstimage,normimage,0,255,NORM_MINMAX,CV_32FC1,Mat());
	convertScaleAbs(normimage,scaledImage);

	for (int j=0;j<normimage.rows;j++)
	{
		for (int i=0;i<normimage.cols;i++)
		{
			if ((int)normimage.at<float>(j,i) > (thresh+80))
			{
				circle(g_srcimage1,Point(i,j),5,Scalar(10,10,255),2,8,0);
				circle(scaledImage,Point(i,j),5,Scalar(0,10,255),2,8,0);
			}
		}
	}

	imshow(WINDOW_SRC,g_srcimage1);
	imshow(WINDOW_DST,scaledImage);
}
*/

//--------------石-Tomasi角点检测+亚像素级角点检测
/*
int g_MaxCornerNum=33;
int g_MaxTrackerbarNum=500;
RNG g_rng(12345);
void on_GoodFeaturesToTrack(int,void*)
{
	if (g_MaxCornerNum<=1)g_MaxCornerNum=1;
    
	vector<Point2f>corners;
	double qualityLevel=0.01;//角点检测可以接受的最小特征值
	double minDistance=10;//角点之间的最小距离
	int blockSize=3;
	double k=0.4;
	Mat copy=srcimage.clone();

	goodFeaturesToTrack(grayimage,corners,g_MaxCornerNum,qualityLevel,minDistance,Mat(),blockSize,false,k);

	cout<<"此次检测到的角点数量为："<<corners.size()<<endl;

	int r=4;
	for (int i=0;i<corners.size();i++)
	{
		circle(copy,corners[i],r,Scalar(g_rng.uniform(0,255),g_rng.uniform(0,255),g_rng.uniform(0,255)),-1,8,0);
	}

	//-------------亚像素角点检测-------------
	Size winSize=Size(5,5);
	Size zeroSize=Size(-1,-1);

	TermCriteria criteria=TermCriteria(TermCriteria::EPS+TermCriteria::MAX_ITER,40,0.001);
	cornerSubPix(grayimage,corners,winSize,zeroSize,criteria);
	for (int i=0;i<corners.size();i++)
		cout<<"\t精确角点的坐标["<<i<<"]("<<corners[i].x<<","<<corners[i].y<<")"<<endl;
	//----------------------------------------
	imshow(WINDOW_SRC,copy);

}

int main()
{
	srcimage=imread("D:\\img\\19.jfif",1);
	cvtColor(srcimage,grayimage,COLOR_BGR2GRAY);
	namedWindow(WINDOW_SRC,WINDOW_AUTOSIZE);
	createTrackbar("最大角点数量",WINDOW_SRC,&g_MaxCornerNum,g_MaxTrackerbarNum,on_GoodFeaturesToTrack);
	imshow(WINDOW_SRC,srcimage);
	on_GoodFeaturesToTrack(0,0);
	waitKey(0);
	return 0;
}
*/

//--------------------SURF特征点检测
/*
int main()
{
	Mat srcimage1=imread("D:\\img\\15.jpg");
	Mat srcimage2=imread("D:\\img\\15.2.jpg");
	imshow("原始图1",srcimage1);
	imshow("原始图2",srcimage2);
	//定义要用到的变量和类
	int minHessian=400;//定义SURF中的Hessian阈值特征点检测算子
	SurfFeatureDetector detector(minHessian);//定义一个特征检测类的对象
	std::vector<KeyPoint>keypoints_1,keypoints_2;

	detector.detect(srcimage1,keypoints_1);
	detector.detect(srcimage2,keypoints_2);

	//绘制特征关键点
	Mat img_keypoints_1;
	Mat img_keypoints_2;

	drawKeypoints(srcimage1,keypoints_1,img_keypoints_1,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
	drawKeypoints(srcimage2,keypoints_2,img_keypoints_2,Scalar::all(-1),DrawMatchesFlags::DEFAULT);

	imshow("特征点检测效果图1",srcimage1);
	imshow("特征点检测效果图2",srcimage2);

	waitKey(0);
	return 0;
}
*/

//--------------SURF特征提取----------

/*
int main()
{
	Mat srcimage1=imread("D:\\img\\15.jpg");
	Mat srcimage2=imread("D:\\img\\15.2.jpg");
	imshow("原始图1",srcimage1);
	imshow("原始图2",srcimage2);
	//定义要用到的变量和类
	int minHessian=400;//定义SURF中的Hessian阈值特征点检测算子
	SurfFeatureDetector detector(minHessian);//定义一个特征检测类的对象
	std::vector<KeyPoint>keypoints_1,keypoints_2;

	detector.detect(srcimage1,keypoints_1);
	detector.detect(srcimage2,keypoints_2);

	//计算描述符
	SurfDescriptorExtractor extractor;
	Mat descriptors1,descriptors2;
	extractor.compute(srcimage1,keypoints_1,descriptors1);
	extractor.compute(srcimage2,keypoints_2,descriptors2);

	//实例化一个匹配器
	BruteForceMatcher<L2<float>>matcher;
	std::vector<DMatch>matches;
	matcher.match(descriptors1,descriptors2,matches);

	//绘制匹配的关键点
	Mat imgMatches;
	drawMatches(srcimage1,keypoints_1,srcimage2,keypoints_2,matches,imgMatches);

	imshow("匹配图",imgMatches);
	waitKey(0);
	return 0;
}

*/
//-------------使用FLANN进行特征点的匹配----
/*
int main()
{
	Mat image1=imread("D:\\img\\15.jpg");
	Mat image2=imread("D:\\img\\15.2.jpg");
	imshow("原始图1",image1);
	imshow("原始图2",image2);
	//定义要用到的变量和类
	int minHessian=300;//定义SURF中的Hessian阈值特征点检测算子
	SURF detector(minHessian);//定义一个特征检测类的对象
	std::vector<KeyPoint>keypoints_1,keypoints_2;

	detector.detect(image1,keypoints_1);
	detector.detect(image2,keypoints_2);
	//计算描述符（特征向量），
	SURF extractor;
	Mat descriptors_1,descriptors_2;
	extractor.compute(image1,keypoints_1,descriptors_1);
	extractor.compute(image2,keypoints_2,descriptors_2);
	//采用FLANN算法匹配描述符向量
	FlannBasedMatcher matcher;
	std::vector<DMatch>matches;
	matcher.match(descriptors_1,descriptors_2,matches);
    

	double max_dist=0;double min_dist=100;
	for (int i=0;i<descriptors_1.rows;i++)
	{
		double dist=matches[i].distance;
		if(dist<min_dist)min_dist=dist;
		if (dist>max_dist)max_dist=dist;
	}
	printf(">最大距离：%f\n",max_dist);
	printf(">最小距离：%f\n",min_dist);

	std::vector<DMatch>good_matches;
	for (int i=0;i<descriptors_1.rows;i++)
		if (matches[i].distance<2*min_dist)good_matches.push_back(matches[i]);
	
	//绘制出符合条件的匹配点
	Mat img_matches;
	drawMatches(image1,keypoints_1,image2,keypoints_2,good_matches,img_matches,Scalar::all(-1),Scalar::all(-1),vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	for (int i=0;i<good_matches.size();i++)
     printf(">符合条件的匹配点【%d】特征点1：%d---特征点2：%d\n",i,good_matches[i].queryIdx,good_matches[i].trainIdx);

	imshow("匹配效果图",img_matches);
	waitKey(0);
	return 0;
}
*/

//----------------寻找已知物体-------
/*
int main()
{
	Mat srcimage1=imread("D:\\img\\15.jpg");
	Mat srcimage2=imread("D:\\img\\15.2.jpg");

	int minHessian=400;
	SurfFeatureDetector detector(minHessian);
	vector<KeyPoint>keypoints_object,keypoints_scene;
    
	detector.detect(srcimage1,keypoints_object);
	detector.detect(srcimage2,keypoints_scene);

	SurfDescriptorExtractor extractor;
	Mat descriptors_object,descriptors_scene;
	extractor.compute(srcimage1,keypoints_object,descriptors_object);
	extractor.compute(srcimage2,keypoints_scene,descriptors_scene);

	FlannBasedMatcher matcher;
	vector<DMatch>matches;
	matcher.match(descriptors_object,descriptors_scene,matches);
	double max_dist=0;double min_dist=100;

	for (int i=0;i<descriptors_object.rows;i++)
	{
		double dist=matches[i].distance;
		if (dist<min_dist)min_dist=dist;
		if (dist>max_dist)max_dist=dist;
	}
	printf(">Max disi 最大距离：%f\n",max_dist);
	printf(">Min disi 最小距离：%f\n",min_dist);

	//存下匹配距离小于3*min_dist的点对
	std::vector<DMatch>good_matches;
	for (int i=0;i<descriptors_object.rows;i++)
	{
		if (matches[i].distance<3*min_dist)
			good_matches.push_back(matches[i]);
	}

	//绘制
	Mat img_matches;
	drawMatches(srcimage1,keypoints_object,srcimage2,keypoints_scene,good_matches,img_matches,
		Scalar::all(-1),Scalar::all(-1),vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	vector<Point2f>obj;
	vector<Point2f>scene;
	
	//从匹配成功的匹配对中获取关键点
	for (unsigned int i=0;i<good_matches.size();i++)
	{
		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	}
	
	Mat H=findHomography(obj,,scene,CV_RANSAC);//计算透视变换

	//从待测图片中获得角点
	vector<Point2f>obj_corners(4);
	obj_corners[0]=CvPoint(0,0);
	obj_corners[1]=CvPoint(srcimage1.cols,0);
	obj_corners[2]=CvPoint(0,srcimage1.rows);
	obj_corners[3]=CvPoint(srcimage1.cols,srcimage1.rows);
	vector<Point2f>scene_corners(4);

	//进行透视变化
	perspectiveTransform(obj_corners,scene_corners,H);

	//绘制出角点之间的直线
	line(img_matches,scene_corners[0]+Point2f(static_cast<float>(srcimage1.cols),0),
		scene_corners[1]+Point2f(static_cast<float>(srcimage1.cols),0),Scalar(255,0,123),4);
	line(img_matches,scene_corners[1]+Point2f(static_cast<float>(srcimage1.cols),0),
		scene_corners[2]+Point2f(static_cast<float>(srcimage1.cols),0),Scalar(255,0,123),4);
	line(img_matches,scene_corners[2]+Point2f(static_cast<float>(srcimage1.cols),0),
		scene_corners[3]+Point2f(static_cast<float>(srcimage1.cols),0),Scalar(255,0,123),4);
	line(img_matches,scene_corners[3]+Point2f(static_cast<float>(srcimage1.cols),0),
		scene_corners[0]+Point2f(static_cast<float>(srcimage1.cols),0),Scalar(255,0,123),4);

	imshow("GOOD Matches & Object detetcion",img_matches);
	waitKey(0);
	return 0;
}
*/

//----------ORB------------------、
/*
int main()
{
	srcimage=imread("15.jpg");
	imshow(WINDOW_SRC,srcimage);
	cvtColor(srcimage,grayimage,COLOR_BGR2GRAY);
	
	//------检测SIFT特征点并在图像中提取物体的描述符
	OrbFeatureDetector featureDetector;
    vector<KeyPoint>keypoints;
	Mat descriptors;

	featureDetector.detect(grayimage,keypoints);
    
	OrbDescriptorExtractor featureExtractor;
	featureExtractor.compute(grayimage,keypoints,descriptors);

	//基于FLANN的描述符对象匹配
	flann::Index flannIndex(descriptors,flann::LshIndexParams(12,20,2),cvflann::FLANN_DIST_HAMMING);
	//初始化视频采集对象
	VideoCapture cap(0);
	cap.set(CV_CAP_PROP_FRAME_WIDTH,360);//采集视频的宽度
	cap.set(CV_CAP_PROP_FRAME_HEIGHT,900);//采集视频的高度
	unsigned int frameCount=0;//帧数

	while (1)
	{
		double time0=static_cast<double>(getTickCount());//记录起始时间
		Mat captureImage,captureImage_gray;
		cap>>captureImage;//采集视频帧
		if (captureImage.empty())continue;//采集为空的处理
		cvtColor(captureImage,captureImage_gray,COLOR_BGR2GRAY);
        
		vector<KeyPoint>captureKeyPoints;
		Mat captureDescription;
		featureDetector.detect(captureImage_gray,captureKeyPoints);
		featureExtractor.compute(captureImage_gray,captureKeyPoints,captureDescription);

		//匹配和测试描述符，获取两个最邻近的描述符
		Mat matchIndex(captureDescription.rows,2,CV_32SC1),matchDistance(captureDescription.rows,2,CV_32FC1);
		flannIndex.knnSearch(captureDescription,matchIndex,matchDistance,2,flann::SearchParams());//调用K邻近算法

		//Lowe's algorithm选出优秀的匹配
		vector<DMatch>goodMatches;
		for (int i=0;i<matchDistance.rows;i++)
		{
			if (matchDistance.at<float>(i,0)<0.6*matchDistance.at<float>(i,1))
			{
				DMatch dmatches(i,matchIndex.at<int>(i,0),matchDistance.at<float>(i,0));
				goodMatches.push_back(dmatches);
			}
		}
		//绘制
		Mat resultImage;
	    drawMatches(captureImage,captureKeyPoints,srcimage,keypoints,goodMatches,resultImage);
		imshow("匹配窗口",resultImage);

		//显示帧率
		cout<<">帧率："<<getTickFrequency()/(getTickCount()-time0)<<endl;
		if((char)(waitKey(1))==27)break;
	}
	return 0;
}
*/

//-----------------滤波---------------
/*
Mat g_srcimage1,g_dstimage1,g_dstimage2,g_dstimage3,g_dstimage4,g_dstimage5;
int g_nBoxFilterValue=6;//方框滤波内核值
int g_nMeanBlurValue=10;//均值滤波内核值
int g_nGussainBlurValue=6;//高斯滤波内核值
int g_MedianBlurValue=10;//中值滤波参数值
int g_nBilaterFilterValue=10;//双边滤波参数值

static void on_BoxFilter(int,void*);
static void on_MeanBlur(int,void*);
static void on_GaussianBlur(int,void*);
static void on_MedianBlur(int,void*);
static void on_BilaterFilter(int,void*);

int main()
{
	srcimage=imread("D:\\img\\3.png");
	g_dstimage1=srcimage.clone();
	g_dstimage2=srcimage.clone();
	g_dstimage3=srcimage.clone();
	g_dstimage4=srcimage.clone();
	g_dstimage5=srcimage.clone();
	namedWindow(WINDOW_SRC,WINDOW_AUTOSIZE);
    imshow(WINDOW_SRC,srcimage);

	//---------------------方框滤波---------
	namedWindow("【方框滤波】",WINDOW_AUTOSIZE);
	createTrackbar("内核值","【方框滤波】",&g_nBoxFilterValue,50,on_BoxFilter);
	on_BoxFilter(g_nBoxFilterValue,0);
	//---------------------均值滤波---------
	namedWindow("【均值滤波】",WINDOW_AUTOSIZE);
	createTrackbar("内核值","【均值滤波】",&g_nMeanBlurValue,50,on_MeanBlur);
	on_MeanBlur(g_nMeanBlurValue,0);
	
	//---------------------高斯滤波---------
	namedWindow("【高斯滤波】",WINDOW_AUTOSIZE);
	createTrackbar("内核值","【高斯滤波】",&g_nGussainBlurValue,50,on_GaussianBlur);
	on_GaussianBlur(g_nGussainBlurValue,0);
	
	//---------------------中值滤波---------
	namedWindow("【中值滤波】",WINDOW_AUTOSIZE);
	createTrackbar("内核值","【中值滤波】",&g_MedianBlurValue,50,on_MedianBlur);
	on_MedianBlur(g_MedianBlurValue,0);
	
	//---------------------双边滤波---------
	namedWindow("【双边滤波】",WINDOW_AUTOSIZE);
	createTrackbar("内核值","【双边滤波】",&g_nBilaterFilterValue,50,on_BilaterFilter);
	on_BilaterFilter(g_nBilaterFilterValue,0);
	

	while (char(waitKey(1))!='q'){};
	return 0;
	
}
static void on_BoxFilter(int,void*)
{
	boxFilter(srcimage,g_dstimage1,-1,Size(g_nBoxFilterValue+1,g_nBoxFilterValue+1));
	imshow("【方框滤波】",g_dstimage1);
}
static void on_MeanBlur(int,void*)
{
	blur(srcimage,g_dstimage2,Size(g_nMeanBlurValue+1,g_nMeanBlurValue+1),Point(-1,-1));
	imshow("【均值滤波】",g_dstimage2);
}

static void on_GaussianBlur(int,void*)
{
	GaussianBlur(srcimage,g_dstimage3,Size(g_nGussainBlurValue*2+1,g_nGussainBlurValue*2+1),0,0);
	imshow("【高斯滤波】",g_dstimage3);
}

static void on_MedianBlur(int,void*)
{
	medianBlur(srcimage,g_dstimage4,g_MedianBlurValue*2+1);
	imshow("【中值滤波】",g_dstimage4);
}
static void on_BilaterFilter(int,void*)
{
	bilateralFilter(srcimage,g_dstimage5,g_nBilaterFilterValue,g_nBilaterFilterValue*2,g_nBilaterFilterValue/2);
	imshow("【双边滤波】",g_dstimage5);
}

*/

//---------形态学滤波
/*
int g_nElementShape=MORPH_RECT;

int g_nMaxIterationNum=10;
int g_nOpenCloseNum=0;
int g_nErodeDilationNum=0;
int g_nTopBlackHatNum=0;

static void on_OpenClose(int,void*);
static void on_ErodeDilate(int ,void*);
static void on_TopBlackHat(int ,void*);

int main()
{
	srcimage=imread("D:\\img\\3.png");
	namedWindow(WINDOW_SRC,WINDOW_AUTOSIZE);
	imshow(WINDOW_SRC,srcimage);
	
	namedWindow("【开运算/闭运算】",1);
	namedWindow("【腐蚀/膨胀】",1);
	namedWindow("【顶帽/黑帽】",1);

	g_nOpenCloseNum=9;
	g_nErodeDilationNum=9;
	g_nTopBlackHatNum=2;

	createTrackbar("迭代值","【开运算/闭运算】",&g_nOpenCloseNum,g_nMaxIterationNum*2+1,on_OpenClose);
	createTrackbar("迭代值","【腐蚀/膨胀】",&g_nErodeDilationNum,g_nMaxIterationNum*2+1,on_ErodeDilate);
	createTrackbar("迭代值","【顶帽/黑帽】",&g_nTopBlackHatNum,g_nMaxIterationNum*2+1,on_TopBlackHat);
	while (1)
	{
		int c;
		on_OpenClose(g_nOpenCloseNum,0);
		on_ErodeDilate(g_nErodeDilationNum,0);
		on_TopBlackHat(g_nTopBlackHatNum,0);

		c=waitKey(0);
		if ((char)c=='q'||(char)c==27)break;
		if ((char)c==49)g_nElementShape=MORPH_ELLIPSE;
		else if((char)c==50)g_nElementShape=MORPH_RECT;
		else if((char)c==51)g_nElementShape=MORPH_CROSS;
		else if ((char)c==' ')g_nElementShape=(g_nElementShape+1)%3;
	}
	return 0;
}
static void on_OpenClose(int,void*)
{
	int offset=g_nOpenCloseNum-g_nMaxIterationNum;
	int Absolute_offset=offset>0?offset:-offset;
	Mat element=getStructuringElement(g_nElementShape,Size(Absolute_offset*2+1,Absolute_offset*2+1),Point(Absolute_offset,Absolute_offset));
	if (offset<0)morphologyEx(srcimage,dstimage,MORPH_OPEN,element);
	else morphologyEx(srcimage,dstimage,MORPH_CLOSE,element);
	imshow("【开运算/闭运算】",dstimage);
}

static void on_ErodeDilate(int ,void*)
{
	int offset=g_nErodeDilationNum-g_nMaxIterationNum;
	int Absolute_offset=offset>0?offset:-offset;
	Mat element=getStructuringElement(g_nElementShape,Size(Absolute_offset*2+1,Absolute_offset*2+1),Point(Absolute_offset,Absolute_offset));
	if (offset<0)erode(srcimage,dstimage,element);
	else dilate(srcimage,dstimage,element);
	imshow("【腐蚀/膨胀】",dstimage);
}

static void on_TopBlackHat(int ,void*)
{
	int offset=g_nTopBlackHatNum-g_nMaxIterationNum;
	int Absolute_offset=offset>0?offset:-offset;
	Mat element=getStructuringElement(g_nElementShape,Size(Absolute_offset*2+1,Absolute_offset*2+1),Point(Absolute_offset,Absolute_offset));
	if (offset<0)morphologyEx(srcimage,dstimage,MORPH_TOPHAT,element);
	else morphologyEx(srcimage,dstimage,MORPH_BLACKHAT,element);
	imshow("【顶帽/黑帽】",dstimage);
}
*/

//simpleblobdetector
/*
int main()
{
	Mat srcimage;
	srcimage=imread("D:\\img\\20.jfif",0);
	SimpleBlobDetector::Params params;
	
	
	//阈值控制
	params.minThreshold = 50;
	params.maxThreshold = 100;
	//像素面积大小控制
	params.filterByArea = true;
	params.minArea = 80;
	//形状（凸）
	params.filterByCircularity = true;
	params.minCircularity = 0.5;
	//形状（凹）
	params.filterByConvexity = false;
	params.minConvexity = 0.9;
	//形状（园）
	params.filterByInertia = true;
	params.minInertiaRatio = 0.5;

	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
	vector<KeyPoint> keypoints;
	detector->detect(srcimage,keypoints);
	Mat img_with_keypoints;
	drawKeypoints(srcimage,keypoints,img_with_keypoints,Scalar(0,0,255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("keypoints",img_with_keypoints);
	waitKey(0);
	return 0;
}

*/
