#include <iostream>
#include "opencv2/opencv.hpp"              // OpenCV 관련 헤더 파일을 포함
#include <ctime>
#include <fstream>

using namespace cv;                        // cv와 std 네임스페이스를 사용하도록 설정
using namespace std;

const int32_t Width = 640;
const int32_t Height = 480;
const int32_t Y_offset = 400;

vector<Vec4f> imageProcess(Mat& frame);
void drawLine(Mat& img, Point start_p,Point end_p, Scalar color = Scalar(0, 255, 255));
void drawCircle(Mat& img, Point center, Scalar color = Scalar(255, 255, 0));
string getCSVfilename();
void addDataCSV(string filename, int index_, int frame, int lposl, int lposr,int rposl, int rposr);
vector<int> laneDetection(Mat& frame, vector<Vec4f> all_lines);

int main()
{
    //영상 파일 열기 
    VideoCapture cap;    
    cap.open("resource/Sub_project.avi");
    cap.set(CAP_PROP_POS_FRAMES, 0);
	cap.set(CAP_PROP_FPS, 30);

    if (!cap.isOpened()){
        cerr<<"Camera open failed!"<<endl;
        return -1 ;
    }

    Mat frame;   
    string filename=getCSVfilename();
    int index=0, total_frame = cap.get(CAP_PROP_FRAME_COUNT), fps=0;
    while(total_frame>fps){        
        cap >> frame;
        fps=cap.get(cv::CAP_PROP_POS_FRAMES);
        if (frame.empty()){
            cerr<<"Frame empty!"<<endl;
            return -1 ;
        }
        
        vector<int> all_pos = laneDetection(frame,imageProcess(frame));
        
        if (fps % 30 == 0)
		{
			addDataCSV(filename,index,fps,all_pos[0],all_pos[1],all_pos[2],all_pos[3]);			
            ++index;
		}
        
        imshow("frame",frame);
       
        if (waitKey(1)==27)
            break;
    }
    cap.release();
    destroyAllWindows();
}

vector<Vec4f> imageProcess(Mat& frame)
{
	Mat img_gray, img_histo, img_blur, img_edge, roi, thresframe;
    vector<Vec4f> lines;

    Mat output;
    Mat mask = Mat::zeros(frame.size(), CV_8UC1);

    vector<Point> polygon;
    
    polygon.push_back(Point(0, frame.rows));
    polygon.push_back(Point(50, 340));
    polygon.push_back(Point(580, 340));
    polygon.push_back(Point(frame.cols, frame.rows));

    fillConvexPoly(mask, &polygon[0],4, Scalar(255));

    vector<Point> square;
    square.push_back(Point(220,frame.rows));
    square.push_back(Point(220, 390)); 
    square.push_back(Point(430,390)); 
    square.push_back(Point(430,frame.rows)); 

    fillConvexPoly(mask, &square[0], 4, Scalar(0));

    cvtColor(frame, img_gray, COLOR_BGR2GRAY);	

    GaussianBlur(img_gray, img_blur, Size(), 1.0);
    Canny(img_blur, img_edge, 70, 150);


    bitwise_and(img_edge, mask, output, mask= mask);    

    threshold(output,thresframe,190,255,THRESH_BINARY);

    HoughLinesP(thresframe, lines, 1, CV_PI / 180, 50, 70, 30);


    imshow("thresframe", thresframe);
    imshow("mask", mask);
	imshow("img_edge",img_edge);
	return lines;
}

vector<int> laneDetection(Mat& frame, vector<Vec4f> all_lines)
{    
    vector<Point> left_points, right_points;
    Point lposl, lposr, rposl, rposr;
    vector<int> left_x_at_Y_offset;
    vector<int> right_x_at_Y_offset;

    for (Vec4f line_ : all_lines) {
        Point pt1(line_[0], line_[1]);
        Point pt2(line_[2], line_[3]);
        double slope = (double)(pt2.y - pt1.y) / (pt2.x - pt1.x + 0.0001);

        if (abs(slope) < 5) { 
            int x_at_Y_offset;
            if (pt1.x != pt2.x) {
                x_at_Y_offset = (Y_offset - pt1.y) / slope + pt1.x;
            } else {
                x_at_Y_offset = pt1.x; 
            }

            if (slope < 0) {
                left_x_at_Y_offset.push_back(x_at_Y_offset);
            } else if (slope > 0) {
                right_x_at_Y_offset.push_back(x_at_Y_offset);
            }
        }
        drawLine(frame, pt1, pt2);
    }

    if (!left_x_at_Y_offset.empty()) {
        int lposl_x = *min_element(left_x_at_Y_offset.begin(), left_x_at_Y_offset.end());
        int lposr_x = *max_element(left_x_at_Y_offset.begin(), left_x_at_Y_offset.end());
        lposl = Point(lposl_x, Y_offset);
        lposr = Point(lposr_x, Y_offset);
        drawCircle(frame, (lposl+lposr)/2, Scalar(0, 255, 0)); 
        putText(frame, format("(%d, %d)", (lposl.x + lposr.x) / 2, Y_offset), 
            (lposl + lposr) / 2 + Point(-50, -20), 
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1, LINE_AA);
    }

    if (!right_x_at_Y_offset.empty()) {
        int rposl_x = *min_element(right_x_at_Y_offset.begin(), right_x_at_Y_offset.end());
        int rposr_x = *max_element(right_x_at_Y_offset.begin(), right_x_at_Y_offset.end());
        rposl = Point(rposl_x, Y_offset);
        rposr = Point(rposr_x, Y_offset);
        drawCircle(frame, (rposl+rposr)/2, Scalar(0, 255, 0)); 
        putText(frame, format("(%d, %d)", (rposl.x + rposr.x) / 2, Y_offset), 
            (rposl + rposr) / 2 + Point(-50, -20), 
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1, LINE_AA);
    }
    vector<int> all_pos = {lposl.x,lposr.x,rposl.x,rposr.x };
    
    return all_pos;
}

void drawLine(Mat& frame, Point start_p,Point end_p, Scalar color)
{		
	line(frame,start_p, end_p, color, 2, LINE_AA);
}

void drawCircle(Mat& frame, Point center, Scalar color)
{
    circle(frame,center,5,color,2, LINE_AA);
}

string getCSVfilename(){
    string str_buf;            
    time_t curTime = time(NULL); 
    struct tm* pLocal = localtime(&curTime);
    
    str_buf=to_string(pLocal->tm_year + 1900)+to_string(pLocal->tm_mon + 1)+to_string(pLocal->tm_mday)+ "_" + to_string(pLocal->tm_hour) + to_string(pLocal->tm_min) + to_string(pLocal->tm_sec)+"_data.csv";
    return str_buf;
}

void addDataCSV(string filename, int index_, int frame, int lposl, int lposr,int rposl, int rposr)
{ 
    ifstream check_file;
    check_file.open(filename);
    if(!check_file)
    {   
        ofstream outfile(filename,ios_base::out |ios_base::app); 
        outfile<<"index,frame,lposl,lposr,rposl,rposr,new lposl, new lposr, new rposl, new rposr"<<endl;
        outfile.close();
    }
    check_file.close();
    ofstream outfile(filename,ios_base::out |ios_base::app);    
	
    int new_lposl=lposl-2, new_lposr=lposr+2,new_rposl=rposl-2,new_rposr=rposr+2;
    if (new_lposl < 0) new_lposl = 0;
    if (new_lposr > 640 ) new_lposr = 640;
    if (new_rposl < 0) new_rposl = 0;
    if (new_rposr > 640) new_rposr = 640;
    outfile<<index_<<","<<frame<<","<<lposl<<","<<lposr<<","<<rposl<<","<<rposr<<","<<new_lposl<<","<<new_lposr<<","<<new_rposl<<","<<new_rposr<<endl; 
    outfile.close();      
}
