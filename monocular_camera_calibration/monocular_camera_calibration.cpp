
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <sstream> 

#include <unistd.h>

using namespace cv;  
using namespace std;  
 
const static int NUM_CHESSBOARD = 16;
const static int NUM_VERIFY_CHESSBOARD = 1;

//According to chessboard direction to set below value.
const static int NUM_ROW_CORNERS = 8;
const static int NUM_COL_CORNERS = 11;

const static int CHESS_SQUARE_WIDTH = 3;  //cm
const static int CHESS_SQUARE_HEIGTH = 3;  //cm

const static char calibration_images_path[] = "./caliberation_data.txt";
const static char output_calibrated_detail_result_path[] = "./caliberation_result.txt";

int main()   
{  
    /*Images with chessboard to calibrate*/ 
    ifstream fin();            
    /* Save the result of calibration*/   	
    ofstream fout(output_calibrated_detail_result_path,ios::out|ios::trunc);  

    cout<<"Extracting corners……"<<endl;
    /* The bumber of images for calibrating */ 	
    int image_count=0;  
    Size image_size;
	/*The number of column and row for corners*/
    Size board_size = Size(NUM_COL_CORNERS,NUM_ROW_CORNERS);  
    /* The number of the detected corners for each image */      
    vector<Point2f> image_points_buf;  
	/* The total number of the detected corners for all calibration images */
    vector<vector<Point2f> > image_points_seq;
    string filename;  
	/*for counting the detected corners*/
    int count= 0 ;
		
    while( image_count < NUM_CHESSBOARD )  
    {  
        image_count++;        

        cout<<"image_count = "<<image_count<<endl;          
        cout<<"-->count = "<<count<<endl; 

		stringstream  str;
		str<<image_count<<endl;
        str>>filename;

		filename += ".png";
        cout<<"filename:"<<filename<<endl;
        Mat imageInput = imread("./camera_test/" + filename);
        if(imageInput.empty())
        {
            cout<<"The image("<<filename<<") is empty or could no read!"<<endl;
            continue;
        }
        if (image_count == 1)
        {  
            image_size.width = imageInput.cols;  
            image_size.height =imageInput.rows;           
            cout<<"image_size.width = "<<image_size.width<<endl;  
            cout<<"image_size.height = "<<image_size.height<<endl;  
        }  
  
        /* Extracting corners*/  
        if (0 == findChessboardCorners(imageInput,board_size,image_points_buf))  
        {             
            cout<<"Could not find chessboard corners!\n";
            exit(1);  
        }   
        else   
        {  
            Mat view_gray;  
            cvtColor(imageInput,view_gray,CV_RGB2GRAY);  
            /* subpixel */  
            find4QuadCornerSubpix(view_gray,image_points_buf,Size(5,5));  //It seems this more precise than cornerSubPix
            //cornerSubPix(view_gray,image_points_buf,Size(5,5),Size(-1,-1),TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,30,0.1));  
            image_points_seq.push_back(image_points_buf); 
			count += image_points_buf.size();
            /* view only*/  
            drawChessboardCorners(view_gray,board_size,image_points_buf,false);  
            imshow("Camera Calibration",view_gray);
            waitKey(300);//0.3S         
        }  
    } 
	
    int total = image_points_seq.size();  
    cout<<"total = "<<total<<endl;  
    int CornerNum=board_size.width*board_size.height;  
	
    for (int ii=0 ; ii<total;ii++)  
    {  
        if (0 == ii%CornerNum)   
        {     
            int i = -1;  
            i = ii/CornerNum;  
            int j=i+1;  
            cout<<"\nThe "<<j <<" image data --> : "<<endl;  
        }  
        if (0 == ii%3) 
        {  
            cout<<endl;  
        }  
        else  
        {  
            cout.width(10);  
        }  
        //all corners 
        cout<<" -->"<<image_points_seq[0][ii].x;  
        cout<<" -->"<<image_points_seq[0][ii].y;  
    }     
    cout<<"\n\nExtracting corners done!"<<endl; 
	cout<<"Already extracted corners: "
	    <<count
		<<" ?= "
		<<"total corners: "
		<<NUM_CHESSBOARD*NUM_ROW_CORNERS*NUM_COL_CORNERS
		<<endl;
		
    cout<<"=============================="<<endl<<endl;
	
    //begin to calibrate 
    cout<<"Start calibrating……"<<endl;  

    Size square_size = Size(CHESS_SQUARE_HEIGTH,CHESS_SQUARE_WIDTH); 
	/* Save 3D coordinate for corner*/
    vector<vector<Point3f> > object_points;  
 
    /*camera instrinsic parameters*/
    Mat cameraMatrix=Mat(3,3,CV_32FC1,Scalar::all(0)); 
    vector<int> point_counts; 
	
	/*If distCoeffs is of length 4, the coefficients(k1,k2,p1,p2);
	 *If the length is 5 or 8, then the coefficients(k1,k2,p1,p2 and k3)
	 *   or (k1,k2,p1,p2,k3,k4,k5,and k6)
     */ 
    Mat distCoeffs=Mat(1,5,CV_32FC1,Scalar::all(0));  
    vector<Mat> tvecsMat; 
    vector<Mat> rvecsMat; 
  
    int i,j,t;  
    for (t=0;t<image_count;t++)   
    {  
        vector<Point3f> tempPointSet;  
        for (i=0;i<board_size.height;i++)   
        {  
            for (j=0;j<board_size.width;j++)   
            {  
                Point3f realPoint;  
                /* assume z = 0 */  
                realPoint.x = i*square_size.width;  
                realPoint.y = j*square_size.height;  
                realPoint.z = 0;  
                tempPointSet.push_back(realPoint);  
            }  
        }  
        object_points.push_back(tempPointSet);  
    }  

    for (i=0;i<image_count;i++)  
    {  
        point_counts.push_back(board_size.width*board_size.height);  
    }     

    calibrateCamera(object_points,image_points_seq,image_size,cameraMatrix,distCoeffs,rvecsMat,tvecsMat,0);  
    cout<<"Calibration done！"<<endl;  
    cout<<"=============================="<<endl<<endl;
	
    cout<<"Start evaluating calibrated result……"<<endl;  
    double total_err = 0.0;   
    double err = 0.0;  
    vector<Point2f> image_points2;  
    cout<<"Each image calibrated error："<<endl; 
	
	
	fout<<"\n##############################################"<<endl;
	fout<<"####  Monocular Camera Calibration Report"<<" ####"<<endl;
	fout<<"##############################################"<<endl<<endl;
	
	char name[128];   
    gethostname(name, 128);

	fout<<"Author: "<<name<<endl;
	
	time_t rawtime;
    time(&rawtime);
	fout<<"Date:"<<asctime(localtime(&rawtime))<<endl;
	
    fout<<"Each image calibrated error：\n";  

    for (i=0;i<image_count;i++)  
    {  
        vector<Point3f> tempPointSet=object_points[i];  
        /* projecting*/  
        projectPoints(tempPointSet,rvecsMat[i],tvecsMat[i],cameraMatrix,distCoeffs,image_points2);  
        /* computing the error*/  
        vector<Point2f> tempImagePoint = image_points_seq[i];  
        Mat tempImagePointMat = Mat(1,tempImagePoint.size(),CV_32FC2);  
        Mat image_points2Mat = Mat(1,image_points2.size(), CV_32FC2);  
		
        for (unsigned int j = 0 ; j < tempImagePoint.size(); j++)  
        {  
            image_points2Mat.at<Vec2f>(0,j) = Vec2f(image_points2[j].x, image_points2[j].y);  
            tempImagePointMat.at<Vec2f>(0,j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);  
        }  
		
        err = norm(image_points2Mat, tempImagePointMat, NORM_L2);  
        total_err += err/=  point_counts[i];     
        std::cout<<"The "<<i+1<<" image average error："<<err<<" pixel"<<endl;     
        fout<<"The "<<i+1<<" image average error："<<err<<" pixel"<<endl;     
    }     
    std::cout<<"\nTotal average ERROR: "<<total_err/image_count<<" pixel"<<endl;     
    fout<<"\nTotal average error："<<total_err/image_count<<" pixel"<<endl<<endl;     
    std::cout<<"Evaluation done!"<<endl;    
    cout<<"=============================="<<endl<<endl;
	
    std::cout<<"Start to save calibrated result……"<<endl;         
    Mat rotation_matrix = Mat(3,3,CV_32FC1, Scalar::all(0));
	fout<<"    | Fx  0  Cx |"<<endl;
	fout<<"M = | 0   Fy Cy |"<<endl;
	fout<<"    | 0   0   1 |"<<endl;
    fout<<"Camera instrinsic matrix："<<endl;     
    fout<<cameraMatrix<<endl<<endl; 
	
	string coeffs4("DistCoeffs:k1,k2,p1,p2");
	string coeffs5("DistCoeffs:k1,k2,p1,p2,k3");
	string coeffs8("DistCoeffs:k1,k2,p1,p2,k3,k4,k5,k6");
	if(distCoeffs.cols == 4)
	{
       fout<<coeffs4<<endl;  
	}else if(distCoeffs.cols == 5)
	{
		fout<<coeffs5<<endl; 
	}else if(distCoeffs.cols == 8)
	{
		fout<<coeffs8<<endl;
	}
    fout<<"Distort："<<endl;     
    fout<<distCoeffs<<endl<<endl<<endl;   
	#if 1
    for (int i=0; i<image_count; i++)   
    {   
        fout<<"The "<<i+1<<" image Translation（centimeter) vector："<<endl;     
        fout<<tvecsMat[i]<<endl;      
  
        Rodrigues(tvecsMat[i],rotation_matrix);     
        fout<<"The "<<i+1<<" image rotation matrix："<<endl;     
        fout<<rotation_matrix<<endl;     
        fout<<"The "<<i+1<<" image Rotation vector："<<endl;     
        fout<<rvecsMat[i]<<endl<<endl;     
    } 
    #endif	
    std::cout<<"Saving done!"<<endl;
	cout<<"=============================="<<endl<<endl;
	
    fout<<endl;  
    /************************************************************************   
    To view the undistorted images using calibrated result
    *************************************************************************/  
    Mat mapx = Mat(image_size,CV_32FC1);  
    Mat mapy = Mat(image_size,CV_32FC1);  
    Mat R = Mat::eye(3,3,CV_32F);  
    std::cout<<"Save distorted image."<<endl;  
    string imageFileName;  
    std::stringstream StrStm;  
    for (int i = 0 ; i < NUM_VERIFY_CHESSBOARD ; i++)  
    {  
        std::cout<<"image #"<<i+1<<"..."<<endl;
		
        initUndistortRectifyMap(cameraMatrix,distCoeffs,R,cameraMatrix,image_size,CV_32FC1,mapx,mapy);        
        StrStm.clear();  
        imageFileName.clear();  
        string filePath="./toVerify/";  
        StrStm<<i+1;  
        StrStm>>imageFileName;  
        filePath+=imageFileName;  
        filePath+=".png";  
        Mat imageSource = imread(filePath);  
        Mat newimage = imageSource.clone();  
 
        undistort(imageSource,newimage,cameraMatrix,distCoeffs); 
        //Another do not need transfering matrix		
        //remap(imageSource,newimage,mapx, mapy, INTER_LINEAR);         
        StrStm.clear();  
        filePath.clear(); 
		
		string outputfilePath="./undistorted/"; 
        StrStm<<i+1;  
        StrStm>>imageFileName;  
        imageFileName += "_undistorted.jpg";  
		outputfilePath += imageFileName;
        imwrite(outputfilePath,newimage);  
    }  
	
    std::cout<<"Distorted image done！"<<endl<<endl;   
	
    return 0;  
}  
