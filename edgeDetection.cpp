///*
/// @author halil erden
/// @time 13.12.2025
/// @about Basic Motion Detection With opencv

// #include <QCoreApplication>


#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/tracking.hpp>
#include <opencv4/opencv2/core/ocl.hpp>
#include <unistd.h>
#include <chrono>
#include <thread>

using namespace cv;
using namespace std;





int edgeDetection()
{
    Mat frame, grayFrame, equalizedFrame, cannyEdges, sobelEdges;
    VideoCapture camera(0, CAP_V4L2);

    if (!camera.isOpened()) {
        cerr << "Error: Unable to access the camera" << endl;
        return -1;
    }

    // Kamera boyutlarını ayarla
    camera.set(CAP_PROP_FRAME_WIDTH, 512);
    camera.set(CAP_PROP_FRAME_HEIGHT, 288);

    while (camera.read(frame)) {
        // 1. Gri tonlamaya çevir
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

        // 2. Histogram eşitleme uygula
        equalizeHist(grayFrame, equalizedFrame);

        // 3. Canny Kenar Algılama
        Canny(grayFrame, cannyEdges, 100, 200);

        // 4. Sobel Kenar Algılama
        Mat gradX, gradY;
        Sobel(grayFrame, gradX, CV_16S, 1, 0, 3);
        Sobel(grayFrame, gradY, CV_16S, 0, 1, 3);
        Mat absGradX, absGradY;
        convertScaleAbs(gradX, absGradX);
        convertScaleAbs(gradY, absGradY);
        addWeighted(absGradX, 0.5, absGradY, 0.5, 0, sobelEdges);

        // Görüntüleri göster
        imshow("Original Grayscale Frame", grayFrame);
        imshow("Equalized Frame", equalizedFrame);
        imshow("Canny Edges", cannyEdges);
        imshow("Sobel Edges", sobelEdges);

        // Çıkış kontrolü (ESC tuşu ile çıkış)
        if (waitKey(1) == 27) {
            break;
        }
    }

    camera.release();
    destroyAllWindows();
    return 0;
}

int main() {
    edgeDetection();
    // motionDetectLearning();
    // greyColor_blackWhite();
    // greyColor_blackWhite_Gaus_median();
    // saltPaperMedian();
    destroyAllWindows();
    return 0;
}
