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



int motionDetectBase()
{

    Mat frame, gray, frameDelta, thresh, thresh_10,thresh_20, output;
    VideoCapture camera(0, CAP_V4L2);
    vector<vector<Point>> cnts;
    if (!camera.isOpened()) {
        cerr << "Error: Unable to access the camera" << endl;
        return -1;
    }

    // Kamera ayarları
    camera.set(CAP_PROP_FRAME_WIDTH, 512);
    camera.set(CAP_PROP_FRAME_HEIGHT, 288);

    Mat prevGray;
    if (!camera.read(frame)) {
        cerr << "Error: Unable to read from the camera" << endl;
        return -1;
    }

    // İlk kareyi işleme
    cvtColor(frame, prevGray, COLOR_BGR2GRAY);
    GaussianBlur(prevGray, prevGray, Size(21, 21), 0);

    while (camera.read(frame)) {
        // Gri tonlama ve bulanıklaştırma
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(21, 21), 0);

        // Fark hesaplama ve eşikleme
        absdiff(prevGray, gray, frameDelta);
        threshold(frameDelta, thresh, 25, 255, THRESH_BINARY);
        threshold(frameDelta, thresh_20, 20, 255, THRESH_BINARY);
        threshold(frameDelta, thresh_10, 10, 255, THRESH_BINARY);

        // Çıktı görüntüsünü oluştur
        frame.copyTo(output);
        dilate(thresh, thresh, Mat(), Point(-1, -1), 2);
        findContours(thresh, cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (int i = 0; i < cnts.size(); i++) {
            if (contourArea(cnts[i]) < 500) {
                continue;
            }

            putText(output, "Motion Detected", Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
        }

        // Görüntüleri göstermek
        imshow("Original Frame", frame);
        imshow("Grayscale Frame", gray);
        imshow("Thresholded Frame", thresh);
        imshow("Thresholded Frame delta", frameDelta);
        imshow("Thresholded Frame 10", thresh_10);
        imshow("Output Frame", output);

        // Pencereleri düzenlemek (isteğe bağlı)
        moveWindow("Original Frame", 0, 0);
        moveWindow("Grayscale Frame", 400, 0);
        moveWindow("Thresholded Frame", 0, 350);
        moveWindow("Thresholded Frame delta", 0, 700);
        moveWindow("Thresholded Frame 10", 0, 1050);
        moveWindow("Output Frame", 400, 350);

        // Bir sonraki döngü için önceki kareyi güncelle
        gray.copyTo(prevGray);

        // Çıkış kontrolü
        if (waitKey(1) == 27) { // ESC tuşu ile çıkış
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1000/20));
    }

    camera.release();

}



int histogram()
{
    Mat frame, grayFrame, equalizedFrame;
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

        // Görüntüleri göster
        imshow("Original  Frame", frame);

        imshow("Original Grayscale Frame", grayFrame);
        imshow("Equalized Frame", equalizedFrame);

        // Çıkış kontrolü (ESC tuşu ile çıkış)
        if (waitKey(1) == 27) {
            break;
        }
    }
}

int main() {
    histogram();
    // motionDetectLearning();
    // greyColor_blackWhite();
    // greyColor_blackWhite_Gaus_median();
    // saltPaperMedian();
    destroyAllWindows();
    return 0;
}
