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

int motionDetectLearning()
{


    Mat frame, gray, blurred, prevGray, frameDelta, thresh, dilateThresh, output, contourVisualization;
    VideoCapture camera(0, CAP_V4L2);

    if (!camera.isOpened()) {
        cerr << "Error: Unable to access the camera" << endl;
        return -1;
    }

    // Kamera boyutlarını ayarla
    camera.set(CAP_PROP_FRAME_WIDTH, 512);
    camera.set(CAP_PROP_FRAME_HEIGHT, 288);

    // İlk kareyi oku ve gri tonlamaya çevir
    if (!camera.read(frame)) {
        cerr << "Error: Unable to read from the camera" << endl;
        return -1;
    }

    cvtColor(frame, prevGray, COLOR_BGR2GRAY);
    GaussianBlur(prevGray, prevGray, Size(21, 21), 0);

    while (camera.read(frame)) {
        // 1. Orijinal görüntüyü göster
        imshow("1- Original Frame (Before Processing)", frame);

        // 2. Gri tonlamaya çevir
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        imshow("2- Grayscale Frame (After Conversion)", gray);

        // 3. Gauss bulanıklığı uygula
        blurred = gray.clone();
        GaussianBlur(blurred, blurred, Size(21, 21), 0);
        imshow("3- Blurred Frame (After Gaussian Blur)", blurred);

        // 4. Önceki gri tonlama görüntüsünü göster
        imshow("4- Previous Grayscale Frame", prevGray);

        // 5. Frame Delta hesapla (önceki gri ve bulanık arasındaki fark)
        absdiff(prevGray, blurred, frameDelta);
        imshow("5- Frame Delta (Difference Between Frames)", frameDelta);

        // 6. Threshold uygula (eşikleme)
        threshold(frameDelta, thresh, 25, 255, THRESH_BINARY);
        imshow("6- Thresholded Frame (After Thresholding)", thresh);

        // 7. Dilate işlemi uygula (genişletme)
        dilateThresh = thresh.clone();
        dilate(dilateThresh, dilateThresh, Mat(), Point(-1, -1), 2);
        imshow("7- Dilated Thresholded Frame (After Dilation)", dilateThresh);

        // 8. Konturları görselleştir ve hareket tespit edilen kareyi işaretle
        frame.copyTo(output);
        frame.copyTo(contourVisualization);
        vector<vector<Point>> contours;
        findContours(dilateThresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            if (contourArea(contour) < 500) continue;
            drawContours(contourVisualization, contours, -1, Scalar(255, 0, 255), 2);
            Rect boundingBox = boundingRect(contour);
            rectangle(output, boundingBox, Scalar(0, 255, 0), 2);
            putText(output, "Motion Detected", Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
        }
        imshow("8- Contours Visualization", contourVisualization);
        imshow("9- Output Frame (Motion Detected)", output);

        // Yeni gri görüntüyü bir sonraki döngü için önceki görüntü olarak sakla
        blurred.copyTo(prevGray);

        // Çıkış kontrolü (ESC tuşu ile çıkış)
        if (waitKey(1) == 27) {
            break;
        }
    }
}


int greyColor_blackWhite()
{
    Mat frame, gray, binary;
    VideoCapture camera(0, CAP_V4L2);

    if (!camera.isOpened()) {
        cerr << "Error: Unable to access the camera" << endl;
        return -1;
    }

    // Kamera boyutlarını ayarla
    camera.set(CAP_PROP_FRAME_WIDTH, 512);
    camera.set(CAP_PROP_FRAME_HEIGHT, 288);

    while (camera.read(frame)) {
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // 2. Siyah beyaz görüntü için eşikleme uygula
        threshold(gray, binary, 128, 255, THRESH_BINARY);

        // Byte boyutlarını hesapla
        size_t colorBytes = frame.total() * frame.elemSize();
        size_t grayBytes = gray.total() * gray.elemSize();
        size_t binaryBytes = binary.total() * binary.elemSize();

        // Görüntünün üzerine yazı ekle (Byte boyutları)
        string colorText = "Color Size: " + to_string(colorBytes/1024) + " kb";
        string grayText = "Gray Size: " + to_string(grayBytes/1024) + " kb";
        string binaryText = "Binary Size: " + to_string(binaryBytes/1024) + " kb";

        putText(frame, colorText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
        putText(gray, grayText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
        putText(binary, binaryText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);

        // Görüntüleri göster
        imshow("1- Original Frame", frame);
        imshow("2- Grayscale Frame", gray);
        imshow("3- Binary Frame", binary);

        // Çıkış kontrolü (ESC tuşu ile çıkış)
        if (waitKey(1) == 27) {
            break;
        }
    }

}

int greyColor_blackWhite_Gaus_median()
{
    Mat frame, gray, binary, frame_gaus, gray_gaus, binary_gaus, frame_median, gray_median, binary_median;
    VideoCapture camera(0, CAP_V4L2);

    if (!camera.isOpened()) {
        cerr << "Error: Unable to access the camera" << endl;
        return -1;
    }

    // Kamera boyutlarını ayarla
    camera.set(CAP_PROP_FRAME_WIDTH, 512);
    camera.set(CAP_PROP_FRAME_HEIGHT, 288);

    while (camera.read(frame)) {


        cvtColor(frame, gray, COLOR_BGR2GRAY);

        GaussianBlur(frame,frame_gaus,Size(15,15),0);

        GaussianBlur(gray,gray_gaus,Size(15,15),0);

        // 2. Siyah beyaz görüntü için eşikleme uygula
        threshold(gray, binary, 128, 255, THRESH_BINARY);
        GaussianBlur(binary, binary_gaus, Size(15, 15), 0);
        // 6. MedianBlur uygula (frame)
        medianBlur(frame, frame_median, 15);

        // 7. MedianBlur uygula (gray)
        medianBlur(gray, gray_median, 15);

        // 8. MedianBlur uygula (binary)
        medianBlur(binary, binary_median, 15);
        // Byte boyutlarını hesapla
        size_t colorBytes = frame.total() * frame.elemSize();
        size_t grayBytes = gray.total() * gray.elemSize();
        size_t binaryBytes = binary.total() * binary.elemSize();

        // Görüntünün üzerine yazı ekle (Byte boyutları)
        string colorText = "Color Size: " + to_string(colorBytes/1024) + " kb";
        string grayText = "Gray Size: " + to_string(grayBytes/1024) + " kb";
        string binaryText = "Binary Size: " + to_string(binaryBytes/1024) + " kb";

        putText(frame, colorText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
        putText(gray, grayText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
        putText(binary, binaryText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);

        putText(frame_gaus, colorText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
        putText(gray_gaus, grayText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
        putText(binary_gaus, binaryText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);

        // Median filtre uygulanmış görüntülerin üzerine yazı ekle
        putText(frame_median, "Median Filter", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
        putText(gray_median, "Median Filter", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
        putText(binary_median, "Median Filter", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);

        // Görüntüleri göster
        imshow("1- Original Frame", frame);
        imshow("1_1- Original Frame GAUS", frame_gaus);
        imshow("2- Grayscale Frame", gray);
        imshow("2_1 Grayscale Frame GAUS", gray_gaus);
        imshow("3- Binary Frame", binary);
        imshow("3_1 Binary Frame gaus", binary_gaus);
        imshow("1_2 Original Frame (Median Filtered)", frame_median);
        imshow("2_2 Grayscale Frame (Median Filtered)", gray_median);
        imshow("3_3 Binary Frame (Median Filtered)", binary_median);

        // Çıkış kontrolü (ESC tuşu ile çıkış)
        if (waitKey(1) == 27) {
            break;
        }
    }

}

int saltPaperMedian()
{
    Mat inputImage, medianFiltered,gray;

    // JPEG görüntüyü yükle
    inputImage = imread("/home/halilerden/Documents/workFiles/00-Qt_Assets/MotionDetection/colorlena-300x300.jpg");

    cvtColor(inputImage,gray,COLOR_BGR2GRAY);

    if (inputImage.empty()) {
        cerr << "Error: Unable to load the image." << endl;
        return -1;
    }

    // Median filtre uygula
    medianBlur(inputImage, medianFiltered, 5);

    // Orijinal ve median filtre uygulanmış görüntüleri göster
    imshow("Original Image", inputImage);
     imshow("gray Image", gray);
    imshow("Median Filtered Image", medianFiltered);

    // Çıkış kontrolü (ESC tuşu ile çıkış)
    waitKey(0);
}

int main() {
    motionDetectBase();
    // motionDetectLearning();
    // greyColor_blackWhite();
    // greyColor_blackWhite_Gaus_median();
    // saltPaperMedian();
    destroyAllWindows();
    return 0;
}
