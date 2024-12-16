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



int imageDetaction(){
    Mat rawImage, greyImage ,cannyEdges ,gaussianCannyEdge, blurredFrame ,dilatedEdges;

    rawImage = imread("/home/halilerden/Documents/workFiles/06-imageProcess/imageProcess/opencv-motionDetect/objs.png");
    if (rawImage.empty()) {
        cerr << "Error: Unable to load the image." << endl;
        return -1;
    }
    cvtColor(rawImage,greyImage,COLOR_BGR2GRAY);
    // imshow("raw image", rawImage);
     imshow("grey image", greyImage);

    // GaussianBlur(greyImage,greyImage,Size(3, 3), 0);
    // kenar bulma uygulamalıyız
    // imshow("GaussianBlur image", greyImage);

    // gaus uygulanmadan önceki hali
    //canny uygulayalım.
    Canny(greyImage,cannyEdges,100,200);
    imshow("Canny image", cannyEdges);


    // 4. Kenarları genişletme (Dilate işlemi)
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(cannyEdges, dilatedEdges, kernel);





    imshow("dilatedEdges image", dilatedEdges);

    // 5. Kontur Bulma (Canny üzerinde)
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(cannyEdges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    // 5.5. Konturları çizecek yeni bir görüntü oluştur

    Mat contourImage = Mat::zeros(rawImage.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++) {
        // Her bir konturu farklı bir renkle çiz
        Scalar color = Scalar(rand() % 256, rand() % 256, rand() % 256);
        drawContours(contourImage, contours, static_cast<int>(i), color, 2);
    }
    imshow("Contours", contourImage);


    // 6. Nesneleri işaretleme

    Mat contourFrame = rawImage.clone();

    int counter=0;

    for (const auto& contour : contours) {
        if (contourArea(contour) > 100) { // Küçük alanları filtrele

            Rect boundingBox = boundingRect(contour);
            rectangle(contourFrame, boundingBox, Scalar(0, 255, 0), 3);
            putText(contourFrame, "Object"+to_string(counter), boundingBox.tl() - Point(0, 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
            // Orta noktayı hesapla
            Point center = (boundingBox.tl() + boundingBox.br()) * 0.5;

            // Orta noktaya text yaz
            putText(contourFrame, to_string(counter), center, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);

            counter++;
        }
    }


    imshow("Object Detection", contourFrame);


    // Şekil Algılama
    Mat shapeFrame = rawImage.clone();
    for (size_t i = 0; i < contours.size(); i++) {
        if (contourArea(contours[i]) > 500) { // Küçük konturları filtrele
            // Çokgen yaklaştırması
            vector<Point> approx;
            approxPolyDP(contours[i], approx, 0.02 * arcLength(contours[i], true), true);

            // Konturu çevreleyen dikdörtgen
            Rect boundingBox = boundingRect(contours[i]);

            // Şekil tespiti
            string shape = "Unknown";
            if (approx.size() == 3) {
                shape = "Triangle"; // Üçgen
            } else if (approx.size() == 4) {
                // En-boy oranını hesapla
                double aspectRatio = (double)boundingBox.width / boundingBox.height;
                if (aspectRatio >= 0.95 && aspectRatio <= 1.05) {
                    shape = "Square"; // Kare
                } else {
                    shape = "Rectangle"; // Dikdörtgen
                }
            } else if (approx.size() > 4) {
                // Yuvarlaklık hesapla
                double area = contourArea(contours[i]);
                double perimeter = arcLength(contours[i], true);
                double circularity = (4 * M_PI * area) / (perimeter * perimeter);
                if (circularity >= 0.9) {
                    shape = "Circle"; // Daire
                } else {
                    shape = "Ellipse"; // Elips
                }
            }

            // Şekli görüntüye yaz
            Point center = (boundingBox.tl() + boundingBox.br()) * 0.5;
            putText(shapeFrame, shape, center, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

            // Konturu çiz
            drawContours(shapeFrame, contours, (int)i, Scalar(0, 0, 255), 2);
        }
    }

    // Görüntüleri göster
    imshow("Original Frame", rawImage);
    imshow("Detected Shapes", shapeFrame);


    // Çıkış kontrolü
    waitKey(0);

    destroyAllWindows();
    return 0;
}

int objDetection()
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

        // 5. Kontur Bulma (Canny üzerinde)
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(cannyEdges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // 6. Nesneleri işaretleme
        Mat contourFrame = frame.clone();
        for (const auto& contour : contours) {
            if (contourArea(contour) > 500) { // Küçük alanları filtrele
                Rect boundingBox = boundingRect(contour);
                rectangle(contourFrame, boundingBox, Scalar(0, 255, 0), 2);
                putText(contourFrame, "Object", boundingBox.tl() - Point(0, 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
            }
        }

        // Görüntüleri göster
        imshow("Original Grayscale Frame", grayFrame);
        imshow("Equalized Frame", equalizedFrame);
        imshow("Canny Edges", cannyEdges);
        imshow("Sobel Edges", sobelEdges);
        imshow("Object Detection", contourFrame);



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
    // objDetection();
    imageDetaction();

    // motionDetectLearning();
    // greyColor_blackWhite();
    // greyColor_blackWhite_Gaus_median();
    // saltPaperMedian();
    destroyAllWindows();
    return 0;
}
