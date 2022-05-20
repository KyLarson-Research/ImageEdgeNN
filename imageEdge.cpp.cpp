#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>
using namespace cv;
using namespace std;
Mat img, gray, blurred, edge;
// Laplacian
int kernel_size = 3;
int ddepth = CV_16S;
// Canny Edge Detection
int lowerThreshold = 0;
int max_lowThreshold = 100; 

void laplacianDetection() {
    //Method Introduced By Nicolai Neilsen Nov 15, 2020 Published on youtube.com
    GaussianBlur(gray,
        blurred,
        cv::Size(3, 3),
        3);

    Laplacian(blurred,//inferrior to Canny "Canny edge detection function"
        edge,
        ddepth,
        kernel_size);
    convertScaleAbs(edge, edge);
}
//useful operations
int dot(int* A, int* B, int n) {
    int dot = 0;
#
    for (int i = 0; i < n; i++) {
        dot += A[i] * B[i];
    }
    return dot;
}

float sigmoid(float x) {
    return 1 / (1 + exp(-x));//may be a good spot to try some computations using exp to compare to python
}// e^x as in euler's function not some exponent but sometimes 2^x is substituted?

float sigmoid_derivative(float x) {
    return x * (1 - x);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
    }

    Mat image;
    image = imread(argv[1], IMREAD_COLOR); // Read the file

    if (!image.data) // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    cvtColor(image, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(3, 3), 3,0);
    Canny(blurred, edge, 25, 75);
    //cv::namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
    //---------
    ////EDGE DETECTION
    //cv::namedWindow("Original", WINDOW_AUTOSIZE);
    //cv::namedWindow("Gray", WINDOW_AUTOSIZE);
    //cv::namedWindow("Blurred", WINDOW_AUTOSIZE);
    //cv::namedWindow("Edge Detection", WINDOW_AUTOSIZE);
    //-----------
    ////laplacianDetection();
    //-----------
    //
    //cv::imshow("Display window", image); // Show our image inside it.
    //cv::imshow("Original", image); // Show our image inside it.
    //cv::imshow("Gray", gray); // Show our image inside it.
    //cv::imshow("Blurred", blurred); // Show our image inside it.
    //cv::imshow("Edge Detection", edge); // Show our image inside it.
    
    
    auto start = std::chrono::system_clock::now();

    cout<<"__________OVER HEEEERE____________"<<edge.dot(edge)<<"size"<<edge.size<<"type"<<edge.type();
    Mat layer_1_weights;
    layer_1_weights.create(800,800,0);
    cout << "tot"<<layer_1_weights.total();
    cout <<layer_1_weights;

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed Time: " << elapsed.count() << std::endl;


    vector<int> compression_params;
    //compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    //compression_params.push_back(9);
    bool result = false;
    try
    {
        result = cv::imwrite("C:/Users/admin/source/repos/imageEdge/OpenCV_Edge-/alpha_no_lines.png", edge);
    }//, compression_params
    catch (const cv::Exception& ex)
    {
        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
    }

    cv::waitKey(0); // Wait for a keystroke in the window
    return 0;
}