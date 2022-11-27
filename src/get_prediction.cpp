#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "saved_model_loader.h"


int main(int argc, char* argv[]){
	if (argc != 4){
		std::cout << "Error! Usage: <path/to_saved_model_dir> <path/to/input/input.png> <path/to/output/output.png>" << std::endl;
		return 1;
	}

	// Make a Prediction instance
    Tensor out_pred;

	const string model_path = argv[1]; 
	const string test_image_file  = argv[2];
	const string test_prediction_image = argv[3];

	// Load the saved_model
	ModelLoader model(model_path);

	//Predict on the input image
	model.predict(test_image_file, out_pred);

    //Save a tensor to an image(https://github.com/PatWie/tensorflow-cmake/issues/1)
    cv::Mat rotMatrix(out_pred.dim_size(1), out_pred.dim_size(2), CV_32FC1, out_pred.flat<float>().data());
    rotMatrix += 0.5;
    cv::Mat_<uint8_t> gray_mask = (rotMatrix >= 1);
    // Same as above
    //tensorflow::TTypes<float>::Flat output = out_pred.flat<float>();
    //float *data_ptr = output.data();
    ////fill data to a cv rotMatrix
    //cv::Mat_<int> gray_mask = cv::Mat_<int>::ones(out_pred.dim_size(1), out_pred.dim_size(2));
    //for (int x = 0; x < out_pred.dim_size(1); x++) {
    //    for (int y = 0; y < out_pred.dim_size(2); y++) {
    //        gray_mask.at<int>(x,y) = static_cast<int>(floor(*(data_ptr+224*x+y) + 0.5));
    //    }
    //}
    gray_mask *= 255;

    //Find fit ellipse
    cv::Mat canny_output;
    const double threshold = 100;
    cv::Canny(gray_mask, canny_output, threshold, threshold*2);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(canny_output, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    std::vector<cv::RotatedRect> minEllipse(contours.size());
    //cv::RNG rng(1024);
    //cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256));
    cv::Scalar color = cv::Scalar(0, 255, 0);
    //cv::Mat color_mask;
    //cv::cvtColor(gray_mask, color_mask, cv::COLOR_GRAY2RGB);

    cv::Mat gray_img = imread(test_image_file, cv::IMREAD_GRAYSCALE);
    cv::Mat color_img;
    cv::cvtColor(gray_img, color_img, cv::COLOR_GRAY2RGB);

    cv::Mat color_out = color_img; //color_mask
    for (size_t i = 0; i < contours.size(); i++)
    {
        minEllipse[i] = cv::fitEllipse(contours[i]);
        cv::ellipse(color_out, minEllipse[i], color, 2, cv::LINE_AA);
    }

    cv::imwrite(test_prediction_image, color_out);
    return 0;
}
