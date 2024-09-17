#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

void onChangeBlurLevel(int value, void* userData) {
	// Ensure blur level is odd
	if (value % 2 == 0) {
		value = std::max(value - 1, 1); // Set to nearest odd number
		cv::setTrackbarPos("Blur Level", "Trackbars", value);
	}
}

int main()
{
	constexpr int waitTime = 1;
	constexpr int cameraID = 0;
	constexpr double scaleFactor = 2;

	int blurLevel = 99;

	cv::VideoCapture cap(cameraID);
	if (!cap.isOpened()) {
		std::cerr << "Error: Could not open the video file." << std::endl;
		return -1;
	}

	cv::Mat currentFrame;
	cv::Mat faceROI;
	cv::Mat imgHSV;
	cv::Mat mask;

	cv::CascadeClassifier faceCascade;
	if (!faceCascade.load("Resources/haarcascade_frontalface_default.xml")) {
		std::cerr << "XML file not loaded" << std::endl;
		return 1;
	}

	cv::namedWindow("Trackbars", (640, 200));
	cv::createTrackbar("Blur Level", "Trackbars", &blurLevel, 99, onChangeBlurLevel);

	while (true)
	{
		bool isSuccess = cap.read(currentFrame);
		if (!isSuccess || currentFrame.empty()) {
			std::cerr << "Error: Failed to read the frame." << std::endl;
			break;
		}

		cv::Mat gray;
		cv::cvtColor(currentFrame, gray, cv::COLOR_BGR2GRAY);
		cv::equalizeHist(gray, gray);
		
		std::vector<cv::Rect> faces;
		faceCascade.detectMultiScale(currentFrame, faces, 1.1, 10);

		if (faces.size() <= 0) {
			cv::GaussianBlur(currentFrame, currentFrame, cv::Size(blurLevel, blurLevel), 50);
		}
		else
		{
			for (const auto& face : faces) {
				int newWidth = static_cast<int>(face.width * scaleFactor);
				int newHeight = static_cast<int>(face.height * scaleFactor);
				int newX = std::max(0, face.x - (newWidth - face.width) / 2);
				int newY = std::max(0, face.y - (newHeight - face.height) / 2);
				newWidth = std::min(newWidth, currentFrame.cols - newX);
				newHeight = std::min(newHeight, currentFrame.rows - newY);
		
				cv::Rect enlargedFace(newX, newY, newWidth, newHeight);
		
				faceROI = currentFrame(enlargedFace);
				cv::GaussianBlur(faceROI, faceROI, cv::Size(blurLevel, blurLevel), 50);
			}
		}

		cv::imshow("Image", currentFrame);

		int key = cv::waitKey(waitTime);
		if (key == 'q' || key == 27) // 'q' or ESC key
			break;
	}

	cap.release();
	cv::destroyAllWindows();

	return 0;
}