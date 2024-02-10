// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits> 

#define C 9
#define BIN_NUMBER 32
#define K 29
#define TRAINSET 8541

char* classes[C] = {"Australia", "Brazil", "Canada", "Finland", "France", "Japan", "Russia", "South-Africa", "Spain"};

std::vector<double> calcHist(Mat_<Vec3b> img) {
	std::vector<double> hist(BIN_NUMBER * 3, 0.0);

	int binSize = 256 / BIN_NUMBER;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b pixel = img(i, j);

			for (int channel = 0; channel < 3; channel++) {
				int intensity = pixel[channel];
				int bin = min(intensity / binSize, BIN_NUMBER - 1);
				hist[channel * BIN_NUMBER + bin]++;
			}
		}
	}

	return hist;
}

std::vector<double> getQuadrantHistograms(Mat_<Vec3b> img) {

	Mat_<Vec3b> imgHSV;
	cvtColor(img, imgHSV, COLOR_BGR2HSV);

	Mat_<Vec3b> upperLeft(imgHSV.rows / 2, imgHSV.cols / 2);
	Mat_<Vec3b> upperRight(imgHSV.rows / 2, imgHSV.cols / 2);
	Mat_<Vec3b> lowerLeft(imgHSV.rows / 2, imgHSV.cols / 2);
	Mat_<Vec3b> lowerRight(imgHSV.rows / 2, imgHSV.cols / 2);

	for (int i = 0; i < imgHSV.rows; i++) {
		for (int j = 0; j < imgHSV.cols; j++) {

			if (i < img.rows / 2 && j < img.cols / 2) {	
				upperLeft(i, j) = imgHSV(i, j);
			}
			else if (i < img.rows / 2) {				
				upperRight(i, j - imgHSV.cols / 2) = imgHSV(i, j);
			}
			else if (j < img.cols / 2) {				
				lowerLeft(i - imgHSV.rows / 2, j) = imgHSV(i, j);
			}
			else {										
				lowerRight(i - imgHSV.rows / 2, j - imgHSV.cols / 2) = imgHSV(i, j);
			}
		}
	}

	std::vector<double> histUpperLeft = calcHist(upperLeft);
	std::vector<double> histUpperRight = calcHist(upperRight);
	std::vector<double> histLowerLeft = calcHist(lowerLeft);
	std::vector<double> histLowerRight = calcHist(lowerRight);

	histUpperLeft.insert(histUpperLeft.end(), histUpperRight.begin(), histUpperRight.end());
	histUpperLeft.insert(histUpperLeft.end(), histLowerLeft.begin(), histLowerLeft.end());
	histUpperLeft.insert(histUpperLeft.end(), histLowerRight.begin(), histLowerRight.end());

	return histUpperLeft;
}

std::vector<double> getVerticalHistograms(Mat_<Vec3b> img) {
	Mat_<Vec3b> imgHSV;
	cvtColor(img, imgHSV, COLOR_BGR2HSV);

	int segmentWidth = imgHSV.cols / 4;

	Mat_<Vec3b> leftmost(imgHSV.rows, segmentWidth);
	Mat_<Vec3b> centerLeft(imgHSV.rows, segmentWidth);
	Mat_<Vec3b> centerRight(imgHSV.rows, segmentWidth);
	Mat_<Vec3b> rightmost(imgHSV.rows, segmentWidth);

	for (int i = 0; i < imgHSV.rows; i++) {
		for (int j = 0; j < imgHSV.cols; j++) {
			if (j < segmentWidth) { 
				leftmost(i, j) = imgHSV(i, j);
			}
			else if (j < segmentWidth * 2) {  
				centerLeft(i, j - segmentWidth) = imgHSV(i, j);
			}
			else if (j < segmentWidth * 3) {  
				centerRight(i, j - segmentWidth * 2) = imgHSV(i, j);
			}
			else {  
				rightmost(i, j - segmentWidth * 3) = imgHSV(i, j);
			}
		}
	}

	std::vector<double> histLeftmost = calcHist(leftmost);
	std::vector<double> histCenterLeft = calcHist(centerLeft);
	std::vector<double> histCenterRight = calcHist(centerRight);
	std::vector<double> histRightmost = calcHist(rightmost);

	histLeftmost.insert(histLeftmost.end(), histCenterLeft.begin(), histCenterLeft.end());
	histLeftmost.insert(histLeftmost.end(), histCenterRight.begin(), histCenterRight.end());
	histLeftmost.insert(histLeftmost.end(), histRightmost.begin(), histRightmost.end());

	return histLeftmost;
}

std::vector<double> getHorizontalHistograms(Mat_<Vec3b> img) {
	Mat_<Vec3b> imgHSV;
	cvtColor(img, imgHSV, COLOR_BGR2HSV);

	int segmentHeight = imgHSV.rows / 4;

	Mat_<Vec3b> topmost(segmentHeight, imgHSV.cols);
	Mat_<Vec3b> upperMiddle(segmentHeight, imgHSV.cols);
	Mat_<Vec3b> lowerMiddle(segmentHeight, imgHSV.cols);
	Mat_<Vec3b> bottommost(segmentHeight + imgHSV.rows % 4, imgHSV.cols);

	for (int i = 0; i < imgHSV.rows; i++) {
		for (int j = 0; j < imgHSV.cols; j++) {
			if (i < segmentHeight) {
				topmost(i, j) = imgHSV(i, j);
			}
			else if (i < segmentHeight * 2) {
				upperMiddle(i - segmentHeight, j) = imgHSV(i, j);
			}
			else if (i < segmentHeight * 3) {
				lowerMiddle(i - segmentHeight * 2, j) = imgHSV(i, j);
			}
			else {
				bottommost(i - segmentHeight * 3, j) = imgHSV(i, j);
			}
		}
	}

	std::vector<double> histTopmost = calcHist(topmost);
	std::vector<double> histUpperMiddle = calcHist(upperMiddle);
	std::vector<double> histLowerMiddle = calcHist(lowerMiddle);
	std::vector<double> histBottommost = calcHist(bottommost);

	histTopmost.insert(histTopmost.end(), histUpperMiddle.begin(), histUpperMiddle.end());
	histTopmost.insert(histTopmost.end(), histLowerMiddle.begin(), histLowerMiddle.end());
	histTopmost.insert(histTopmost.end(), histBottommost.begin(), histBottommost.end());

	return histTopmost;
}

std::tuple<Mat_<double>, Mat_<double>> readDataset() {
	char fname[MAX_PATH];
	Mat_<double> X(TRAINSET, BIN_NUMBER * 3 * 4);
	X.setTo(0);
	Mat_<double> Y(TRAINSET, 1);
	Y.setTo(0);

	int row = 0;
	Mat_<Vec3b> upperLeft(1536 / 2, 662 / 2);
	Mat_<Vec3b> upperRight(1536 / 2, 662 / 2);
	Mat_<Vec3b> lowerLeft(1536 / 2, 662 / 2);
	Mat_<Vec3b> lowerRight(1536 / 2, 662 / 2);
	for (int classIdx = 0; classIdx < C; classIdx++) {
		int fileNr = 1;
		std::cout << classes[classIdx] << "\n";
		while (true) {
			if (fileNr % 100 == 0) {
				std::cout << fname << "\n";
			}

			sprintf(fname, "dataset/train-kaggle/%s/%s_%d.jpg",classes[classIdx], classes[classIdx], fileNr++);

			Mat_<Vec3b> img = imread(fname, IMREAD_COLOR);

			if (img.cols == 0) {
				break;
			}

			std::vector<double> hist = getVerticalHistograms(img);
			for (int i = 0; i < hist.size(); i++) {
				X(row, i) = hist.at(i);
			}

			Y(row++, 0) = classIdx;
		}
	}

	FILE* xFile = fopen("x.csv", "w");
	for (int i = 0; i < X.rows; i++) {
		for (int j = 0; j < X.cols; j++) {
			fprintf(xFile, "%f,", X(i,j));
		}
		fprintf(xFile, "\n");
	}
	fclose(xFile);

	FILE* yFile = fopen("y.csv", "w");
	for (int i = 0; i < Y.rows; i++) {
		fprintf(yFile, "%f\n", Y(i, 0));
	}
	fclose(yFile);

	return { X, Y };
}

struct candidate {
	double distance;
	int classIndex;
};

int knnClassifier(std::vector<double> hist, Mat_<double> X, Mat_<uchar> Y) {
	std::vector<candidate> v;
	std::vector<int> voting(C, 0);

	for (int i = 0; i < X.rows; i++) {
		double distance = 0;

		for (int j = 0; j < X.cols; j++) {
 			distance += abs(hist.at(j) - X(i, j));
		}

		double maxElement = -1;
		for (int j = 0; j < v.size(); j++) {
			double elem = v.at(j).distance;
			if (elem > maxElement) {
				maxElement = elem;
			}
		}

		if (v.size() <= K || distance < maxElement) {
			struct candidate candidate;
			candidate.distance = distance;
			candidate.classIndex = Y(i, 0);
			v.push_back(candidate);

			if (v.size() > K) {
				int index = 0;
				for (int k = 0; k < v.size(); k++) {
					if (v.at(k).distance == maxElement) {
						v.erase(v.begin() + k);
					}
				}
			}
		}
	}

	for (int i = 0; i < v.size(); i++) {
		int index = v.at(i).classIndex;
		voting.at(index)++;
	}

	int maxim = 0, predictedClass = 0;
	for (int i = 0; i < C; i++) {
		if (voting.at(i) > maxim) {
			maxim = voting.at(i);
			predictedClass = i;
		}
	}

	return predictedClass;
}

Mat_<double> loadCSVtoMat(char* filename) {
	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Failed to open the CSV file." << std::endl;
		return Mat();
	}

	std::string fname(filename);
	int cols = 0;
	if (fname.find('x') != std::string::npos) {
		cols = BIN_NUMBER * 3 * 4;
	}
	else if (fname.find('y') != std::string::npos) {
		cols = 1;
	}
	else {
		std::cerr << "Invalid file name: cannot determine matrix dimensions." << std::endl;
		return Mat();
	}

	Mat_<double> mat(TRAINSET, cols, CV_64F);

	std::string line;
	int row = 0;
	while (std::getline(file, line) && row < TRAINSET) {
		std::istringstream iss(line);
		double value;
		int col = 0;
		while (iss >> value && col < cols) {
			mat(row, col) = value;
			if (iss.peek() == ',') {
				iss.ignore();
			}
			col++;
		}
		row++;
	}

	file.close();
	return mat;
}

void testData(int testNumber) {

	char csvX[7];
	switch (testNumber) {
	case 2:
		strcpy(csvX, "x2.csv");
		break;
	case 3:
		strcpy(csvX, "x3.csv");
		break;
	default:
		strcpy(csvX, "x.csv");
		break;
	}

	Mat_<double> X;
	X = loadCSVtoMat(csvX);
	Mat_<double> Y;
	Y = loadCSVtoMat("y.csv");
	Mat_<double> M(C, C);
	M.setTo(0);

	char fname[MAX_PATH];
	for (int classIdx = 0; classIdx < C; classIdx++) {
		int fileNr = 1;
		std::cout << classes[classIdx] << "\n";

		while (true) {
			sprintf(fname, "dataset/test-kaggle/%s/%s_%d.jpg", classes[classIdx], classes[classIdx], fileNr++);

			Mat_<Vec3b> img = imread(fname, IMREAD_COLOR);

			if (img.cols == 0) {
				break;
			}

			std::vector<double> hist;

			switch (testNumber) {
			case 2:
				hist = getVerticalHistograms(img);
				break;
			case 3:
				hist = getHorizontalHistograms(img);
				break;
			default:
				hist = getQuadrantHistograms(img);
				break;
			}

			int predictedClass = knnClassifier(hist, X, Y);

			M(predictedClass, classIdx)++;

		}
	}

	double accuracy = 0.0;
	for (int i = 0; i < C; i++) {
		accuracy += M(i, i);
	}

	double sum = 0.0;
	for (int i = 0; i < C; i++) {
		for (int j = 0; j < C; j++) {
			sum += M(i, j);
		}
	}

	accuracy /= sum;
	std::cout << "Accuracy: " << accuracy * 100 << "%\n";
}

void testDataQuadrants() {
	testData(1);
	while (true);
}

void testDataVerticalSegments() {
	testData(2);
	while (true);
}

void testDataHorizontalSegments() {
	testData(3);
	while (true);
}

void testAll() {
	std::cout << "----------QUADRANT METHOD----------\n";
	testData(1);
	std::cout << "----------VERTICAL METHOD----------\n";
	testData(2);
	std::cout << "----------HORIZONTAL METHOD----------\n";
	testData(3);
	while (true);
}

void testImageQuadrant(char* fname) {
	Mat_<Vec3b> img = imread(fname, IMREAD_COLOR);
	Mat_<double> X = loadCSVtoMat("x.csv");
	Mat_<double> Y = loadCSVtoMat("y.csv");

	std::vector<double> hist = getQuadrantHistograms(img);
	int predictedQuadrant = knnClassifier(hist, X, Y);

	std::cout << "Class predicted using quadrant division: " << classes[predictedQuadrant] << "\n";
}

void testImageVertical(char* fname) {
	Mat_<Vec3b> img = imread(fname, IMREAD_COLOR);
	Mat_<double> X = loadCSVtoMat("x2.csv");
	Mat_<double> Y = loadCSVtoMat("y.csv");

	std::vector<double> hist = getVerticalHistograms(img);
	int predictedVertical = knnClassifier(hist, X, Y);

	std::cout << "Class predicted using vertical division: " << classes[predictedVertical] << "\n";
}

void testImageHorizontal(char* fname) {
	Mat_<Vec3b> img = imread(fname, IMREAD_COLOR);
	Mat_<double> X = loadCSVtoMat("x3.csv");
	Mat_<double> Y = loadCSVtoMat("y.csv");

	std::vector<double> hist = getHorizontalHistograms(img);
	int predictedHorizontal = knnClassifier(hist, X, Y);

	std::cout << "Class predicted using horizontal division: " << classes[predictedHorizontal] << "\n";
}


void testImage() {
	char folder[50]; 
	int number;
	char fname[MAX_PATH]; 

	for (int i = 0; i < C; ++i) {
		std::cout << classes[i] << "\n";
	}
	std::cout << "Enter the country (folder) name from the previous options:";
	std::cin >> folder;

	bool isValidFolder = false;
	for (int i = 0; i < 9; ++i) {
		if (strcmp(classes[i], folder) == 0) {
			isValidFolder = true;
			break;
		}
	}

	if (!isValidFolder) {
		std::cerr << "Invalid folder name entered.\n";
		return; 
	}

	std::cout << "Enter the image number (1-100): ";
	std::cin >> number;

	if (number < 1 || number > 100) {
		std::cerr << "Invalid number entered. Must be between 1 and 100.\n";
		return; 
	}

	sprintf(fname, "dataset/test-kaggle/%s/%s_%d.jpg", folder, folder, number);

	testImageQuadrant(fname);
	testImageVertical(fname);
	testImageHorizontal(fname);
	Mat_<Vec3b> img = imread(fname, IMREAD_COLOR);
	imshow("image", img);
	waitKey();
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Test data using quadrant's histograms\n");
		printf(" 2 - Test data using vertical segments' histograms\n");
		printf(" 3 - Test data using horizontal segments' histograms\n");
		printf(" 4 - Test data using all methods\n");
		printf(" 5 - Test one image (from testing set) using all methods\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testDataQuadrants();
				break;
			case 2:
				testDataVerticalSegments();
				break;
			case 3:
				testDataHorizontalSegments();
				break;
			case 4:
				testAll();
				break;
			case 5:
				testImage();
				break;
			default:
				break;
		}
	}
	while (op!=0);
	return 0;
}