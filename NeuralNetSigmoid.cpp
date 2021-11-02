/*	NeuralNetSigmoid.cpp
	This file contains the implementation of the Item class and the Sigmoid class
*/

//Used for cmath M_E
#define _USE_MATH_DEFINES

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>	//E
#include <math.h>	//pow
#include <string>
#include <iomanip>

class Item;

using namespace std;

//Taken from my A1P2 source code
class Item {
	//Vector to store a variable number of values of attributes.
private: vector<int> attVal = {};
		 //Variable to store the class value. This will be set by the constructor if a class value is known from the input, or assigned by classification.
private: int classVal = -1;

		 //Known class value constructor
public: Item(vector<int> values, int cVal) {
	attVal = values;
	classVal = cVal;
}

		//Unknown class value constructor
public: Item(vector<int> values) { attVal = values; }

		//Get function for attVal
public: vector<int> getAttVal() { return attVal; }

		//Get function for classVal
public: int getClassVal() { return classVal; }

};



class Sigmoid {
private: vector<double> w = {};
private: vector<Item> itemList = {};
private: vector<string> attributes = {};
private: int numTotalItems = 0;
private: double alpha = 1.0;

		 //Sigmoid constructor
public: Sigmoid(string fileName, double learnRate) {
	//Inputs the data from the training file
	input(fileName);
	//Set the weights to random weights
	for (size_t i = 0; i < attributes.size(); i++) {
		w.push_back(0.0);
	}
	alpha = learnRate;
}
	
		//Taken from my A1P2 source code and modified
public: void input(string fileName) {
	//Make sure itemList is empty an numTotalItems is 0
	itemList = {};
	numTotalItems = 0;

	//Vector of items in the input file
	//vector<Item> itemList = {};
	//Input stream to read from file
	ifstream read(fileName);
	string line;
	string token;
	//Counter to know the number of attrbutes before class
	int count = 0;

	//Populate line for use in instantiatingthe string stream
	getline(read, line);
	//String stream for parsing lines from the file
	stringstream ss;
	ss << line;
	//Populate line with the line of attributes

	//Populate attributes vector and count them
	while (!ss.eof()) {
		ss >> token;
		if (token != "class") {
			attributes.push_back(token);
			token = "";
			count++;
		}

	}

	//Populate itemList
	while (getline(read, line)) {
		vector<int> attV;
		ss.str("");
		ss.clear();
		ss << line;
		attV = {};
		int temp = 0;
		int i = 0;
		while (i < count) {
			ss >> temp;
			attV.push_back(temp);
			i++;
		}
		int classV = -1;
		ss >> classV;

		Item it(attV, classV);
		itemList.push_back(it);
		numTotalItems++;
	}
}

public: void printSigmoid() {
	cout << fixed;
	cout << setprecision(4);
	for (size_t i = 0; i < w.size(); i++) {
		cout << "w(" << attributes[i] << ") = " << w[i] << ", ";
	}
}
		 //Helper function to learn
		 //Updates the weight value of the sigmoid unit
private: void updateWeight(double O, int T, Item k) {
	//wi = wi + (alpha * error * sigmoidDerivative(dotProduct) * attributevaluei[of Item k]
	//sigmoidDerivative = sigmoid(dotProduct) * (1 - sigmoid[dotProduct])
	//sigmoid(dotProduct) = predict(Item k)

	//Update all weights
	vector<int> attVals = k.getAttVal();
	vector<double> tempWeights = w;
	for (size_t i = 0; i < tempWeights.size(); i++) {
		//Calculate the components of the weight update rule
		double sigK = predict(k);
		double error = (double)k.getClassVal() - sigK;
		double sigmoidDerivative = sigK * (1.0 - sigK);

		//Update the weight
		tempWeights[i] = w[i] + (alpha * error * sigmoidDerivative * attVals[i]);
	}
	for (size_t i = 0; i < w.size(); i++) {
		w[i] = tempWeights[i];
	}
	//All weights are updated
}

		 //Helper function for learn and test
		 //Outputs a predicted class value for the item
		 //aka the Sigmoid function
private: double predict(Item k) {
	//Calculate the dot product
	double dotProduct = 0;
	vector<int> attVals = k.getAttVal();
	for (size_t i = 0; i < attVals.size(); i++) {
		dotProduct += (double)w[i] * attVals[i];
	}
	//Sigmoid uses negative dot product
	dotProduct = dotProduct * -1.0;
	double denominator = 1 + pow(M_E, dotProduct);	//1 + e^(dot product)
	return (double)1.0 / denominator;
}

		//Trains the sigmoid unit on the input training set
public: void learn(int numIterations) {
	//Runs for the number of iterations specified in the command line
	for (int i = 0; i < numIterations;) {
		for (size_t j = 0; j < itemList.size(); j++) {
			//Exit if the user's desired number of iterations is reached
			if (i >= numIterations)
				return;

			Item k = itemList[j];
			double O = predict(k);
			int T = k.getClassVal();
			updateWeight(O, T, k);
			cout << "After iteration " << i + 1 << ": ";
			printSigmoid();
			cout << "output = " << predict(k) << endl;
			i++;
		}
	}
}
		//Tests the sigmoid unit on the data set and calculates the accuracy
public: double test() {
	int numCorrect = 0;
	//Loop through all test items in itemList
	for (size_t i = 0; i < itemList.size(); i++) {
		double O = predict(itemList[i]);
		int T = itemList[i].getClassVal();
		//If predicted value is class = 1 and actual class value is class = 1
		if (O >= 0.5 && T == 1)
			numCorrect++;
		//If predicted value is class = 0 and actual class value is class = 0
		else if (O < 0.5 && T == 0)
			numCorrect++;
	}
	//Calculate accuracy and return it
	double accuracy = (double)numCorrect / numTotalItems;
	return 100.0 * accuracy;
}

		//Returns the total number of items
public: int getNumItems() {
	return numTotalItems;
}
};



int main(int argc, char** argv) {
	//Inputs in order: training file, test file, learning rate, number of iterations to run
	//If there is an improper number of arguments, then terminate.
	if (argc != 5) {
		cout << "Invalid number of arguments. Requires a training file, a test file, a learning rate, and the number of iterations." << endl;
		return 1;
	}

	//Variables to store file names
	string trainFile = argv[1];
	string testFile = argv[2];

	//Convert learning rate from char* to double
	double learningRate = stod(argv[3]);	//argv[3] is the learning rate
	Sigmoid nn(trainFile, learningRate);

	//Convert number of iterations from char* to int
	int numIterations = stoi(argv[4]);
	//Learn on the training data set
	nn.learn(numIterations);	

	//Classify training data and output accuracy
	double trainingAccuracy = nn.test();
	cout << fixed;
	cout << setprecision(1);
	cout << endl;
	cout << "Accuracy on training set (" << nn.getNumItems() << " instances): " << trainingAccuracy << "%" << endl << endl;

	//Input test data
	nn.input(testFile);

	//Classify test data and output accuracy
	double testAccuracy = nn.test();
	cout << "Accuracy on test set (" << nn.getNumItems() << " instances): " << testAccuracy << "%" << endl;

	return 0;
}