#pragma once

/*
Developer: Sushkov Maxim
Data: 11.05.2020
Product: student project

In this project was been used method of back propagation. 
*/

#include "Neuron.h"

#include <fstream>
#include <string>

double Neuron::epsi = 40;
double Neuron::alpha = 0.5;

class Net
{
private:
	typedef std::vector<Neuron> Layer;//������ �������� - ����
	std::vector<Layer> layers;//������ ����� - ��
	double Error;//������ ����������

public:
	Net(std::vector<int>& topology);
	void feedForward(std::vector<double>& inputVals);//���������� ������ � ��
	void back_Prop(const std::vector<double>& targetVals);//�������� ���������� ����
	void getResults();//�������� �������� ������ ����
	double findError(const std::vector<double>& targetVals);
	void save_weight_and_topology(std::string path);//���������� ��������� �� � ����� �������� � ��������� ����
	void trening(std::vector<double>& inputVals, std::vector<double>& targetVals, unsigned int epoch);//�������� �� �� ������

	void setEpsi(double value);//������ epsi
	void setAlpha(double value);//������ alpha
};


Net::Net(std::vector<int>& topology)
{
	int layersNum = topology.size();
	for (int i = 0; i < layersNum; ++i)
	{
		layers.push_back(Layer());
		unsigned numOutputs = i == topology.size() - 1 ? 0 : topology[i + 1];

		for (unsigned int neuroneNum = 0; neuroneNum < topology[i]; ++neuroneNum)
		{
			layers.back().push_back(Neuron(numOutputs, neuroneNum));
			layers.back().back().setOutputVals(neuroneNum);
		}
	}

	Neuron::epsi = 40;
	Neuron::alpha = 0.5;

	this->Error = 0.0;
}

void Net::feedForward(std::vector<double>& inputVals)//���������� ������ � ��
{
	assert(inputVals.size() == layers[0].size());

	for (int i = 0; i < layers[0].size(); ++i)
	{
		layers[0][i].setOutputVals(inputVals[i]);
	}

	for (int layerNum = 1; layerNum < layers.size(); ++layerNum)
	{
		Layer& prevLayer = layers[layerNum - 1];
		for (int neuronNum = 0; neuronNum < layers[layerNum].size(); ++neuronNum)
		{
			layers[layerNum][neuronNum].feedForvard(prevLayer);
		}
	}
}

void Net::getResults()//�������� �������� ������ ���� - ����� ����� �� �����
{
	int lastLayer = layers.size() - 1;;//���������� ����� = ����� ���������� ����
	int sizeOfLastLayers = layers.back().size();;//���������� �������� � ��������� ����
	std::cout << " Result:\n";
	for (int numNeurons = 0; numNeurons < sizeOfLastLayers; ++numNeurons)
	{
		std::cout << numNeurons + 1 << ")" << std::to_string(layers[lastLayer][numNeurons].getOutoutValue()) << " ";
	}
}

double Net::findError(const std::vector<double>& targetVals)//���������� ������ �������� �������� ��
{
	assert(layers.back().size() == targetVals.size());

	int sizeOfLastLayers = layers.back().size();
	int lastLayer = layers.size() - 1;
	this->Error = 0.0;

	for (int i = 0; i < sizeOfLastLayers; ++i)
	{
		this->Error += pow(targetVals[i] - layers[lastLayer][i].getOutoutValue(), 2);
	}

	this->Error /= 2;
	//this->Error /= layers.back().size();
	return this->Error;
}

void Net::back_Prop(const std::vector<double>& targetVals)//�������� ���������� ����
{
	assert(layers.back().size() == targetVals.size());

	//������ �����: ���������� ������ �� ��������� ���� ��
	int sizeOfLastLayers = layers.back().size();//������ ���������� ����
	int lastLayer = layers.size() - 1;//������ ���������� ����
	for (int i = 0; i < sizeOfLastLayers; ++i)
	{
		double value = targetVals[i] - layers[lastLayer][i].getOutoutValue();
		layers[lastLayer][i].setDelta(value);//������������� ������ ��� ������� ��������� �������
	}

	//������ �����: ���������� ������ �� ��������� ����� ��
	int startLayer = layers.size() - 2;//������������� ���� ��
	for (int numLayer = startLayer; numLayer >= 0; --numLayer)
	{
		Layer& prevLayer = layers[numLayer + 1];//���������� �������������� ����
		Layer& presentLayer = layers[numLayer];//�������� ����
		int sizeOfLayer = layers[numLayer].size();//������ �������� ����
		for (int numNeuron = 0; numNeuron < sizeOfLayer; ++numNeuron)
		{
			//���������� ������ �������
			//value - ������ ����������, ������� ����������� ��������, ������������ ��������;
			//presentLayer[numNeuron] - �������� ������.
			double value = presentLayer[numNeuron].getMultWeightDelta(prevLayer);
			presentLayer[numNeuron].setDelta(value);
			//���������� ��������a
			presentLayer[numNeuron].setGradient(prevLayer);
			//����������� ����� ���������� ������� ����
			presentLayer[numNeuron].changeWeights();
		}
	}
}

void Net::save_weight_and_topology(std::string path)//Saving the Neuron Network topology and synapse weights to a text file\���������� ��������� �� � ����� �������� � ��������� ����
{
	std::ofstream file;
	file.open(path);
	file << "topology: ";
	//������ ��������� ��
	for (int i = 0; i < layers.size(); ++i)
	{
		file << layers[i].size() << " ";
	}
	//������ ����� �������� ��
	for (int numLayer = 0; numLayer < layers.size(); ++numLayer)//���� �����
	{
		file << "\n";
		for (int numNeuron = 0; numNeuron < layers[numLayer].size(); ++numNeuron)//���� ��������
		{
			for (int numSynapse = 0; numSynapse < layers[numLayer][numNeuron].getNumOutputs(); ++numSynapse)//���� ��������
			{
				file << layers[numLayer][numNeuron].getWeightOfSynapse(numSynapse) << " ";
			}
		}
	}
}

void Net::trening(std::vector<double>& inputVals, std::vector<double>& targetVals, unsigned int epoch)//�������� �� �� ������
{
	assert(inputVals.size() % targetVals.size() == layers[0].size() % layers.back().size());

	int extra_in = 0;
	int extra_out = 0;

	std::vector<double> input;
	std::vector<double> output;
	/*
	for (; it != inputVals.end(); it++)//���������� ������� ������� ������ �������
	{
		input.push_back(it->second);
	}

	for (; its != targetVals.end(); its++)//���������� ������� �������� ������ �������
	{
		output.push_back(its->second);
	}
	*/
	for (unsigned int numEpoch = 0; numEpoch < epoch; numEpoch++)
	{
		extra_in = 0;
		extra_out = 0;
		for (int numSet = 0; numSet < (inputVals.size() / layers[0].size()); numSet++)
		{
			input.clear();
			output.clear();
			for (int numNeuron = 0 + extra_in; numNeuron < (layers[0].size() + extra_in); ++numNeuron)
			{
				input.push_back(inputVals[numNeuron]);
			}
			for (int numNeuron = 0 + extra_out; numNeuron < (layers.back().size() + extra_out); ++numNeuron)
			{
				output.push_back(targetVals[numNeuron]);
			}
			extra_in += layers[0].size();
			extra_out += layers.back().size();
			feedForward(input);
			if (numEpoch == epoch - 1)
			{
				getResults();
				std::cout << "\nError = " << findError(output) * 100 << "%\n";
				for (int numNeuron = 0; numNeuron < layers[0].size(); ++numNeuron)
				{
					std::cout << "\nInput value " << numNeuron + 1 << ": " << input[numNeuron];
				}
				for (int numNeuron = 0; numNeuron < layers.back().size(); ++numNeuron)
				{
					std::cout << " | Ideal output value " << numNeuron + 1 << ": " << output[numNeuron];
				}
				std::cout << "\nIterration number " << numSet + 1 << "----------------------------------------------\n\n";
			}
				back_Prop(output);
		}
	}
	std::string answer;
	std::cout << "\nSave weights of synapses? (yes/no)" << std::endl;
	std::cin >> answer;
	if (answer == "yes" || answer == "Yes" || answer == "YES")
	{
		save_weight_and_topology("Weights.txt");
	}
	else
	{
		return;
	}
}


void Net::setEpsi(double value)
{
	Neuron::epsi = value;
}

void Net::setAlpha(double value)
{
	Neuron::alpha = value;
}
