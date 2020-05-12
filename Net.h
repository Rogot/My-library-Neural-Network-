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


double Neuron::epsi = 0.7;
double Neuron::alpha = 0.3;

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
		Layer prevLayer = layers[layerNum - 1];
		for (int neuronNum = 0; neuronNum < layers[layerNum].size(); ++neuronNum)
		{
			layers[layerNum][neuronNum].feedForvard(prevLayer);
		}
	}
}

void Net::getResults()//�������� �������� ������ ���� - ����� ����� �� �����
{
	int sizeOfNEt = layers.size();//���������� ����� = ����� ���������� ����
	int numNeurons = layers.back().size();//���������� �������� � ��������� ����
	for (int i = 0; i < numNeurons; ++i)
	{
		std::cout << layers[sizeOfNEt - 1][numNeurons].getOutoutValue() << " ";
	}
}

double Net::findError(const std::vector<double>& targetVals)//���������� ������ �������� �������� ��
{
	assert(layers.back().size() == targetVals.size());

	int sizeOfLastLayers = layers.back().size();
	int lastLayer = layers.size() - 1;

	for (int i = 0; i < sizeOfLastLayers; ++i)
	{
		this->Error += targetVals[i] - layers[lastLayer][i].getOutoutValue();
	}

	this->Error /= layers.back().size();
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
	for (int numLayer = startLayer; numLayer > 0; --numLayer)
	{
		Layer prevLayer = layers[numLayer + 1];//���������� ������������� ����
		Layer presentLayer = layers[numLayer];//�������� ����
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

void Net::save_weight_and_topology(std::string path)//���������� ��������� �� � ����� �������� � ��������� ����
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
