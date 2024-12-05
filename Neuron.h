#pragma once

#include <vector>
#include <iostream>
#include <cassert>
#include <cmath>

#include "random.h"

class Neuron
{
private:
	double input_value;
	double output_value;
	std::vector<double> synapses;
	std::vector<double> lastGangeWeight;
	int neuronIndex;
	int numOutputs;
	double delta;
	std::vector<double> gradient;

public:
	Neuron(int numOutputs, int neuronNum);

	static double epsi;//�������� �������� ��
	static double alpha;//������ - ��������� �����������

	void feedForvard(const std::vector<Neuron>& prevLayer);//���������� �������� �������
	double getMultWeightDelta(const std::vector<Neuron>& prevLayer);//������������ ������������ ����� � ����� ����
	void setOutputVals(double vals) { this->output_value = vals; }//��������� ��������� �������� �������
	double getOutoutValue() const { return this->output_value; }//���������� ������� ������� �������
	void setWeightOfSynapse(int numOutputs);//��������� ��������� �������� ����� �� �������
	double getNumOutputs() { return this->numOutputs; }//��������� ���������� ������� �������
	double getWeightOfSynapse(int neuronIndex) const { return this->synapses[neuronIndex]; }//��������� ���� �������
	void setDelta(double value) { this->delta = value * DerivativeActivatFunc(this->output_value); }//������ ������ ��� ������� �������
	double getDelta() const { return this->delta; }//��������� ������
	void setGradient(const std::vector<Neuron>& prevLayer);//������� ������ ���������
	double getGradient(int index) { return this->gradient[index]; }//������� ��������� ���������
	void changeWeights();//��������� ����� �������� ��� �������

	static double activateFunc(double value);//������� ���������
	static double DerivativeActivatFunc(double value);//����������� ������� ���������

};

typedef std::vector<Neuron> Layer;

Neuron::Neuron(int numOutputs = int(), int neuronIndex = int())
{
	this->neuronIndex = neuronIndex;
	this->output_value = 0.0;
	this->input_value = 0.0;
	this->delta = 0.0;
	this->numOutputs = numOutputs;
	setWeightOfSynapse(numOutputs);
}

void Neuron::setWeightOfSynapse(int numOutputs)//��������� ��������� �������� ����� �� �������
{
	for (int i = 0; i < numOutputs; ++i)
	{
		double weight = fRand(0.0, 1.0);
		this->synapses.push_back(weight);
		this->lastGangeWeight.push_back(0.0);
		this->gradient.push_back(0.0);
	}
}

double Neuron::activateFunc(double value)//���������� ������� ���������
{
	//output rang [0...1]
	return 1 / (1 + pow(2.718, -value));
}

double Neuron::DerivativeActivatFunc(double value)//����������� ������� ���������
{
	return (1 - value) * value;
}


void Neuron::feedForvard(const Layer& prevLayer)//���������� ������ � ��
{
	double sum = 0.0;

	for (int neuronNum = 0; neuronNum < prevLayer.size(); ++neuronNum)
	{
		sum += prevLayer[neuronNum].getOutoutValue() * prevLayer[neuronNum].getWeightOfSynapse(neuronIndex);
	}

	this->output_value = Neuron::activateFunc(sum);
}

double Neuron::getMultWeightDelta(const std::vector<Neuron>& prevLayer)
{
	double sum = 0.0;

	for (int i = 0; i < prevLayer.size(); ++i)
	{
		sum += this->getWeightOfSynapse(i) * prevLayer[i].getDelta();
	}

	return sum;
}

void Neuron::setGradient(const std::vector<Neuron>& prevLayer)//������� ������ ���������
{
	double value;

	for (int numNeuron = 0; numNeuron < prevLayer.size(); ++numNeuron)
	{
		value = this->getOutoutValue() * prevLayer[numNeuron].getDelta();
		gradient[numNeuron] = value;
	}
}

void Neuron::changeWeights()//��������� ����� �������� ��� �������
{
	for (int i = 0; i < synapses.size(); ++i)
	{
		double change = epsi * gradient[i] + alpha * lastGangeWeight[i];//������������ ���������
		lastGangeWeight[i] = change;//������ ��������� ��������� ���� ��������
		this->synapses[i] += change;
	}
}
