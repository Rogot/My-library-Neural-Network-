#include <iostream>
#include <ctime>
#include <cstdlib>
#include <stdlib.h>
#include <time.h>


double fRand(double fMin, double fMax);
void randomize(void);

double fRand(double fMin, double fMax)
{
	double f = (double)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

void randomize(void)//������ �� �������� ���������� �������
{
	srand((unsigned int)time(NULL));
}
