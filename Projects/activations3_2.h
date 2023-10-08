#include <iostream>
#include <cmath>
#include "constants.h"
#ifndef ACTIVATIONS2_2_H_
#define ACTIVATIONS2_2_H_

double ReLU(double x)									// ReLU����
{
	if (x > 0)
		return x;
	else
		return 0.0001*x;
}

double dReLU(double x)									// ���򴫲�����y��s 
{
	if (x > 0)
		return 1;
	else
		return 0.0001;
}

void rReLU(double** ds, double** y, double** Y) 		// ���򴫲�����Y��s 
{
	for (int m = 0; m < MD; m++)
		for (int j = 0; j < N; j++)						// ʵ����û�о���ReLU���˴�rReLUָ����������� 
			ds[j][m] = y[j+1][m] - Y[j][m];				// ��ƽ����ʧ������Ϊ���������
}

double sigmoid(double x)								// ��sigmoid���׵����������Ϊ������ 
{														// ��� rsigmoidΪ rSoftmax��������� 
	return 1 / (1 + exp(-x));
}

double dsigmoid(double y)								// ���򴫲�����y��s 
{
	return y * (1 - y);
}

void Softmax(double** y) 
{
	for (int m = 0; m < MD; m++)
	{
		double sum = 0.0;
		for (int j = 1; j <= N; j++) {
			y[j][m] = exp(y[j][m]);								// ָ���� 
			sum += y[j][m];
		}
		for (int j = 1; j <= N; j++) {
			y[j][m] /= sum;									// ʹ����Ӻ�Ϊ 1 
		}
	}
	
}

void rSoftmax(double** ds, double** y, double** Y)	// Softmax�ķ��򴫲�����Y��s��
{
	for (int m = 0; m < MD; m++)
		for (int j = 0; j < N; j++)
			ds[j][m] = y[j+1][m] - Y[j][m];

}

double tanh(double x)
{
	return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

double dtanh(double y)									// ���򴫲�����y��s 
{
	return 1 - y * y;
}

void rtanh(double** ds, double** y, double** Y)		// ���򴫲�����Y��s��y�� Y�������Ϊ������ 
{
	for (int m = 0; m < MD; m++)
		for (int j = 0; j < N; j++)
			ds[j][m] = -Y[j][m]*(1 - pow(y[j+1][m],2))/y[j+1][m] + (1-Y[j][m])*(1+y[j+1][m]);
}

#endif
