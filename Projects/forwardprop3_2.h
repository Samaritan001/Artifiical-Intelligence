#include <iostream>
#include "constants.h"
#include "activations3_1.h"
#ifndef FORWARDPROP2_2_H_
#define FORWARDPROP2_2_H_

void f(double** s, double** x, double** W, int I, int J)
{
	for (int m = 0; m < MD; m++)
		for (int j = 0; j < J; j++) {
			s[j][m] = 0.0;									// 初始化 
			for (int i = 0; i <= I; i++) {
				s[j][m] += W[j][i] * x[i][m];				// 乘权重 
			}
		}
	
}

void h(double** y, double** s, int J, int type)
{
	for (int m = 0; m < MD; m++)
	{
		y[0][m] = 1.0;
	}
	switch (type)
	{
		case RELU:
			for (int m = 0; m < MD; m++)
				for (int j = 1; j <= J; j++)
					y[j][m] = ReLU(s[j-1][m]);						// 过 ReLU函数
			break;
		case SOFTMAX:
			Softmax(y);
			break;
		case SIGMOID:
			for (int m = 0; m < MD; m++)
				for (int j = 1; j <= J; j++)
					y[j][m] = sigmoid(s[j-1][m]);
			break;
		case TANH:
			for (int m = 0; m < MD; m++)
				for (int j = 1; j <= J; j++)
					y[j][m] = tanh(s[j-1][m]);
			break;
		default:
			for (int m = 0; m < MD; m++)
				for (int j = 1; j <= J; j++)
					y[j][m] = s[j-1][m];
			break;
	}
}

void rnorm(double** y, double** Y)
{
	for (int i = 0; i < N; i++)
	{
		double sigma = 0.0;
		double mean = 0.0;
		for (int m = 0; m < MD; m++)
			mean += Y[i][m];
		mean /= MD;
		for (int m = 0; m < MD; m++)
			sigma += pow(Y[i][m] - mean, 2);
		if (sigma != 0)
		{
			sigma = sqrt(sigma/MD);
			for (int m = 0; m < MD; m++)
				y[i+1][m] = y[i+1][m] * sigma + mean;
		}
		//std::cout << mean << " " << sigma << std::endl;
	}
}


#endif
