#include <iostream>
#include <cmath>
#include "constants.h"
#ifndef ACTIVATIONS2_2_H_
#define ACTIVATIONS2_2_H_

double ReLU(double x)									// ReLU函数
{
	if (x > 0)
		return x;
	else
		return 0.0001*x;
}

double dReLU(double x)									// 反向传播，从y到s 
{
	if (x > 0)
		return 1;
	else
		return 0.0001;
}

void rReLU(double** ds, double** y, double** Y) 		// 反向传播，从Y到s 
{
	for (int m = 0; m < MD; m++)
		for (int j = 0; j < N; j++)						// 实际上没有经过ReLU，此处rReLU指的是拟合问题 
			ds[j][m] = y[j+1][m] - Y[j][m];				// 将平方损失函数作为输出处理函数
}

double sigmoid(double x)								// 和sigmoid配套的输出处理函数为交叉熵 
{														// 因此 rsigmoid为 rSoftmax的特殊情况 
	return 1 / (1 + exp(-x));
}

double dsigmoid(double y)								// 反向传播，从y到s 
{
	return y * (1 - y);
}

void Softmax(double** y) 
{
	for (int m = 0; m < MD; m++)
	{
		double sum = 0.0;
		for (int j = 1; j <= N; j++) {
			y[j][m] = exp(y[j][m]);								// 指数化 
			sum += y[j][m];
		}
		for (int j = 1; j <= N; j++) {
			y[j][m] /= sum;									// 使各项加和为 1 
		}
	}
	
}

void rSoftmax(double** ds, double** y, double** Y)	// Softmax的反向传播，从Y到s。
{
	for (int m = 0; m < MD; m++)
		for (int j = 0; j < N; j++)
			ds[j][m] = y[j+1][m] - Y[j][m];

}

double tanh(double x)
{
	return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

double dtanh(double y)									// 反向传播，从y到s 
{
	return 1 - y * y;
}

void rtanh(double** ds, double** y, double** Y)		// 反向传播，从Y到s，y到 Y输出函数为交叉熵 
{
	for (int m = 0; m < MD; m++)
		for (int j = 0; j < N; j++)
			ds[j][m] = -Y[j][m]*(1 - pow(y[j+1][m],2))/y[j+1][m] + (1-Y[j][m])*(1+y[j+1][m]);
}

#endif
