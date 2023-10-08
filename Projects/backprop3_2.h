#include <cmath>
#include "activations3_1.h"
#include "constants.h"
#ifndef BACKPROP2_2_H_
#define BACKPROP2_2_H_

void reverse(double** ds, double** y, double** Y, int type)
{
	switch (type)
	{
		case SIGMOID:
		case SOFTMAX:
			rSoftmax(ds, y, Y);
			break;
		case TANH:
			rtanh(ds, y, Y);
			break;
		default:
			rReLU(ds, y, Y);
			break;		
	}
}

void s2y(double** dy, double** W, double** ds, int J, int K)
{
	for (int m = 0; m < MD; m++)
		for (int j = 1; j <= J; j++) {	
			dy[j-1][m] = 0.0;									// 初始化 
			for (int k = 0; k < K; k++)							// 用后一层的 W和 ds更新当前层的dy 
				dy[j-1][m] += W[k][j] * ds[k][m]; 
		}
}

void y2s(double** ds, double** s, double** y, double** dy, int J, int type)
{	
	switch (type)
	{
		case RELU:
			for (int m = 0; m < MD; m++)
				for (int j = 0; j < J; j++)
					ds[j][m] = dReLU(s[j][m]) * dy[j][m]; 					// 从dy推到ds
			break;
		case SIGMOID:
			for (int m = 0; m < MD; m++)
				for (int j = 0; j < J; j++)
					ds[j][m] = dsigmoid(y[j][m]) * dy[j][m];
			break;
		case TANH:
			for (int m = 0; m < MD; m++)
				for (int j = 0; j < J; j++)
					ds[j][m] = dtanh(y[j][m]) * dy[j][m];
			break;
		default:
			for (int m = 0; m < MD; m++)
				for (int j = 0; j < J; j++)
					ds[j][m] = dy[j][m];
			break;
	}
	
}

void s2W(double** dW, double** ds, double** x, int I, int J)
{
	for (int j = 0; j < J; j++)
		for (int i = 0; i <= I; i++)
		{
			dW[j][i] = 0.0;
			for (int m = 0; m < MD; m++)
			{
				dW[j][i] += ds[j][m] * x[i][m];								// 由ds算出dW
			}
			dW[j][i] /= MD;
		}
}

void update(double** W, double** dW, int I, int J)
{
	for (int j = 0; j < J; j++) {
		for (int i = 0; i <= I; i++) {
			W[j][i] -= alpha * dW[j][i];
		}
	}
}

#endif
