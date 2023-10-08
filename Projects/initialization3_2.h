#include <iostream> 
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <cmath>
#include <string>
#include "constants.h"
#ifndef INITIAILZATION2_2_H_
#define INITIALIZATION2_2_H_

using namespace std;

void init(double*** W, double** X, double** Y, double*** x, double*** s, double*** y, 
	double*** dW, double*** ds, double*** dy, int* nl, double w_min, double w_max, const char* address)
{
	// 创建 W的动态数组 
	for (int l = 1; l <= L; l++) {						
		W[l] = new double*[nl[l]];
		dW[l] = new double*[nl[l]];
		for (int j = 0; j < nl[l]; j++) {
			W[l][j] = new double[nl[l-1] + 1];
			dW[l][j] = new double[nl[l-1] + 1];
		}
	}
	
	// 创建其他变量动态数组 
	for (int l = 1; l <= L; l++)
	{
		x[l] = new double*[nl[l]+1];
		s[l] = new double*[nl[l]];
		
		ds[l] = new double*[nl[l]];
		dy[l] = new double*[nl[l]];
		
		x[l][0] = new double[MD];
		for (int j = 0; j < nl[l]; j++)
		{
			x[l][j+1] = new double[MD];
			s[l][j] = new double[MD];
			ds[l][j] = new double[MD];
			dy[l][j] = new double[MD];
		}
		for (int m = 0; m < MD; m++)
			x[l][0][m] = 1.0;
	}
	for (int j = 0; j <= M; j++)
		X[j] = new double[MD];
	for (int n = 0; n < N; n++)
		Y[n] = new double[MD];
	
	switch (INIT_W)												// W初始化方式 
	{
		case 1:
			// 随机初始化 W 
			srand((unsigned)time(NULL));							// 生成随机种子	
			for (int l = 1; l < L+1; l++)
				for (int j = 0; j < nl[l]; j++)
					for (int i = 0; i <= nl[l-1]; i++)			// i = 0为兼容齐次化保留 
						W[l][j][i] = ((rand() % int(w_max-w_min)) + w_min) / 100.0; // 赋予权重矩阵 W 随机值
			break;
		case 2:														// 从外部加载模型
			{
			ifstream cin(address);
			for (int l = 1; l < L+1; l++)
				for (int j = 0; j < nl[l]; j++)
					for (int i = 0; i <= nl[l-1]; i++)
						cin >> W[l][j][i];
			break;
			}
		case 0:  
			for (int l = 1; l < L+1; l++)
				for (int j = 0; j < nl[l]; j++)
					for (int i = 0; i <= nl[l-1]; i++)
						W[l][j][i] = 0.0;
			break;
	}
	
	
}

void load_data(double** X, const char* address)
{
	ifstream cin(address);
	for (int m = 0; m < MD; m++)							// 输入txt中样本水平排列
	{
		X[0][m] = 1.0; 										// 齐次化 
		for (int i = 1; i <= M; i++) {
			cin >> X[i][m];									// 读取输入
		}
	}
}

void norm1(double** X)
{
	// 归一化
	for (int i = 1; i <= M; i++)
	{
		double sigma = 0.0;
		double mean = 0.0;
		for (int m = 0; m < MD; m++)
			mean += X[i][m];
		mean /= MD;
		for (int m = 0; m < MD; m++)
			sigma += pow(X[i][m] - mean, 2);
		if (sigma != 0)
		{
			sigma = sqrt(sigma/MD);
			for (int m = 0; m < MD; m++)
				X[i][m] = (X[i][m] - mean) / sigma;
		}
	}
}

void norm2(double** Y)
{
	// 归一化
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
				Y[i][m] = (Y[i][m] - mean) / sigma;
		}
	}
}

void load_train(double** Y, const char* address2)
{
	ifstream cin(address2);							// 输入txt中样本水平排列 
	for (int m = 0; m < MD; m++)
	{
		for (int j = 0; j < N; j++) {
			cin >> Y[j][m];									// 读取训练数据 
		}
	}
	
}


#endif
