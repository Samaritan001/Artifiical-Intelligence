#include <iostream>
#include <cmath>
#include <fstream>
#include "constants.h" 									// L, M, N, Max_nl��������ṹ���� 
#include "initialization3_1.h"							// ��ʼ�� 
#include "activations3_1.h"								// ����� 
#include "forwardprop3_1.h"								// ���򴫲� 
#include "backprop3_1.h"								// ���򴫲� 

#define ROUND 2000										// ѵ������ 
#define address1 "E:\\����ʦ��__����\\���ݹ���\\������\\input.txt"
#define address2 "E:\\����ʦ��__����\\���ݹ���\\������\\train.txt"
#define address3 "E:\\����ʦ��__����\\���ݹ���\\������\\Weight.txt"
#define address4 "E:\\����ʦ��__����\\���ݹ���\\������\\Log��־.txt"
#define address5 "E:\\����ʦ��__����\\���ݹ���\\������\\Weight_stored.txt"

using namespace std;

// չʾ 
void iter(double*** W, double** X, double** Y, double*** x, double*** s, double*** y, 
	double*** dW, double*** ds, double*** dy, const int* nl, int* types, double loss[][MD], int r);
void dispose(double*** W, double** Y, double*** x, double*** s, double*** dW, double*** ds, double*** dy, int* nl);
void save_model(double*** W, int* nl); 
void show(double*** W, const int* nl);
void Log(double*** W, double*** x, double*** s, double*** y, double*** dW, 
	double*** ds, double*** dy, int* nl, int* types, ofstream & outlog, int r, double* loss); 

// ��Ҫ������Ϳ���
// init (new): load() (load_model), random(), zeros(); dispose (delete)
// save() (save_model)
// forward(): input(): load_data() X, Y; predict()
// log()
// bp = True
// debug = 0

int main()
{
	// �������� 
	int nl[L+1] = {M, 3, 2, N};     	   				// ÿ���е�Ԫ�ػ���Ԫ���� 
	double* X[M+1];										// ������, (M+1,MD)
	double* Y[N];										// ѵ��ֵ, (N,MD)
	double** x[L+1];									// ������, (L+1,n[l]+1,MD)
	double** s[L+1];									// ��Ȩ�ؼӺͺ�, (L+1,n[l],MD)
	double*** y = x;									// �����, (L+1,n[l]+1,MD)	
	double** W[L+1];									// Ȩ�ؾ���, (L+1,nl[l],nl[l-1]+1)
	int types[L] = {3, 3, 0}; 								// ��������� 
	
	double** ds[L+1];									// ���� s��(L+1,n[l],MD)
	double** dy[L+1];									// ���� y��(L+1,n[l],MD) 
	double** dW[L+1];									// ���� W��(L+1,n[l],n[l-1]+1) 
	
	double loss[ROUND][MD];
	
	int w_min = 0;										// �����ʼ����Χ 
	int w_max = 99;
	init(W, X, Y, x, s, y, dW, ds, dy, nl, w_min, w_max, address5);
	show(W, nl);
	
	// �������� X 
	load_data(X, address1);
	norm1(X);
	cout << "X" << endl;
	for (int m = 0; m < MD; m++)
		cout << X[1][m] << " ";
	cout << endl;
	
	load_train(Y, address2);
	/*
	switch (QTYPE)										// ���Ϊ����������׼��Y 
	{
		case 1:
			norm2(Y);
			break;
		case 2:
			break;
	}
	*/
	cout << "Y" << endl;
	for (int m = 0; m < MD; m++)
		cout << Y[0][m] << " ";
	cout << endl;
	
	ofstream outlog(address4);
	
	if (BP)
		for (int r = 0; r < ROUND; r++)
		{
			if ((r+1) % 50 == 0)
				cout << "ROUND: " << r+1 << endl;
			iter(W, X, Y, x, s, y, dW, ds, dy, nl, types, loss, r);				// ����һ�ε�����һ��ѵ�� 
			Log(W, x, s, y, dW, ds, dy, nl, types, outlog, r, loss[r]);
			for (int l = L; l >= 1; l--) {
				int I = nl[l-1];
				int J = nl[l];
				// ���²���
				update(W[l], dW[l], I, J);
			}
		}
	else
	{
		iter(W, X, Y, x, s, y, dW, ds, dy, nl, types, loss, 0);				// ֻ��һ������
		Log(W, x, s, y, dW, ds, dy, nl, types, outlog, 0, loss[0]);
	}

	
	// ����ģ�Ͳ��� 
	if (SAVE) 
		save_model(W, nl);
	
	// �ͷŶ�̬�����ڴ� 
	dispose(W, Y, x, s, dW, ds, dy, nl);
	
	return 0;
}

void iter(double*** W, double** X, double** Y, double*** x, double*** s, double*** y, 
	double*** dW, double*** ds, double*** dy, const int* nl, int* types, double loss[][MD], int r)
{
	
	x[0] = X;
	
	// ���򴫲�
	for (int l = 1; l <= L; l++) {						// ѭ��ÿ�� 
		int I = nl[l-1];								// �������� 
		int J = nl[l];									// ��ǰ����Ԫ���� 
		int K = nl[l+1]; 
		 
		f(s[l], x[l-1], W[l], I, J);					// ����ֵ��Ȩ�ؾ��󣬼�ƫֵ��ʡ����g����
		h(y[l], s[l], J, types[l-1]);					// �������
	}
	/*
	cout << "y " << r << endl;
	for (int m = 0; m < MD; m++)
		cout << y[L][1][m] << " ";
	cout << endl;
	*/
	rnorm(y[L], Y);
	
	
	// ��¼��ʧ
	switch (QTYPE) 
	{
		case 1:
			for (int m = 0; m < MD; m++)
				loss[r][m] = 0.5 * pow(y[L][1][m] - Y[0][m], 2);
			break;
		case 2:
			break;	
	}
	
	if (r == ROUND-1)
	{
		cout << "y_hat: ";
		for (int m = 0; m < MD; m++)
			cout << y[L][1][m] << " ";
		cout << endl;
		cout << "loss: ";
		for (int m = 0; m < MD; m++)
			cout << 0.5 * pow(y[L][1][m] - Y[0][m], 2) << " ";
		cout << endl;
		show(W, nl);
	}
	
	// ���򴫲�
	if (!BP)
		return;
	
	for (int l = L; l >= 1; l--) {
		int I = nl[l-1];
		int J = nl[l];
		int K = 0;

		if (l == L) {									// ���Ϊ����㣬�������ʧ��������Ԥ��ֵ��ƫ�� 
			reverse(ds[L], y[L], Y, types[L-1]);		// ֱ���Ƴ�ds
		}
		else {
			K = nl[l+1];
			s2y(dy[l], W[l+1], ds[l+1], J, K);			// �ú�һ��� W�� ds���µ�ǰ���dy
			y2s(ds[l], s[l], y[l], dy[l], J, types[l-1]);					// ��dy�Ƶ�ds
		}
		
		s2W(dW[l], ds[l], x[l-1], I, J);				// ��ds���dW
		
		// ���²���
		// update(W[l], dW[l], I, J);
	}

}

void dispose(double*** W, double** Y, double*** x, double*** s, double*** dW, double*** ds, double*** dy, int* nl)
{
	for (int l = 1; l <= L; l++)
	{
		delete [] x[l][0];								// x��y���õ�ַ 
		for (int j = 0; j < nl[l]; j++)
		{
			delete [] x[l][j+1];
			delete [] s[l][j];
			delete [] W[l][j];
			delete [] ds[l][j];
			delete [] dW[l][j]; 
		}
	}
	for (int j = 0; j <= M; j++)						// �ͷ� X���ڴ� 
		delete [] x[0][j];
	for (int n = 0; n < N; n++)
		delete [] Y[n];
}

void save_model(double*** W, int* nl)
{
	// ������� W 
	ofstream cout(address3);
	for (int l = 1; l <= L; l++)
	{
		for (int j = 0; j < nl[l]; j++)
		{
			for (int i = 0; i <= nl[l-1]; i++)
				cout << W[l][j][i] << " ";
			cout << endl; 
		}
		cout << endl;
	}
}

void show(double*** W, const int* nl)							// ��ӡ�����Ĳ��Ժ��� 
{
	cout << "W" << endl;
	for (int l = 1; l <= L; l++) {
		cout << "layer = " << l << endl;
		for (int j = 0; j < nl[l]; j++) {
			for (int i = 0; i <= nl[l-1]; i++)
				cout << W[l][j][i] << " ";
			cout << endl;
		}
		cout << endl;
	}
}

// ��־������������־�ļ� 
void Log(double*** W, double*** x, double*** s, double*** y, double*** dW, 
	double*** ds, double*** dy, int* nl, int* types, ofstream & outlog, int r, double* loss)
{
	if (r == 0) {
		outlog << "LOG" << endl;
		outlog << "Debug(0-3): " << debug << endl;
		outlog << "L: " << L << "  M: " << M << "  N: " << N << "  MD: " << MD;
		outlog << "  alpha: " << alpha << " ROUND: " << ROUND << endl << endl;
	}

	outlog << "ROUND" << " " << r+1 << endl;
	for (int l = 1; l <= L; l++) {
		if (debug >= 1) {
			outlog << "Forward  Layer " << l << endl;
		}
		
		if (l == 1 && debug >= 2) {
			outlog << "x" << endl;
			for (int i = 1; i <= nl[l-1]; i++) {
				for (int m = 0; m < MD; m++)
					outlog << x[l-1][i][m] << " ";
				outlog << endl;
			}
			outlog << endl;
		}
		
		if (debug >= 1) {
			outlog << "W" << endl;;
			for (int j = 0; j < nl[l]; j++) {
				for (int i = 0; i <= nl[l-1]; i++)
					outlog << W[l][j][i] << " ";
				outlog << endl;
			}
			outlog << endl;
		}	
		
		if (debug >= 3) {
			outlog << "s" << endl;
			for (int j = 0; j < nl[l]; j++) {
				for (int m = 0; m < MD; m++)
					outlog << s[l][j][m] << " ";
				outlog << endl;
			}
			outlog << endl;
		}
		
		if (debug >= 2 || (l == L && debug >= 1) || (l == L && r == ROUND-1) || BP == false) {
			outlog << "activation function: " << types[l-1] << endl << endl;
			outlog << "y" << endl;
			for (int j = 1; j <= nl[l]; j++) {
				for (int m = 0; m < MD; m++)
					outlog << y[l][j][m] << " ";
				outlog << endl;
			}
			outlog << endl;
		}
		if (l != L && debug >= 1)
			outlog << endl;
	}
	
	outlog << "Loss" << endl;
	double total_loss = 0.0;
	for (int m = 0; m < MD; m++) {
		outlog << loss[m] << " ";
		total_loss += loss[m];
	}
	outlog << endl << "Total Loss: " << total_loss << endl << endl;
	
	if (debug == 0)	
		return;
	outlog << endl;
	for (int l = L; l >= 1; l--) {
		outlog << "Backward  Layer " << l << endl;
		if (l != L && debug >= 2) {
			outlog << "dy" << endl;
			for (int j = 0; j < nl[l]; j++) {
				for (int m = 0; m < MD; m++)
					outlog << dy[l][j][m] << " ";
				outlog << endl;
			}
			outlog << endl;
		}
		
		if (debug >= 3 || (debug >= 2 && l == L)) {
			outlog << "ds" << endl;
			for (int j = 0; j < nl[l]; j++) {
				for (int m = 0; m < MD; m++)
					outlog << ds[l][j][m] << " ";
				outlog << endl;
			}
			outlog << endl;
		}
		
		if (debug >= 1) {
			outlog << "dW" << endl;;
			for (int j = 0; j < nl[l]; j++) {
				for (int i = 0; i <= nl[l-1]; i++)
					outlog << dW[l][j][i] << " ";
				outlog << endl;
			}
			outlog << endl << endl;
		}
		if (l == 1)
			outlog << endl;
	}
	
}

