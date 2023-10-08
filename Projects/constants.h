#ifndef CONSTANTS_H_
#define CONSTANTS_H_

#define L 3												// 隐藏层数量 
#define M 1												// 输入数量 
#define N 1												// 输出数量 
#define MD 101											// 数据组数 
#define alpha 0.01										// 学习速率

#define NONE 0
#define RELU 1
#define SOFTMAX 2
#define SIGMOID 3
#define TANH 4
#define LOG 5 

// 各种开关 
#define INIT_W 1										// W初始化选项，0为全0，1为随机，2为载入 
#define BP true 										// 是否优化参数 
#define debug 0											// debug等级0-3，决定Log日志记录多少信息 
#define SAVE true										// 是否保存参数 
#define QTYPE 1											// 问题类型，1为拟合，2为分类 

#endif
