#include"SimpleGraph.h"
#include<stdio.h>
int main()
{
	int weight = matrix_variable(matrix_scanData(3, 2));
	int data = matrix_placeholder(2, 3);
	int trans = matrix_mul(weight, data);
	int c = matrix_relu(trans);
	int out = matrix_meanSquar(c);
	matrix_fillIn(data, matrix_scanData(2, 3));
	matrix_forwardFlow();
	matrix_backFlow(out);
	matrix_printGrad(weight);
}