#include"SimpleGraph.h"
#include<stdio.h>
int main()
{
	int weight = matrix_variable(matrix_scanData(2,3));
	int data = matrix_placeholder(2,2);
	int trans = matrix_mul(data, weight);
	int out = matrix_meanSquar(trans);
	Node x=matrix_scanData(2, 2);
	matrix_fillIn(data,x);
	matrix_forwardFlow();
	matrix_backFlow(out);
	matrix_optimize(weight,1);
	matrix_printData(weight);
}