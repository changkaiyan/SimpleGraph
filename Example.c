#include"SimpleGraph.h"
#include<stdio.h>
int main()
{
	int weight = matrix_variable(matrix_scanData(1,1));
	int data = matrix_placeholder(12, 1);
	int trans = matrix_mul(data, weight);
	int bias=matrix_variable(matrix_scanData(12,1));
	int temp1=matrix_add(trans,bias);
	int temp=matrix_constant(matrix_scanData(12,1));
	int temp2=matrix_sub(temp1,temp);
	int out = matrix_meanSquar(temp2);
	
	Node x=matrix_scanData(12, 1);
	for(int i=0;i<10000;++i)
	{
		matrix_fillIn(data,x);
		matrix_forwardFlow();
		matrix_backFlow(out);
		matrix_optimize(weight,0.3);
		matrix_optimize(bias,0.3);
	}
	matrix_fillIn(data,x);
	matrix_forwardFlow();
	matrix_printData(temp1);
}