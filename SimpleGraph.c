/**
 * @author ChangKaiyan (changkaiyan@qq.com)
 * @brief A Computational Graph for Deep learning from UESTC Software Engineering
 * @version 0.1
 * @date 2019-05-15
 * @copyright Copyright 2018 Kaiyan Chang
 *
 */

#include<stdio.h>
#include<math.h>
#include<assert.h>
#include<malloc.h>
#include"SimpleGraph.h"
#include<stdbool.h>

#define ERRORFUNC fprintf(stderr,"In function %s , in line %d ,"\
					"an error occurs.\nCheck it for the details:\n",\
					__FUNCTION__,__LINE__)
#ifdef DEBUG
	#define DEBUGSHOW printf("Running in function %s, in line %d:\n",__FUNCTION__,__LINE__)
#endif
#ifndef DEBUGSHOW
	#define DEBUGSHOW
#endif

static Node mul(Node*a, Node*b);
static Node add(Node*a, Node*b);
static Node sub(Node*a, Node*b);
static void backward(int node);
static void cleargrad();
static Node matgraph[512];//
static int graphpoint = 0;//
static Node grad[512];//
bool has_forward = false;
/**
 * @brief
 *
 * @param a
 * @param b
 * @return Node
 */
static Node add(Node*a, Node*b)
{
	assert(a->m == b->m&&a->n == b->n);
	Node temp;
	temp.m = a->m;
	temp.n = a->n;
	for (int i = 0; i < a->m; ++i)
		for (int j = 0; j < a->n; ++j)
		{
			temp.data[i][j] = a->data[i][j] + b->data[i][j];
		}
	return temp;
}

/**
 * @brief
 *
 * @param a
 * @param b
 * @return Node
 */
static Node sub(Node*a, Node*b)
{
	assert(a->m == b->m&&a->n == b->n);
	Node temp;
	temp.m = a->m;
	temp.n = a->n;
	for (int i = 0; i < a->m; ++i)
		for (int j = 0; j < a->n; ++j)
		{
			temp.data[i][j] = a->data[i][j] - b->data[i][j];
		}
	return temp;
}

/**
 * @brief
 *
 * @param a
 * @param b
 * @return Node
 */
static Node mul(Node*a, Node*b)
{
	assert(a->n == b->m);
	Node temp;
	temp.m = a->m;
	temp.n = b->n;
	for (int i = 0; i < a->m; ++i)
		for (int j = 0; j < b->n; ++j)
		{
			temp.data[i][j] = 0;
			for (int k = 0; k < a->n; ++k)
				temp.data[i][j] += a->data[i][k] * b->data[k][j];
		}
	return temp;
}

/**
 * @brief
 *
 * @param x Which must have specific shape and data
 * @return int
 */
int matrix_constant(Node x)
{
	matgraph[graphpoint] = x;
	grad[graphpoint].type = matgraph[graphpoint].type = CONSTANT;
	matgraph[graphpoint].lnode = matgraph[graphpoint].rnode = -1;
	grad[graphpoint].parentGrad = grad[graphpoint].parentGrad_=0;
	grad[graphpoint].data = (double**)calloc(matgraph[graphpoint].m, sizeof(double*));
	for (int j = 0; j < matgraph[graphpoint].m; ++j)
	{
		grad[graphpoint].data[j] = (double*)calloc(matgraph[graphpoint].n, sizeof(double));
	}
	grad[graphpoint].m = matgraph[graphpoint].m;
	grad[graphpoint].n = matgraph[graphpoint].n;
	return graphpoint++;
}

/**
 * @brief
 *
 * @param x
 * @return int
 */
int matrix_variable(Node x)
{
	matgraph[graphpoint] = x;
	grad[graphpoint].type = matgraph[graphpoint].type = VARIABLE;
	matgraph[graphpoint].lnode = matgraph[graphpoint].rnode = -1;
	grad[graphpoint].parentGrad = grad[graphpoint].parentGrad_=0;
	grad[graphpoint].data = (double**)calloc(matgraph[graphpoint].m, sizeof(double*));
	for (int j = 0; j < matgraph[graphpoint].m; ++j)
	{
		grad[graphpoint].data[j] = (double*)calloc(matgraph[graphpoint].n, sizeof(double));
	}
	grad[graphpoint].m = matgraph[graphpoint].m;
	grad[graphpoint].n = matgraph[graphpoint].n;
	return graphpoint++;
}

/**
 * @brief 
 *
 * @param m
 * @param n
 * @return int
 */
int matrix_placeholder(int m, int n)//需要输入矩阵型号,确保前向传播矩阵型号一致统一,构建计算图时,反向传播统一分配所有内存
{
	grad[graphpoint].m = matgraph[graphpoint].m = m;
	grad[graphpoint].n = matgraph[graphpoint].n = n;
	grad[graphpoint].type = matgraph[graphpoint].type = PLACEHOLDER;
	matgraph[graphpoint].lnode = matgraph[graphpoint].rnode = -1;
	matgraph[graphpoint].data = NULL;
	grad[graphpoint].parentGrad = grad[graphpoint].parentGrad_ = 0;
	grad[graphpoint].data = (double**)calloc(matgraph[graphpoint].m, sizeof(double*));
	for (int j = 0; j < matgraph[graphpoint].m; ++j)
	{
		grad[graphpoint].data[j] = (double*)calloc(matgraph[graphpoint].n, sizeof(double));
	}
	return graphpoint++;
}

/**
 * @brief
 *
 * @param lchild
 * @param rchild
 * @return int
 */
int matrix_add(int lchild, int rchild)
{
	grad[graphpoint].m = matgraph[graphpoint].m = matgraph[lchild].m;
	grad[graphpoint].n = matgraph[graphpoint].n = matgraph[rchild].n;
	matgraph[graphpoint].data = (double**)calloc(matgraph[graphpoint].m, sizeof(double*));
	for (int j = 0; j < matgraph[graphpoint].m; ++j)
	{
		matgraph[graphpoint].data[j] = (double*)calloc(matgraph[graphpoint].n, sizeof(double));
	}
	grad[graphpoint].data = (double**)calloc(matgraph[graphpoint].m, sizeof(double*));
	for (int j = 0; j < matgraph[graphpoint].m; ++j)
	{
		grad[graphpoint].data[j] = (double*)calloc(matgraph[graphpoint].n, sizeof(double));
	}
	grad[graphpoint].type = matgraph[graphpoint].type = ADD;
	matgraph[graphpoint].lnode = lchild;
	matgraph[graphpoint].rnode = rchild;
	grad[lchild].parentGrad++;
	grad[rchild].parentGrad++;
	grad[lchild].parentGrad_++;
	grad[rchild].parentGrad_++;
	grad[graphpoint].parentGrad = grad[graphpoint].parentGrad_=0;
	return graphpoint++;
}

/**
 * @brief
 *
 * @param lchild
 * @param rchild
 * @return int
 */
int matrix_mul(int lchild, int rchild)
{
	DEBUGSHOW;
	grad[graphpoint].m = matgraph[graphpoint].m = matgraph[lchild].m;
	grad[graphpoint].n = matgraph[graphpoint].n = matgraph[rchild].n;
	matgraph[graphpoint].data = (double**)calloc(matgraph[graphpoint].m, sizeof(double*));
	for (int j = 0; j < matgraph[graphpoint].m; ++j)
	{
		matgraph[graphpoint].data[j] = (double*)calloc(matgraph[graphpoint].n, sizeof(double));
	}
	grad[graphpoint].data = (double**)calloc(matgraph[graphpoint].m, sizeof(double*));
	for (int j = 0; j < matgraph[graphpoint].m; ++j)
	{
		grad[graphpoint].data[j] = (double*)calloc(matgraph[graphpoint].n, sizeof(double));
	}
	grad[graphpoint].type = matgraph[graphpoint].type = MULTIPLY;
	matgraph[graphpoint].lnode = lchild;
	matgraph[graphpoint].rnode = rchild;
	grad[lchild].parentGrad++;
	grad[rchild].parentGrad++;
	grad[lchild].parentGrad_++;
	grad[rchild].parentGrad_++;
	grad[graphpoint].parentGrad = grad[graphpoint].parentGrad_=0;
	return graphpoint++;
}

Node matrix_zero(int m, int n)
{
	Node matrix;
	matrix.m = m;
	matrix.n = n;
	matrix.data = (double**)calloc(m, sizeof(double*));
	for (int i = 0; i < m; ++i)
		matrix.data[i] = (double*)calloc(n, sizeof(double));
	return matrix;
}

/**
 * @brief
 *
 * @param lchild
 * @param rchild
 * @return int
 */
int matrix_sub(int lchild, int rchild)
{
	grad[graphpoint].m = matgraph[graphpoint].m = matgraph[lchild].m;
	grad[graphpoint].n = matgraph[graphpoint].n = matgraph[rchild].n;
	matgraph[graphpoint].data = (double**)calloc(matgraph[graphpoint].m, sizeof(double*));
	for (int j = 0; j < matgraph[graphpoint].m; ++j)
	{
		matgraph[graphpoint].data[j] = (double*)calloc(matgraph[graphpoint].n, sizeof(double));
	}
	grad[graphpoint].data = (double**)calloc(matgraph[graphpoint].m, sizeof(double*));
	for (int j = 0; j < matgraph[graphpoint].m; ++j)
	{
		grad[graphpoint].data[j] = (double*)calloc(matgraph[graphpoint].n, sizeof(double));
	}
	grad[graphpoint].type = matgraph[graphpoint].type = SUB;
	matgraph[graphpoint].lnode = lchild;
	matgraph[graphpoint].rnode = rchild;
	grad[lchild].parentGrad++;
	grad[rchild].parentGrad++;
	grad[lchild].parentGrad_++;
	grad[rchild].parentGrad_++;
	grad[graphpoint].parentGrad = grad[graphpoint].parentGrad_=0;
	return graphpoint++;
}

int matrix_relu(int lchild)
{
	grad[graphpoint].m = matgraph[graphpoint].m = matgraph[lchild].m;
	grad[graphpoint].n = matgraph[graphpoint].n = matgraph[lchild].n;
	matgraph[graphpoint].data = (double**)calloc(matgraph[graphpoint].m, sizeof(double*));
	for (int j = 0; j < matgraph[graphpoint].m; ++j)
	{
		matgraph[graphpoint].data[j] = (double*)calloc(matgraph[graphpoint].n, sizeof(double));
	}
	grad[graphpoint].data = (double**)calloc(matgraph[graphpoint].m, sizeof(double*));
	for (int j = 0; j < matgraph[graphpoint].m; ++j)
	{
		grad[graphpoint].data[j] = (double*)calloc(matgraph[graphpoint].n, sizeof(double));
	}
	grad[graphpoint].type = matgraph[graphpoint].type = RELU;
	matgraph[graphpoint].lnode = lchild;
	matgraph[graphpoint].rnode = -1;
	grad[lchild].parentGrad++;
	grad[lchild].parentGrad_++;
	grad[graphpoint].parentGrad = grad[graphpoint].parentGrad_=0;
	return graphpoint++;
}
/**
 * @brief
 *
 * @param lchild
 * @return int
 */
int matrix_meanSquar(int lchild)
{
	grad[graphpoint].m = matgraph[graphpoint].m = 1;
	grad[graphpoint].n = matgraph[graphpoint].n = 1;
	matgraph[graphpoint].data = (double**)calloc(1, sizeof(double*));
	matgraph[graphpoint].data[0] = (double*)calloc(1, sizeof(double));
	grad[graphpoint].type = matgraph[graphpoint].type = MEANSQUAR;
	matgraph[graphpoint].lnode = lchild;
	matgraph[graphpoint].rnode = -1;
	grad[lchild].parentGrad++;
	grad[lchild].parentGrad_++;
	grad[graphpoint].parentGrad = 0;
	grad[graphpoint].parentGrad_ = 0;
	grad[graphpoint].data = (double**)calloc(matgraph[graphpoint].m, sizeof(double*));
	for (int j = 0; j < matgraph[graphpoint].m; ++j)
	{
		grad[graphpoint].data[j] = (double*)calloc(matgraph[graphpoint].n, sizeof(double));
	}
	grad[graphpoint].m = matgraph[graphpoint].m;
	grad[graphpoint].n = matgraph[graphpoint].n;
	return graphpoint++;
}

/**
 * @brief
 *
 * @param node
 * @param x
 */
void matrix_fillIn(int node, Node x)
{
	if (matgraph[node].type != PLACEHOLDER)
	{
		ERRORFUNC;
		fprintf(stderr, "Can not fill in Data to a not-placeholder node!");
	}
	else
		matgraph[node].data = x.data;
}

/**
 * @brief
 *
 */
void deletegraph()
{
	for (int i = 0; i < graphpoint; ++i)
	{
		for (int j = 0; j < matgraph[i].m; ++j)
		{
			free(matgraph[i].data[j]);
		}
		free(matgraph[i].data);
		for (int j = 0; j < grad[i].m; ++j)
		{
			free(grad[i].data[j]);
		}
		free(grad[i].data);
	}
}


/**
 * @brief Make the graph run, caculate the forward.Only one optimized node and a fixed shape graph.
 * 
 */
void matrix_forwardFlow()
{
	cleargrad();
	has_forward = true;
	for (int index = 0; index < graphpoint; ++index)
	{
		switch (matgraph[index].type)
		{
		case ADD:
		{
			for (int i = 0; i < matgraph[index].m; ++i)
				for (int j = 0; j < matgraph[index].n; ++j)
					matgraph[index].data[i][j] = matgraph[matgraph[index].lnode].data[i][j] + matgraph[matgraph[index].rnode].data[i][j];
			break;
		}
		case MULTIPLY:
		{
			for (int i = 0; i < matgraph[index].m; ++i)
				for (int j = 0; j < matgraph[index].n; ++j)
					for (int k = 0; k < matgraph[matgraph[index].lnode].n; ++k)
						matgraph[index].data[i][j] = matgraph[matgraph[index].lnode].data[i][k] * matgraph[matgraph[index].rnode].data[k][j];
			break;
		}
		case SUB:
		{
			for (int i = 0; i < matgraph[index].m; ++i)
				for (int j = 0; j < matgraph[index].n; ++j)
					matgraph[index].data[i][j] = matgraph[matgraph[index].lnode].data[i][j] - matgraph[matgraph[index].rnode].data[i][j];
			break;
		}
		case MEANSQUAR:
		{
			matgraph[index].data[0][0] = 0;
			for (int i = 0; i < matgraph[matgraph[index].lnode].m; ++i)
				for (int j = 0; j < matgraph[matgraph[index].lnode].n; ++j)
					matgraph[index].data[0][0] += 0.5* matgraph[matgraph[index].lnode].data[i][j] * matgraph[matgraph[index].lnode].data[i][j];
			matgraph[index].data[0][0] /= matgraph[matgraph[index].lnode].m*matgraph[matgraph[index].lnode].n;
			break;
		}
		case PLACEHOLDER:
		{
			if (matgraph[index].data == NULL)
			{
				ERRORFUNC;
				fprintf(stderr, "Must assign to placeholder before putting forward!\n");
				assert(matgraph[index].data != NULL);
			}
			break;
		}
		case RELU:
		{
			for (int i = 0; i < matgraph[index].m; ++i)
				for (int j = 0; j < matgraph[index].n; ++j)
					matgraph[index].data[i][j] = matgraph[matgraph[index].lnode].data[i][j] > 0 ? matgraph[matgraph[index].lnode].data[i][j] : 0;
			break;
		}
		default:;//默认情况下是叶子节点,什么也不需要做
		}
	}
}

/**
 * @brief
 *
 * @param node
 */
void matrix_backFlow(int node)
{
	if (matgraph[node].type != MEANSQUAR)
	{
		ERRORFUNC;
		fprintf(stderr, "Error in backFlow. The original node must output scaler!\n");
		assert(matgraph[node].type == MEANSQUAR);
	}
	else if (has_forward == false)
	{
		ERRORFUNC;
		fprintf(stderr, "On every epoch, forward flow must before back flow.\n");
		assert(has_forward != false);
	}
	else
	{
		backward(node);
	}
}

/**
 * @brief
 *
 * @param node
 */
static void backward(int node)
{
	DEBUGSHOW;
	if (matgraph[node].type==MEANSQUAR)//梯度源点,标量对矩阵求导,这个节点仅有左孩子
	{
		for (int i = 0; i < grad[matgraph[node].lnode].m; ++i)
			for (int j = 0; j < grad[matgraph[node].lnode].n; ++j)
				grad[matgraph[node].lnode].data[i][j] = matgraph[matgraph[node].lnode].data[i][j];//均方误差导数
		grad[matgraph[node].lnode].parentGrad--;
		backward(matgraph[node].lnode);
	}
	else if (grad[node].parentGrad >= 1)//仅余一个父节点没有传递导数,或者两个父节点都没有传递导数
	{
		return;//等待剩余父节点传递导数
	}
	else//所有父节点导数均传递到此
	{
		if (matgraph[node].lnode == -1 && matgraph[node].rnode == -1)//是叶子节点
		{
			return;//梯度求解完成
		}
		else//非叶子节点
		{
			switch (grad[node].type)
			{
			case ADD:
			{
				for (int i = 0; i < matgraph[node].m; ++i)
					for (int j = 0; j < matgraph[node].n; ++j)
					{
						grad[matgraph[node].lnode].data[i][j] += grad[node].data[i][j];//更新左孩子的梯度
						grad[matgraph[node].rnode].data[i][j] += grad[node].data[i][j];//更新右孩子的梯度
					}
				grad[matgraph[node].lnode].parentGrad--;
				grad[matgraph[node].rnode].parentGrad--;
				backward(matgraph[node].lnode);
				backward(matgraph[node].rnode);
				break;
			}
			case MULTIPLY:
			{
				for (int i = 0; i < grad[node].m; ++i)
					for (int j = 0; j < matgraph[matgraph[node].rnode].m; ++j)
					{
						for (int k = 0; k < grad[node].n; ++k)
							grad[matgraph[node].lnode].data[i][j] += grad[node].data[i][k] * matgraph[matgraph[node].rnode].data[j][k];//更新左孩子的梯度
					}

				for (int i = 0; i < matgraph[matgraph[node].lnode].n; ++i)
					for (int j = 0; j < grad[node].n; ++j)
					{
						for (int k = 0; k < grad[node].m; ++k)
							grad[matgraph[node].rnode].data[i][j] += grad[node].data[k][j] * matgraph[matgraph[node].lnode].data[k][i];//更新右孩子的梯度
					}
				grad[matgraph[node].lnode].parentGrad--;
				grad[matgraph[node].rnode].parentGrad--;
				backward(matgraph[node].lnode);
				backward(matgraph[node].rnode);
				break;
			}
			case SUB:
			{
				for (int i = 0; i < matgraph[node].m; ++i)
					for (int j = 0; j < matgraph[node].n; ++j)
					{
						grad[matgraph[node].lnode].data[i][j] += grad[node].data[i][j];//更新左孩子的梯度
						grad[matgraph[node].rnode].data[i][j] -= grad[node].data[i][j];//更新右孩子的梯度
					}
				grad[matgraph[node].lnode].parentGrad--;
				grad[matgraph[node].rnode].parentGrad--;
				backward(matgraph[node].lnode);
				backward(matgraph[node].rnode);
				break;
			}
			case RELU:
			{
				for (int i = 0; i < matgraph[node].m; ++i)
					for (int j = 0; j < matgraph[node].n; ++j)
					{
						grad[matgraph[node].lnode].data[i][j] += grad[node].data[i][j] * (matgraph[matgraph[node].lnode].data[i][j] > 0 ? 1 : 0);//更新左孩子的梯度
					}
				grad[matgraph[node].lnode].parentGrad--;
				backward(matgraph[node].lnode);
				break;
			}
			}
		}
	}
}

/**
 * @brief Clear the gradient of all the node and reset the count which has 
 * 			not been derived in its parent nodes.
 *
 */
static void cleargrad()//清除所有节点的梯度以及重置双亲节点中尚未求导的个数
{
	DEBUGSHOW;
	for (int i = 0; i < graphpoint; ++i)
	{
		for (int index = 0; index < grad[i].m; ++index)
		{
			for (int j = 0; j < grad[i].n; ++j)
			{
				grad[i].data[index][j] = 0.0;
			}
		}
		grad[i].parentGrad = grad[i].parentGrad_;
	}
}

/**
 * @brief Print data matrix
 * 
 * @param node The index of a node
 */
void matrix_printData(int node)
{
	DEBUGSHOW;
	for (int i = 0; i < matgraph[node].m; ++i)
	{
		for (int j = 0; j < matgraph[node].n; ++j)
			printf("%7.3lf ", matgraph[node].data[i][j]);
		putchar('\n');
	}
}

void matrix_optimize(int vari_node, double learningrate)
{
	DEBUGSHOW;
	if (matgraph[vari_node].type != VARIABLE && learningrate >= 0)
	{
		ERRORFUNC;
		fprintf(stderr, "The optimize node is not a Variable node!!\n");
		assert(matgraph[vari_node].type == VARIABLE);
	}
	double sum = 0.0;
	for (int i = 0; i < matgraph[vari_node].m; ++i)
		for (int j = 0; j < matgraph[vari_node].n; ++j)
		{
			sum += grad[vari_node].data[i][j] * grad[vari_node].data[i][j];
		}
	sum = sqrt(sum);
	if (sum != 0)
	{
		for (int i = 0; i < matgraph[vari_node].m; ++i)
			for (int j = 0; j < matgraph[vari_node].n; ++j)
				matgraph[vari_node].data[i][j] -= learningrate * (grad[vari_node].data[i][j] / sum);
	}
}


/**
 * @brief Print gradiend matrix
 * 
 * @param node The index of the graph
 */
void matrix_printGrad(int node)
{
	for (int i = 0; i < grad[node].m; ++i)
	{
		for (int j = 0; j < grad[node].n; ++j)
			printf("%7.3f ", grad[node].data[i][j]);
		putchar('\n');
	}
}


/**
 * @brief Input matrix data and convert it to Node type.
 *
 * @param m Row numbers
 * @param n Col numbers
 * @return Node Data matrix
 */
Node matrix_scanData(int m, int n)
{
	DEBUGSHOW;
	Node temp;
	temp.data = (double**)malloc(sizeof(double*)*m);
	for (int i = 0; i < m; ++i)
	{
		temp.data[i] = (double*)malloc(sizeof(double)*n);
	}
	for (int i = 0; i < m; ++i)
		for (int j = 0; j < n; ++j)
			scanf("%lf", &temp.data[i][j]);
	temp.m = m;
	temp.n = n;
	return temp;
}
Node matrix_creatNode (int origin)
{
	Node no;
	no=matgraph[origin];
	return no;
}
Node matrix_scanDataFromCsv(const char* filename,int m,int n)
{
	Node temp;
	FILE* file=fopen(filename,"r");
	if(file==NULL)
	{
		ERRORFUNC;
		fprintf(stderr,"File errors!\n");
		return temp;
	}
	temp.data = (double**)malloc(sizeof(double*)*m);
	for (int i = 0; i < m; ++i)
	{
		temp.data[i] = (double*)malloc(sizeof(double)*n);
	}
	if(n>1)
	{
		for (int i = 0; i < m; ++i)
		{	
			for (int j = 0; j < n-1; ++j)
				fscanf(file," %lf,", &temp.data[i][j]);
			fscanf(file," %lf",&temp.data[i][n-1]);
		}
	}
	else if(n==1)
	{
		for (int i = 0; i < m; ++i)
		{	
			fscanf(file," %lf",&temp.data[i][n-1]);
		}
	}
	else
	{
		ERRORFUNC;
		fprintf(stderr,"Error in the input format! Cols must more than 1. \n");
	}
	
	temp.m = m;
	temp.n = n;
	return temp;
}

