/* The interface of the SimpleGraph.
   Copyright 2018 Kaiyan Chang.

This file is the header of the SimpleGraph.

SimpleGraph is free software; you can redistribute it and/or modify it under
the terms of the Apache 2.0 License , or (at your option) any later
version.

SimpleGraph is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the Apache License
for more details.

You should have received a copy of the Apache 2.0 License along with SimpleGraph; 
see the file License.   
*/

/**
 * @brief All types of nodes.
 * 
 */
enum Type { CONSTANT, VARIABLE, ADD, SUB, MULTIPLY, PLACEHOLDER, MEANSQUAR,RELU };

/**
 * @brief A Node in the Computational Graph
 * 
 */
typedef struct Node {
	double**data;//Store Data and Gradient
	int m;//The height of matrix
	int n;//The length of matrix
	int lnode;//Left child index in forward table.First parent index in backward table.
	int rnode;//Right child index in forward table.Second parent index in backward table. 
	enum Type type;//Type of the node.
	int parentGrad;//Only use in Gradient table,calculate the rest parent which has not been derivated.
}Node;

/**
 * @brief Make the gradient backflow
 * 
 * @param node The node index you want to optimize(Must be a scaler).
 */
void matrix_backFlow(int node);

/**
 * @brief Create a constant node
 * 
 * @param x A node that contains m,n and data.
 * @return int The index of the node in the default graph.
 */
int matrix_constant(Node x);

/**
 * @brief 
 * 
 * @param x 
 * @return int 
 */
int matrix_variable(Node x);
int matrix_add(int lchild, int rchild);
int matrix_mul(int lchild, int rchild);
int matrix_sub(int lchild, int rchild);
int matrix_meanSquar(int lchild);
void matrix_forwardFlow();
int matrix_relu(int lchild);
void deletegraph();
void matrix_fillIn(int node, Node x);
void matrix_printGrad(int node);
static void backward(int node);
static void cleargrad();
Node matrix_scanData(int m, int n);
void matrix_printData(int node);
int matrix_placeholder(int m, int n);