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
 * @brief Create a variable node
 * 
 * @param x A node that contains m,n and data.
 * @return int The index of the node in the default graph.
 */
int matrix_variable(Node x);
/**
 * @brief A computation node that can add two matrix
 * 
 * @param lchild A index that contains left node to add
 * @param rchild A index that contains right node to add
 * @return int The index of the ADD node
 */
int matrix_add(int lchild, int rchild);

/**
 * @brief A computation node that can multiply two matrix
 * 
 * @param lchild A index that provides left multiplicator
 * @param rchild A index that provides right multiplicator
 * @return int The index of the multiply node
 */
int matrix_mul(int lchild, int rchild);

/**
 * @brief A computation node that can substract two matrix
 * 
 * @param lchild  A index that contains left matrix to substract
 * @param rchild  A index that contains right matrix to substract
 * @return int The index of the sub node
 */
int matrix_sub(int lchild, int rchild);

/**
 * @brief A computation node that can translate a matrix to scaler
 * 
 * @note A graph must have and only have one of this node.It can output mean square error
 * @param lchild A index that contains matrix to process
 * @return int  The index of node
 */
int matrix_meanSquar(int lchild);
/**
 * @brief Make the graph run, caculate the forward.
 * 
 */
void matrix_forwardFlow();
/**
 * @brief A node can activate the matrix data
 * 
 * @param lchild The index of data source
 * @return int A index
 */
int matrix_relu(int lchild);
/**
 * @brief Delete this SimpleGraph when you have already used
 * 
 */
void deletegraph();
/**
 * @brief Fill in the placeholder node
 * 
 * @note This function must run before forwardFlow
 * @param node The index of a placeholder node
 * @param x The data that you want to fill in.
 */
void matrix_fillIn(int node, Node x);
/**
 * @brief Print gradiend matrix
 * @param node The index of the graph
 */
void matrix_printGrad(int node);
/**
 * @brief Caculate the gradient in graph .
 * 
 * @param node The begin node index. Must be a scaler-output node
 */

Node matrix_scanData(int m, int n);
/**
 * @brief Print data matrix
 * 
 * @param node The index of a node
 */
void matrix_printData(int node);
/**
 * @brief  Create a placeholder node
 * 
 * @param m The height of the matrix
 * @param n The length of the matrix
 * @return int The index of the node
 */
int matrix_placeholder(int m, int n);
/**
 * @brief Optimize a variable node
 * 
 * @param vari_node The index pf the node
 * @param learningrate You can change it beyond the scope
 */
void matrix_optimize(int vari_node,double learningrate);
/**
 * @brief Derive a zero matrix 
 * 
 * @param m The height of the matrix
 * @param n The length of the matrix
 * @return Node The zero matrix
 */
Node matrix_zero(int m,int n);
