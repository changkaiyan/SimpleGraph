enum Type { CONSTANT, VARIABLE, ADD, SUB, MULTIPLY, PLACEHOLDER, MEANSQUAR,RELU };
typedef struct Node {
	double**data;//存储数据,在matgraph中是前向传播的计算数据,在grad中是反向传播的梯度
	int m;
	int n;
	int lnode;//梯度表中左双亲,前向表中左孩子
	int rnode;//梯度表中右双亲,前向表中右孩子
	enum Type type;//节点类型
	int parentGrad;//指示双亲结点尚未求导的个数,为零时可以对此节点求导.前向图禁止使用.
}Node;

void matrix_backFlow(int node);
int matrix_constant(Node x);
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