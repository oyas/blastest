#include <cstdio>
#include <vector>
#include <openblas/cblas.h>

using namespace std;

typedef vector<vector<double>> Matrix;

// ベクトルの内積
inline double dot(const vector<double> &a, const vector<double> &b){
	if( a.size() != b.size() ) return {};
	int N = a.size();
	double c;
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			1, 1, N,	// Aの行数、Bの列数、Aの列数(Bの行数)
			1.0,	// alpha
			&a[0], N,	// A
			&b[0], 1,	// B
			0.0,	// beta
			&c, 1	// C
		);
	return c;
}

// 行列xベクトル
vector<double> mul(const Matrix &A, const vector<double> &b){
	if( A.size() != b.size() ) return {};
	int N = b.size();
	for(auto a: A) if( N != a.size() ) return {};
	vector<double> _A;
	for(auto a: A) std::copy(a.begin(), a.end(), std::back_inserter(_A));
	vector<double> C(N);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			N, 1, N,	// Aの行数、Bの列数、Aの列数(Bの行数)
			1.0,	// alpha
			&_A[0], N,	// A
			&b[0], 1,	// B
			0.0,	// beta
			&C[0], 1	// C
		);
	return C;
}

template<typename T>
vector<T> operator+(const vector<T>& t1, const vector<T>& t2){
	vector<T> ret( max(t1.size(),t2.size()) );
	for(int i=0; i<t1.size(); i++) ret[i] += t1[i];
	for(int i=0; i<t2.size(); i++) ret[i] += t2[i];
	return ret;
}

template<typename T>
vector<T> operator-(const vector<T>& t1, const vector<T>& t2){
	vector<T> ret( max(t1.size(),t2.size()) );
	for(int i=0; i<t1.size(); i++) ret[i] += t1[i];
	for(int i=0; i<t2.size(); i++) ret[i] -= t2[i];
	return ret;
}

template<typename T>
vector<T> operator*(const double& b, const vector<T>& t2){
	vector<T> ret = t2;
	for(auto &&t: ret) t *= b;
	return ret;
}

template<typename T>
T operator*(const vector<T>& t1, const vector<T>& t2){
	return dot(t1, t2);
}

template<typename T>
vector<T> operator*(const Matrix& A, const vector<T>& b){
	return mul(A, b);
}

template<typename T>
vector<T> calc(const Matrix &A, const vector<T> &b){
	int N = b.size();
	if( N != A.size() ) return {};
	for(auto a: A) if( N != a.size() ) return {};

	printf("CG solver start\n");
	const double EPS = 1e-30;

	// 共役勾配法で、Ax = b を解く
	vector<T> x(N, 0.0);		// 解
	double ro0, ro1;
	auto r = b - A*x;
	auto p = r;
	for(int i=0; i<50; i++){
		ro0 = ro1;
		ro1 = r*r;
		if( i > 0 ){
			double beta = ro1 / ro0;
			p = r + beta * p;
		}
		auto q = A * p;
		double alpha = ro1 / (p*q);
		x = x + alpha*p;
		r = r - alpha*q;
		printf("%d\tError: %.5e\n", i, ro1);
		if( ro1 < EPS ) break;
	}

	return x;
}

int main(){
//	Matrix A = {
//		{1.0, 2.0, 3.0},
//		{1.0, 2.0, 3.0},
//		{1.0, 2.0, 3.0},
//	};
//	vector<double> b = {
//		1.0, 2.0, 3.0,
//	};
	Matrix A = {
		{5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{2.0, 5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{0.0, 2.0, 5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 2.0, 5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 2.0, 5.0, 2.0, 0.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 0.0, 2.0, 5.0, 2.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 5.0, 2.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 5.0, 2.0, 0.0},
		{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 5.0, 2.0},
		{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 5.0}
	};
	vector<double> b = {3.0, 1.0, 4.0, 0.0, 5.0, -1.0, 6.0, -2.0, 7.0, -15.0};
	
	// print A
	printf("\nA =\n");
	for(auto a: A){
		for(auto t: a) printf("%.2e ", t);
		printf("\n");
	}

	// print b
	printf("\nb =\n");
	for(auto t: b) printf("%.2e ", t); printf("\n");

	printf("\n Ax = b\n");
	auto x = calc(A, b);
	
	// print x
	printf("\nx =\n");
	for(auto t: x) printf("%.2e ", t); printf("\n");

	return 0;
}

/*
// alpha*A*B + beta*C
cblas_dgemm( const enum CBLAS_ORDER order,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_TRANSPOSE TransB,
    const int M,                        // 行列Aの行数
    const int N,                        // 行列Bの列数
    const int K,                        // 行列Aの列数、行列Bの行数
    const double alpha,                 // 行列の積に掛けるスカラ値(なければ1を設定)
    const double *A,                    // 行列A
    const int ldA,                      // Aのleading dimension (通常は行数を指定すれば良い）
    const double *B,                    // 行列B
    const int ldB,                      // Bのleading dimension
    const double beta,                  // 行列Cに掛けるスカラ値(なければ0を設定)
    double *C,                          // 行列C（ＡとＢの積） !破壊され結果が代入される
    const int ldc );                    // Cのleading dimension

// Orderには行列の形式を指定します。
enum CBLAS_ORDER {
	CblasRowMajor=101,		// 行形式
	CblasColMajor=102		// 列形式
};
// TransAおよびTransBには積を求める前に行列を転置するかどうかを指定します。
enum CBLAS_TRANSPOSE {
	CblasNoTrans=111,		// 転置なし
	CblasTrans=112,			// 転置
	CblasConjTrans=113		// 共役転置
};
*/
