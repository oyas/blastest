
Xgemm() 行列x行列演算
--------------------------

**alpha A B + beta C** の演算 (結果は、Cへ上書きされる)

	// alpha*A*B + beta*C
	void cblas_dgemm(
		const enum CBLAS_ORDER order,
	    const enum CBLAS_TRANSPOSE TransA,	// 行列Aに対するオペレータ
	    const enum CBLAS_TRANSPOSE TransB,	// 行列Bに対するオペレータ
	    const int M,                        // 行列Aの行数
	    const int N,                        // 行列Bの列数
	    const int K,                        // 行列Aの列数、行列Bの行数
	    const double alpha,                 // 行列の積に掛けるスカラ値
	    const double *A,                    // 行列A
	    const int ldA,                      // Aのleading dimension (通常は行数を指定すれば良い）
	    const double *B,                    // 行列B
	    const int ldB,                      // Bのleading dimension
	    const double beta,                  // 行列Cに掛けるスカラ値
	    double *C,                          // 行列C !破壊され結果が代入される
	    const int ldc                       // Cのleading dimension
	);

	// Orderには行列の形式を指定します。
	enum CBLAS_ORDER {
		CblasRowMajor  =101,	// 行形式
		CblasColMajor  =102 	// 列形式
	};
	// TransAおよびTransBには積を求める前に行列を転置するかどうかを指定します。
	enum CBLAS_TRANSPOSE {
		CblasNoTrans     =111,	// 転置なし
		CblasTrans       =112,	// 転置
		CblasConjTrans   =113,	// 共役転置
		CblasConjNoTrans =114 	// 共役
	};


## 解説

今回は、Row-major order についてのみ扱う。(Column-majorは行と列が逆になるだけ)

行列Aを以下とする。

	A =
	  1.0  2.0  3.0
	  4.0  5.0  6.0

このとき、行数(M)=2, 列数(K)=3 である。

行列Bを以下とする。

	B =
	  1.0  2.0
	  3.0  4.0
	  5.0  6.0

このとき、行数(K)=3, 列数(N)=2 である。


### Kが共通である理由

この2つを掛け合わせる(**AB**)とき、Aの列数とBの行数が一致していなければならないことがわかる。
なので、Kは共通な値となっているのである。


### ldA, ldB, ldCについて

メモリアドレスは一次元に並んでいる。なので、2次元配列を扱う場合は、ある周期で折り返してやる必要がある。
具体的には、以下の式で行列要素の値を格納するアドレスを決める。

要素(x,y)のアドレス = 先頭 + (Row * y + x)

配列Aの添字として書くと、`A[Row*y + x]`

この、`Row`がつまりは、idAなのである。(なお、今回の式はRow-majorであることを前提としている)

配列Aのメモリ配置と行列の配置の対応を表すと以下のようになる。

	A =
	  A[0]  A[1]  A[2]
	  A[3]  A[4]  A[5]

(idA = 3)を用いて表すと、

	A =
	  A[idA\*0 + 0]  A[idA\*0 + 1]  A[idA\*0 + 2]
	  A[idA\*1 + 0]  A[idA\*1 + 1]  A[idA\*1 + 2]


#### ハマったエラー

行列Cを格納する配列のメモリが足りなくてエラーがでたことがあった。
行列A,Bは上のものを使って以下のコードを動かした。


	double C[4];
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			3, 2, 3,	// Aの行数、Bの列数、Aの列数(Bの行数)
			1.0,	// alpha
			&A[0], 3,	// A
			&B[0], 2,	// B
			0.0,	// beta
			&C[0], 4	// C
		);

このプログラムは存在しないメモリにアクセスしようとして、エラーを起こす。
注目して欲しいのは、`idC`の値である。

これは、以下のような意味である。
行列Cのためのメモリは以下のように確保されている。

	行列C =
	  C[0]  C[1]  C[2]  C[3]


このとき**AB**の結果は、2x2の行列になるので、

	  C[idC\*0+0]  C[idC\*0+1]
	  C[idC\*1+0]  C[idC\*1+1]

へ結果を代入しようとする。このとき、`idC\*1 = 4`なので、2行目でアクセスエラーが起こってしまったのである。

正しくは、`idC=2`とするべきであった。

この結果は、`Row-major`かつ、オペレータで`NoTrans`を選んだ場合のことである。


## 参考

https://software.intel.com/en-us/intel-mkl/documentation

http://d.hatena.ne.jp/spadeAJ/20101116/1289908636

http://www.intel.co.jp/content/dam/www/public/ijkk/jp/ja/documents/developer/mklman52\_j.pdf

