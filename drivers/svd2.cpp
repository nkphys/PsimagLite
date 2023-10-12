#include "Svd.h"
#include "Matrix.h"
#include "Vector.h"

#include <cstdlib> 
#include <iostream> 
#include <time.h> 
using namespace std;

int main()
{
	//typedef PsimagLite::Matrix<std::complex<double>> MatrixType;
	typedef PsimagLite::Matrix<double> MatrixType;

	srand(43); 
	

	int size_row=4000;
	int size_col=size_row;
	MatrixType a(size_row, size_col);
	
	for(int i=0;i<size_row;i++){
	for(int j=0;j<size_col;j++){
	//a(i, j)=std::complex<double>(  (rand()/(1.0*RAND_MAX))  , (rand()/(1.0*RAND_MAX)) );
	a(i,j)=rand()/(1.0*RAND_MAX);
	}}

	/*	
	std::cout << "A\n";
	std::cout << a;
	*/

	cout<<"-------- matrix a ----------"<<endl;
	for(int i=0;i<4;i++){
	for(int j=0;j<4;j++){
	cout<<a(i,j)<<" ";
	}
	cout<<endl;
	}

	MatrixType m(a);

	PsimagLite::Vector<double>::Type s;
	MatrixType vt;

	//PsimagLite::Svd<std::complex<double>> svd;
	PsimagLite::Svd<double> svd;
	svd('A', a, s, vt);


	/*
	std::cout << "U\n";
	std::cout << a;
	std::cout << "S\n";
	std::cout << s;
	*/

	cout<<"-------- matrix u ----------"<<endl;
        for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
        cout<<a(i,j)<<" ";
        }
	cout<<endl;
        }

	cout<<"-------- array s ----------"<<endl;
        for(int i=0;i<100;i++){
        cout<<s[i]<<"  ";
        }
	cout<<endl;




}
