/*
Copyright (c) 2009-2012-2018, UT-Battelle, LLC
All rights reserved

[PsimagLite, Version 2.]
[by G.A., Oak Ridge National Laboratory]

UT Battelle Open Source Software License 11242008

OPEN SOURCE LICENSE

Subject to the conditions of this License, each
contributor to this software hereby grants, free of
charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), a
perpetual, worldwide, non-exclusive, no-charge,
royalty-free, irrevocable copyright license to use, copy,
modify, merge, publish, distribute, and/or sublicense
copies of the Software.

1. Redistributions of Software must retain the above
copyright and license notices, this list of conditions,
and the following disclaimer.  Changes or modifications
to, or derivative works of, the Software should be noted
with comments and the contributor and organization's
name.

2. Neither the names of UT-Battelle, LLC or the
Department of Energy nor the names of the Software
contributors may be used to endorse or promote products
derived from this software without specific prior written
permission of UT-Battelle.

3. The software and the end-user documentation included
with the redistribution, with or without modification,
must include the following acknowledgment:

"This product includes software produced by UT-Battelle,
LLC under Contract No. DE-AC05-00OR22725  with the
Department of Energy."

*********************************************************
DISCLAIMER

THE SOFTWARE IS SUPPLIED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER, CONTRIBUTORS, UNITED STATES GOVERNMENT,
OR THE UNITED STATES DEPARTMENT OF ENERGY BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.

NEITHER THE UNITED STATES GOVERNMENT, NOR THE UNITED
STATES DEPARTMENT OF ENERGY, NOR THE COPYRIGHT OWNER, NOR
ANY OF THEIR EMPLOYEES, REPRESENTS THAT THE USE OF ANY
INFORMATION, DATA, APPARATUS, PRODUCT, OR PROCESS
DISCLOSED WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.

*********************************************************

*/
/** \ingroup DMRG */
/*@{*/

/*! \file CrsMatrix.h
 *
 *  A class to represent a sparse matrix in Compressed Row Storage
 *
 */

#ifndef CRSMATRIX_HEADER_H
#define CRSMATRIX_HEADER_H
#include "BLAS.h"
#include "Complex.h"
#include "Io/IoSerializerStub.h"
#include "Matrix.h"
#include "Mpi.h"
#include "Sort.h"
#include "loki/TypeTraits.h"
#include <algorithm>
#include <cassert>
#include <fstream>

namespace PsimagLite
{

//! A Sparse Matrix in Compressed Row Storage (CRS) format.
/**
	The CRS format puts the subsequent nonzero elements of the matrix rows
	in contiguous memory locations. We create 3 vectors: one for complex
   numbers containing the values of the matrix entries and the other two for
   integers ($colind$ and $rowptr$). The vector $values$ stores the values of
   the non-zero elements of the matrix, as they are traversed in a row-wise
   fashion. The $colind$ vector stores the column indices of the elements of the
   $values$ vector. That is, if $values[k] = a[i][j]$ then $colind[k] = j$. The
   $rowptr$ vector stores the locations in the $values$ vector that start a row,
   that is $values[k] = a[i][j]$ if $rowptr[i] \le i < rowptr[i + 1]$. By
   convention, we define $rowptr[N_{dim}]$ to be equal to the number of non-zero
   elements, $n_z$, in the matrix. The storage savings of this approach are
   significant because instead of
	storing $N_{dim}^2$ elements, we need only $2n_z + N_{dim} + 1$ storage
   locations.\\ To illustrate how the CRS format works, consider the
   non-symmetric matrix defined by \begin{equation}
		A=\left[\begin{tabular}{llllll}

		10 &  0 & 0 & 0  & -2 & 0 \\
		3 &  9 &  0 &  0 &  0 &  3 \\
		0 &  7 &  8 &  7 &  0 &  0 \\
		3 &  0 &  8 &  7  & 5 &  0 \\
		0 &   8 &  0 &  9 &  9 & 13 \\
		0 &  4 &  0 &  0 &  2&  -1 \\
	\end{tabular}\right]\end{equation}
	The CRS format for this matrix is then specified by the arrays:\\
	\begin{tt}
		values = [10 -2  3  9  3  7  8  7  3 ... 9 13  4  2 -1 ]\\
		colind = [ 0  4  0  1  5  1  2  3  0 ... 4  5  1  4  5 ]\\
		rowptr = [ 0  2  5  8 12 16 19 ]\\
	\end{tt}
	*/
template <class T>
class CrsMatrix
{

public:

	typedef T MatrixElementType;
	typedef T value_type;

	CrsMatrix()
	    : nrow_(0)
	    , ncol_(0)
	{
	}

	~CrsMatrix() { }

	CrsMatrix(SizeType nrow, SizeType ncol)
	    : nrow_(nrow)
	    , ncol_(ncol)
	{
		resize(nrow, ncol);
	}

	CrsMatrix(SizeType nrow, SizeType ncol, SizeType nonzero)
	    : nrow_(nrow)
	    , ncol_(ncol)
	{
		resize(nrow, ncol, nonzero);
	}

	template <typename S>
	CrsMatrix(const CrsMatrix<S>& a)
	{
		colind_ = a.colind_;
		rowptr_ = a.rowptr_;
		values_ = a.values_;
		nrow_ = a.nrow_;
		ncol_ = a.ncol_;
	}

	template <typename S>
	CrsMatrix(const CrsMatrix<std::complex<S>>& a)
	{
		colind_ = a.colind_;
		rowptr_ = a.rowptr_;
		values_ = a.values_;
		nrow_ = a.nrow_;
		ncol_ = a.ncol_;
	}

	explicit CrsMatrix(const Matrix<T>& a)
	{
		int counter = 0;
		double eps = 0;

		resize(a.rows(), a.cols());

		for (SizeType i = 0; i < a.rows(); i++) {
			setRow(i, counter);
			for (SizeType j = 0; j < a.cols(); j++) {
				if (PsimagLite::norm(a(i, j)) <= eps)
					continue;
				pushValue(a(i, j));
				pushCol(j);
				++counter;
			}
		}

		setRow(a.rows(), counter);
	}

	CrsMatrix(SizeType rank, // square matrix ONLY for now
	    const Vector<SizeType>::Type& rows2,
	    const Vector<SizeType>::Type& cols,
	    const typename Vector<T>::Type& vals)
	    : rowptr_(rank + 1)
	    , nrow_(rank)
	    , ncol_(rank)
	{
		Sort<Vector<SizeType>::Type> s;
		Vector<SizeType>::Type iperm(rows2.size());
		Vector<SizeType>::Type rows = rows2;
		s.sort(rows, iperm);
		SizeType counter = 0;
		SizeType prevRow = rows[0] + 1;
		for (SizeType i = 0; i < rows.size(); i++) {
			SizeType row = rows[i];
			if (prevRow != row) {
				// add new row
				rowptr_[row] = counter++;
				prevRow = row;
			}

			colind_.push_back(cols[iperm[i]]);
			values_.push_back(vals[iperm[i]]);
		}

		SizeType lastNonZeroRow = rows[rows.size() - 1];
		for (SizeType i = lastNonZeroRow + 1; i <= rank; ++i)
			rowptr_[i] = counter;
	}

	// start closure ctors

	CrsMatrix(const std::ClosureOperator<
	    CrsMatrix,
	    CrsMatrix,
	    std::ClosureOperations::OP_MULT>& c)
	{
		CrsMatrix& x = *this;
		const CrsMatrix& y = c.r1;
		const CrsMatrix& z = c.r2;
		multiply(x, y, z);
	}

	CrsMatrix(const std::ClosureOperator<
	    T,
	    CrsMatrix,
	    std::ClosureOperations::OP_MULT>& c)
	{
		*this = c.r2;
		this->values_ *= c.r1;
	}

	// end all ctors

	template <typename SomeMemResolvType>
	SizeType memResolv(SomeMemResolvType& mres, SizeType, String msg = "") const
	{
		String str = msg;
		str += "CrsMatrix";

		const char* start = reinterpret_cast<const char*>(this);
		const char* end = reinterpret_cast<const char*>(&colind_);
		SizeType total = mres.memResolv(&rowptr_, end - start, str + " rowptr");

		start = end;
		end = reinterpret_cast<const char*>(&values_);
		total += mres.memResolv(&colind_, end - start, str + " colind");

		start = end;
		end = reinterpret_cast<const char*>(&nrow_);
		total += mres.memResolv(&values_, end - start, str + " values");

		start = end;
		end = reinterpret_cast<const char*>(&ncol_);
		total += mres.memResolv(&nrow_, end - start, str + " nrow");

		total += mres.memResolv(&ncol_, sizeof(*this) - total, str + " ncol");

		return total;
	}

	void resize(SizeType nrow, SizeType ncol)
	{
		colind_.clear();
		values_.clear();
		rowptr_.clear();
		rowptr_.resize(nrow + 1);
		nrow_ = nrow;
		ncol_ = ncol;
	}

	void clear()
	{
		colind_.clear();
		values_.clear();
		rowptr_.clear();
		nrow_ = ncol_ = 0;
	}

	void resize(SizeType nrow, SizeType ncol, SizeType nonzero)
	{
		nrow_ = nrow;
		ncol_ = ncol;

		// ------------------------------------
		// Note arrays are not cleared out
		// arrays should retain original values
		// ------------------------------------
		rowptr_.resize(nrow_ + 1);
		colind_.resize(nonzero);
		values_.resize(nonzero);
	}

	void reserve(SizeType nonzero)
	{
		// -------------------------------------------------
		// increase internal capacity
		// to avoid repeated allocation expansion and copies
		// -------------------------------------------------
		colind_.reserve(nonzero);
		values_.reserve(nonzero);
	}

	void setRow(SizeType n, SizeType v)
	{
		assert(n < rowptr_.size());
		rowptr_[n] = v;
	}

	void setCol(int n, int v) { colind_[n] = v; }

	void setCol_check(int n, int v)
	{
		if (((size_t)n) == (colind_.size() + 1)) {
			colind_.push_back(v);
		} else {
			colind_[n] = v;
		};
	}

	void setValues(int n, const T& v) { values_[n] = v; }

	void setValues_check(int n, const T& v)
	{
		if (((size_t)n) == (values_.size() + 1)) {
			values_.push_back(v);
		} else {
			values_[n] = v;
		};
	}

	void operator*=(T x) { values_ *= x; }

	bool operator==(const CrsMatrix<T>& op) const
	{
		return (nrow_ == op.nrow_ && ncol_ == op.ncol_ && rowptr_ == op.rowptr_ && colind_ == op.colind_ && values_ == op.values_);
	}

	template <typename VerySparseMatrixType>
	typename EnableIf<!std::IsClosureLike<VerySparseMatrixType>::True,
	    void>::Type
	operator=(const VerySparseMatrixType& m)
	{
		if (!m.sorted())
			throw RuntimeError(
			    "CrsMatrix: VerySparseMatrix must be sorted\n");

		clear();
		SizeType nonZeros = m.nonZeros();
		resize(m.rows(), m.cols(), nonZeros);

		SizeType counter = 0;
		for (SizeType i = 0; i < m.rows(); ++i) {
			setRow(i, counter);

			while (counter < nonZeros && m.getRow(counter) == i) {
				colind_[counter] = m.getColumn(counter);
				values_[counter] = m.getValue(counter);
				counter++;
			}
		}

		setRow(m.rows(), counter);
		checkValidity();
	}

	void operator+=(const CrsMatrix& m)
	{
		CrsMatrix c;
		static const typename Real<T>::Type f1 = 1.0;
		add(c, m, f1);
		*this = c;
	}

	SizeType nonZeros() const
	{
		if (nrow_ >= 1) {
			assert(rowptr_.size() == 1 + nrow_);

			assert(static_cast<SizeType>(rowptr_[nrow_]) == colind_.size());
			assert(static_cast<SizeType>(rowptr_[nrow_]) == values_.size());

			return colind_.size();
		} else {
			return 0;
		};
	}

	/** performs x = x + A * y
	 ** where x and y are vectors and A is a sparse matrix in
	 ** row-compressed format */
	template <typename VectorLikeType>
	void matrixVectorProduct(VectorLikeType& x,
	    const VectorLikeType& y) const
	{
		assert(x.size() == y.size());
		for (SizeType i = 0; i < y.size(); i++) {
			assert(i + 1 < rowptr_.size());
			for (int j = rowptr_[i]; j < rowptr_[i + 1]; j++) {
				assert(SizeType(j) < values_.size());
				assert(SizeType(j) < colind_.size());
				assert(SizeType(colind_[j]) < y.size());
				x[i] += values_[j] * y[colind_[j]];
			}
		}
	}

#ifndef NO_DEPRECATED_ALLOWED
	int nonZero() const
	{
		return colind_.size();
	} // DEPRECATED, use nonZeros()
#endif

	SizeType rows() const
	{
		return nrow_;
	}

	SizeType cols() const { return ncol_; }

	void swap(CrsMatrix& other)
	{
		this->rowptr_.swap(other.rowptr_);
		this->colind_.swap(other.colind_);
		this->values_.swap(other.values_);
		SizeType nrow = this->nrow_;
		this->nrow_ = other.nrow_;
		other.nrow_ = nrow;
		SizeType ncol = this->ncol_;
		this->ncol_ = other.ncol_;
		other.ncol_ = ncol;
	}

	void pushCol(SizeType i) { colind_.push_back(i); }

	void pushValue(T const& value) { values_.push_back(value); }

	//! Make a diagonal CRS matrix with value "value"
	void makeDiagonal(SizeType row, T const& value = 0)
	{
		nrow_ = row;
		ncol_ = row;
		rowptr_.resize(row + 1);
		values_.resize(row);
		colind_.resize(row);

		for (SizeType i = 0; i < row; i++) {
			values_[i] = value;
			colind_[i] = i;
			rowptr_[i] = i;
		}

		rowptr_[row] = row;
	}

	const int& getRowPtr(SizeType i) const
	{
		assert(i < rowptr_.size());
		return rowptr_[i];
	}

	const int& getCol(SizeType i) const
	{
		assert(i < colind_.size());
		return colind_[i];
	}

	const T& getValue(SizeType i) const
	{
		assert(i < values_.size());
		return values_[i];
	}

	void conjugate()
	{
		SizeType n = values_.size();
		for (SizeType i = 0; i < n; ++i)
			values_[i] = PsimagLite::conj(values_[i]);
	}

	Matrix<T> toDense() const
	{
		Matrix<T> m;
		crsMatrixToFullMatrix(m, *this);
		return m;
	}

	void checkValidity() const
	{
#ifndef NDEBUG
		SizeType n = nrow_;
		assert(n + 1 == rowptr_.size());
		assert(static_cast<SizeType>(rowptr_[nrow_]) == colind_.size());
		assert(values_.size() == colind_.size());
		assert(nrow_ > 0 && ncol_ > 0);
		typename Vector<SizeType>::Type p(ncol_, 0);
		for (SizeType i = 0; i < n; i++) {
			assert(rowptr_[i] <= rowptr_[i + 1]);
			for (int k = rowptr_[i]; k < rowptr_[i + 1]; k++) {
				SizeType col = colind_[k];
				assert(col < p.size());
				assert(p[col] == 0);
				p[col] = 1;
			}

			for (int k = rowptr_[i]; k < rowptr_[i + 1]; k++)
				p[colind_[k]] = 0;
		}
#endif
	}

	// closures operators start

	template <typename T1>
	typename EnableIf<Loki::TypeTraits<T1>::isArith || IsComplexNumber<T1>::True,
	    CrsMatrix>::Type
	operator=(const std::ClosureOperator<
	    T1,
	    CrsMatrix,
	    std::ClosureOperations::OP_MULT>& c)
	{
		*this = c.r2;
		this->values_ *= c.r1;
		return *this;
	}

	template <typename T1>
	typename EnableIf<Loki::TypeTraits<T1>::isArith || IsComplexNumber<T1>::True,
	    CrsMatrix>::Type
	operator+=(const std::ClosureOperator<
	    T1,
	    CrsMatrix,
	    std::ClosureOperations::OP_MULT>& c)
	{
		CrsMatrix s;
		add(s, c.r2, c.r1);
		this->swap(s);
		return *this;
	}

	template <typename T1>
	typename EnableIf<Loki::TypeTraits<T1>::isArith || IsComplexNumber<T1>::True,
	    CrsMatrix>::Type
	operator+=(const std::ClosureOperator<
	    std::ClosureOperator<T1, CrsMatrix, std::ClosureOperations::OP_MULT>,
	    CrsMatrix,
	    std::ClosureOperations::OP_MULT>& c)
	{
		CrsMatrix s;
		multiply(s, c.r1.r2, c.r2);
		CrsMatrix s2;
		add(s2, s, c.r1.r1);
		this->swap(s2);
		return *this;
	}

	// closures operators end

	void send(int root, int tag, MPI::CommType mpiComm)
	{
		MPI::send(nrow_, root, tag, mpiComm);
		MPI::send(ncol_, root, tag + 1, mpiComm);
		MPI::send(rowptr_, root, tag + 2, mpiComm);
		MPI::send(colind_, root, tag + 3, mpiComm);
		MPI::send(values_, root, tag + 4, mpiComm);
	}

	void recv(int root, int tag, MPI::CommType mpiComm)
	{
		MPI::recv(nrow_, root, tag, mpiComm);
		MPI::recv(ncol_, root, tag + 1, mpiComm);
		MPI::recv(rowptr_, root, tag + 2, mpiComm);
		MPI::recv(colind_, root, tag + 3, mpiComm);
		MPI::recv(values_, root, tag + 4, mpiComm);
	}

	void
	write(String label, IoSerializer& ioSerializer, IoSerializer::WriteMode mode = IoSerializer::NO_OVERWRITE) const
	{
		if (nrow_ > 0)
			checkValidity();
		if (mode != IoSerializer::ALLOW_OVERWRITE)
			ioSerializer.createGroup(label);
		ioSerializer.write(label + "/nrow_", nrow_, mode);
		ioSerializer.write(label + "/ncol_", ncol_, mode);
		if (nrow_ == 0 || ncol_ == 0)
			return;
		ioSerializer.write(label + "/rowptr_", rowptr_, mode);
		assert(rowptr_.size() == nrow_ + 1);
		if (rowptr_[nrow_] == 0)
			return;
		ioSerializer.write(label + "/colind_", colind_, mode);
		ioSerializer.write(label + "/values_", values_, mode);
	}

	void overwrite(String label, IoSerializer& ioSerializer) const
	{
		write(label, ioSerializer, IoSerializer::ALLOW_OVERWRITE);
	}
	void read(String label, IoSerializer& ioSerializer)
	{
		ioSerializer.read(nrow_, label + "/nrow_");
		ioSerializer.read(ncol_, label + "/ncol_");
		if (nrow_ == 0 || ncol_ == 0)
			return;
		ioSerializer.read(rowptr_, label + "/rowptr_");
		assert(rowptr_.size() == nrow_ + 1);
		if (rowptr_[nrow_] == 0)
			return;
		ioSerializer.read(colind_, label + "/colind_");
		ioSerializer.read(values_, label + "/values_");
		checkValidity();
	}

	template <typename S>
	friend bool isZero(const CrsMatrix<S>&, double);

	template <typename S>
	friend typename Real<S>::Type norm2(const CrsMatrix<S>& m);

	template <typename S>
	friend std::ostream& operator<<(std::ostream& os,
	    const CrsMatrix<S>& m);

	template <class S>
	friend void difference(const CrsMatrix<S>& A, const CrsMatrix<S>& B);

	template <typename S>
	friend void MpiBroadcast(CrsMatrix<S>* v, int rank);

	template <typename S>
	friend void MpiSend(CrsMatrix<S>* v, int iproc, int i);

	template <typename S>
	friend void MpiRecv(CrsMatrix<S>* v, int iproc, int i);

	template <typename CrsMatrixType>
	friend std::istream& operator>>(std::istream& is,
	    CrsMatrix<CrsMatrixType>& m);

	template <typename S>
	friend void bcast(CrsMatrix<S>& m);

private:

	template <typename T1>
	typename std::enable_if<
	    std::is_same<T1, typename Real<T>::Type>::value || std::is_same<T1, T>::value,
	    void>::type
	add(CrsMatrix<T>& c, const CrsMatrix<T>& m, const T1& t1) const
	{
		assert(m.rows() == m.cols());
		const T1 one = 1.0;
		if (nrow_ >= m.rows())
			operatorPlus(c, *this, one, m, t1);
		else
			operatorPlus(c, m, t1, *this, one);
	}

	typename Vector<int>::Type rowptr_;
	typename Vector<int>::Type colind_;
	typename Vector<T>::Type values_;
	SizeType nrow_;
	SizeType ncol_;
}; // class CrsMatrix

// Companion functions below:

template <typename T>
std::ostream& operator<<(std::ostream& os, const CrsMatrix<T>& m)
{
	SizeType n = m.rows();
	if (n == 0) {
		os << "0 0\n";
		return os;
	}

	os << n << " " << m.cols() << "\n";
	for (SizeType i = 0; i < n + 1; i++)
		os << m.rowptr_[i] << " ";
	os << "\n";

	SizeType nonzero = m.nonZeros();
	os << nonzero << "\n";
	for (SizeType i = 0; i < nonzero; i++)
		os << m.colind_[i] << " ";
	os << "\n";

	os << nonzero << "\n";
	for (SizeType i = 0; i < nonzero; i++)
		os << m.values_[i] << " ";
	os << "\n";

	return os;
}

template <typename T>
std::istream& operator>>(std::istream& is, CrsMatrix<T>& m)
{
	int n;
	is >> n;
	if (n < 0)
		throw RuntimeError(
		    "is>>CrsMatrix(...): Rows must be positive\n");

	int ncol = 0;
	if (ncol < 0)
		throw RuntimeError(
		    "is>>CrsMatrix(...): Cols must be positive\n");
	is >> ncol;

	if (n == 0 || ncol == 0)
		return is;

	m.resize(n, ncol);
	for (SizeType i = 0; i < m.rowptr_.size(); i++)
		is >> m.rowptr_[i];

	SizeType nonzero;
	is >> nonzero;
	m.colind_.resize(nonzero);
	for (SizeType i = 0; i < m.colind_.size(); i++)
		is >> m.colind_[i];

	is >> nonzero;
	m.values_.resize(nonzero);
	for (SizeType i = 0; i < m.values_.size(); i++)
		is >> m.values_[i];

	return is;
}

template <typename T>
class IsMatrixLike<CrsMatrix<T>>
{
public:

	enum { True = true };
};

template <typename S>
void bcast(CrsMatrix<S>& m)
{
	MPI::bcast(m.rowptr_);
	MPI::bcast(m.colind_);
	MPI::bcast(m.values_);
	MPI::bcast(m.nrow_);
	MPI::bcast(m.ncol_);
}

//! Transforms a Compressed-Row-Storage (CRS) into a full Matrix (Fast version)
template <typename T>
void crsMatrixToFullMatrix(Matrix<T>& m, const CrsMatrix<T>& crsMatrix)
{
	m.resize(crsMatrix.rows(), crsMatrix.cols(), 0);
	for (SizeType i = 0; i < crsMatrix.rows(); i++) {
		//  for (SizeType k=0;k<crsMatrix.cols();k++) m(i,k)=0;
		for (int k = crsMatrix.getRowPtr(i);
		     k < crsMatrix.getRowPtr(i + 1);
		     k++)
			m(i, crsMatrix.getCol(k)) = crsMatrix.getValue(k);
	}
}

//! Transforms a full matrix into a Compressed-Row-Storage (CRS) Matrix
// Use the constructor if possible
template <typename T>
void fullMatrixToCrsMatrix(CrsMatrix<T>& crsMatrix, const Matrix<T>& a)
{
	const T zval = 0.0;
	SizeType rows = a.rows();
	SizeType cols = a.cols();
	SizeType nonZeros = rows * cols;

	const bool use_push = true;

	if (use_push) {
		// ------------------------------
		// avoid filling array with zeros
		// ------------------------------
		crsMatrix.resize(rows, cols);
		crsMatrix.reserve(nonZeros);
	} else {
		crsMatrix.resize(rows, cols, nonZeros);
	};

	SizeType counter = 0;
	for (SizeType i = 0; i < rows; ++i) {
		crsMatrix.setRow(i, counter);
		for (SizeType j = 0; j < cols; ++j) {
			const T& val = a(i, j);
			if (val == zval)
				continue;

			if (use_push) {
				crsMatrix.pushValue(val);
				crsMatrix.pushCol(j);
			} else {
				crsMatrix.setValues(counter, val);
				crsMatrix.setCol(counter, j);
			};
			++counter;
		}
	}

	crsMatrix.setRow(rows, counter);
	crsMatrix.checkValidity();
}

/** If order==false then
		creates B such that
   B_{i1+j1*nout,i2+j2*nout)=A(j1,j2)\delta_{i1,i2} if order==true then creates
   B such that B_{i1+j1*na,i2+j2*na)=A(i1,i2)\delta_{j1,j2} where na=rank(A)
	  */

template <typename T, typename VectorLikeType>
typename EnableIf<
    IsVectorLike<VectorLikeType>::True && Loki::TypeTraits<typename VectorLikeType::value_type>::isFloat,
    void>::Type
externalProduct(CrsMatrix<T>& B, const CrsMatrix<T>& A, SizeType nout, const VectorLikeType& signs, bool order, const PsimagLite::Vector<SizeType>::Type& permutationFull)
{
	if (A.rows() > 0)
		A.checkValidity();
	// -------------------------------------
	//  B = kron(eye, A)   if (is_A_fastest)
	//  B = kron(A, eye)   otherwise
	// -------------------------------------
	SizeType nrow_A = A.rows();
	SizeType ncol_A = A.cols();
	SizeType n = nout;
	SizeType nrow_eye = n;
	SizeType ncol_eye = n;
	SizeType nnz_A = A.nonZeros();

	SizeType nrow_B = n * nrow_A;
	SizeType ncol_B = n * ncol_A;
	SizeType nnz_B = n * nnz_A;

	B.resize(nrow_B, ncol_B, nnz_B);

	bool is_A_fastest = order;

	if (nrow_A != ncol_A)
		throw RuntimeError(
		    "externalProduct: matrices must be square\n");

	// -----------------------
	// setup row pointers in B
	// Note: if (is_A_fastest)  then
	//          B( [ia,ie], [ja,je] ) = A(ia,ja) * eye(ie,je)
	//       else
	//          B( [ie,ia], [je,ja] ) = A(ia,ja) * eye(ie,je)
	//       endif
	//
	//  where [ia,ie] = ia + ie * nrow_A,   [ja,je] = ja + je * ncol_A
	//        [ie,ia] = ie + ia * nrow_eye, [je,ja] = je + ja * ncol_eye
	// -----------------------

	// -------------------------------------------------
	// calculate the number of nonzeros in each row of B
	// -------------------------------------------------
	std::vector<int> nnz_B_row(nrow_B);

	assert(nrow_A * nrow_eye <= permutationFull.size());

	for (SizeType ia = 0; ia < nrow_A; ia++) {

		SizeType nnz_row = A.getRowPtr(ia + 1) - A.getRowPtr(ia);

		for (SizeType ie = 0; ie < nrow_eye; ie++) {
			SizeType ib = (is_A_fastest)
			    ? permutationFull[ia + ie * nrow_A]
			    : permutationFull[ie + ia * nrow_eye];
			nnz_B_row[ib] = nnz_row;
		};
	};

	// -------------------------------
	// setup row pointers in matrix B
	// -------------------------------
	std::vector<SizeType> B_rowptr(nrow_B);

	SizeType ip = 0;
	for (SizeType ib = 0; ib < nrow_B; ib++) {

		B_rowptr[ib] = ip;
		B.setRow(ib, ip);

		ip += nnz_B_row[ib];
	};
	assert(ip == nnz_B);
	B.setRow(nrow_B, nnz_B);

	// ---------------------------
	// copy entries into matrix B
	// ---------------------------

	// ----------------------------------------------
	// single pass over non-zero entries of matrix A
	// ----------------------------------------------
	for (SizeType ia = 0; ia < nrow_A; ia++) {
		for (int k = A.getRowPtr(ia); k < A.getRowPtr(ia + 1); k++) {

			// --------------------
			// entry aij = A(ia,ja)
			// --------------------
			SizeType ja = A.getCol(k);
			T aij = A.getValue(k);

			for (SizeType ie = 0; ie < nrow_eye; ie++) {
				SizeType je = ie;

				SizeType ib = (is_A_fastest)
				    ? ia + ie * nrow_A
				    : ie + ia * nrow_eye;

				SizeType jb = (is_A_fastest)
				    ? ja + je * ncol_A
				    : je + ja * ncol_eye;

				// --------------------
				// entry bij = B(ib,jb)
				// --------------------
				int alpha = ie;
				T bij = (is_A_fastest) ? aij : aij * signs[alpha];

				SizeType ip = B_rowptr[permutationFull[ib]];

				assert(jb < permutationFull.size());
				B.setCol(ip, permutationFull[jb]);
				B.setValues(ip, bij);

				++B_rowptr[permutationFull[ib]];
			};
		};
	};

	if (nrow_B != 0)
		B.checkValidity();
}

//-------

/** If order==false then
		creates C such that C_{i1+j1*nout,i2+j2*nout)=A(j1,j2)B_{i1,i2}
		if order==true then
		creates C such that C_{i1+j1*na,i2+j2*na)=A(i1,i2)B_{j1,j2}
		where na=rank(A) and nout = rank(B)
	  */

template <typename T, typename VectorLikeType>
typename EnableIf<
    IsVectorLike<VectorLikeType>::True && Loki::TypeTraits<typename VectorLikeType::value_type>::isFloat,
    void>::Type
externalProduct(CrsMatrix<T>& C, const CrsMatrix<T>& A, const CrsMatrix<T>& B, const VectorLikeType& signs, bool order, const PsimagLite::Vector<SizeType>::Type& permutationFull)
{
	const SizeType nfull = permutationFull.size();

	Vector<SizeType>::Type perm(nfull);
	for (SizeType i = 0; i < nfull; ++i)
		perm[permutationFull[i]] = i;

	const SizeType nout = B.rows();
	const SizeType na = A.rows();
	const SizeType noutOrNa = (!order) ? nout : na;
	const CrsMatrix<T>& AorB = (!order) ? A : B;
	const CrsMatrix<T>& BorA = (!order) ? B : A;
	assert(A.rows() == A.cols());
	assert(B.rows() == B.cols());
	assert(nout * na == nfull);
	assert(signs.size() == noutOrNa);
	C.resize(nfull, nfull);
	SizeType counter = 0;
	for (SizeType i = 0; i < nfull; ++i) {
		C.setRow(i, counter);
		const SizeType ind = perm[i];
		div_t q = div(ind, noutOrNa);
		for (int k1 = BorA.getRowPtr(q.rem);
		     k1 < BorA.getRowPtr(q.rem + 1);
		     ++k1) {
			const SizeType col1 = BorA.getCol(k1);
			for (int k2 = AorB.getRowPtr(q.quot);
			     k2 < AorB.getRowPtr(q.quot + 1);
			     ++k2) {
				const SizeType col2 = AorB.getCol(k2);
				SizeType j = permutationFull[col1 + col2 * noutOrNa];
				C.pushCol(j);
				C.pushValue(BorA.getValue(k1) * AorB.getValue(k2) * signs[q.rem]);
				++counter;
			}
		}
	}

	C.setRow(nfull, counter);
	C.checkValidity();
}

template <typename T>
void printFullMatrix(const CrsMatrix<T>& s, const String& name, SizeType how = 0, double eps = 1e-20)
{
	Matrix<T> fullm(s.rows(), s.cols());
	crsMatrixToFullMatrix(fullm, s);
	std::cout << "--------->   " << name;
	std::cout << " rank=" << s.rows() << "x" << s.cols()
		  << " <----------\n";
	try {
		if (how == 1)
			mathematicaPrint(std::cout, fullm);
		if (how == 2)
			symbolicPrint(std::cout, fullm);
	} catch (std::exception& e) {
	}

	if (how == 0)
		fullm.print(std::cout, eps);
}

//! C = A*B,  all matrices are CRS matrices
//! idea is from http://web.maths.unsw.edu.au/~farid/Papers/Hons/node23.html
template <typename S, typename S3, typename S2>
void multiply(CrsMatrix<S>& C, CrsMatrix<S3> const& A, CrsMatrix<S2> const& B)
{
	int j, s, mlast, itemp, jbk;
	SizeType n = A.rows();
	typename Vector<int>::Type ptr(B.cols(), -1), index(B.cols(), 0);
	typename Vector<S>::Type temp(B.cols(), 0);
	S tmp;

	C.resize(n, B.cols());

	// mlast pointer to the last place we updated in the C vector
	mlast = 0;
	// for (SizeType l=0;l<n;l++) ptr[l] = -1;
	// over the rows of A
	for (SizeType i = 0; i < n; i++) {
		C.setRow(i, mlast);
		// start calculations for row
		itemp = 0;
		for (j = A.getRowPtr(i); j < A.getRowPtr(i + 1); j++) {
			SizeType istart = B.getRowPtr(A.getCol(j));
			SizeType iend = B.getRowPtr(A.getCol(j) + 1);
			for (SizeType k = istart; k < iend; k++) {
				jbk = B.getCol(k);
				tmp = A.getValue(j) * B.getValue(k);
				if (ptr[jbk] < 0) {
					ptr[jbk] = itemp;
					temp[ptr[jbk]] = tmp;
					index[ptr[jbk]] = jbk;
					itemp++;
				} else {
					temp[ptr[jbk]] += tmp;
				}
			}
		}
		// before you leave this row update array c , jc
		for (s = 0; s < itemp; s++) {
			C.pushValue(temp[s]);
			C.pushCol(index[s]);
			ptr[index[s]] = -1;
		}
		mlast += itemp;
	}
	C.setRow(n, mlast);
	C.checkValidity();
}

// vector2 = sparseMatrix * vector1
template <class S>
void multiply(typename Vector<S>::Type& v2, const CrsMatrix<S>& m, const typename Vector<S>::Type& v1)
{
	SizeType n = m.rows();
	v2.resize(n);
	for (SizeType i = 0; i < n; i++) {
		v2[i] = 0;
		for (int j = m.getRowPtr(i); j < m.getRowPtr(i + 1); j++) {
			v2[i] += m.getValue(j) * v1[m.getCol(j)];
		}
	}
}

//! Sets B=transpose(conjugate(A))
template <typename S, typename S2>
void transposeConjugate(CrsMatrix<S>& B, const CrsMatrix<S2>& A)
{
	SizeType nrowA = A.rows();
	SizeType ncolA = A.cols();
	SizeType nrowB = ncolA;
	SizeType ncolB = nrowA;

	SizeType nnz_A = A.nonZeros();
	SizeType nnz_B = nnz_A;

	B.resize(nrowB, ncolB, nnz_B);

	std::vector<SizeType> nnz_count(ncolA, 0);

	// ----------------------------------------------------
	// 1st pass to count number of nonzeros per column in A
	// which is equivalent to number of nonzeros
	// per row in B = transpose(conjugate(A))
	// ----------------------------------------------------
	for (SizeType ia = 0; ia < nrowA; ia++) {
		for (int k = A.getRowPtr(ia); k < A.getRowPtr(ia + 1); k++) {
			SizeType ja = A.getCol(k);
			++nnz_count[ja];
		};
	};

	// -----------------------
	// setup row pointers in B
	// -----------------------
	SizeType ipos = 0;
	for (SizeType ib = 0; ib < nrowB; ib++) {
		B.setRow(ib, ipos);
		ipos += nnz_count[ib];
	};
	assert(ipos == nnz_B);
	B.setRow(nrowB, nnz_B);

	// -------------------------------
	// setup row pointers in B matrix
	// -------------------------------
	std::vector<SizeType> B_rowptr(nrowB);
	for (SizeType ib = 0; ib < nrowB; ib++) {
		B_rowptr[ib] = B.getRowPtr(ib);
	};

	// -------------------------------------------
	// 2nd pass over matrix A to assign values to B
	// -------------------------------------------
	for (SizeType ia = 0; ia < nrowA; ia++) {
		for (int k = A.getRowPtr(ia); k < A.getRowPtr(ia + 1); k++) {

			SizeType ja = A.getCol(k);
			S2 aij = A.getValue(k);

			// ---------------------------------
			// B(ib=ja,jb=ia) = conj( A(ia,ja) )
			// ---------------------------------
			SizeType ib = ja;
			SizeType jb = ia;

			SizeType ip = B_rowptr[ib];

			// B.colind_[ ip ] = jb;
			// B.values_[ ip ] = PsimagLite::conj( aij );

			B.setCol(ip, jb);
			B.setValues(ip, PsimagLite::conj(aij));

			++B_rowptr[ib];
		};
	};
}

//! Sets A=B*b1+C*c1, restriction: B.size has to be larger or equal than C.size
template <typename T, typename T1>
void operatorPlus(CrsMatrix<T>& A, const CrsMatrix<T>& B, T1& b1, const CrsMatrix<T>& C, T1& c1)
{
	const T zero = static_cast<T>(0.0);

	SizeType nrow_B = B.rows();
	SizeType ncol_B = B.cols();
	SizeType nrow_C = C.rows();
	SizeType ncol_C = C.cols();

	// ------------------------------
	// nrow_A = MAX( nrow_B, nrow_C )
	// ncol_A = MAX( ncol_B, ncol_C )
	// ------------------------------
	SizeType nrow_A = (nrow_B >= nrow_C) ? nrow_B : nrow_C;
	SizeType ncol_A = (ncol_B >= ncol_C) ? ncol_B : ncol_C;

	A.resize(nrow_A, ncol_A);

	// ------------------------------------------------------
	// TODO: using A.resize(nrow_A,ncol_A,nnz_A) may not work correctly
	// ------------------------------------------------------
	const bool set_nonzeros = true;
	if (set_nonzeros) {
		SizeType nnz_B = B.nonZeros();
		SizeType nnz_C = C.nonZeros();

		// -----------------------------------------------
		// worst case when no overlap in sparsity pattern
		// between matrix B and matrix C
		// -----------------------------------------------
		SizeType nnz_A = nnz_B + nnz_C;

		A.reserve(nnz_A);
	};

	// ------------------------------------------
	// temporary vectors to accelerate processing
	// ------------------------------------------
	std::vector<T> valueTmp(ncol_A, zero);
	std::vector<bool> is_examined_already(ncol_A, false);

	SizeType counter = 0;
	for (SizeType irow = 0; irow < nrow_A; irow++) {
		A.setRow(irow, counter);

		const bool is_valid_B_row = (irow < nrow_B);
		const bool is_valid_C_row = (irow < nrow_C);

		const int kstart_B = (is_valid_B_row) ? B.getRowPtr(irow) : 0;
		const int kend_B = (is_valid_B_row) ? B.getRowPtr(irow + 1) : 0;

		const int kstart_C = (is_valid_C_row) ? C.getRowPtr(irow) : 0;
		const int kend_C = (is_valid_C_row) ? C.getRowPtr(irow + 1) : 0;

		// --------------------------------
		// check whether there is work to do
		// --------------------------------
		const bool has_work = ((kend_B - kstart_B) + (kend_C - kstart_C) >= 1);
		if (!has_work)
			continue;

		// -------------------------------
		// add contributions from matrix B and matrix C
		// -------------------------------
		for (int k = kstart_B; k < kend_B; k++) {
			const T bij = B.getValue(k);
			const SizeType jcol = B.getCol(k);

			assert(jcol < ncol_A);

			valueTmp[jcol] += (bij * b1);
		};

		for (int k = kstart_C; k < kend_C; k++) {
			const T cij = C.getValue(k);
			const SizeType jcol = C.getCol(k);

			assert(jcol < ncol_A);

			valueTmp[jcol] += (cij * c1);
		};

		// --------------------
		// copy row to matrix A
		// --------------------

		for (int k = kstart_B; k < kend_B; k++) {
			const SizeType jcol = B.getCol(k);
			if (!is_examined_already[jcol]) {
				is_examined_already[jcol] = true;

				const T aij = valueTmp[jcol];
				const bool is_zero = (aij == zero);
				if (!is_zero) {
					A.pushCol(jcol);
					A.pushValue(aij);
					counter++;
				};
			};
		};

		for (int k = kstart_C; k < kend_C; k++) {
			const SizeType jcol = C.getCol(k);
			if (!is_examined_already[jcol]) {
				is_examined_already[jcol] = true;

				const T aij = valueTmp[jcol];
				const bool is_zero = (aij == zero);
				if (!is_zero) {
					A.pushCol(jcol);
					A.pushValue(aij);
					counter++;
				};
			};
		};

		// --------------------------------------------------
		// reset vectors valueTmp[] and is_examined_already[]
		// --------------------------------------------------

		for (int k = kstart_B; k < kend_B; k++) {
			const SizeType jcol = B.getCol(k);
			valueTmp[jcol] = zero;
			is_examined_already[jcol] = false;
		};

		for (int k = kstart_C; k < kend_C; k++) {
			const SizeType jcol = C.getCol(k);
			valueTmp[jcol] = zero;
			is_examined_already[jcol] = false;
		};

	}; // end for irow

	A.setRow(nrow_A, counter);

	// ----------------------------------------
	// set exact number of nonzeros in matrix A
	// ----------------------------------------
	SizeType nnz_A = counter;
	A.resize(nrow_A, ncol_A, nnz_A);

	A.checkValidity();
}

//! Sets A=B0*b0+B1*b1 + ...
template <typename T, typename T1>
void sum(CrsMatrix<T>& A, const std::vector<const CrsMatrix<T>*>& Bmats, const std::vector<T1>& bvec)
{
	SizeType Bmats_size = Bmats.size();

	// ------------------------------
	// nrow_A = MAX( nrow_B(:) )
	// ncol_A = MAX( ncol_B(:) )
	// ------------------------------
	SizeType nrow_A = 0;
	SizeType ncol_A = 0;
	SizeType nnz_sum = 0;
	SizeType nnz_max = 0;

	for (SizeType imat = 0; imat < Bmats_size; ++imat) {
		assert(imat < Bmats.size());
		const CrsMatrix<T>& thisMat = *(Bmats[imat]);
		SizeType nrow_B = thisMat.rows();
		SizeType ncol_B = thisMat.cols();
		SizeType nnz_B = thisMat.nonZeros();

		nrow_A = (nrow_B > nrow_A) ? nrow_B : nrow_A;
		ncol_A = (ncol_B > ncol_A) ? ncol_B : ncol_A;

		nnz_max = (nnz_B > nnz_max) ? nnz_B : nnz_max;
		nnz_sum += nnz_B;
	}

	A.resize(nrow_A, ncol_A);

	// ---------------------------------------------------
	// lower bound for total number of nonzeros is nnz_max
	// upper bound for total number of nonzeros is nnz_sum
	// initially set it to 2 * nnz_max
	// ---------------------------------------------------
	A.reserve((2 * nnz_max > nnz_sum) ? nnz_sum : 2 * nnz_max);

	// ------------------------------------------------------
	// TODO: using A.resize(nrow_A,ncol_A,nnz_A) may not work correctly
	// ------------------------------------------------------
	const bool set_nonzeros = true;
	if (set_nonzeros) {
		// -----------------------------------------------
		// worst case when no overlap in sparsity pattern
		// among matrices
		// -----------------------------------------------

		A.reserve(nnz_sum);
	}

	// ------------------------------------------
	// temporary vectors to accelerate processing
	// ------------------------------------------
	std::vector<T> valueTmp(ncol_A);
	std::vector<bool> is_examined_already(ncol_A, false);

	std::vector<SizeType> column_index;
	column_index.reserve(ncol_A);

	SizeType counter = 0;
	for (SizeType irow = 0; irow < nrow_A; ++irow) {
		A.setRow(irow, counter);

		column_index.clear();

		for (SizeType imat = 0; imat < Bmats_size; ++imat) {
			assert(imat < Bmats.size());
			const CrsMatrix<T>& thisMat = *(Bmats[imat]);
			const SizeType nrow_B = thisMat.rows();
			const bool is_valid_B_row = (irow < nrow_B);

			const SizeType kstart_B = (is_valid_B_row) ? thisMat.getRowPtr(irow) : 0;
			const SizeType kend_B = (is_valid_B_row) ? thisMat.getRowPtr(irow + 1) : 0;

			// --------------------------------
			// check whether there is work to do
			// --------------------------------
			const bool has_work = ((kend_B - kstart_B) >= 1);
			if (!has_work)
				continue;

			// -------------------------------
			// add contributions from matrix Bmats[i]
			// -------------------------------
			const T1 b1 = bvec[imat];
			for (SizeType k = kstart_B; k < kend_B; ++k) {
				const T bij = thisMat.getValue(k);
				const SizeType jcol = thisMat.getCol(k);

				assert(jcol < ncol_A);

				if (is_examined_already[jcol]) {
					valueTmp[jcol] += (bij * b1);
				} else {
					// ------------------------------------
					// new column entry not examined before
					// ------------------------------------

					is_examined_already[jcol] = true;

					valueTmp[jcol] = (bij * b1);

					column_index.push_back(jcol);
				}
			}
		} // end for imat

		// --------------------------------------
		// copy row to matrix A and reset vectors
		// --------------------------------------

		const SizeType kmax = column_index.size();
		for (SizeType k = 0; k < kmax; ++k) {
			const SizeType jcol = column_index[k];

			A.pushCol(jcol);
			A.pushValue(valueTmp[jcol]);
			is_examined_already[jcol] = false;
		}

		counter += kmax;
	} // end for irow

	SizeType nnz_A = counter;
	A.setRow(nrow_A, nnz_A);

	// ----------------------------------------
	// set exact number of nonzeros in matrix A
	//
	// note: this might be expensive in allocating another copy
	// and copying all the non-zero entries
	// ----------------------------------------
	const bool set_exact_nnz = true;
	if (set_exact_nnz)
		A.resize(nrow_A, ncol_A, nnz_A);

	A.checkValidity();
}

template <typename T>
bool isHermitian(const CrsMatrix<T>& A, bool verbose = false)
{
	if (A.rows() != A.cols())
		return false;
	Matrix<T> dense;
	crsMatrixToFullMatrix(dense, A);
	return isHermitian(dense, verbose);
}

template <typename T>
bool isAntiHermitian(const CrsMatrix<T>& A)
{
	if (A.rows() != A.cols())
		return false;
	Matrix<T> dense;
	crsMatrixToFullMatrix(dense, A);
	return isAntiHermitian(dense);
}

template <typename T>
void fromBlockToFull(CrsMatrix<T>& Bfull, const CrsMatrix<T>& B, SizeType offset)
{
	const bool use_push = true;
	int nrows_Bfull = Bfull.rows();
	int ncols_Bfull = Bfull.cols();
	int nnz_Bfull = B.nonZeros();
	Bfull.clear();

	if (use_push) {
		Bfull.resize(nrows_Bfull, ncols_Bfull);
		Bfull.reserve(nnz_Bfull);
	} else {
		Bfull.resize(nrows_Bfull, ncols_Bfull, nnz_Bfull);
	};

	int counter = 0;
	for (SizeType i = 0; i < offset; ++i)
		Bfull.setRow(i, counter);

	for (SizeType ii = 0; ii < B.rows(); ++ii) {
		SizeType i = ii + offset;
		Bfull.setRow(i, counter);
		for (int jj = B.getRowPtr(ii); jj < B.getRowPtr(ii + 1); ++jj) {
			SizeType j = B.getCol(jj) + offset;
			T tmp = B.getValue(jj);
			if (use_push) {
				Bfull.pushCol(j);
				Bfull.pushValue(tmp);
			} else {
				Bfull.setCol(counter, j);
				Bfull.setValues(counter, tmp);
			};
			counter++;
		}
	}

	for (SizeType i = B.rows() + offset; i < Bfull.rows(); ++i)
		Bfull.setRow(i, counter);

	Bfull.setRow(Bfull.rows(), counter);
	Bfull.checkValidity();
}

template <class T>
bool isDiagonal(const CrsMatrix<T>& A, double eps = 1e-6, bool checkForIdentity = false)
{
	if (A.rows() != A.cols())
		return false;
	SizeType n = A.rows();
	const T f1 = (-1.0);
	for (SizeType i = 0; i < n; i++) {
		for (int k = A.getRowPtr(i); k < A.getRowPtr(i + 1); k++) {
			SizeType col = A.getCol(k);
			const T& val = A.getValue(k);
			if (checkForIdentity && col == i && PsimagLite::norm(val + f1) > eps) {
				return false;
			}
			if (col != i && PsimagLite::norm(val) > eps) {
				return false;
			}
		}
	}
	return true;
}

template <class T>
bool isTheIdentity(const CrsMatrix<T>& A, double eps = 1e-6)
{
	return isDiagonal(A, eps, true);
}

template <typename T>
typename Real<T>::Type norm2(const CrsMatrix<T>& m)
{
	T val = 0;
	for (SizeType i = 0; i < m.values_.size(); i++)
		val += PsimagLite::conj(m.values_[i]) * m.values_[i];

	return PsimagLite::real(val);
}

template <typename T>
Matrix<T> multiplyTc(const CrsMatrix<T>& a, const CrsMatrix<T>& b)
{

	CrsMatrix<T> bb, c;
	transposeConjugate(bb, b);
	multiply(c, a, bb);
	Matrix<T> cc;
	crsMatrixToFullMatrix(cc, c);
	return cc;
}

template <typename T>
bool isZero(const CrsMatrix<T>& A, double eps = 0)
{
	SizeType n = A.values_.size();
	for (SizeType i = 0; i < n; ++i) {
		if (std::abs(A.values_[i]) > eps)
			return false;
	}

	return true;
}

} // namespace PsimagLite
/*@}*/
#endif
