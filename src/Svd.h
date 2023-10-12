#ifndef SVD_H
#define SVD_H
#include "Matrix.h"
#include "Profiling.h"
#include <fstream>
#include <iostream>
#include <sys/types.h>
#include <unistd.h>
#ifdef USE_MAGMA
#include "magma_v2.h"
#include <cuda.h>
#endif

namespace PsimagLite
{

template <typename ComplexOrRealType>
class Svd
{

public:

	typedef typename Real<ComplexOrRealType>::Type RealType;

	Svd(String name = "gesdd")
	    : name_(name)
	{
	}

	bool canTryAgain() const { return (name_ == "gesdd"); }

	String name() const { return name_; }

	void operator()(char jobz, Matrix<ComplexOrRealType>& a, typename Vector<RealType>::Type& s, Matrix<ComplexOrRealType>& vt)
	{
		if (jobz != 'A' && jobz != 'S') {
			String msg("svd: jobz must be either A or S");
			String jobzString = " ";
			jobzString[0] = jobz;
			throw RuntimeError(msg + ", not " + jobzString + "\n");
		}

		Profiling profiling("OnlySvd", std::cout);

		int m = a.rows();
		int n = a.cols();
		int min = (m < n) ? m : n;

		s.resize(min);
		int ldu = m;
		int ucol = (jobz == 'A') ? m : min;
		Matrix<ComplexOrRealType> u(ldu, ucol);
		int ldvt = (jobz == 'A') ? n : min;
		vt.resize(ldvt, n);

#ifdef USE_MAGMA
		PerformMagmaSvd(&jobz, &m, &n, (&a(0, 0)), (&u(0, 0)), &(s[0]), &(vt(0, 0)));
#else
		int lda = m;
		int lrwork = 2.0 * min * std::max(5 * min + 7, 2 * std::max(m, n) + 2 * min + 1);
		typename Vector<typename Real<ComplexOrRealType>::Type>::Type
		    rwork(lrwork, 0.0);

		typename Vector<ComplexOrRealType>::Type work(100, 0);
		int info = 0;
		Vector<int>::Type iwork(8 * min, 0);

		// query optimal work
		int lwork = -1;
		mycall(&jobz, &m, &n, &(a(0, 0)), &lda, &(s[0]), &(u(0, 0)), &ldu, &(vt(0, 0)), &ldvt, &(work[0]), &lwork, &(rwork[0]), &(iwork[0]), &info);
		if (info != 0) {
			String str(__FILE__);
			str += " svd(...) failed at workspace size calculation ";
			str += "with info=" + ttos(info) + "\n";
			throw RuntimeError(str.c_str());
		}

		RealType lworkReal = PsimagLite::real(work[0]);
		lwork = static_cast<int>(lworkReal) + (m + n) * 256;
		work.resize(lwork + 10);

		// real work:
		mycall(&jobz, &m, &n, &(a(0, 0)), &lda, &(s[0]), &(u(0, 0)), &ldu, &(vt(0, 0)), &ldvt, &(work[0]), &lwork, &(rwork[0]), &(iwork[0]), &info);
		if (info != 0) {
			if (info < 0)
				throw RuntimeError(String(__FILE__) + ": " + ttos(__LINE__) + " info= " + ttos(info));
			if (info > 0)
				std::cerr << "WARNING " << __FILE__ << ": "
					  << __LINE__ << " info= " << info
					  << "\n";
		}
#endif
		a = u;
	}

private:

	void mycall(char* jobz, int* m, int* n,
	    ComplexOrRealType* a, // T*,
	    int* lda,
	    RealType* s,
	    ComplexOrRealType* u, // T*,
	    int* ldu,
	    ComplexOrRealType* vt, // T*,
	    int* ldvt,
	    ComplexOrRealType* work, // T*,
	    int* lwork,
	    RealType* rwork, // nothing
	    int* iwork,
	    int* info)
	{
		if (name_ == "gesdd") {
			psimag::LAPACK::GESDD(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, info);
		} else if (name_ == "gesvd") {
			psimag::LAPACK::GESVD(jobz, jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info);
		} else {
			throw PsimagLite::RuntimeError("Unknown backend " + name_ + "\n");
		}
	}

#ifdef USE_MAGMA
	void PerformMagmaSvd(char* jobz, int* m, int* n,
	    ComplexOrRealType* a, // T*,
	    ComplexOrRealType* u, // T*,
	    RealType* s,
	    ComplexOrRealType* vt // T*,
	)
	{
		magma_init(); // initialize Magma

		int min_mn = (m[0] <= n[0]) ? m[0] : n[0];
		int max_mn = (m[0] >= n[0]) ? m[0] : n[0];
		magma_int_t n_, m_;
		n_ = n[0];
		m_ = m[0];

		int lda = m_;
		int ldu = m_;
		int ldvt = (jobz[0] == 'A') ? n_ : min_mn;

		if (name_ == "gesdd") {

			magma_int_t info;
			ComplexOrRealType* h_work; // h_work - workspace
			magma_int_t lwork; // workspace size
			magma_int_t liwork;
			magma_int_t* iwork; // workspace

			// may be dgesvd_nb, cgesvd_nb, zgesvd_nb should be used for respective ComplexOrRealTypes
			magma_int_t nb = magma_get_sgesvd_nb(m_, n_); // optim . block size

			lwork = min_mn * min_mn + 2 * min_mn + 2 * min_mn * nb;
			liwork = 8 * min_mn;
			iwork = (magma_int_t*)malloc(liwork * sizeof(magma_int_t));

			ComplexOrRealType aux_work[1];

			int lrwork = std::max(5 * min_mn * min_mn + 5 * min_mn, 2 * max_mn * min_mn + 2 * min_mn * min_mn + min_mn) + max_mn;
			RealType* rwork;
			rwork = (RealType*)malloc(lrwork * sizeof(RealType));

			magma_Xgesdd(MagmaSomeVec, m_, n_, a, lda, s, u, ldu, vt, ldvt, aux_work, -1, rwork, iwork, &info);
			lwork = (magma_int_t)abs(aux_work[0]);

			magma_Xmalloc_pinned(&h_work, lwork);
			magma_Xgesdd(MagmaSomeVec, m_, n_, a, lda, s, u, ldu, vt, ldvt, h_work, lwork, rwork, iwork, &info);

		} else if (name_ == "gesvd") {
			assert(false);
		} else {
			throw PsimagLite::RuntimeError("Unknown backend " + name_ + "\n");
		}

		magma_finalize(); // finalize Magma
	}

	magma_int_t magma_Xgesdd(magma_vec_t jobz, magma_int_t m_, magma_int_t n_,
	    std::complex<float>* a, magma_int_t lda, float* s, std::complex<float>* u,
	    magma_int_t ldu, std::complex<float>* vt, magma_int_t ldvt, std::complex<float>* aux_work,
	    magma_int_t lwork_, float* rwork, magma_int_t* iwork, magma_int_t* info)
	{
		magma_int_t temp_int = magma_cgesdd(jobz, m_, n_, (magmaFloatComplex*)a, lda, s, (magmaFloatComplex*)u, ldu,
		    (magmaFloatComplex*)vt, ldvt, (magmaFloatComplex*)aux_work, lwork_, rwork, iwork, info);
		return temp_int;
	}
	magma_int_t magma_Xgesdd(magma_vec_t jobz, magma_int_t m_, magma_int_t n_,
	    float* a, magma_int_t lda, float* s, float* u,
	    magma_int_t ldu, float* vt, magma_int_t ldvt, float* aux_work,
	    magma_int_t lwork_, float* rwork, magma_int_t* iwork, magma_int_t* info)
	{
		magma_int_t temp_int = magma_sgesdd(jobz, m_, n_, a, lda, s, u, ldu, vt, ldvt, aux_work, lwork_, iwork, info);
		return temp_int;
	}
	magma_int_t magma_Xgesdd(magma_vec_t jobz, magma_int_t m_, magma_int_t n_,
	    std::complex<double>* a, magma_int_t lda, double* s, std::complex<double>* u,
	    magma_int_t ldu, std::complex<double>* vt, magma_int_t ldvt, std::complex<double>* aux_work,
	    magma_int_t lwork_, double* rwork, magma_int_t* iwork, magma_int_t* info)
	{
		magma_int_t temp_int = magma_zgesdd(jobz, m_, n_, (magmaDoubleComplex*)a, lda, s, (magmaDoubleComplex*)u, ldu,
		    (magmaDoubleComplex*)vt, ldvt, (magmaDoubleComplex*)aux_work, lwork_, rwork, iwork, info);
		return temp_int;
	}
	magma_int_t magma_Xgesdd(magma_vec_t jobz, magma_int_t m_, magma_int_t n_,
	    double* a, magma_int_t lda, double* s, double* u,
	    magma_int_t ldu, double* vt, magma_int_t ldvt, double* aux_work,
	    magma_int_t lwork_, double* rwork, magma_int_t* iwork, magma_int_t* info)
	{
		magma_int_t temp_int = magma_dgesdd(jobz, m_, n_, a, lda, s, u, ldu, vt, ldvt, aux_work, lwork_, iwork, info);
		return temp_int;
	}

	magma_int_t magma_Xmalloc_pinned(float** h_work, magma_int_t lwork)
	{
		magma_int_t temp_int = magma_smalloc_pinned(h_work, lwork);
		return temp_int;
	}

	magma_int_t magma_Xmalloc_pinned(double** h_work, magma_int_t lwork)
	{
		magma_int_t temp_int = magma_dmalloc_pinned(h_work, lwork);
		return temp_int;
	}

	magma_int_t magma_Xmalloc_pinned(std::complex<float>** h_work, magma_int_t lwork)
	{
		magma_int_t temp_int = magma_cmalloc_pinned((magmaFloatComplex**)h_work, lwork);
		return temp_int;
	}

	magma_int_t magma_Xmalloc_pinned(std::complex<double>** h_work, magma_int_t lwork)
	{
		magma_int_t temp_int = magma_zmalloc_pinned((magmaDoubleComplex**)h_work, lwork);
		return temp_int;
	}

#endif

	Svd(const Svd&);

	Svd& operator=(const Svd&);

	String name_;
};

} // namespace PsimagLite

#endif // SVD_H
