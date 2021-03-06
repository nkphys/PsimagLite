#ifndef PARALLELIZER2OPENMP_H
#define PARALLELIZER2OPENMP_H
#include "CodeSectionParams.h"
#include "Vector.h"
#include <omp.h>

namespace PsimagLite
{

template <typename = int>
class Parallelizer2
{

public:

	Parallelizer2(const CodeSectionParams& codeParams)
	    : threads_(codeParams.npthreads)
	{
		omp_set_num_threads(codeParams.npthreads);
	}

	template <typename SomeLambdaType>
	void parallelFor(SizeType start, SizeType end, const SomeLambdaType& lambda)
	{
#pragma omp parallel for
		for (SizeType i = start; i < end; ++i)
			lambda(i, omp_get_thread_num());
	}

	SizeType numberOfThreads() const { return omp_get_num_threads(); }

	String name() const { return "openmp"; }

private:

	SizeType threads_;
};
} // namespace PsimagLite
#endif // PARALLELIZER2OPENMP_H
