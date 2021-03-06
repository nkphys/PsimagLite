#ifndef PSI_LANCZOS_SOLVER_H
#define PSI_LANCZOS_SOLVER_H
#include "LanczosCore.h"
#include "LanczosOrDavidsonBase.h"
#include "Profiling.h"
#include "Vector.h"

namespace PsimagLite
{

template <typename SolverParametersType, typename MatrixType_, typename VectorType_>
class LanczosSolver : public LanczosOrDavidsonBase<SolverParametersType,
			  MatrixType_,
			  VectorType_>
{

public:

	typedef LanczosOrDavidsonBase<SolverParametersType, MatrixType_, VectorType_>
	    BaseType;
	typedef LanczosCore<SolverParametersType, MatrixType_, VectorType_>
	    LanczosCoreType;
	typedef typename LanczosCoreType::TridiagonalMatrixType
	    TridiagonalMatrixType;
	typedef typename LanczosCoreType::RealType RealType;
	typedef typename LanczosCoreType::VectorType VectorType;
	typedef typename Vector<VectorType>::Type VectorVectorType;
	typedef typename LanczosCoreType::VectorRealType VectorRealType;
	typedef MatrixType_ MatrixType;
	typedef ContinuedFraction<TridiagonalMatrixType> PostProcType;
	typedef SolverParametersType ParametersSolverType;

	LanczosSolver(const MatrixType& mat, const SolverParametersType& params)
	    : ls_(mat, params, BaseType::isReorthoEnabled(params))
	{
	}

	void computeOneState(RealType& energy, VectorType& z, const VectorType& initialVector, SizeType excited)
	{
		Profiling profiling("LanczosSolver", std::cout);

		TridiagonalMatrixType ab;
		ls_.decomposition(initialVector, ab, excited);

		VectorRealType eigs(ab.size());
		typename LanczosCoreType::DenseMatrixType ritz;
		ab.buildDenseMatrix(ritz);
		diag(ritz, eigs, 'V');

		energy = eigs[excited];
		ls_.excitedVector(z, ritz, excited);

		String str = "LanczosSolver: computeOneState: ";
		if (norm(z) < 1e-6)
			throw RuntimeError(str + " norm is zero\n");

		const RealType norma = norm(initialVector);
		const SizeType iter = ls_.steps();

		if (norma < 1e-5 || norma > 100)
			std::cerr << "norma=" << norma << "\n";

		OstringStream msg(std::cout.precision());
		String what = "lowest";
		if (excited > 0)
			what = ttos(excited) + " excited";
		msg() << "Found " << what << " eigenvalue= " << energy
		      << " after " << iter;
		msg() << " iterations, "
		      << " orig. norm=" << norma << " excited=" << excited;
		profiling.end(msg().str());
	}

	void computeAllStatesBelow(VectorRealType& eigs, VectorVectorType& z, const VectorType& initialVector, SizeType excited)
	{
		TridiagonalMatrixType ab;
		ls_.decomposition(initialVector, ab, excited);

		if (excited > ab.size())
			throw RuntimeError("Excited too big\n");

		typename LanczosCoreType::DenseMatrixType ritz;
		ab.buildDenseMatrix(ritz);
		diag(ritz, eigs, 'V');

		SizeType n = z.size();
		if (n > excited + 1)
			n = excited + 1;
		for (SizeType i = 0; i < n; ++i)
			ls_.excitedVector(z[i], ritz, i);
	}

	void decomposition(const VectorType& initVector,
	    TridiagonalMatrixType& ab)
	{
		return ls_.decomposition(initVector, ab, ls_.params().eigsForStop);
	}

	void lanczosVectorsSwap(typename LanczosCoreType::DenseMatrixType& V)
	{
		ls_.lanczosVectorsSwap(V);
	}

	SizeType steps() const { return ls_.steps(); }

private:

	LanczosCoreType ls_;
};
} // namespace PsimagLite
#endif // PSI_LANCZOS_SOLVER_H
