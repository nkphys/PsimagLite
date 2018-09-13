#ifndef PSI_PSIMAGLITE_H
#define PSI_PSIMAGLITE_H

#include <iostream>
#include <utility>
#include "Concurrency.h"
#include "AnsiColors.h"
#include "TypeToString.h"
#include "Vector.h"
#include "Random48.h"

namespace PsimagLite {

std::ostream& operator<<(std::ostream&,const std::pair<SizeType,SizeType>&);

std::istream& operator>>(std::istream&,std::pair<SizeType,SizeType>&);

SizeType log2Integer(SizeType x);

void err(String);

struct MatchPathSeparator {
    bool operator()(char ch) const
    {
        return (ch == '/');
    }
};

template<typename T>
void fillRandom(T& v, typename EnableIf<IsVectorLike<T>::True, int>::Type = 0)
{
	SizeType n = v.size();
	if (n == 0)
		throw std::runtime_error("fillRandom must be called with size > 0\n");

	Random48<typename T::value_type> myrng(time(0));
	typename PsimagLite::Real<typename T::value_type>::Type sum = 0;
	const typename T::value_type zeroPointFive = 0.5;
	for (SizeType i = 0; i < n; ++i) {
		v[i] = myrng() - zeroPointFive;
		sum += PsimagLite::real(v[i]*PsimagLite::conj(v[i]));
	}

	sum = 1.0/sqrt(sum);
	for (SizeType i = 0; i < n; ++i) v[i] *= sum;
}

void split(Vector<String>::Type& tokens, String str, String delimiters = " ");

String basename(const String&);

class PsiApp {
public:

	PsiApp(String appName, int* argc, char*** argv, int nthreads)
	    : concurrency_(argc,argv,nthreads), appName_(basename(appName))
	{
		chekSizeType();
	}

	const String& name() const { return appName_; }

private:

	void chekSizeType()
	{
		if (sizeof(SizeType) == libSizeOfSizeType_) return;
		std::string msg("PsimagLite compiled with -DUSE_LONG but");
		msg += "application without. Or viceversa.\n";
		throw std::runtime_error(msg);
	}

	static const int libSizeOfSizeType_;

	Concurrency concurrency_;
	String appName_;
};
} // namespace PsimagLite

void err(PsimagLite::String);

#endif // PSI_PSIMAGLITE_H

