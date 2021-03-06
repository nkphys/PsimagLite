#include "PredicateAwesome.h"
#include <cstdlib>

int main(int argc, char** argv)
{
	if (argc < 3)
		return 1;
	PsimagLite::String predicate(argv[1]);

	PsimagLite::replaceAll(predicate, "c", "5");
	PsimagLite::PredicateAwesome<> pAwesome(predicate);
	std::cout << pAwesome.isTrue("l", atoi(argv[2]));
}
