#include "Ainur.h"
#include <complex>
#include <vector>

void partiallyReadSomething(const PsimagLite::Ainur& ainur)
{
	SizeType n = 0;
	ainur.readValue(n, "TotalNumberOfSites");
	std::cout << "Read: TotalNumberOfSites=" << n << "\n";

	PsimagLite::String m;
	ainur.readValue(m, "Model");
	std::cout << m << "\n";

	std::string v2;
	ainur.readValue(v2, "useit");
	std::cout << v2;
	std::cout << std::endl;

	std::vector<std::string> v3;
	ainur.readValue(v3, "myvector");
	double t = 0.3;
	std::vector<double> v3double(v3.size());
	for (SizeType i = 0; i < v3.size(); ++i) {
		PsimagLite::String str = v3[i];
		PsimagLite::replaceAll(str, "%t", std::to_string(t));
		str = ainur.resolve(str);
		v3double[i] = PsimagLite::atof(str);
	}

	std::cout << v3double.size() << "\n";
	if (v3double.size() == 2) {
		std::cout << v3double[0] << " " << v3double[1] << "\n";
	}
}

int main(int argc, char** argv)
{
	if (argc == 1)
		return 1;

	PsimagLite::String str;
	for (int i = 1; i < argc; ++i) {
		std::ifstream fin(argv[i]);
		PsimagLite::String str2;

		fin.seekg(0, std::ios::end);
		str2.reserve(fin.tellg());
		fin.seekg(0, std::ios::beg);

		str2.assign((std::istreambuf_iterator<char>(fin)),
		    std::istreambuf_iterator<char>());
		fin.close();

		str += str2;
	}

	PsimagLite::Ainur ainur(str);
	// ainur.printAll(std::cout);
	partiallyReadSomething(ainur);
}
