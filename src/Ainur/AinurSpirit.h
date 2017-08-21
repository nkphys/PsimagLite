#ifndef _AINUR_SPIRIT_H_
#define _AINUR_SPIRIT_H_
#include "../Vector.h"
#include "../TypeToString.h"
#include "../PsimagLite.h"
#include <iostream>
#include <string>
#include <fstream>

namespace PsimagLite {

class Ainur {

	struct Action {

		Action(const char *name)
		    : name_(name)
		{}

		template <typename A, typename ContextType, typename PassType>
		void operator()(A& attr,
		                ContextType& context,
		                PassType hit) const;

	private:

		const char* name_;
	};

	struct myprint
	{
		template <typename T>
		void operator()(const T &t) const
		{
			std::cout << "|  " << std::boolalpha << t << '\n';
		}
	};

public:

	typedef std::string::iterator IteratorType;
	typedef Vector<String>::Type VectorStringType;
	//typedef AinurStatements AinurStatementsType;
	//typedef AinurStatementsType::AinurLexicalType AinurLexicalType;

	Ainur(String str);

	String& prefix() { return dummy_; }

	const String& prefix() const { return dummy_; }

	void printUnused(std::ostream& os) const
	{
		os<<"PRINT UNUSED\n";
	}

	template<typename SomeType>
	void readValue(SomeType& t, String label) const
	{
		std::cerr<<"readValue called for label="<<label<<"\n";
		err("Ainur isn't ready, throwing...\n");
	}

private:

	String dummy_;
}; // class AinurSpirit

}
#endif // _AINUR_SPIRIT_H_