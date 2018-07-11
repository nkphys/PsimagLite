#include "PsimagLite.h"

namespace PsimagLite {

std::ostream& operator<<(std::ostream& os,const std::pair<SizeType,SizeType>& p)
{
	os<<p.first<<" "<<p.second<<" ";
	return os;
}

std::istream& operator>>(std::istream& is,std::pair<SizeType,SizeType>& pair)
{
	is>>pair.first;
	is>>pair.second;
	return is;
}

void err(String s)
{
	throw RuntimeError(s);
}

SizeType log2Integer(SizeType x)
{
	SizeType count = 0;
	while (x > 0) {
		x >>= 1;
		++count;
	}

	return count;
}

void split(Vector<String>::Type& tokens, String str, String delimiters)
{
	// Skip delimiters at beginning.
	String::size_type lastPos = str.find_first_not_of(delimiters, 0);
	// Find first "non-delimiter".
	String::size_type pos     = str.find_first_of(delimiters, lastPos);

	while (String::npos != pos || String::npos != lastPos)
	{
		// Found a token, add it to the vector.
		tokens.push_back(str.substr(lastPos, pos - lastPos));
		// Skip delimiters.  Note the "not_of"
		lastPos = str.find_first_not_of(delimiters, pos);
		// Find next "non-delimiter"
		pos = str.find_first_of(delimiters, lastPos);
	}
}

String basename(const String& path)
{
	return String(std::find_if(path.rbegin(),
	                           path.rend(),
	                           MatchPathSeparator()).base(),
	              path.end());
}

SizeType indexOfItemOrMinusOne(const Vector<SizeType>::Type& v, SizeType x)
{
	SizeType n = v.size();
	for (SizeType i = 0; i < n; ++i)
		if (v[i] == x) return i;

	return -1;
}

SizeType indexOfItem(const Vector<SizeType>::Type& v, SizeType x)
{
	int y = indexOfItemOrMinusOne(v, x);
	if (y >= 0) return y;

	throw RuntimeError("indexOfItem(): item not found " + typeToString(x) + "\n");
}
const int PsiApp::libSizeOfSizeType_ = sizeof(SizeType);

} // namespace PsimagLite

void err(PsimagLite::String s)
{
	PsimagLite::err(s);
}

