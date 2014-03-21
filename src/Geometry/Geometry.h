/*
Copyright (c) 2009-2103, UT-Battelle, LLC
All rights reserved

[PsimagLite, Version 1.0.0]
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
/** \ingroup PsimagLite */
/*@{*/

/*! \file Geometry.h
 *
 *  DOC NEEDED FIXME
 */
#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "GeometryTerm.h"
#include "String.h"
#include "GeometryEx.h"

namespace PsimagLite {

template<typename ComplexOrRealType_,typename InputType,typename ProgramGlobalsType>
class Geometry {

public:

	typedef ComplexOrRealType_ ComplexOrRealType;
	typedef GeometryTerm<ComplexOrRealType,InputType> GeometryTermType;
	typedef typename Vector<SizeType>::Type BlockType;
	typedef typename GeometryTermType::AdditionalDataType AdditionalDataType;
	typedef GeometryEx<GeometryTermType> GeometryExType;
	typedef typename GeometryExType::VectorRealType VectorRealType;

	Geometry(InputType& io,bool debug=false)
	: geometryEx_(linSize_,terms_)
	{
		int x;
		io.readline(x,"TotalNumberOfSites=");
		if (x<0) throw RuntimeError("TotalNumberOfSites<0 is an error\n");
		linSize_ = x;

		io.readline(x,"NumberOfTerms=");
		if (x<0) throw RuntimeError("NumberOfTerms<0 is an error\n");

		terms_.resize(x,0);
		for (SizeType i=0;i<terms_.size();i++) {
			terms_[i] = new GeometryTermType(io,i,linSize_,debug);
		}
	}

	~Geometry()
	{
		for (SizeType i=0;i<terms_.size();i++)
			if (terms_[i]) delete terms_[i];
	}

	String label(SizeType i) const { return terms_[i]->label(); }

	SizeType connectionKind(SizeType smax,SizeType ind,SizeType jnd) const
	{
		SizeType middle = smax + 1;
		if (ind<middle && jnd>=middle) return ProgramGlobalsType::SYSTEM_ENVIRON;
		if (jnd<middle && ind>=middle) return ProgramGlobalsType::ENVIRON_SYSTEM;
		if (ind<middle) return ProgramGlobalsType::SYSTEM_SYSTEM;
		return ProgramGlobalsType::ENVIRON_ENVIRON;
	}

	ComplexOrRealType operator()
	(SizeType smax,SizeType emin,
	 SizeType i1,SizeType edof1,SizeType i2, SizeType edof2,SizeType term) const
	{
		if (smax+1==emin) return terms_[term]->operator()(i1,edof1,i2,edof2);
		return terms_[term]->operator()(smax,emin,i1,edof1,i2,edof2);
	}

	ComplexOrRealType operator()
	(SizeType i1,SizeType edof1,SizeType i2, SizeType edof2,SizeType term) const
	{
		return terms_[term]->operator()(i1,edof1,i2,edof2);
	}

	// needs to check all terms FIXME:
	bool connected(SizeType smax,SizeType emin,SizeType i1,SizeType i2) const
	{
		if (smax+1==emin) return terms_[0]->connected(i1,i2); // any term will do
		return terms_[0]->connected(smax,emin,i1,i2); // any term will do
	}

	SizeType terms() const { return terms_.size(); }

	SizeType numberOfSites() const { return linSize_; }

	void split(SizeType sitesPerBlock,
	           BlockType& S,
	           typename Vector<BlockType>::Type& X,
	           typename Vector<BlockType>::Type& Y,
	           BlockType& E) const
	{
		SizeType middle = linSize_/2;
		if (linSize_& 1 ||
		    linSize_ % sitesPerBlock!=0 ||
		    SizeType(linSize_/sitesPerBlock)<3) {
			String str(__FILE__);
			str += " " + ttos(__LINE__) + "\n";
			str += "split error, linSize_=" + ttos(linSize_);
			str += " sitesPerBlock=" + ttos(sitesPerBlock) + "\n";
			throw RuntimeError(str.c_str());
		}

		SizeType i=0;
		while (i<sitesPerBlock) {
			S.push_back(i);
			i++;
		}

		while (i<middle) {
			typename Vector<SizeType>::Type tmpV(sitesPerBlock);
			for (SizeType j=0;j<sitesPerBlock;j++)
				tmpV[j] = i+j;
			X.push_back(tmpV);
			i+=sitesPerBlock;
		}

		SizeType lastMiddle=linSize_-sitesPerBlock;
		while (i<lastMiddle) {
			typename Vector<SizeType>::Type tmpV(sitesPerBlock);
			for (SizeType j=0;j<sitesPerBlock;j++) {
				SizeType jj = sitesPerBlock-1-j;
				tmpV[j] = (linSize_-1-i-jj)+(middle-sitesPerBlock);
				assert(tmpV[j]<linSize_);
			}
			Y.push_back(tmpV);
			i+=sitesPerBlock;
		}

		while (i<linSize_) {
			E.push_back(i);
			i++;
		}
	}

	SizeType maxConnections(SizeType termId = 0) const
	{
		return terms_[termId]->maxConnections();
	}

	void fillAdditionalData(AdditionalDataType& additionalData,
	                        SizeType term,
	                        SizeType ind,
	                        SizeType jnd) const
	{
		terms_[term]->fillAdditionalData(additionalData,ind,jnd);
	}

	SizeType findReflection(SizeType site,SizeType termId) const
	{
		return terms_[termId]->findReflection(site);
	}

	SizeType length(SizeType i,SizeType termId) const
	{
		return terms_[termId]->length(i);
	}

	SizeType translate(SizeType site,SizeType dir, SizeType amount,SizeType termId) const
	{
		return terms_[termId]->translate(site,dir,amount);
	}

	void print(std::ostream& os) const
	{
		for (SizeType i=0;i<terms_.size();i++)
			terms_[i]->print(os,linSize_);
	}

	// extended functions
	
	SizeType dimension(SizeType i = 0) const { return geometryEx_.dimension(i); }
	
	void index2Rvector(SizeType r,VectorRealType& rvector) const
	{
		return geometryEx_.index2Rvector(r,rvector);
	}
	
	void index2Kvector(SizeType k,VectorRealType& kvector) const
	{
		return geometryEx_.index2Kvector(k,kvector);
	}
	
	SizeType nGroupK() const
	{
		return geometryEx_.nGroupK();
	}
	
	SizeType ickequ(SizeType j,SizeType op)  const
	{
		return geometryEx_.ickequ(j,op) ;
	}
	
	// friends
	template<typename RealType2,typename InputType2, typename PgType>
	friend std::ostream& operator<<(std::ostream& os,
	                                const Geometry<RealType2,InputType2,PgType>& g);

private:

	SizeType linSize_;
	typename Vector<GeometryTermType*>::Type terms_;
	GeometryExType geometryEx_;

}; // class Geometry

template<typename ComplexOrRealType,typename InputType,typename PgType>
std::ostream& operator<<(std::ostream& os,
                         const Geometry<ComplexOrRealType,InputType,PgType>& g)
{
	os<<"#GeometrySize="<<g.linSize_<<"\n";
	os<<"#GeometryTerms="<<g.terms_.size()<<"\n";
	for (SizeType i=0;i<g.terms_.size();i++) os<<*(g.terms_[i]);
	return os;
}
} // namespace PsimagLite 

/*@}*/
#endif // GEOMETRY_H

