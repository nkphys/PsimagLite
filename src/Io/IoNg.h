/*
Copyright (c) 2009-2018, UT-Battelle, LLC
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

/** \ingroup PsimagLite */
/*@{*/

/*! \file IoNg.h
 *
 *  This class handles Input/Output for PsimagLite
 */

#ifndef PSI_IO_NG_H
#define PSI_IO_NG_H

#include <iostream>
#include "Vector.h"
#include "Matrix.h"
#include "Stack.h"
#include "Map.h"
#include "H5Cpp.h"
#include <typeinfo>
#include "IoNgSerializer.h"

namespace PsimagLite {

template<typename T>
struct IsRootUnDelegated {
	enum {True = Loki::TypeTraits<T>::isArith ||
		  IsVectorLike<T>::True ||
		  IsPairLike<T>::True ||
		  IsEnum<T>::True};
};

class IoNg {

public:

	enum WriteMode {NO = IoNgSerializer::NO,
		            ALLOW_OVERWRITE = IoNgSerializer::ALLOW_OVERWRITE};

	class Out {

	public:

		typedef std::vector<String> VectorStringType;

		Out(const String& fn)
		    : filename_(fn),
		      hdf5File_(new H5::H5File(fn, H5F_ACC_TRUNC)),
		      groupDef_(new H5::Group(hdf5File_->createGroup("/Def"))),
		      ioNgSerializer_(hdf5File_, groupDef_)
		{}

		~Out()
		{
			filename_ = "";
			delete groupDef_; // should I close something first? FIXME
			groupDef_ = 0;
			delete hdf5File_; // should I close something first? FIXME
			hdf5File_ = 0;
		}

		const String& filename() const
		{
			return filename_;
		}

		void open(String const &fn,
		          std::ios_base::openmode mode)
		{
			if (hdf5File_) delete hdf5File_;

			filename_ = fn;
			// deal with mode
			hdf5File_ = new H5::H5File(fn, H5F_ACC_TRUNC);

			throw RuntimeError("IoNg:: open cannot handle mode yet\n");
			throw RuntimeError("IoNg:: open cannot handle serializer object yet\n");
		}

		void close()
		{
			// deal with the serializer object FIXME
			delete groupDef_; // should I close something first? FIXME
			groupDef_ = 0;
			delete hdf5File_; // should I close something first? FIXME
			hdf5File_ = 0;
			filename_ = "";
		}

		void createGroup(String groupName)
		{
			ioNgSerializer_.createGroup(groupName);
		}

		template<typename T>
		void writeLabel(T x,
		                PsimagLite::String str,
		                PsimagLite::OstringStream&,
		                SizeType counter)
		{
			if (counter == 0) createGroup(str);

			ioNgSerializer_.write(str + "/" + ttos(counter), x);
			ioNgSerializer_.write(str + "/Size", counter, (counter == 0) ?
			                          IoNgSerializer::NO : IoNgSerializer::ALLOW_OVERWRITE);
		}

		template<typename T>
		void write(const T& what,
		           String name2,
		           typename EnableIf<IsRootUnDelegated<T>::True, int>::Type = 0)
		{
			ioNgSerializer_.write(name2, what);
		}

		template<typename T>
		void write(const T& what,
		           String name2,
		           typename EnableIf<!IsRootUnDelegated<T>::True, int>::Type = 0)
		{
			what.write(name2, ioNgSerializer_);
		}

		int rank() { throw RuntimeError("IoNg:: not implemented\n"); }

		void flush()
		{ throw RuntimeError("IoNg:: not implemented\n"); }

		void setPrecision(SizeType x)
		{ throw RuntimeError("IoNg:: not implemented\n"); }

		template<typename ActionType>
		void action(ActionType& a)
		{ throw RuntimeError("IoNg:: not implemented\n"); }

		template<typename T>
		friend Out& operator<<(Out& io, const T& t)
		{ throw RuntimeError("IoNg:: not implemented\n"); }

	private:

		String filename_;
		H5::H5File* hdf5File_;
		H5::Group* groupDef_;
		IoNgSerializer ioNgSerializer_;
	};

	class In {

	public:

		typedef int long LongIntegerType;
		static const LongIntegerType LAST_INSTANCE=-1;
		typedef unsigned int long LongSizeType;

		In() : hdf5File_(0), groupDef_(0),ioNgSerializer_(hdf5File_, groupDef_)
		{}

		In(String const &fn)
		    : filename_(fn),
		      hdf5File_(new H5::H5File(fn, H5F_ACC_RDONLY)),
		      groupDef_(new H5::Group(hdf5File_->openGroup("Def"))),
		      ioNgSerializer_(hdf5File_, groupDef_)
		{}

		~In()
		{
			delete groupDef_; // should I close something first? FIXME
			groupDef_ = 0;
			delete hdf5File_; // should I close something first? FIXME
			hdf5File_ = 0;
			filename_ = "";
		}

		void open(String const &fn)
		{
			if (hdf5File_) delete hdf5File_;

			filename_ = fn;
			// deal with mode
			hdf5File_ = new H5::H5File(fn, H5F_ACC_RDONLY);
		}

		void close()
		{ throw RuntimeError("IoNg:: not implemented\n"); }

		template<typename SomeType>
		SizeType readline(SomeType &x,const String &s,LongIntegerType = 0)
		{
			ioNgSerializer_.read(x, s);
			return 0;
		}

		template<typename T>
		void read(T& what,
		          String name,
		          typename EnableIf<IsRootUnDelegated<T>::True, int>::Type = 0)
		{
			ioNgSerializer_.read(what, name);
		}

		template<typename T>
		void read(T& what,
		          String name,
		          typename EnableIf<!IsRootUnDelegated<T>::True, int>::Type = 0)
		{
			what.read(name, ioNgSerializer_);
		}

		template<typename X>
		std::pair<String,SizeType> readKnownSize(X &x,
		                                         String const &s,
		                                         LongIntegerType level=0)
		{ throw RuntimeError("IoNg:: not implemented\n"); }

		std::pair<String,SizeType> advance(String const &s,
		                                   LongIntegerType level=0,
		                                   bool beQuiet=false)
		{ throw RuntimeError("IoNg:: not implemented\n"); }

		void readFullLine(String& temp)
		{ throw RuntimeError("IoNg:: not implemented\n"); }

		LongSizeType advanceToLine(LongSizeType line)
		{ throw RuntimeError("IoNg:: not implemented\n"); }

		SizeType count(const String& s)
		{ throw RuntimeError("IoNg:: not implemented\n"); }

		template<typename X,template<typename> class SomeType>
		void readSparseVector(SomeType<X> &x,
		                      String const &s,
		                      LongIntegerType level=0)
		{ throw RuntimeError("IoNg:: not implemented\n"); }

		void rewind()
		{ throw RuntimeError("IoNg:: not implemented\n"); }

		void move(int x)
		{ throw RuntimeError("IoNg:: not implemented\n"); }

		bool eof() const { throw RuntimeError("IoNg:: not implemented\n"); }

		const char* filename() const
		{ throw RuntimeError("IoNg:: not implemented\n"); }


		template<typename T>
		friend void operator>>(In& io, T& t)
		{ throw RuntimeError("IoNg:: not implemented\n"); }

	private:

		template<typename T>
		void internalRead(void* ptr, String label, H5::DataSet& dataset) const
		{
			H5T_class_t typeClass = dataset.getTypeClass();
			if (typeClass != TypeToH5<T>::super)
				throw RuntimeError("Reading " + label + " has incorrect type\n");
			// H5::FloatType ft = dataset.getFloatType(); // <-- check correct subtype FIXME

			dataset.read(ptr, TypeToH5<T>::type);
		}

		String filename_;
		H5::H5File* hdf5File_;
		H5::Group* groupDef_;
		IoNgSerializer ioNgSerializer_;
	};
}; //class IoNg

template<>
struct IsInputLike<IoNg::In> {
	enum {True = true};
};

template<>
struct IsOutputLike<IoNg::Out> {
	enum {True = true};
};

} // namespace PsimagLite

/*@}*/
#endif
