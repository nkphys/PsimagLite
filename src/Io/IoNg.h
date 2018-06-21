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
#include "AllocatorCpu.h"

namespace PsimagLite {

/* PSIDOC IsRootUnDelegated
Root-undelegated types are one of the following.
\begin{lstlisting}
PSIDOCCOPY IsRootUnDelegatedCode
\end{lstlisting}
Root-undelegateds are always at least partially written to by \code{IoNg},
and, if needed, parts of it are delegated.
For example, all native types are written by \code{IoNg} directly into a single dataset.
\code{std::complex<T>} where \code{T}
is a native type is written directly by doubling the size of the array into a single dataset.
*/
// PSIDOC_CODE_START IsRootUnDelegatedCode
template<typename T>
struct IsRootUnDelegated {
	enum {True = Loki::TypeTraits<T>::isArith ||
		  IsVectorLike<T>::True ||
		  IsStackLike<T>::True ||
		  IsPairLike<T>::True ||
		  IsEnum<T>::True ||
		  IsStringLike<T>::True};
};
// PSIDOC_CODE_END

class IoNg {

public:

	enum OpenMode {ACC_TRUNC, ACC_EXCL, ACC_RDONLY, ACC_RDW};

	/*
		H5F_ACC_TRUNC - Truncate file, if it already exists,
		erasing all data previously stored in the file.
		H5F_ACC_EXCL - Fail if file already exists. H5F_ACC_TRUNC
		and H5F_ACC_EXCL are mutually exclusive
		H5F_ACC_RDONLY - Open file as read-only, if it already exists, and fail, otherwise
		H5F_ACC_RDWR - Open file for read/write, if it already exists, and fail, otherwise
	*/

	class Out {

	public:

		typedef IoNgSerializer Serializer;
		typedef std::vector<String> VectorStringType;

		Out(const String& fn, OpenMode mode = ACC_TRUNC)
		    : filename_(fn),
		      hdf5File_(new H5::H5File(fn, modeToH5(mode))),
		      ioNgSerializer_(hdf5File_)
		{
#ifdef NDEBUG
			H5::Exception::dontPrint();
#endif
			if (mode == ACC_TRUNC)
				ioNgSerializer_.createGroup("");
		}

		~Out()
		{
			filename_ = "";
			delete hdf5File_;
			hdf5File_ = 0;
		}

		bool ng() const { return true; }

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
			delete hdf5File_;
			hdf5File_ = 0;
			filename_ = "";
		}

		void createGroup(String groupName)
		{
			ioNgSerializer_.createGroup(groupName);
		}

		template<typename T>
		void writeVectorEntry(T x,
		                      PsimagLite::String str,
		                      SizeType counter)
		{
			if (counter == 0) createGroup(str);

			ioNgSerializer_.write(str + "/" + ttos(counter), x);
			ioNgSerializer_.write(str + "/Size", counter, (counter == 0) ?
			                          IoNgSerializer::NO_OVERWRITE :
			                          IoNgSerializer::ALLOW_OVERWRITE);
		}

		template<typename T>
		void write(std::stack<T>& what,
		           String name2,
		           IoNgSerializer::WriteMode mode = IoNgSerializer::NO_OVERWRITE,
		           typename EnableIf<!IsRootUnDelegated<T>::True, int>::Type = 0)
		{
			ioNgSerializer_.write(name2, what, mode);
		}

		template<typename T>
		void write(const T& what,
		           String name2,
		           IoNgSerializer::WriteMode mode = IoNgSerializer::NO_OVERWRITE,
		           typename EnableIf<IsRootUnDelegated<T>::True, int>::Type = 0)
		{
			ioNgSerializer_.write(name2, what, mode);
		}

		template<typename T>
		void write(const T& what,
		           String name2,
		           typename EnableIf<!IsRootUnDelegated<T>::True, int>::Type = 0)
		{
			what.write(name2, ioNgSerializer_);
		}

	private:

		Out(const Out&);

		Out& operator=(const Out&);

		static unsigned int modeToH5(OpenMode mode)
		{
			switch (mode) {
			case ACC_TRUNC:
				return H5F_ACC_TRUNC;
			case ACC_EXCL:
				return H5F_ACC_EXCL;
			case ACC_RDONLY:
				return H5F_ACC_RDONLY;
			case ACC_RDW:
				return H5F_ACC_RDWR;
			}

			throw RuntimeError("IoNg:: wrong open mode\n");
		}

		String filename_;
		H5::H5File* hdf5File_;
		IoNgSerializer ioNgSerializer_;
	};

	class In {

	public:

		typedef int long LongIntegerType;
		typedef unsigned int long LongSizeType;

		In(String const &fn)
		    : filename_(fn),
		      hdf5File_(new H5::H5File(fn, H5F_ACC_RDONLY)),
		      ioNgSerializer_(hdf5File_)
		{
#ifdef NDEBUG
			H5::Exception::dontPrint();
#endif
		}

		~In()
		{
			delete hdf5File_;
			hdf5File_ = 0;
			filename_ = "";
		}

		const String& filename() const { return filename_; }

		void open(String const &fn)
		{
			if (hdf5File_) delete hdf5File_;

			filename_ = fn;
			// deal with mode
			hdf5File_ = new H5::H5File(fn, H5F_ACC_RDONLY);
		}

		void close()
		{
			filename_ = "";
			hdf5File_->close();
		}

		bool ng() const { return true; }

		template<typename SomeType>
		void readLastVectorEntry(SomeType &x, String s)
		{
			int total = 0;
			ioNgSerializer_.read(total, s + "/Size");

			if (total <= 0)
				throw RuntimeError("Error reading last instance of " + s + "\n");

			ioNgSerializer_.read(x, s + "/" + ttos(--total));
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

	private:

		In(const In&);

		In& operator=(const In&);

		String filename_;
		H5::H5File* hdf5File_;
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

