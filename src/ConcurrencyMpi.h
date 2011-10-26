// BEGIN LICENSE BLOCK
/*
Copyright (c) 2009, UT-Battelle, LLC
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
// END LICENSE BLOCK
/** \ingroup PsimagLite */
/*@{*/

/*! \file ConcurrencyMpi.h
 *
 * Implements the Concurrency.h interface for MPI parallelization mode
 * Implements the Concurrency.h interface for MPI parallelization mode,
 * supports load-balancing
 */
#ifndef CONCURRENCY_MPI_HEADER_H
#define CONCURRENCY_MPI_HEADER_H
#include "ConcurrencyMpiFunctions.h"
#include "Concurrency.h"
#include "TypeToString.h"

namespace PsimagLite {
	template<typename FieldType>
	class ConcurrencyMpi : public Concurrency<FieldType> {

	public:

		typedef MPI_Comm CommType;
		typedef std::pair<CommType,CommType> CommPairType;

		static const CommType  COMM_WORLD;

		ConcurrencyMpi(int argc, char *argv[])
		{
			MPI_Init(&argc,&argv);
		}

		std::string name() const { return "mpi"; }

		~ConcurrencyMpi()
		{
			MPI_Finalize();
			for (size_t i=0;i<garbage_.size();i++)
				delete garbage_[i];
		}

		int nprocs(CommType mpiComm=COMM_WORLD) 
		{ 
			int tmp;
			MPI_Comm_size(mpiComm,&tmp);
			return tmp;
		}

		int rank(CommType mpiComm=COMM_WORLD) 
		{
			int tmp;
			MPI_Comm_rank(mpiComm,&tmp);
			return tmp; 
		}

		bool root(CommType mpiComm=COMM_WORLD) 
		{
			if (rank(mpiComm)==0) return true;
			return false;
		}
		
		CommPairType newCommFromSegments(size_t numberOfSegments,CommType mpiComm=COMM_WORLD)
		{
			size_t procs = nprocs(mpiComm);
			if (procs%numberOfSegments !=0) {
				std::string s("Segment size must be a divisor of nprocs ");
				s += std::string("__FUNCTION__") + __FILE__+" : " + ttos(__LINE__);
				throw std::runtime_error(s.c_str());
			}
			/* Extract the original group handle */ 
			MPI_Group origGroup;
			MPI_Comm_group(mpiComm, &origGroup); 
			
			/* Divide tasks into procs/x distinct groups based upon rank */ 
			size_t segmentSize = size_t(procs/numberOfSegments);
			size_t r = rank(mpiComm);
			std::vector<int> rv;

			getSegmentsDirect(rv,numberOfSegments,segmentSize,r);
			CommType comm1 = getCommFromSegments(origGroup,rv);
			
			getSegmentsAdjuct(rv,numberOfSegments,segmentSize,r);
			CommType comm2 =getCommFromSegments(origGroup,rv);

			return CommPairType(comm1,comm2);
		}

		void reduce(double& v)
		{
			double w = 0;
			int x = MPI_Reduce(&v,&w,1,MPI_DOUBLE,MPI_SUM,0,COMM_WORLD);
			if (x!=MPI_SUCCESS) {
				std::string s = "ConcurrencyMpi: reduce(Vector) failed\n";
				throw std::runtime_error(s.c_str());
			}
			if (rank()==0) v = w;
		}

		void reduce(std::vector<double>& v,CommType mpiComm=COMM_WORLD)
		{
			std::vector<double> w(v.size());
			
			int x = MPI_Reduce(&(v[0]),&(w[0]),v.size(),
			                     MPI_DOUBLE,MPI_SUM,0,mpiComm);
			if (x!=MPI_SUCCESS) {
				std::string s = "ConcurrencyMpi: reduce(Vector) failed\n";
				throw std::runtime_error(s.c_str());
			}
			
			if (root(mpiComm)) v = w;
		}

		void reduce(std::vector<std::complex<double> >& v)
		{
			std::vector<std::complex<double> > w(v.size());
			
			int x = MPI_Reduce(&(v[0]),&(w[0]),2*v.size(),
			                     MPI_DOUBLE,MPI_SUM,0,COMM_WORLD);
			if (x!=MPI_SUCCESS) {
				std::string s = "ConcurrencyMpi: reduce(Vector) failed\n";
				throw std::runtime_error(s.c_str());
			}
			
			if (rank()==0) v = w;
		}

		void reduce(PsimagLite::Matrix<double>& m)
		{
			PsimagLite::Matrix<double> w(m.n_row(),m.n_col());
			int n = m.n_row()*m.n_col();
			int x = MPI_Reduce(&(m(0,0)),&(w(0,0)),n,MPI_DOUBLE,MPI_SUM,0,COMM_WORLD);
			if (x!=MPI_SUCCESS) {
				std::string s = "ConcurrencyMpi: reduce(Matrix) failed\n";
				throw std::runtime_error(s.c_str());
			}
			if (rank()==0) m = w;
		}
		
		void gather(std::vector<PsimagLite::Matrix<double> > &v,CommType mpiComm=COMM_WORLD)
		{
			std::string s = "You hit an unimplemented function.\n";
			s += "Contribute to PsimagLite development and make a difference!\n";
			s += "Implement this function!\n";
			s += ttos(__FUNCTION__) + __FILE__ + " : " + ttos(__LINE__) + "\n"; 
			throw std::runtime_error(s.c_str());
		}

// 		void gather(std::vector<std::vector<std::complex<double> > > &v,CommType mpiComm=COMM_WORLD) 
// 		{
// 			int i,x;
// 			std::vector<std::complex<double> > tmpVec;
// 			MPI_Status status;
// 			int tag=999;
// 
// 			if (!assigned_) return;
// 
// 			if (total_>v.size()  || myIndices_.size()<=0) { 
// 				std::cerr<<"total_="<<total_<<" v.size()="<<v.size()<<" myindices.size="<<myIndices_.size()<<" line="<<__LINE__<<"\n";
// 				throw std::runtime_error("ConcurrencyMpi::gather() loopCreate() must be called before.\n"); 
// 			}
// 			if (rank()>0) {
// 				for (step_=0;step_<myIndices_.size();step_++) {
// 					i=myIndices_[step_];
// 					if (i>=total_) break;
// 					x=v[i].size();
// 					MPI_Send(&x,1,MPI_INTEGER,0,i,mpiComm);
// 					MPI_Send(&(v[i][0]),2*x,MPI_DOUBLE,0,i,mpiComm);
// 				
// 				}
// 			} else {
// 				int nprocs1 = nprocs();
// 				for (int r=1;r<nprocs1;r++) {
// 					for (step_=0;step_<indicesOfThisProc_[r].size();step_++) {
// 						i = indicesOfThisProc_[r][step_];
// 						if (i>=total_) continue;
// 						
// 						MPI_Recv(&x,1,MPI_INTEGER,r,i,mpiComm,&status);
// 						tmpVec.resize(x);
// 						MPI_Recv(&(tmpVec[0]),2*x,MPI_DOUBLE,r,i,mpiComm,&status);
// 						v[i]=tmpVec;
// 					}
// 				}
// 			}
// 			step_= -1;
// 		}
// 
// 		void gather(std::vector<std::vector<double> > &v,CommType mpiComm=COMM_WORLD) 
// 		{
// 			int x;
// 			size_t i;
// 			std::vector<double> tmpVec;
// 			MPI_Status status;
// 
// 			if (!assigned_) return;
// 
// 			if (total_>v.size() ||  myIndices_.size()<=0) { 
// 				std::cerr<<"total_="<<total_<<" v.size()="<<v.size()<<" myindices.size="<<myIndices_.size()<<" line="<<__LINE__<<"\n";
// 				throw std::runtime_error("ConcurrencyMpi::gather() loopCreate() must be called before.\n"); 
// 			}
// 			if (rank()>0) {
// 				for (step_=0;step_<int(myIndices_.size());step_++) {
// 					i=myIndices_[step_];
// 					if (i>=total_) break;
// 					x=v[i].size();
// 					MPI_Send(&x,1,MPI_INTEGER,0,i,mpiComm);
// 					MPI_Send(&(v[i][0]),x,MPI_DOUBLE,0,i,mpiComm);
// 				}
// 			} else {
// 				for (int r=1;r<nprocs();r++) {
// 					for (step_=0;step_<int(indicesOfThisProc_[r].size());step_++) {
// 						i=indicesOfThisProc_[r][step_];
// 						if (i>=total_) continue;
// 						
// 						MPI_Recv(&x,1,MPI_INTEGER,r,i,mpiComm,&status);
// 						tmpVec.resize(x);
// 						MPI_Recv(&(tmpVec[0]),x,MPI_DOUBLE,r,i,mpiComm,&status);
// 						v[i]=tmpVec;
// 					}
// 				}
// 			}
// 			step_= -1;
// 		}
// 
// 		template<typename T>
// 		void gather(std::vector<T> &v,CommType mpiComm=COMM_WORLD) 
// 		{
// 			size_t i;
// 
// 			if (!assigned_) return;
// 
// 			if (total_>v.size() ||  myIndices_.size()<=0) { 
// 				std::cerr<<"total_="<<total_<<" v.size()="<<v.size()<<" myIndices_.size()="<<myIndices_.size()<<"\n";
// 				throw std::runtime_error("ConcurrencyMpi::broadcast() loopCreate() must be called before.\n"); 
// 			}
// 
// 			if (rank()>0) {
// 				for (step_=0;step_<int(myIndices_.size());step_++) {
// 					i=myIndices_[step_];
// 					if (i>=total_) break;
// 					MpiSend(&(v[i]),rank(),i);
// 				
// 				}
// 			} else {
// 				for (int r=1;r<nprocs();r++) {
// 					for (step_=0;step_<int(indicesOfThisProc_[r].size());step_++) {
// 						i=indicesOfThisProc_[r][step_];
// 						if (i>=total_) continue;
// 						MpiRecv(&(v[i]),r,i);
// 					}
// 				}
// 			}
// 			step_= -1;
// 		}
// 
// 		template<typename T>
// 		void gather(std::vector<PsimagLite::Matrix<T> > &v,CommType mpiComm=COMM_WORLD) 
// 		{
// 			size_t i;
// 
// 			if (!assigned_) return;
// 
// 			if (total_>v.size() ||  myIndices_.size()<=0) { 
// 				std::cerr<<"total_="<<total_<<" v.size()="<<v.size()<<" myIndices_.size()="<<myIndices_.size()<<"\n";
// 				throw std::runtime_error("ConcurrencyMpi::broadcast() loopCreate() must be called before.\n"); 
// 			}
// 			int r1 = rank();
// 			if (r1>0) {
// 				for (step_=0;step_<int(myIndices_.size());step_++) {
// 					i=myIndices_[step_];
// 					if (i>=total_) break;
// 					int nrow = v[i].n_row();
// 					int ncol = v[i].n_col();
// 					MpiSend(&nrow,r1,i);
// 					MpiSend(&ncol,r1,i);
// 					MpiSend(&(v[i]),r1,i);
// 				
// 				}
// 			} else {
// 				for (int r=1;r<nprocs();r++) {
// 					for (step_=0;step_<int(indicesOfThisProc_[r].size());step_++) {
// 						i=indicesOfThisProc_[r][step_];
// 						if (i>=total_) continue;
// 						int nrow,ncol;
// 						MpiRecv(&nrow,r,i);
// 						MpiRecv(&ncol,r,i);
// 						//std::cerr<<"r="<<r<<" nrow="<<nrow<<" ncol="<<ncol<<" i="<<i<<"\n";
// 						v[i].resize(nrow,ncol);
// 						MpiRecv(&(v[i]),r,i);
// 					}
// 				}
// 			}
// 			step_= -1;
// 		}
// 
// 		template<typename T>
// 		void gather(std::vector<T*> &v,CommType mpiComm=COMM_WORLD) 
// 		{
// 			size_t i;
// 
// 			if (!assigned_) return;
// 
// 			if (total_>v.size() ||  myIndices_.size()<=0) { 
// 				std::cerr<<"total_="<<total_<<" v.size()="<<v.size()<<" myIndices_.size()="<<myIndices_.size()<<"\n";
// 				throw std::runtime_error("ConcurrencyMpi::broadcast() loopCreate() must be called before.\n"); 
// 			}
// 
// 			int r1 = rank();
// 			if (r1>0) {
// 				for (step_=0;step_<int(myIndices_.size());step_++) {
// 					i=myIndices_[step_];
// 					if (i>=total_) break;
// 					MpiSend(v[i],r1,i);
// 				
// 				}
// 			} else {
// 				for (int r=1;r<nprocs();r++) {
// 					for (step_=0;step_<int(indicesOfThisProc_[r].size());step_++) {
// 						i=indicesOfThisProc_[r][step_];
// 						if (i>=total_) continue;
// 						MpiRecv(v[i],r,i);
// 					}
// 				}
// 			}
// 			step_= -1;
// 		}

		template<typename T>
		void broadcast(std::vector<std::vector<T> > &v,CommType mpiComm=COMM_WORLD) 
		{ 
			for (size_t i=0;i<v.size();i++) MpiBroadcast(&(v[i]),0);
		}

		template<typename DataType>
		void broadcast(std::vector<DataType> &v,CommType mpiComm=COMM_WORLD) 
		{ 
			for (size_t i=0;i<v.size();i++) MpiBroadcast(&(v[i]),0);
		}
		
		template<typename DataType>
		void broadcast(std::vector<DataType*> &v,CommType mpiComm=COMM_WORLD) 
		{ 
			for (size_t i=0;i<v.size();i++) MpiBroadcast(v[i],0);
		}

		void barrier(CommType mpiComm=COMM_WORLD)
		{
			MPI_Barrier(mpiComm);
		
		}

	private:
		std::vector<CommType*> garbage_;

		void MpiGather(std::vector<double> &vrec,double vsend,int iproc,CommType mpiComm)
		{
			MPI_Gather(&vsend,1,MPI_DOUBLE,&(vrec[0]),1,MPI_DOUBLE,iproc,mpiComm);
		}

		void MpiGather(std::vector<FieldType> &vrec,FieldType &vsend,int iproc,CommType mpiComm)
		{
			MPI_Gather(&vsend,2,MPI_DOUBLE,&(vrec[0]),2,MPI_DOUBLE,iproc,mpiComm);
		}

		void MpiGather(std::vector<std::vector<FieldType> > &vrec,std::vector<FieldType> &vsend,int iproc,CommType mpiComm)
		{
			int x = vsend.size();
			MPI_Gather(&(vsend[0]),2*x,MPI_DOUBLE,&(vrec[0][0]),2*x,MPI_DOUBLE,iproc,mpiComm);
		}

		void getSegmentsDirect(std::vector<int>& rv,size_t numberOfSegments,size_t segmentSize,size_t r)
		{
			std::vector<std::vector<int> > ranks;
			size_t thisSegment = 0;
			for (size_t i=0;i<segmentSize;i++) {
				std::vector<int> tmp(numberOfSegments);
				for (size_t j=0;j<numberOfSegments;j++) {
					tmp[j] = j*segmentSize+i;
					if (r==size_t(tmp[j])) thisSegment=i;
				}
				ranks.push_back(tmp);
			}
			rv = ranks[thisSegment];
		}

		void getSegmentsAdjuct(std::vector<int>& rv,size_t numberOfSegments,size_t segmentSize,size_t r)
		{
			
			std::vector<std::vector<int> > ranks;
			size_t thisSegment = 0;
			for (size_t i=0;i<numberOfSegments;i++) {
				std::vector<int> tmp;
				size_t start = i*segmentSize;
				size_t end = (i+1)*segmentSize;
				if (r>=start && r<end) thisSegment = i;
				for (size_t j=start;j<end;j++) tmp.push_back(j);
				ranks.push_back(tmp);
			}
			rv = ranks[thisSegment];
		}

		CommType getCommFromSegments(MPI_Group origGroup,std::vector<int>& rv)
		{
			MPI_Group newGroup;
			int status = MPI_Group_incl(origGroup,rv.size(),&(rv[0]),&newGroup);
			if (status!=MPI_SUCCESS) throw std::runtime_error("getCommFromSegments\n");
			CommType* newComm = new CommType;
			garbage_.push_back(newComm);
			MPI_Comm_create(COMM_WORLD, newGroup, newComm);
			return *newComm;
		}
	}; // class ConcurrencyMpi

	template<typename FieldType>
	const typename ConcurrencyMpi<FieldType>::CommType
	ConcurrencyMpi<FieldType>::COMM_WORLD = MPI_COMM_WORLD;
	
} // namespace Dmrg

/*@}*/
#endif
