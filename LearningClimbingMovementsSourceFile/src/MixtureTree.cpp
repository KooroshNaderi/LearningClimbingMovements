#include "MixtureTree.h"
#include <Eigen/Eigen> 
#include <Eigen/SparseCore> 
#include "Debug.h"
#include <limits>
#include "ClippedGaussianSampling.h"

using namespace Eigen;
namespace AaltoGames
{
	enum PruningCriterion
	{
		pcMarginalProbability,pcInformation,pcErrorSum
	};
	static const PruningCriterion pruningCriterion=pcErrorSum;
	static const float multiplicativeNoise=0.0f;
	static const float dropOutRate=0.0f;
	static const float logScaleThreshold=-10.0f;
	static const bool useVariances=true;
	static const bool sharedVariances=false;
	static const bool scalarVariances=false;
	static const float varianceMultiplier=0.5f;
	static const bool useGating=false;
	static const float minLogInit=-1e10;
	static const bool superSparse=false;  //with discrete inputs, replace dot products with scalar products
	static const float lidstoneSmoothing=-1e4;
	static const bool useThreads=true;


	static void colWiseMakePdf( MixtureTree::MatrixX &m )
	{
		for (int i=0; i<m.cols(); i++)
		{
			m.col(i)/=m.col(i).sum();
			if (!m.col(i).allFinite())
				m.col(i).setConstant(1.0f/(MixtureTree::real)(m.rows()));
		}
	}
	static void columnPdfMakeSparse( int c, MixtureTree::MatrixX &m,MixtureTree::real sparsityThreshold )
	{
		m.col(c)=m.col(c).unaryExpr([sparsityThreshold](MixtureTree::real d){return d<sparsityThreshold ? 0 : d;}); 
		//for (int r=0; r<m.rows(); r++)
		//{
		//	if (m(r,c)<sparsityThreshold)
		//	{
		//		m(r,c)=0;
		//	}
		//}
		m.col(c)/=m.col(c).sum();
	}
	//static void colwisePdfMakeSparse( MixtureTree::MatrixX &m,float sparsityThreshold )
	//{
	//	for (int c=0; c<m.cols(); c++)
	//	{
	//		columnPdfMakeSparse(c,m,sparsityThreshold);
	//	}
	//}

	static void columnLogToPdf( int colIdx, MixtureTree::real minThreshold,MixtureTree::MatrixX &m)
	{
		int i=colIdx;
		//Use the trick from Deep Learning book: the softmax function can be stabilized by first
		//subtracting the max element from all elements. This does not affect the result but ensures no inf or -inf.
		MixtureTree::real maxVal=m.col(i).maxCoeff();
		m.col(i)=(m.col(i).array()-maxVal);
		m.col(i)=m.col(i).array().exp();
		m.col(i)/=m.col(i).sum();

		//MixtureTree::VectorX::Index k;
		//MixtureTree::real maxVal=m.col(i).maxCoeff(&k);
		//const MixtureTree::real maxThreshold=logf(1e20f);
		//if (maxVal>maxThreshold)
		//	m.col(i)=(maxThreshold/maxVal)*m.col(i);	//scale to prevent overflows (possible in Bernoulli mixtures)
		//else if (maxVal<minThreshold)
		//	m.col(i)=(minThreshold/maxVal)*m.col(i);	//scale to prevent underflows
		//m.col(i)=m.col(i).array().exp();
		//m.col(i)/=m.col(i).sum();
	}
	static int sampleFromVector( const MixtureTree::VectorX &v )
	{
		double total=v.sum();
		if (total==0)
		{
			return randInt(0,v.rows()-1);
		}
		double acc=0;
		int idx=0;
		total*=random();
		while (acc<total && idx<v.rows())
		{
			acc+=v[idx];
			idx++;
		}
		return max(0,idx-1);
	}

	//static void colwiseLogToPdf( MixtureTree::MatrixX &m)
	//{
	//	for (int i=0; i<m.cols(); i++)
	//	{
	//		columnLogToPdf(i,m);
	//	}
	//}

	MixtureTree::MixtureTree()
	{
		nInputs=0;
		parent=NULL;
		root=this;
		depth=0;
		nMeans=0;
		nTotalInputVariables=0;
		convolutionMode=false;
		convStride=1;
		manager=NULL;
		useMarginalProbabilities=true;
	}

	

	MixtureTree::~MixtureTree()
	{
		delete manager;
		if (children.size()!=0)
		{
			for (int i=0; i<nChildrenPerNode; i++)
			{
				delete children[i];
			}
			children.clear();
		}
	}

	int MixtureTree::init( int nChildrenPerNode,int maxDepth/*=1*/, float sparsityThreshold, int nThreads)
	{
		if (nThreads>0 && useThreads)
			manager=new WorkerThreadManager<MixtureTree *>(nThreads);
		nThreads=nThreads;
		this->nChildrenPerNode=nChildrenPerNode;
		this->maxDepth=maxDepth;
		this->sparsityThreshold=sparsityThreshold;
		marginalProbabilityAcc=VectorX::Zero(nChildrenPerNode);
		marginalProbabilities=VectorX::Zero(nChildrenPerNode);
		logMarginalProbabilities=VectorX::Constant(nChildrenPerNode,1e-4f);
		nInitializedUnits=nChildrenPerNode;  //incremental training not yet implemented
		if (this==root)
		{
			int nLeaves=1;
			int nNodes=1;
			for (int d=1; d<maxDepth; d++)
			{
				nLeaves*=nChildrenPerNode;
				nNodes+=nLeaves;
			}
			leaves.reserve(nLeaves);
			leaves.clear();
			allNodes.reserve(nNodes);
			allNodes.clear();
			nMeans=nLeaves*nChildrenPerNode;
			prunes=std::vector<int>(nMeans,-1);
		}
		if (depth<maxDepth-1)
		{
			if (children.size()!=0)
			{
				for (int i=0; i<nChildrenPerNode; i++)
				{
					delete children[i];
				}
				children.clear();
			}
			children.resize(nChildrenPerNode);
			for (int i=0; i<nChildrenPerNode; i++)
			{
				children[i]=new MixtureTree();
				children[i]->parent=this;
				children[i]->root=root;
				children[i]->depth=depth+1;
				children[i]->init(nChildrenPerNode,maxDepth,sparsityThreshold);
			}
		}
		else 
		{
			leafBaseIdx=nChildrenPerNode*(int)root->leaves.size();
			root->leaves.push_back(this);
		}
		root->allNodes.push_back(this);
		nInitializedComponents=nMeans; //TODO: implement incremental init, set this initially to zero

		return nMeans;
	}

	void MixtureTree::addInput( int nVariables, InputTypes type, float kernelWidth/*=1.0f*/, const real *minBounds/*=NULL*/, const real *maxBounds/*=NULL*/ )
	{
		if (convolutionMode)
			Debug::throwError("Combining convolutional and non-convolutional inputs not supported");
		inputVectorLengths.push_back(nVariables);
		nTotalInputVariables+=nVariables;
		inputTypes.push_back(type);
		kernelWidths.push_back(kernelWidth);
		W.push_back(MatrixX(nVariables,nChildrenPerNode));
		Wsum.push_back(MatrixX(nVariables,nChildrenPerNode));
		if (useVariances)
		{
			WsumSq.push_back(MatrixX(nVariables,nChildrenPerNode));
			Wsd.push_back(MatrixX(nVariables,nChildrenPerNode));
			WvarLog.push_back(MatrixX(nVariables,nChildrenPerNode));
		}
		logW.push_back(MatrixX(nVariables,nChildrenPerNode));
		inputLogMemberships.push_back(MatrixX(nChildrenPerNode,1));

		//init with random values. Various accumulators set to zero so that first incoming training samples will fully replace the initial values
		real weightPerSample=((real)1.0)/(real)nChildrenPerNode;
		for (int k=0; k<nChildrenPerNode; k++)
		{
			marginalProbabilityAcc[k]=0;
			marginalProbabilities[k]=weightPerSample;
			for (int j=0; j<inputVectorLengths[nInputs]; j++)
			{
				if (type==Continuous)
					W[nInputs](j,k)=minBounds[j]+randomf()*(maxBounds[j]-minBounds[j]);
				else
					W[nInputs](j,k)=randomf();
				logW[nInputs](j,k)=max(lidstoneSmoothing,log(W[nInputs](j,k)));
				Wsum[nInputs](j,k)=0;
				if (type==Continuous  && useVariances)
				{
					WsumSq[nInputs](j,k)=0;
					Wsd[nInputs](j,k)=kernelWidth;
					WvarLog[nInputs](j,k)=log(squared(kernelWidth));
				}
			}
		}
		if (inputTypes[nInputs]==Discrete)
			colWiseMakePdf(W[nInputs]);
		for (MixtureTree *child : children)
		{
			child->addInput(nVariables,type,kernelWidth,minBounds,maxBounds);
		}
		//init parent based on children
		if (children.size()!=0)
		{
			for (int i=0; i<nChildrenPerNode; i++)
			{
				W[nInputs].col(i)=children[i]->W[nInputs].rowwise().mean();
				Wsum[nInputs].col(i).setZero();
				marginalProbabilityAcc[i]=0;
				if (type==Continuous  && useVariances)
				{
					//weighed variance: var(x)=sum(w/w_tot*(x-mean)^2)=sum(w/w_tot*x^2-2*w/w_tot*x*mean + w/w_tot*mean^2)
					//=sum(w/w_tot*x^2)-2*mean*sum(w/w_tot*x) + mean^2*sum(w/w_tot)
					//because sum(w/w_tot*x)=mean =>
					//var(x)=sum(w*x^2)/w_tot - mean^2
					WsumSq[nInputs].col(i).setZero();
					Wsd[nInputs].col(i).setConstant(kernelWidth);
					WvarLog[nInputs].col(i).setConstant(log(squared(kernelWidth)));
				}
			}
		}
		nInputs++;
	}
	void MixtureTree::addConvolutionalInput(int nVarsPerPixel, int nRows, int nCols, int kernelRows, int kernelCols, int stride, InputTypes type, float minSd, const real *minBounds, const real *maxBounds)
	{
		if (maxDepth>1)
			Debug::throwError("Convolutional inputs not supported with tree-based models!");
		if (nInputs!=0)
		{
			Debug::throwError("Only 1 input supported in convolutional mode!");
		}
		kernelRows=min(kernelRows,nRows);
		kernelCols=min(kernelCols,nCols);
		for (int r=0; r<kernelRows; r++)
		{
			for (int c=0; c<kernelCols; c++)
			{
				addInput(nVarsPerPixel,type,minSd,minBounds,maxBounds);
				Wtranspose.push_back(MatrixX(nVarsPerPixel,nChildrenPerNode));
				Wtranspose[nInputs-1]=W[nInputs-1].transpose();
			}
		}
		convolutionMode=true;

		convInRows=nRows;
		convInCols=nCols;
		convKernelRows=kernelRows;
		convKernelCols=kernelCols;
		convStride=stride;
		convOutRows=convInRows/convStride;
		convOutCols=convInCols/convStride;
	}

	void MixtureTree::computeSampleMemberships( int sampleIdx, bool trainingMode, const MatrixX *samples[], const MatrixX *mask[])
	{
		visited[sampleIdx]=true; 
		//The membership probabilities for all inputs are first computed in log-domain and added together for more numerical precision
		//(The squared distances from unit means overflow less likely than exp(-sqdist) underflows.)
		for (int inputIdx=0; inputIdx<nInputs; inputIdx++)
		{
			MatrixX::ConstColXpr sample=samples[inputIdx]->col(sampleIdx);
			real maskMean=1.0f;
			if (mask!=NULL)
				maskMean=mask[inputIdx]->col(sampleIdx).mean();  //NULL mask corresponds to a mask matrix of all ones
			if (maskMean==0)
			{
				inputLogMemberships[inputIdx].col(sampleIdx).setZero();
				continue;
			}
			if (inputTypes[inputIdx]==Continuous)
			{
				//Key to the algorithm performing is that we handle two cases that may fail due to numerical imprecision:
				//1. Even if the exponentiation of distances (or log probabilities) leads to underflow, the closest 
				//   units should have larger activations
				//2. Despite the numerical scale, if all activations are equal, the resulting normalized pdf should be uniform
				//   (this is important especially at initialization when units have not learned). The dropout mechanism will anyway 
				//   result in non-uniform adaptation.
				if (maskMean>0.999f)
				{
					//if (children.size()!=0)
					//if (nChildrenToVisit!=1)  //variances only used during inference, or in middle tree layers 
					if (useVariances && (!trainingMode))// || children.size()!=0))
					{
						for (int k=0; k<nInitializedUnits; k++)
						{
							//real dist=(sample-W[inputIdx].col(k)).squaredNorm()/squared(kernelWidths[inputIdx]);
							real dist=-0.5f*((sample-W[inputIdx].col(k)).cwiseQuotient(Wsd[inputIdx].col(k))).squaredNorm();
							//pdf scaling for diagonal Gaussian = 1/(2*pi*product(variances))^0.5
							//log scale=-0.5f*(2+PI+sum(log(variances))
							dist+=-0.5f*(logf(2.0f*PI)+WvarLog[inputIdx].col(k).sum());
							inputLogMemberships[inputIdx](k,sampleIdx)=dist;
						}
					}
					else
					{
						for (int k=0; k<nInitializedUnits; k++)
						{
							real dist=-0.5f*(sample-W[inputIdx].col(k)).squaredNorm()/squared(kernelWidths[inputIdx]);
							inputLogMemberships[inputIdx](k,sampleIdx)=dist;
						}
					}
				}
				else
				{
					//if (children.size()!=0)
					//if (nChildrenToVisit!=1)  //variances only used during inference
					if (useVariances && !trainingMode)
					{
						for (int k=0; k<nInitializedUnits; k++)
						{
							//real dist=((sample-W[inputIdx].col(k)).array()*mask[inputIdx].col(sampleIdx).array()).matrix().squaredNorm()/squared(kernelWidths[inputIdx]);
							real dist=-0.5f*((sample-W[inputIdx].col(k)).cwiseQuotient(Wsd[inputIdx].col(k))).cwiseProduct(mask[inputIdx]->col(sampleIdx)).squaredNorm();
							//pdf scaling for diagonal Gaussian = 1/(2*pi*product(variances))^0.5
							//log scale=-0.5f*(2+PI+sum(log(variances))
							dist+=-0.5f*(logf(2.0f*PI)+WvarLog[inputIdx].col(k).cwiseProduct(mask[inputIdx]->col(sampleIdx)).sum());
							inputLogMemberships[inputIdx](k,sampleIdx)=dist;
						}
					}
					else
					{
						for (int k=0; k<nInitializedUnits; k++)
						{
							real dist=-0.5f*((sample-W[inputIdx].col(k)).cwiseProduct(mask[inputIdx]->col(sampleIdx))).squaredNorm()/squared(kernelWidths[inputIdx]);
							inputLogMemberships[inputIdx](k,sampleIdx)=dist;
						}
					}
				}

			}
			else if (inputTypes[inputIdx]==Bernoulli)//Bernoulli and discrete pdf handled as dot products. Note however that logW computing differs 
			{
				for (int k=0; k<nInitializedUnits; k++)
				{
					//see the logW computing above
					real dotProduct;
					if (maskMean>0.999f)
						dotProduct=-logW[inputIdx].col(k).dot(sample);
					else
						dotProduct=-logW[inputIdx].col(k).dot(sample.cwiseProduct(mask[inputIdx]->col(sampleIdx))); 
					inputLogMemberships[inputIdx].col(sampleIdx)[k]=-dotProduct;
				}
			}
			else  //discrete pdf
			{
				if (superSparse)// && trainingMode)
				{
					VectorX::Index maxLoc=sampleFromVector(sample);
					//VectorX::Index maxLoc;
					sample.maxCoeff(&maxLoc);
					real maxVal=sample[maxLoc];
					for (int k=0; k<nInitializedUnits; k++)
					{
						real dotProduct;
						if (maskMean>0.999f)
							dotProduct=maxVal*W[inputIdx](maxLoc,k);
						else
						{
							real maskVal=mask[inputIdx]->col(sampleIdx)[maxLoc];
							dotProduct=maxVal*maskVal*W[inputIdx](maxLoc,k);
						}
						inputLogMemberships[inputIdx].col(sampleIdx)[k]=max(lidstoneSmoothing,log(dotProduct));
					}
				}
				else
				{
					const bool mutuallyExclusive=true;
					if (mutuallyExclusive)
					{
						for (int k=0; k<nInitializedUnits; k++)
						{
							real dotProduct;
							if (maskMean>0.999f)
								dotProduct=W[inputIdx].col(k).dot(sample);
							else
								dotProduct=W[inputIdx].col(k).dot(sample.cwiseProduct(mask[inputIdx]->col(sampleIdx))); 
							inputLogMemberships[inputIdx].col(sampleIdx)[k]=max(lidstoneSmoothing,log(dotProduct));
						}
					}
					else
					{
						for (int k=0; k<nInitializedUnits; k++)
						{
							real dotProduct;
							if (maskMean>0.999f)
								dotProduct=logW[inputIdx].col(k).dot(sample);
							else
								dotProduct=logW[inputIdx].col(k).dot(sample.cwiseProduct(mask[inputIdx]->col(sampleIdx))); 
							inputLogMemberships[inputIdx].col(sampleIdx)[k]=dotProduct;
						}
						//real uniformDensityValue=1.0f/(float)inputVectorLengths[inputIdx];
						//for (int k=0; k<nInitializedUnits; k++)
						//{
						//	real dotProduct;
						//	if (maskMean>0.999f)
						//		dotProduct=logW[inputIdx].col(k).dot(sample);
						//	else
						//	{
						//		//in effect, we treat the input as a confidence-weighted mixture of a uniform density and the input samples,
						//		//i.e., we want to compute logW[inputIdx].col(k).dot(sample.cwiseProduct(conf)+uniform.cwiseProduct(1-conf));
						//		dotProduct=logW[inputIdx].col(k).dot(sample.cwiseProduct(mask[inputIdx]->col(sampleIdx))); 
						//		dotProduct+=uniformDensityValue*(logW[inputIdx].col(k).sum()-logW[inputIdx].col(k).dot(mask[inputIdx]->col(sampleIdx))); //same as logW[inputIdx].col(k).dot(VectorX::Ones(inputVectorLengths[inputIdx])-mask);  
						//	}
						//	inputLogMemberships[inputIdx].col(sampleIdx)[k]=dotProduct;//std::max(lidstoneSmoothing,log(dotProduct));
						//}
					}

				}
			}
		}   //for each input

		//sum input log probabilities together
		logMemberships.col(sampleIdx).setZero();
		for (int inputIdx=0; inputIdx<nInputs; inputIdx++)
		{
			logMemberships.col(sampleIdx)+=inputLogMemberships[inputIdx].col(sampleIdx);
		}

		//Add marginal probability in inference mode
		if (!trainingMode && useMarginalProbabilities)
			logMemberships.col(sampleIdx)+=logMarginalProbabilities;

		if (children.size()!=0)
		{
			VectorX::Index bestChild;
			real bestLog=logMemberships.col(sampleIdx).maxCoeff(&bestChild);
			if (trainingMode)
			{
				children[bestChild]->computeSampleMemberships(sampleIdx,trainingMode,samples,mask);
			}
			else
			{
				for (int childIdx=0; childIdx<nChildrenPerNode; childIdx++)
				{
					real logm=logMemberships.col(sampleIdx)[childIdx];
					//if ((childIdx==bestChild) || (bestLog - logm < logMultiplier - log(sparsityThreshold)))  //the unit's membership > sparsityThreshold * best unit membership
					if ((childIdx==bestChild) || (logm > bestLog + log(sparsityThreshold)))  //the unit's membership > sparsityThreshold * best unit membership
						//children[childIdx]->computeSampleMemberships(sampleIdx,trainingMode,samples,mask,logm + logMultiplier);
						children[childIdx]->computeSampleMemberships(sampleIdx,trainingMode,samples,mask);//,logMultiplier);
				}
			}
		}
		//if (children.size()!=0)
		//{
		//	if (useGating)
		//	{
		//		//if (nChildrenToVisit>1)
		//		//{
		//		//	//Convert to pdf. Note that this is temporary, only to determine which children to visit.
		//		//	//We will do this again after all leaves have been processed, so that we apply uniform scale to all 
		//		//	//memberships to avoid overflows and underflows
		//		//	memberships.col(sampleIdx)=logMemberships.col(sampleIdx);
		//		//	columnLogToPdf(sampleIdx,-5.0f,memberships); //+log(multiplier)
		//		//}
		//		int nVisited=std::min(nChildrenToVisit,(int)children.size());
		//		real bestLog=0;
		//		for (int i=0; i<nVisited; i++)
		//		{
		//			VectorX::Index bestChild;
		//			real logm=logMemberships.col(sampleIdx).maxCoeff(&bestChild);
		//			if (i==0)
		//				bestLog=logm;
		//			logMemberships(bestChild,sampleIdx)=-std::numeric_limits<real>::infinity();	//so that we will not visit this child again
		//		
		//			if (nChildrenToVisit==1)
		//			{
		//				children[bestChild]->computeSampleMemberships(sampleIdx,trainingMode,samples,mask,nChildrenToVisit,sparsityThreshold,multiplier);
		//			}
		//			else
		//			{
		//				real m=expf(-0.5f*squared((float)i)/squared(0.5f*(float)nVisited));///expf(-0.5f*fabs(logm/bestLog-1.0f));  
		//				//real m=memberships(bestChild,sampleIdx);
		//				if (i==0 || (m*multiplier>sparsityThreshold/((float)nChildrenPerNode)))
		//					children[bestChild]->computeSampleMemberships(sampleIdx,trainingMode,samples,mask,nChildrenToVisit,sparsityThreshold,m*multiplier);
		//				else 
		//					break;
		//			}
		//		}
		//	} //useGating
		//	else
		//	{
		//		int nVisited=std::min(nChildrenToVisit,(int)children.size());
		//		for (int i=0; i<nVisited; i++)
		//		{
		//			VectorX::Index bestChild;
		//			logMemberships.col(sampleIdx).maxCoeff(&bestChild);
		//			logMemberships(bestChild,sampleIdx)=-std::numeric_limits<real>::infinity();	//so that we will not visit this child again				
		//			children[bestChild]->computeSampleMemberships(sampleIdx,trainingMode,samples,mask,nChildrenToVisit,sparsityThreshold,multiplier);
		//		}
		//	}
		//} 
	}

	void MixtureTree::resizeBuffers( int nSamples )
	{
		logMemberships.resize(nChildrenPerNode,nSamples);
		logFullMemberships.resize(nChildrenPerNode*leaves.size(),nSamples);
		memberships.resize(nChildrenPerNode,nSamples);
		fullMemberships.resize(nChildrenPerNode*leaves.size(),nSamples);
		visited.resize(nSamples);
		for (int i=0; i<nInputs; i++)
		{
			inputLogMemberships[i].resize(nChildrenPerNode,nSamples);
		}
		membershipSum.resize(nSamples);
		if (children.size()!=0)
		{
			for (MixtureTree *child : children)
			{
				child->resizeBuffers(nSamples);
			}
		}
	}

	void MixtureTree::computeMemberships( const MatrixX *samples, const MatrixX *mask, bool trainingMode, bool winnerTakesAll )
	{
		const int maxInputs=256;
		const MatrixX *pSamples[maxInputs];
		const MatrixX *pMask[maxInputs];
		if (nInputs>maxInputs)
			Debug::throwError("Too many inputs!");
		for (int i=0; i<nInputs; i++)
		{
			pSamples[i]=&samples[i];
			pMask[i]=&mask[i];
		}
		computeMemberships(pSamples,pMask,trainingMode,winnerTakesAll);
	}
	void MixtureTree::computeMemberships( const MatrixX *samples[], const MatrixX *mask[], bool trainingMode, bool winnerTakesAll )
	{
		computeLogMemberships(samples,mask,trainingMode,winnerTakesAll);
		onLogMembershipsUpdated(winnerTakesAll);
	}
	enum ConvOperations
	{
		coSqDiff=0,coDotProduct
	};
	static void convProcessKernelPixel(
		const MatrixXf &inputData,
		const MatrixXf &units,	//unit params as cols
		const MatrixXf &unitsTranspose,	//unit params as rows
		const MatrixXf *unitSd,
		const MatrixXf *unitVarLog,
		const MatrixXf *mask,
		float scalarSd,
		MatrixXf &result,
		ConvOperations operation,
		int nInputRows, int nInputCols, 
		int kernelRows, int kernelCols,
		int kernelX, int kernelY,
		int stride)
	{
		if (kernelRows>nInputRows || kernelCols>nInputCols)
			Debug::throwError("Kernel too large for input!");
		int outputIdx=-1;
		int kernelRowShift=kernelRows/2;
		int kernelColShift=kernelCols/2;
		int halfStride=stride/2;
		int nInputImages=inputData.cols()/(nInputRows*nInputCols);
		kernelX-=kernelColShift;
		kernelY-=kernelRowShift;
		for (int inputImageIdx=0; inputImageIdx<nInputImages; inputImageIdx++)
		{
			int inputBaseIdx=inputImageIdx*nInputRows*nInputCols;
			for (int r=halfStride; r<nInputRows; r+=stride)
			{
				int clippedR=clipMinMaxi(r,kernelRowShift,nInputRows-kernelRows+kernelRowShift);
				for (int c=halfStride; c<nInputCols; c+=stride)
				{
					outputIdx++;
					int segmentStart=0;
					int clippedC=clipMinMaxi(c,kernelColShift,nInputCols-kernelCols+kernelColShift);
					int rr=clippedR+kernelY;
					int cc=clippedC+kernelX;
					int inputSampleIdx=inputBaseIdx+rr*nInputCols+cc;
					MatrixXf::ConstColXpr sample=inputData.col(inputSampleIdx);
					bool useMask=mask!=NULL && (mask->col(inputSampleIdx).mean()<0.999f);
					VectorXf::Index maxLoc;
					float maxVal=0;
					if (superSparse)
					{
						maxVal=sample.maxCoeff(&maxLoc);
					}
					for (int k=0; k<units.cols(); k++)
					{
						if (operation==coDotProduct)
						{
							//TODO: we should not really need a mask for discrete pdf:s?
							//if (!useMask)
							{
								if (superSparse)
									result(k,outputIdx)=max(lidstoneSmoothing,logf(maxVal*units.col(k)[maxLoc]));
								else
									result(k,outputIdx)=max(lidstoneSmoothing,logf(units.col(k).dot(sample)));
							}
							//else
							//{
							//	result(k,outputIdx)=std::max(lidstoneSmoothing,logf(units.col(k).dot(sample.cwiseProduct(mask->col(inputSampleIdx)))));
							//}
						}
						else
						{
							if (!useMask)
							{
								//if (children.size()!=0)
								//if (nChildrenToVisit!=1)  //variances only used during inference, or in middle tree layers 
								if (unitSd!=NULL)
								{
									//real dist=(sample-W[inputIdx].col(k)).squaredNorm()/squared(kernelWidths[inputIdx]);
									float dist=-0.5f*((sample-units.col(k)).cwiseQuotient(unitSd->col(k))).squaredNorm();
									//pdf scaling for diagonal Gaussian = 1/(2*pi*product(variances))^0.5
									//log scale=-0.5f*(2+PI+sum(log(variances))
									dist+=-0.5f*(logf(2.0f*PI)+unitVarLog->col(k).sum());
									result(k,outputIdx)=dist;
								}
								else
								{
									float dist=-0.5f*(sample-units.col(k)).squaredNorm();
									dist/=squared(scalarSd);
									result(k,outputIdx)=dist;
								}
							}
							else
							{
								//same as above, but with the mask
								if (unitSd!=NULL)
								{
									//real dist=((sample-W[inputIdx].col(k)).array()*mask[inputIdx].col(sampleIdx).array()).matrix().squaredNorm()/squared(kernelWidths[inputIdx]);
									float dist=-0.5f*((sample-units.col(k)).cwiseQuotient(unitSd->col(k))).cwiseProduct(mask->col(inputSampleIdx)).squaredNorm();
									//pdf scaling for diagonal Gaussian = 1/(2*pi*product(variances))^0.5
									//log scale=-0.5f*(2+PI+sum(log(variances))
									dist+=-0.5f*(logf(2.0f*PI)+unitVarLog->col(k).cwiseProduct(mask->col(inputSampleIdx)).sum());
									result(k,outputIdx)=dist;
								}
								else
								{
									float dist=-0.5f*((sample-units.col(k)).cwiseProduct(mask->col(inputSampleIdx))).squaredNorm()/squared(scalarSd);
									result(k,outputIdx)=dist;
								}
							}
						}
					}
				}
			}
		}
	}
	void MixtureTree::computeLogMemberships(const MatrixX *samples[], const MatrixX *mask[], bool trainingMode, bool winnerTakesAll)
	{
		int nSamples=samples[0]->cols()/(convStride*convStride);
		resizeBuffers(nSamples);
		//logFullMemberships.setConstant(-std::numeric_limits<float>::infinity());//minLogInit);
		logFullMemberships.setConstant(minLogInit);
		fullMemberships.setZero();
		for (MixtureTree * node : allNodes)
		{
			for (int i=0; i<nSamples; i++)
				node->visited[i]=false;
			node->membershipSum.setZero();
			node->logMemberships.setConstant(minLogInit);
			node->memberships.setZero();
		}
		if (convolutionMode)
		{
			//with convolutional inputs, the outer loop is over inputs, 
			//because looping over samples is more efficient on GPU (with discrete inputs, the loop corresponds to one matrix multiplication).
			//The downside of this is that we cannot support a tree structure (the tree traversing order is determined sample by sample
			//after processing all inputs for thesample)
			for (int r=0; r<convKernelRows; r++)
			{
				for (int c=0; c<convKernelCols; c++)
				{
					auto threadFunc=[r,c,trainingMode,samples,mask](MixtureTree *m)
					{
						int inputIdx=r*m->convKernelCols+c;
						ConvOperations co;
						const MatrixX *unitSd=NULL;
						const MatrixX *unitVarLog=NULL;
						if (m->inputTypes[0]==Continuous)
						{
							co=coSqDiff;
							if (useVariances && !trainingMode)
							{
								unitSd=&m->Wsd[inputIdx];
								unitVarLog=&m->WvarLog[inputIdx];
							}
						}
						else if (m->inputTypes[0]==Discrete)
							co=coDotProduct;
						else
							Debug::throwError("Unsupported input type");
						convProcessKernelPixel(*samples[0],m->W[inputIdx],m->Wtranspose[inputIdx],unitSd,unitVarLog,mask==NULL ? NULL : mask[0],
							m->kernelWidths[inputIdx],
							m->inputLogMemberships[inputIdx],co,
							m->convInRows,m->convInCols,m->convKernelRows,m->convKernelCols,c,r,m->convStride); 
					};//threadFunc
					if (manager!=NULL)
						manager->submitJob(threadFunc,this);
					else
						threadFunc(this);
				}
			}
			if (manager!=NULL)
				manager->waitAll();
			//sum the input log memberships and mark the root as visited
			logMemberships.setZero();
			for (int i=0; i<nInputs; i++)
				logMemberships+=inputLogMemberships[i];
			for (int i=0; i<nSamples; i++)
				root->visited[i]=true;	
			//Add marginal probability in inference mode
			if (!trainingMode && useMarginalProbabilities)
				logMemberships=logMemberships.colwise() + logMarginalProbabilities;
		}
		else
		{
			int batchSize=nSamples;
			if (nThreads!=0)
				batchSize=nSamples/max(1,nSamples/nThreads);
			for (int batchStart=0; batchStart<nSamples; batchStart+=batchSize)
			{
				auto threadFunc=[batchStart,batchSize,nSamples,trainingMode,samples,mask](MixtureTree *p)
				{
					int batchEnd=min(nSamples,batchStart+batchSize);
					for (int sampleIdx=batchStart; sampleIdx<batchEnd; sampleIdx++)
					{
						p->computeSampleMemberships(sampleIdx,trainingMode,samples,mask);
					}
				};
				if (manager!=NULL)
					manager->submitJob(threadFunc,this);
				else
					threadFunc(this);
			}
			if (manager!=NULL)
				manager->waitAll();
		}

		for (int sampleIdx=0; sampleIdx<nSamples; sampleIdx++)
		{
			//At this point, each leaf has a vector of log membership probabilities
			//We now concatenate the vectors to a column of logFullMemberships()
			for (size_t leafIdx=0; leafIdx<leaves.size(); leafIdx++)
			{
				if (leaves[leafIdx]->visited[sampleIdx])
				{
					logFullMemberships.col(sampleIdx).segment(leafIdx*nChildrenPerNode,nChildrenPerNode)=leaves[leafIdx]->logMemberships.col(sampleIdx);
				}
				//else
				//	Debug::throwError("Error");
			}
		}
	}

	void MixtureTree::onLogMembershipsUpdated(bool winnerTakesAll)
	{
		int nSamples=logFullMemberships.cols();
		fullMemberships=logFullMemberships;
		for (int sampleIdx=0; sampleIdx<nSamples; sampleIdx++)
		{
			if (winnerTakesAll)
			{
				VectorX::Index maxLoc;
				fullMemberships.col(sampleIdx).maxCoeff(&maxLoc);
				fullMemberships.col(sampleIdx).setZero();
				fullMemberships.col(sampleIdx)[maxLoc]=1;
			}
			else
			{
				columnLogToPdf(sampleIdx,logScaleThreshold,fullMemberships);

				//multiplicative noise
				if (multiplicativeNoise>0)
				{
					real effAmp=multiplicativeNoise/(real)nMeans;
					for (int k=0; k<nInitializedUnits; k++)
					{
						fullMemberships.col(sampleIdx)[k]*=1.0f-effAmp*0.5f+effAmp*randomf();
					}
				}

				//dropout
				if (dropOutRate>0)
				{
					for (int k=0; k<nInitializedUnits; k++)
					{
						if (randomf()<dropOutRate)	
						{
							
							fullMemberships(k,sampleIdx)=0;
						}
					}
					if (fullMemberships.col(sampleIdx).sum()<1e-20)
					{
						if (winnerTakesAll)
						{
							fullMemberships.col(sampleIdx).setZero();
							fullMemberships.col(sampleIdx)[randInt(0,nMeans-1)]=1;
						}
						else
						{
							fullMemberships.col(sampleIdx).setConstant(1.0f/(float)nMeans);
						}
					}
					else
						fullMemberships.col(sampleIdx)/=fullMemberships.col(sampleIdx).sum();
				}

				//sparsity
				columnPdfMakeSparse(sampleIdx,fullMemberships,sparsityThreshold/(float)nMeans);
				//int nNonZero=0;
				//for (int i=0; i<nMeans; i++)
				//{
				//	if (fullMemberships.col(sampleIdx)[i]!=0)
				//		nNonZero++;
				//}
				//printf("Nonzero: %d\n",nNonZero);
			}
		}		
		//copy back to leaf buffers
		for (size_t leafIdx=0; leafIdx<leaves.size(); leafIdx++)
		{
			leaves[leafIdx]->memberships=fullMemberships.middleRows(leafIdx*nChildrenPerNode,nChildrenPerNode);
			leaves[leafIdx]->membershipSum=leaves[leafIdx]->memberships.colwise().sum();
		}
		//loop over all parents (allNodes in child-first order and recursively update memberships and membership sums based on children
		for (size_t i=0; i<allNodes.size(); i++)
		{
			MixtureTree *parent=allNodes[i];
			if (parent->children.size()!=0)
			{
				for (int k=0; k<nChildrenPerNode; k++)
				{
					//parent's unit k membership for each sample = sum of memberships of child k
					parent->memberships.row(k)=parent->children[k]->memberships.colwise().sum();
				}
				parent->membershipSum=parent->memberships.colwise().sum();
				//printf("MembershipSum %f\n",parent->membershipSum);
			}
		}
	}

	void MixtureTree::reconstruct( const MatrixX *samples, const MatrixX *knownVariables, MatrixX *reconstructed )
	{
		int nSamples=samples[0].cols();
		computeMemberships(samples,knownVariables,false,false);
		for (int inputIdx=0; inputIdx<nInputs; inputIdx++)
		{
			//dense version (keep for debug)
			//reconstructed[inputIdx]=W[inputIdx]*memberships;
			//reconstructed[inputIdx]=W[inputIdx]*sparseMemberships;
			
			//sparse reconstruction, skipping the dot products with zero summation weights. TODO check if this is faster than making a sparse Eigen matrix and multiplying with that (which might cause heap allocs...)
			reconstructed[inputIdx].setZero();
			for (int sampleIdx=0; sampleIdx<nSamples; sampleIdx++)
			{
				for (MixtureTree *leaf : leaves)
				{
					if (leaf->membershipSum[sampleIdx]>0)
					{
						for (int k=0; k<nChildrenPerNode; k++)
						{
							float membership=leaf->memberships(k,sampleIdx);
							if (membership!=0)
							{
								reconstructed[inputIdx].col(sampleIdx)+=membership*leaf->W[inputIdx].col(k);
							}
						}
					}
				}
			}
		}
	}

	void MixtureTree::generate( const MatrixX *conditionedOn, const MatrixX *conditionMask, MatrixX *generated )
	{
		if (conditionedOn!=NULL)
			*generated=*conditionedOn; //init with the variables conditioned on
		int nSamples=generated[0].cols();
		if (conditionedOn!=NULL)
			computeMemberships(conditionedOn,conditionMask,false,false);
		else
		{
			for (int sampleIdx=0; sampleIdx<nSamples; sampleIdx++)
			{
				for (MixtureTree *leaf : leaves)
				{
					leaf->membershipSum[sampleIdx]=((float)nChildrenPerNode)*1.0f/(float)nMeans;
					leaf->memberships.setConstant(1.0f/(float)nMeans);
				}
			}
		}
		for (int sampleIdx=0; sampleIdx<nSamples; sampleIdx++)
		{
			//randomly select unit based on the membership probabilities computed above
			float total=0;
			float threshold=randomf();
			MixtureTree *selectedLeaf=leaves[leaves.size()-1];
			int selectedUnit=nChildrenPerNode-1;
			for (MixtureTree *leaf : leaves)
			{
				if (leaf->membershipSum[sampleIdx]>0)
				{
					for (int k=0; k<nChildrenPerNode; k++)
					{
						float membership=leaf->memberships(k,sampleIdx);
						total+=membership;
						if (total>threshold)
						{
							selectedUnit=k;
							break;
						}
					}
					if (total>threshold)
					{
						selectedLeaf=leaf;
						break;
					}
				}
			}
			//draw sample from the selected unit
			for (int inputIdx=0; inputIdx<nInputs; inputIdx++)
			{
				for (int i=0; i<inputVectorLengths[inputIdx]; i++)
				{
					if (conditionMask==NULL || conditionMask[inputIdx](i,sampleIdx)==0)
						generated[inputIdx](i,sampleIdx)=randGaussian(selectedLeaf->W[inputIdx](i,selectedUnit),selectedLeaf->Wsd[inputIdx](i,selectedUnit));
				}
			}		
		}

	}

	void MixtureTree::update( const MatrixX *samples[], const VectorX *sampleWeights, bool winnerTakesAll, float forgetFactor/*=0.99f*/, float relativeContributionThresholdForSplit/*=0.1f*/ )
	{
		//this method is split into two functions, as both of them need to fully recursively browse children
		//TODO: make a tree nodes array to allow browsing without recursion

		//First accumulate all information from samples to parameter matrices
		accumulate(samples,sampleWeights,winnerTakesAll);


		//Update rest of data, no longer directly depends on the samples
		onAccumulatorsUpdated(forgetFactor,relativeContributionThresholdForSplit);
	}
	void MixtureTree::accumulateConvolutional( const MatrixX &inputData, const VectorX *sampleWeights,bool winnerTakesAll )
	{
		//Note that we assume only one input matrix and no tree children
		int kernelRowShift=convKernelRows/2;
		int kernelColShift=convKernelCols/2;
		int halfStride=convStride/2;
		int nInputImages=inputData.cols()/(convInRows*convInCols);
		for (int kernelPixel=0; kernelPixel<nInputs; kernelPixel++)
		{
			int kernelY=kernelPixel/convKernelCols - kernelRowShift;
			int kernelX=(kernelPixel % convKernelCols) - kernelColShift;
			int sampleIdx=-1;
			for (int inputImageIdx=0; inputImageIdx<nInputImages; inputImageIdx++)
			{
				int inputBaseIdx=inputImageIdx*convInRows*convInCols;
				for (int r=halfStride; r<convInRows; r+=convStride)
				{
					int clippedR=clipMinMaxi(r,kernelRowShift,convInRows-convKernelRows+kernelRowShift);
					for (int c=halfStride; c<convInCols; c+=convStride)
					{
						sampleIdx++;
						int segmentStart=0;
						int clippedC=clipMinMaxi(c,kernelColShift,convInCols-convKernelCols+kernelColShift);
						int rr=clippedR+kernelY;
						int cc=clippedC+kernelX;
						int inputSampleIdx=inputBaseIdx+rr*convInCols+cc;
						for (int k=0; k<nChildrenPerNode; k++)
						{
							real w=memberships(k,sampleIdx); 
							if (sampleWeights!=NULL)
								w*=(*sampleWeights)[sampleIdx];
							if (w!=0)
							{
								if (kernelPixel==0) marginalProbabilityAcc[k]+=w;  //only accumulate this once
								MatrixX::ConstColXpr sample=inputData.col(inputSampleIdx);
								Wsum[kernelPixel].col(k)+=w*sample;
								if (inputTypes[kernelPixel]==Continuous && useVariances)
									WsumSq[kernelPixel].col(k)+=varianceMultiplier*w*(sample-W[kernelPixel].col(k)).cwiseAbs2();
							}
						}
					}
				}
			}
		}

	}
	void MixtureTree::accumulate( const MatrixX *samples[], const VectorX *sampleWeights,bool winnerTakesAll )
	{
		if (convolutionMode)
		{
			accumulateConvolutional(*samples[0],sampleWeights,winnerTakesAll);
			return;
		}
		if (children.size()!=0)
		{
			for (MixtureTree *child : children)
			{
				child->accumulate(samples,sampleWeights,winnerTakesAll);
			}
		}
		//Accumulate data (this code only run for leaves)
		for (int sampleIdx=0; sampleIdx<memberships.cols(); sampleIdx++)
		{	
			if (membershipSum[sampleIdx]!=0)
			{
				int numThreads=8;
				int threadBatchSize=max(1,nChildrenPerNode/numThreads);
				for (int unitBaseIdx=0; unitBaseIdx<nChildrenPerNode; unitBaseIdx+=threadBatchSize)
				{
					for (int k=unitBaseIdx; k<min(unitBaseIdx+threadBatchSize,nChildrenPerNode); k++)
					{
						real w=memberships(k,sampleIdx); 
						if (sampleWeights!=NULL)
							w*=(*sampleWeights)[sampleIdx];
						if (w!=0)
						{
							marginalProbabilityAcc[k]+=w;
							for (int i=0; i<nInputs; i++)
							{
								MatrixX::ConstColXpr sample=samples[i]->col(sampleIdx);
								Wsum[i].col(k)+=w*sample;
								if (inputTypes[i]==Continuous && useVariances)
									WsumSq[i].col(k)+=varianceMultiplier*w*(sample-W[i].col(k)).cwiseAbs2();
							}
						}
					}
				}
			} //if memebershipSum !=0 for this sample
		}

	}
	void MixtureTree::splitAndPrune(float relativeContributionThresholdForSplit)
	{
		if (relativeContributionThresholdForSplit!=0)// && children.size()==0)
		{
			Eigen::Map<VectorX> alreadyPruned((real *)alloca(nChildrenPerNode*sizeof(real)),nChildrenPerNode);
			//VectorX alreadyPruned(nChildrenPerNode);
			alreadyPruned.setZero();
			//VectorX importances=marginalProbabilities;
			Eigen::Map<VectorX> importances((real *)alloca(nChildrenPerNode*sizeof(real)),nChildrenPerNode);
			//TODO only prune clusters that have low enough marginal probability - otherwise deleting the cluster
			//causes its samples to be assigned to a new cluster which then will have a larger variance, leading to a chain 
			//of new splits and prunes
			if (pruningCriterion==pcMarginalProbability || !useVariances)
			{
				importances=marginalProbabilities;
			}
			else if (pruningCriterion==pcInformation)
			{
				//Compute importances as Shannon entropy of the distribution modeled by each unit
				//(i.e., amount of information) times the number of samples represented by the unit 
				//(here estimated relative to others as the marginal probability of the unit).
				//We utilize the following properties:
				//- The entropy of a Gaussian equals 0.5*ln((2*PI*e)^K) + 0.5 * ln(the determinant of the covariance matrix), where K is dimensionality
				//- The determinant of a diagonal matrix equals the product of the diagonal elements
				//=> the entropy equals marginalProbability * (0.5*K*ln(2*PI*e) + 0.5 * sum_i(ln(diag_i)))). 
				//For discrete inputs, the entropy is simply -sum_k(p(x==k)log(p(x==k))

				importances.setZero();
				for (int inputIdx=0; inputIdx<nInputs; inputIdx++)
				{
					if (inputTypes[inputIdx]==Continuous)
					{
						real constantTerm=0.5f*inputVectorLengths[inputIdx]*logf(2.0f*PI*2.71828f); 
						for (int k=0; k<nChildrenPerNode; k++)
						{
							importances[k]+=constantTerm+0.5f*(WvarLog[inputIdx].col(k).sum()); 
						}
					}
					else if (inputTypes[inputIdx]==Discrete)
					{
						float K=(float)inputVectorLengths[inputIdx];
						for (int k=0; k<nChildrenPerNode; k++)
						{
							//importances[k]-=1.0f/K*logf(1.0f/K);  //assume uniform distribution (i.e., only use marginal probability for discrete inputs)
							importances[k]-=W[inputIdx].col(k).dot(logW[inputIdx].col(k));
						}
					}
					else
					{
						Debug::throwError("Unsupported input type");
					}
				}
				importances.array()*=marginalProbabilities.array();
			}
			else  
			{
				//pcErrorSum: importance estimated as number of samples * average error,
				//error estimated as sum of variances
				importances.setZero();
				for (int inputIdx=0; inputIdx<nInputs; inputIdx++)
				{
					if (inputTypes[inputIdx]==Continuous)
					{
						for (int k=0; k<nChildrenPerNode; k++)
						{
							importances[k]+=Wsd[inputIdx].col(k).dot(Wsd[inputIdx].col(k)); 
						}
					}
					else if (inputTypes[inputIdx]==Discrete)
					{
						for (int k=0; k<nChildrenPerNode; k++)
						{
							//For Discrete (i.e., categorical) distribution,
							//variance var_i=var(x=i)=p_i*(1-p_i)=p_i-p_i^2.
							//sum_i(p_i)=1 => sum_i(var_i)=1-sum_i(p_i^2)=1-p'p, where p is vector of probabilities
							importances[k]+=1.0f-W[inputIdx].col(k).dot(W[inputIdx].col(k));
						}
					}
					else
					{
						Debug::throwError("Unsupported input type");
					}
				}
				importances.array()*=marginalProbabilities.array();
				importances/=(float)nTotalInputVariables;  //reduce unstability with high-dimensional inputs
			}
			real importanceSum=importances.sum();
			VectorX::Index mostImportant;
			real maxImportance=importances.maxCoeff(&mostImportant);
			//loop over units and prune those with too little information to the highest information one
			for (int k=0; k<nChildrenPerNode; k++)
			{
				if (alreadyPruned[k]==0)
				{
					if (marginalProbabilities[k]<(1.0f/(real)nChildrenPerNode)*relativeContributionThresholdForSplit)
					//if (importances[k]/importanceSum<(1.0f/(real)nChildrenPerNode)*relativeContributionThresholdForSplit)
					//if (importances[k]/maxImportance<relativeContributionThresholdForSplit)
					{
						//if (importances[mostImportant]>2.0f*importances[k])
						{
							float splitNoise=0.01f;
							float srcWeight=clipMinMaxf(0.5f+splitNoise*(1.0f-2.0f*randomf()),0,1);
							splitUnit(mostImportant,k,srcWeight);
							alreadyPruned[k]=1.0f;
							alreadyPruned[mostImportant]=1.0f; //prevent oscillating splits and prunes
							importances[mostImportant]=-1.0f; //we don't want more duplicates of the same
							maxImportance=importances.maxCoeff(&mostImportant);
						}
					}
				}
			}
		}	
		//recurse into subtrees if the subtree was not affected by split (otherwise becomes impossible to track the split variables in further layers)
		if (children.size()!=0)
		{
			for (int k=0; k<nChildrenPerNode; k++)
			{
				//if (alreadySplit[k]==0)
				{
					children[k]->splitAndPrune(relativeContributionThresholdForSplit);
				}
			}
		}
	}
	void MixtureTree::onAccumulatorsUpdated(float forgetFactor, float relativeContributionThresholdForSplit)
	{
		//Based on accumulated information, split units (may also split subtrees)
		for (int k=0; k<nMeans; k++)
		{
			prunes[k]=-1;	//init to indicate no pruning
		}
		splitAndPrune(relativeContributionThresholdForSplit);

		//update units and weights based on accumulated
		for (int i=0; i<nInputs; i++)
		{
			Eigen::Map<VectorX> temp((real *)alloca(inputVectorLengths[i]*sizeof(real)),inputVectorLengths[i]);
			for (int k=0; k<nChildrenPerNode; k++)
			{
				if (marginalProbabilityAcc[k]>1e-20f) //if marginal probability of unit too low, don't update to avoid overflows when dividing
				{
					W[i].col(k)=Wsum[i].col(k)/marginalProbabilityAcc[k];
					if (inputTypes[i]==Continuous && useVariances)
					{
						//Wsd[i].col(k)=WsumSq[i].col(k)/varianceWeightSum[k];// - W[i].col(i).cwiseAbs2();
						Wsd[i].col(k)=WsumSq[i].col(k)/marginalProbabilityAcc[k];// - W[i].col(i).cwiseAbs2();
						Eigen::ArrayWrapper<Eigen::Block<Eigen::MatrixXf, -1, 1, true>>& a = Wsd[i].col(k).array();
						Wsd[i].col(k)= a.max(squared(kernelWidths[i]));
						WvarLog[i].col(k)=Wsd[i].col(k).array().log();
						Wsd[i].col(k)=Wsd[i].col(k).cwiseSqrt();
					}
					if (inputTypes[i]==Discrete) //need to sum-normalize if the class means are discrete pdfs
					{
						W[i].col(k)/=W[i].col(k).sum();
					}
				}
			}

			if (inputTypes[i]==Continuous && useVariances && sharedVariances)
			{
				Wsd[i].col(0)=Wsd[i].rowwise().mean();
				for (int k=1; k<nChildrenPerNode; k++)
				{
					int rows = Wsd[i].col(0).size();
					for (int row = 0; row < rows; row++) {
						Wsd[i](row,k) = Wsd[i](row,0);
					}
				}
			}
			if (inputTypes[i]==Continuous && useVariances && scalarVariances)
			{
				Wsd[i].row(0)=Wsd[i].colwise().mean();
				for (int k=1; k<inputVectorLengths[i]; k++)
				{
					int cols = Wsd[i].row(0).size();
					for (int col = 0; col < cols; col++) {
						Wsd[i](k, col) = Wsd[i](0, col);
					}
				}
			}
			if (convolutionMode)
			{
				Wtranspose[i]=W[i].transpose();
			}
		}
		marginalProbabilities=marginalProbabilityAcc/marginalProbabilityAcc.sum();
		if (!marginalProbabilities.allFinite())
		{
			marginalProbabilityAcc.setZero();
			marginalProbabilities.setZero();//(1.0/(real)nMeans);
		}
		logMarginalProbabilities=marginalProbabilities.array().log();

		//compute log params needed for efficient Bernoulli modeling
		for (int inputIdx=0; inputIdx<nInputs; inputIdx++)
		{
			if (inputTypes[inputIdx]==Bernoulli)
			{
				logW[inputIdx].resizeLike(W[inputIdx]);
				for (int k=0; k<nInitializedUnits; k++)
				{
					for (int r=0; r<inputVectorLengths[inputIdx]; r++)
					{
						//Bernoulli prob: product_i w^(x)(1-w)^(1-x)
						//log_p=sum  x log(w) + (1-x)log(1-w) 
						//     =sum  x log(w) - x log(1-w) + log(1-w)
						//     =sum  x(log(w)-log(1-w)) + log(1-w), the last term may be omitted as it amounts to multiplying 
						//the probability with a constant, and we are only interested in unnormalized pdfs.
						//We store log(w)-log(1-w) in logW
						real logw=log(W[inputIdx](r,k)) - log (1 - W[inputIdx](r,k));
						logW[inputIdx](r,k)=clipMinMaxf(logw,lidstoneSmoothing,-lidstoneSmoothing);
					}
				}
			}
			else if (inputTypes[inputIdx]==Discrete)
			{
				logW[inputIdx]=W[inputIdx].array().log().max(lidstoneSmoothing);
			}
		}

		//recurse into children 
		for (MixtureTree *child : children)
		{
			child->onAccumulatorsUpdated(forgetFactor,relativeContributionThresholdForSplit);
		}

		//forget. (when not training in batches, forgetFactor==0, i.e., the epoch fully determines the units and weights)
		//if (children.size()==0)
		{
			for (int i=0; i<nInputs; i++)
			{
				Wsum[i]*=forgetFactor;
				if (inputTypes[i]==Continuous && useVariances)
				{
					WsumSq[i]*=forgetFactor;
				}
			}
			marginalProbabilityAcc*=forgetFactor;
		}
	}

	void MixtureTree::onlineEM( const MatrixX *samples, const VectorX *sampleWeights, bool winnerTakesAll, float forgetFactor/*=0.99f*/, float relativeContributionThresholdForSplit/*=0.1f*/ )
	{
		const int maxInputs=256;
		const MatrixX *pSamples[maxInputs];
		if (nInputs>maxInputs)
			Debug::throwError("Too many inputs!");
		for (int i=0; i<nInputs; i++)
		{
			pSamples[i]=&samples[i];
		}
		onlineEM(pSamples,sampleWeights,winnerTakesAll,forgetFactor,relativeContributionThresholdForSplit);
	}
	void MixtureTree::onlineEM( const MatrixX *samples[], const VectorX *sampleWeights, bool winnerTakesAll, float forgetFactor/*=0.99f*/, float relativeContributionThresholdForSplit/*=0.1f*/ )
	{
		computeMemberships(samples,NULL,true,winnerTakesAll);
		update(samples,sampleWeights,winnerTakesAll,forgetFactor,relativeContributionThresholdForSplit);
	}


	void MixtureTree::splitUnit( int src, int dst, float srcWeight )
	{
		//TODO: figure out subtree variable ranges, update everything
		for (int i=0; i<nInputs; i++)
		{
			Wsum[i].col(dst)=(1.0f-srcWeight)*Wsum[i].col(src);
			Wsum[i].col(src)*=srcWeight;
			if (inputTypes[i]==Continuous && useVariances)
			{
				WsumSq[i].col(dst)=(1.0f-srcWeight)*WsumSq[i].col(src);
				WsumSq[i].col(src)*=srcWeight;
			}
		}
		marginalProbabilityAcc[dst]=(1.0f-srcWeight)*marginalProbabilityAcc[src];
		marginalProbabilityAcc[src]*=srcWeight;
		if (children.size()!=0)
		{
			children[src]->splitCopyTo(children[dst],srcWeight);

		}
		else
		{
			root->prunes[leafBaseIdx+dst]=leafBaseIdx+src;
		}
	}
	
	void MixtureTree::splitCopyTo( MixtureTree *dst, float srcWeight )
	{
		if (children.size()!=0)
		{
			for (int i=0; i<nChildrenPerNode; i++)
			{
				children[i]->splitCopyTo(dst->children[i],srcWeight);
			}
		}
		else
		{
			for (int k=0; k<nChildrenPerNode; k++)
			{
				root->prunes[dst->leafBaseIdx+k]=leafBaseIdx+k;
			}
		}
		for (int i=0; i<nInputs; i++)
		{
			dst->Wsum[i]=(1.0f-srcWeight)*Wsum[i];
			Wsum[i]*=srcWeight;
			if (inputTypes[i]==Continuous && useVariances)
			{
				dst->WsumSq[i]=(1.0f-srcWeight)*WsumSq[i];
				WsumSq[i]*=srcWeight;
			}
		}
		dst->marginalProbabilityAcc=(1.0f-srcWeight)*marginalProbabilityAcc;
		marginalProbabilityAcc*=srcWeight;
	}

	void MixtureTree::toggleMarginalProbabilities(bool useMarginalProbabilities)
	{
		this->useMarginalProbabilities=useMarginalProbabilities;
	}

}