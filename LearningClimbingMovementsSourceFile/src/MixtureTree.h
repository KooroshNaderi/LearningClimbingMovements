#pragma once
#include <Eigen/Eigen> 
#include <vector>
#include "MathUtils.h"
#include "WorkerThreadManager.h"

namespace AaltoGames
{
	/*
	A tree-based mixture model for modeling the distributions of continuous, discrete or Bernoulli inputs.

	Note that most methods take arrays of matrices as inputs - this is because when this class is used as a basis
	for probabilistic graphical model, each matrix corresponds to the messages of one input connection in the graph. 

	In simple use without a graphical model, one can simplyu have a matrix of samples as column vectors, and
	pass a pointer to the matrix as the input to the methods of this class.
	*/
	class MixtureTree  
	{
	public:
		typedef float real;
		typedef Eigen::Matrix<real,Eigen::Dynamic,Eigen::Dynamic> MatrixX;
		typedef Eigen::Matrix<real,Eigen::Dynamic,1> VectorX;
		enum InputTypes
		{
			Continuous,Bernoulli,Discrete
		};
		/********************************************************

		Main interface functions - typical applications only need these.

		********************************************************/
		//Call this before any of the other methods
		//Params: 
		//nChildrenPerNode		Tree branching factor
		//maxDepth				Tree depth. 
		//sparsityThreshold		In regression and sampling children are browsed only if their membership probability
		//						is greater than sparsityThreshold * max ( membership probabilities)	
		//Note: Shallow and wide trees tend to produce better performance and quality ( child browsing decisions are less uncertain).
		//For example, set nChildrenPerNode=32, maxDepth=2 for a total of 1024 mixture components.
		int init(int nChildrenPerNode,int maxDepth=1,float sparsityThreshold=0.01f, int nThreads=0);

		//Add an input connection. When not building graphical models, you only need to call this once after init().
		//Params:
		//nVariables	Dimensionality of the modeled data (for type==Discrete, the number of different values for the modeled random variable)
		//type			Type of the modeled data
		//minSd			For continuous variables: minimum allowed standard deviation of mixture component (a regularization parameter,
		//				set this according to the desired modeling resolution)
		//minBounds		For continuous variables: minimum values expected to be observed (only used for initialization)
		//maxBounds		For continuous variables: maximum values expected to be observed (only used for initialization)
		void addInput(int nVariables, InputTypes type, float minSd=1.0f, const real *minBounds=NULL, const real *maxBounds=NULL);

		//Performs one minibatch online EM iteration
		//Params:
		//samples			Array of input samples, one matrix per input, samples as column vectors. 
		//					If using only one input, pass a pointer to a single sample matrix
		//sampleWeights		Array of input sample weights. One vector per input, one value per sample.
		//					This is an optional parameter - pass NULL if not used. 
		//winnerTakesAll	If true, each sample affects only the mixture component with highest membership probability
		//					Set to true (default) for faster convergence, false to slightly reduce overfitting.
		//forgetFactor		Learning rate parameter. Good default is 0.99f
		//pruningThreshold	Tree adaptation (pruning & splitting) rate (0...1). A mixture component is pruned if its marginal probability
		//					is lower than pruningThreshold * max(marginal probabilities). 0.25f is a good default. 
		void onlineEM(const MatrixX *samples, const VectorX *sampleWeights, bool winnerTakesAll, float forgetFactor, float pruningThreshold);

		//Performs regression and/or denoising: projects the input to the latent variable space of membership probabilities,
		//and then reconstructs (backprojects) the latent variables back to the input space.
		//Params:
		//samples			Same as above
		//observedVariables	An array of matrices, similar structure as the "samples" parameter. 
		//					The column vectors of the matrices specify whether a variable is observed (1) or unknown(0).
		//					For unknown variables, the corresponding values in the samples matrices have no effect.
		//					For regression, set the values corresponding to regressors to 1, and regressands to 0.
		//reconstructed		Array of matrices for the computed outputs, similar structure as the "samples" parameter.
		void reconstruct(const MatrixX *samples, const MatrixX *observedVariables, MatrixX *reconstructed);

		//Generates samples from the mixture, can be conditioned on any combination of variables.
		//Params:
		//conditionedOn		Similar to the "samples" above, specifies the values for conditioning the generation
		//conditionMask		Similar to the "observedVariables" above, specifies which values of "conditionedOn" have an effect
		//generated			Array of matrices for the computed outputs, similar to "reconstructed" above.		
		void generate(const MatrixX *conditionedOn, const MatrixX *conditionMask, MatrixX *generated);

		/********************************************************

		Other methods and data - kept public for easier debugging 
		and R&D, but avoid using as these might change
		in future versions

		********************************************************/
		~MixtureTree();
		MixtureTree *parent;
		MixtureTree *root;
		//Model parameters are Wsum and marginalProbabilityAcc. Others are computed from them to enable faster operation
		std::vector<MatrixX> Wsum,W,Wtranspose;  //matrix of "weight vectors" as columns. A weight vector can denote the mean of a Gaussian (typically in input layer) or a discrete pdf (in subsequent layers)
		std::vector<MatrixX> WsumSq;
		std::vector<MatrixX> Wsd,WvarLog;
		std::vector<MatrixX> logW; 
		VectorX marginalProbabilities,marginalProbabilityAcc,logMarginalProbabilities,varianceWeightSum;

		std::vector<MatrixX> inputLogMemberships;	//log membership probabilities for each input (in graphical model use, we need them)
		MatrixX logMemberships;					//memberships computed as aggregate of all inputs and marginal probabilities (in graphical model use, these are the beliefs)
		MatrixX memberships;					//memberships computed as aggregate of all inputs and marginal probabilities (in graphical model use, these are the beliefs)
		VectorX membershipSum;					//column sums of memberships, one value per sample. Used for determining when to browse children
		std::vector<MixtureTree *> children; 
		std::vector<MixtureTree *> leaves; //this is only valid for tree root, computed in Init()
		std::vector<MixtureTree *> allNodes; //this is only valid for tree root, computed in Init()
		MatrixX fullMemberships,logFullMemberships; //this is only valid for tree root, computed in computeMemberships()
		int depth;
		int nMeans;
		int nInitializedComponents;
		int nInputs;
		int nThreads;
		std::vector<bool> visited;
		std::vector<float> kernelWidths;  //for each input
		real forgetFactor;
		static const int MAXMEANS=4096;
		float sparsityThreshold;
		bool initialized;
		int leafBaseIdx; //valid for leaves. This is the index of the first unit of the leaf when all leaf units concatenated into a vector such as fullMemberships
		std::vector<InputTypes> inputTypes;
		std::vector<int> inputVectorLengths;
		std::vector<int> prunes; //value per leaf unit, only valid for root. if negative, this unit was not pruned during last adapt(). Otherwise, denotes index of the unit that is split to replace the pruned one. 
		int maxDepth;
		int nChildrenPerNode;
		int nInitializedUnits;
		int nTotalInputVariables;
		int convInCols,convInRows,convOutCols,convOutRows,convKernelRows,convKernelCols,convStride;
		bool convolutionMode;
		bool useMarginalProbabilities;
		WorkerThreadManager<MixtureTree *> *manager;
		MixtureTree();
		//virtual bool allUnitsInitialized();
		//void randomInit();
		//void splitUnit(int src, int dst);
		//void handleInputPrunes();

		//Samples points to an array of sample matrices, one for each input. 
		//updateWeights contains column vectors of unit update weights (one vector per sample). In EM updating, these correspond to membership probabilities
		void update(const MatrixX *samples[], const VectorX *sampleWeights, bool winnerTakesAll, float forgetFactor, float pruningThreshold);
		void computeMemberships(const MatrixX *samples, const MatrixX *mask, bool trainingMode, bool winnerTakesAll);
		void computeMemberships(const MatrixX *samples[], const MatrixX *mask[], bool trainingMode, bool winnerTakesAll);
		void computeLogMemberships( const MatrixX *samples[], const MatrixX *mask[], bool trainingMode, bool winnerTakesAll);
		void onLogMembershipsUpdated(bool winnerTakesAll);
		void onAccumulatorsUpdated(float forgetFactor, float relativeContributionThresholdForSplit );
		void onlineEM(const MatrixX *samples[], const VectorX *sampleWeights, bool winnerTakesAll, float forgetFactor, float pruningThreshold);
		MatrixX::ColXpr getMean(int inputIdx, int unitIdx) const
		{
			return leaves[unitIdx / nChildrenPerNode]->W[inputIdx].col(unitIdx % nChildrenPerNode);
		}
		MatrixX::ColXpr getMeanAcc(int inputIdx, int unitIdx) const
		{
			return leaves[unitIdx / nChildrenPerNode]->Wsum[inputIdx].col(unitIdx % nChildrenPerNode);
		}
		void addConvolutionalInput(int nVarsPerPixel, int nRows, int nCols, int kernelRows, int kernelCols, int stride, InputTypes type, float minSd=1.0f, const real *minBounds=NULL, const real *maxBounds=NULL);
		void accumulate(const MatrixX *samples[], const VectorX *sampleWeights,bool winnerTakesAll);
		void toggleMarginalProbabilities(bool useMarginalProbabilities);
	private:
		void accumulateConvolutional(const MatrixX &samples, const VectorX *sampleWeights,bool winnerTakesAll);
		void splitUnit( int src, int dst, float srcWeight );
		void splitCopyTo(MixtureTree *dst, float srcWeight);
		void splitAndPrune(float relativeContributionThresholdForSplit );
		void computeSampleMemberships(int sampleIdx, bool trainingMode, const MatrixX *samples[], const MatrixX *mask[]);
		void resizeBuffers(int nSamples);
	};
} //AaltoGames
/*
class MixtureTree
{
public:
	typedef float real;
	typedef Eigen::Matrix<real,Eigen::Dynamic,Eigen::Dynamic> MatrixX;
	typedef Eigen::Matrix<real,Eigen::Dynamic,1> VectorX;
	enum PdfTypes
	{
		Gaussian,Bernoulli,SparseBernoulli
	};
	std::vector<PdfTypes> inputTypes;
	std::vector<MatrixX> Wsum,W;  //matrix of "weight vectors" as columns. A weight vector can denote the mean of a Gaussian (typically in input layer) or a discrete pdf (in subsequent layers)
	std::vector<MatrixX> logW; 
	VectorX weights,weightsSum;
	std::vector<int> inputVectorLengths;
	class LinkageData
	{
	public:
		std::vector<int> childIdx;
		bool hasChildren;
	};
	std::vector<LinkageData> linkage;
	int nChildrenPerNode;
	int maxDepth;
	int nInputs;
	MixtureTree()
	{
		nInputs=0;
	}
	void init(int nChildrenPerNode,int maxDepth=1)
	{
		this->nChildrenPerNode=nChildrenPerNode;
		this->maxDepth=maxDepth;
	}
	void addInput(int nVariables, PdfTypes type)
	{
		inputVectorLengths.push_back(nVariables);
		inputTypes.push_back(type);
		nInputs++;
	}
	//Computes the log memberships (without unit weights (i.e. marginal probabilities)) for each unit
	//Samples contains the sample vectors as columns
	void computeLogMemberships(int inputIdx, const MatrixX &samples, VectorX &output)
	{

	}
	//Samples points to an array of sample matrices, one for each input. 
	//updateWeights contains column vectors of unit update weights (one vector per sample)
	void updateComponents(const MatrixX *samples, const MatrixX &updateWeights, float forgetFactor=0.99f, float relativeContributionThresholdForSplit=0.1f)
	{

	}
	//Performs one online EM iteration (i.e, calling computeLogMemberships() for each input, and then updateComponents)
	void onlineEM(const MatrixX *samples, float forgetFactor=0.99f, float relativeContributionThresholdForSplit=0.1f)
	{
	}
	//For each input, errorMask has a column vector defining the boosting weight per variable.
	//The boosting is based on first computing the conditional mean of the variables with 1 in errorMask given the variables with zeros.
	//Then, the difference between the conditional mean and the actual sample values is used as sample weights.
	//This effectively creates ridges around high-error areas in the density modeled by the mixture, meaning that the 
	void errorBoostedOnlineEM(const MatrixX *samples, const MatrixX &errorMask,float forgetFactor=0.99f)
	{
	}
};
*/