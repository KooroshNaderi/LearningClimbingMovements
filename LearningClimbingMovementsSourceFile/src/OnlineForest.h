#pragma once
#include "EigenMathUtils.h"
#include "MathUtils.h"
#include "Debug.h"
#include "IndexRandomizer.h"
#include <stdint.h>
#include "DynamicMinMax.h"

using namespace Eigen;

namespace AaltoGames
{
	template<class T> class PoolAllocator{
	private:
		T *pool;
		T **freeChunks;
		int nextIdx;
		int maxChunks;
	public:
		PoolAllocator (int maxChunks=0)
		{
			this->maxChunks=0;
			if (maxChunks>0)
				init(maxChunks);
		}
		void init(int maxChunks)
		{
			if (this->maxChunks>0)
			{
				delete [] pool;
				delete [] freeChunks;
			}
			pool=new T[maxChunks];
			freeChunks=new T*[maxChunks];
			this->maxChunks=maxChunks;
			for (int i=0; i<maxChunks; i++)
			{
				freeChunks[i]=&pool[i];
			}
			nextIdx=maxChunks-1;
		}
		T *allocate()
		{

			T *result=freeChunks[nextIdx];
			nextIdx--;
			AALTO_ASSERT1(nextIdx>=0);
			return result;
		}
		void free(T *obj)
		{
			AALTO_ASSERT1(obj!=NULL);
			nextIdx++;
			AALTO_ASSERT1(nextIdx<maxChunks);
			freeChunks[nextIdx]=obj;
		}
		void reset()
		{
			for (int i=0; i<maxChunks; i++)
			{
				freeChunks[i]=&pool[i];
			}
			nextIdx=maxChunks-1;
		}
		int numFree()
		{
			return nextIdx;
		}
		int poolSize()
		{
			return maxChunks;
		}
	};


	class OnlineTree
	{
	public:
		OnlineTree *parent;
		OnlineTree *children[2];
		int splitVar;
		float splitLoc;
		int depth;
		const VectorXf *key;
		void *data;
		bool bounded;
		int time;	//time index at insertion
		int nSamples;
		double weight;
		double importance;
		double subTreeIntegral;
		double volume;
		OnlineTree *firstToPrune;
		PoolAllocator<OnlineTree> *allocator;
		OnlineTree(OnlineTree *parent=NULL,PoolAllocator<OnlineTree> *allocator=NULL);
		void setBounds(const VectorXf &minCorner, const VectorXf &maxCorner);
		OnlineTree *findNode(const VectorXf &key);
		void clear();
		void add(const VectorXf &key, void *data, double probability=1, int time=0);
		//Note: this is a convenient way to access all samples, but slow due to std::function causing a heap alloc
		void processAll(const std::function<void (OnlineTree *)> &func);
		void processAll_cb(void(*callback)(OnlineTree *, void *),void *data=NULL);
		int getLeaves(OnlineTree **out_leaves, int nSoFar=0);
		void pruneTo(int maxSamples);
		OnlineTree *prune(OnlineTree *pruned);
		void setImportance(double importance);
		OnlineTree *sample();
		bool OnlineTree::hasChildren()
		{
			return nSamples>1;
		}
	protected:
		void init(const VectorXf &key, void *data, double probability, int time);
		void addToThis(const VectorXf &newKey, void *newData, double newProbability, int newTime);
		OnlineTree *getRoot();
		void propagate();
	};

	class OnlineForest
	{
	public:
		enum PruningModes
		{
			pmRandom=0,
			pmMinDataDiff,
			pmMinVolume
		};
		enum SamplingMethods
		{
			smFastApproximate=0,
			smOnlyNearest
		};
		std::vector<OnlineTree *> trees;
		int nTrees;
		int time;
		int maxSamples;
		int nSamples;
		int rebuildTreeIdx;
		int rebuildSpeed;
		int rebuildSampleIdx;
		PruningModes pruningMode;
		Eigen::VectorXf minKey,maxKey;
		PoolAllocator<OnlineTree> allocator;
		DynamicMinMax minmax;
		class Sample
		{
		public:
			Eigen::VectorXf key;
			Eigen::VectorXf data;
			int time;
			double weight;
			double keyVariance;
			double dataVariance;
		};
		std::vector<Sample> samples;

		IndexRandomizer randomizer;
		OnlineForest();
		void init(int nTrees, int maxSamples, PruningModes pruningMode=pmRandom);
		void setBounds(const VectorXf &minCorner, const VectorXf &maxCorner);
		void addSample(const VectorXf &key, const VectorXf &data, double probability);
		//Use this for batch-inserting multiple samples (results in more optimal tree structure). Keys and data vectors as columns. 
		//void addSamples(const MatrixXf &keys, const MatrixXf &data, VectorXd &probability);
		OnlineTree *sample();
		OnlineTree *sampleWithPrior(const VectorXf &key, const VectorXf &keySd, SamplingMethods method=smFastApproximate, int nIter=1);
		Sample *getNearest(const VectorXf &key, const VectorXf &keySd, float &out_quadraticForm);
		void getNeighbors(int maxReturned, const VectorXf &key, OnlineTree **out_neighbors, int &out_numNeighbors, int searchDepth=2);
		void getNeighborSamples(int maxReturned, const VectorXf &key, Sample **out_neighbors, int &out_numNeighbors, int searchDepth=2);
		void getMeanData(const VectorXf &key,VectorXf &out_data);
		void getMeanDataWithoutOutliers(const VectorXf &key,VectorXf &out_data, float outlierThreshold=2.0f);
		bool getWeightedMeanData(const VectorXf &key,const VectorXf &keySd, VectorXf &out_data);
		void getNearestData(const VectorXf &key, const VectorXf &keySd, VectorXf &out_data);
	protected:
		void updateSampleStatistics(Sample &sample);
		void updateSampleImportanceBasedOnMeanVolume(const VectorXf &key, OnlineTree &tree);
		OnlineTree *sampleWithPriorNearestOnly(const VectorXf &key, const VectorXf &keySd);
	};
} //AaltoGames