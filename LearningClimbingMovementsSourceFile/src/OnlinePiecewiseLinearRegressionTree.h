#pragma once
#include "OnlinePiecewiseLinearRegression.h"
#include <vector>

using namespace Eigen;
namespace AaltoGames
{
	
	class OnlinePiecewiseLinearRegressionTree
	{
	public:
		~OnlinePiecewiseLinearRegressionTree();
		OnlinePiecewiseLinearRegressionTree()
		{
			hasChildren=false;
		}
		OnlinePiecewiseLinearRegression model;
		int keyDim,dataDim;
		OnlinePiecewiseLinearRegressionTree *children[2];
		OnlinePiecewiseLinearRegressionTree *parent;
		int maxDepth;
		int depth;
		bool hasChildren;
		int nSamplesPerLeaf;
		float regularization;
		int cacheSize;
		void init(OnlinePiecewiseLinearRegressionTree* parent, int keyDim, int dataDim, int maxDepth, int nSamplesPerLeaf, float regularization=1e-10, int cacheSize=0);
		void addSample(const VectorXf &key, const VectorXf &data, float weight=1.0f);
		void processAll(const std::function<void (OnlinePiecewiseLinearRegressionTree *)> &func);
		float getDistanceFromTrainingSet(const Eigen::VectorXf &key);
		OnlinePiecewiseLinearRegressionTree *findLeaf(const VectorXf &key);

		//returns the distance of key from data (projection error + distance on axis beyond the accepted standard deviations)
		//TODO: fail if beyond training data
		void getData(const VectorXf &key, VectorXf &out_data);
		void getDataAndClosestKeyMean(const VectorXf &key, VectorXf &out_data, VectorXf &out_keyMean);
		float getDistanceToClosestKeyMean(const Eigen::VectorXf &key);
	}; //OnlinePiecewiseLinearRegressionTree
}// AaltoGames