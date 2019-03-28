#pragma once
#include "Debug.h"
#include "MathUtils.h"
#include <stdint.h>
#include <Eigen/Eigen>

namespace AaltoGames
{

class OnlinePiecewiseLinearRegression
{
public:
	Eigen::VectorXf n,nSum[2];  //in homogenous coordinates, i.e., one can test a new key by padding it with a 1 and computing dot product with the axis
	float b,bSum;
	float bWeightSum;
	Eigen::VectorXf keySum[2],keyMean[2],dataSum[2],dataMean[2],keyTemp;  
	float weightSum[2],nWeightSum[2];
	int keyDim,dataDim;
	int nSamples;
	float regularization;
	float initialForget;
	float alpha;
	Eigen::MatrixXf keyCache;
	Eigen::MatrixXf dataCache;
	Eigen::VectorXf weightCache;
	int cacheSize;
	uint32_t cacheWritePos;
	float normalRegularizationBias,tangentialRegularizationBias;
	void init(int keyDim, int dataDim, float alpha=0.0f, int cacheSize=0, float normalRegularizationBias=0.9f, float tangentialRegularizationBias=0.1f);
	int getCluster(const Eigen::VectorXf &key, const Eigen::VectorXf &data);
	void addSample(const Eigen::VectorXf &key, const Eigen::VectorXf &data, float weight, bool skipNormalUpdate=false);
	void updateNormal();
	bool updateNSums(const Eigen::VectorXf &key, const Eigen::VectorXf &data, float weight, int targetIdx);
	void updateNSums2(const Eigen::VectorXf &key, const Eigen::VectorXf &data, float weight, int targetIdx);
	void getData(const Eigen::VectorXf &key, Eigen::VectorXf &data);
	void getDataAndClosestKeyMean(const Eigen::VectorXf &key, Eigen::VectorXf &data, Eigen::VectorXf &out_keyMean);
	float getDistanceToClosestKeyMean(const Eigen::VectorXf &key);
	float getDistanceFromTrainingSet(const Eigen::VectorXf &key);
	void getDataLerped(const Eigen::VectorXf &key, Eigen::VectorXf &data);
	void setSplitToBetweenKeyMeans();
};

} //AaltoGames