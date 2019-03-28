#include "OnlinePiecewiseLinearRegression.h"

using namespace AaltoGames;
using namespace Eigen;

namespace AaltoGames
{

void OnlinePiecewiseLinearRegression::init( int keyDim, int dataDim, float alpha/*=0.0f*/, int cacheSize/*=0*/, float normalRegularizationBias/*=0.9f*/, float tangentialRegularizationBias/*=0.1f*/ )
{
	this->keyDim=keyDim;
	this->dataDim=dataDim;
	this->alpha=alpha;
	this->cacheSize=cacheSize;
	this->normalRegularizationBias=normalRegularizationBias;
	this->tangentialRegularizationBias=tangentialRegularizationBias;
	for (int i=0; i<2; i++)
	{
		weightSum[i]=0;
		nWeightSum[i]=0;
		keySum[i]=VectorXf::Zero(keyDim);
		dataSum[i]=VectorXf::Zero(dataDim);
		keyMean[i]=VectorXf::Zero(keyDim);
		dataMean[i]=VectorXf::Zero(dataDim);
		nSum[i]=VectorXf::Zero(keyDim);
	}
	keyTemp=VectorXf::Zero(keyDim);
	nSamples=0;
	regularization=1.0f;
	bSum=0;
	b=0;
	bWeightSum=0;
	initialForget=0;
	n=VectorXf::Zero(keyDim);
	cacheWritePos=0;
	if (cacheSize>0)
	{
		keyCache=Eigen::MatrixXf::Zero(keyDim,cacheSize);
		dataCache=Eigen::MatrixXf::Zero(dataDim,cacheSize);
		weightCache=Eigen::VectorXf::Zero(cacheSize);
	}
}

int OnlinePiecewiseLinearRegression::getCluster( const Eigen::VectorXf &key, const Eigen::VectorXf &data )
{
	float dist0=(dataMean[0]-data).squaredNorm();
	dist0+=(keyMean[0]-key).squaredNorm();
	float dist1=(dataMean[1]-data).squaredNorm();
	dist1+=(keyMean[1]-key).squaredNorm();
	if (dist0<dist1)
		return 0;
	else if (dist1<dist0)
		return 1;
	else
		return rand01();
}

void OnlinePiecewiseLinearRegression::addSample( const Eigen::VectorXf &key, const Eigen::VectorXf &data, float weight, bool skipNormalUpdate )
{
	nSamples++;
	const float forgetFactor=0.5f;
	initialForget=forgetFactor*initialForget+(1.0f-forgetFactor)*1.0f;
	weight*=initialForget;
	//update the joint distribution clustering
	int targetIdx;
	if (weightSum[0]==0)
	{
		targetIdx=0;
	}
	else if (weightSum[1]==0)
	{
		targetIdx=1;
	}
	else
	{
		targetIdx=getCluster(key,data);
	}
	/*
	//importance weighting based on how close to volume borders
	float normalDistanceFromBorder=FLT_MAX;
	float proj=n.dot(key)+b;
	float mult=targetIdx==1 ? 1.0f : -1.0f;
	for (int i=0; i<keyDim; i++)
	{
	float v=-(key[i])/(mult*n[i]);
	if (validFloat(v) && v>0)
	normalDistanceFromBorder=_min(normalDistanceFromBorder,v);
	v=(256.0f-(key[i]))/(mult*n[i]);
	if (validFloat(v) && v>0)
	normalDistanceFromBorder=_min(normalDistanceFromBorder,v);
	}
	float samplingVolumeSize=1024.0f;
	normalDistanceFromBorder=_min(normalDistanceFromBorder,samplingVolumeSize)/samplingVolumeSize;
	//float keyShift=0.5f-normalDistanceFromBorder;
	normalDistanceFromBorder=_max(normalDistanceFromBorder,0.001f);
	nWeightSum[targetIdx]+=weight;	//also need to store the non-importance weighed sums
	weight*=1.0f/normalDistanceFromBorder; //importance weighting
	*/
	weightSum[targetIdx]+=weight;
	keySum[targetIdx]+=weight*key;//+(mult*weight*keyShift*samplingVolumeSize)*n;
	keyMean[targetIdx]=keySum[targetIdx]/weightSum[targetIdx];
	dataSum[targetIdx]+=weight*data;
	dataMean[targetIdx]=dataSum[targetIdx]/weightSum[targetIdx];

	//if importance weighting above works, we can simply compute n and b from 
	/*
	if (weightSum[0]!=0 && weightSum[1]!=0)
	{
	n=keyMean[1]-keyMean[0];
	n.normalize();
	//b=-0.5*n.dot(keyMean[1]+keyMean[0]);
	float total=nWeightSum[0] + nWeightSum[1];
	b=-(n.dot(keyMean[0])*nWeightSum[1]/total+n.dot(keyMean[1])*nWeightSum[0]/total);
	}
	return;
	*/
	//update the classifier
	regularization*=alpha;
	bool misclassified=updateNSums(key,data,weight,targetIdx);
	if (cacheSize>0)
	{
		int startIdx=_min(0,cacheWritePos-cacheSize);
		for (int i=startIdx; i<(int)cacheWritePos; i++)
		{
			int wrappedIdx=i % cacheSize;
			updateNSums(keyCache.col(wrappedIdx),dataCache.col(wrappedIdx),weightCache(wrappedIdx),
				getCluster(keyCache.col(wrappedIdx),dataCache.col(wrappedIdx)));
		}
		//There are various rationale for adding a sample to the cache.
		//1: add all, leading to a low-pass filtered behavior on past cacheSize samples
		//2: only add misclassified ones, as samples close to the decision surface are likely to be misclassified again if the surface is adjusted
		//3: reservoir sampling: add sample randomly, probability cacheSize/nSamples
		if (misclassified)
			//if (randomf()<((float)cacheSize)/((float)nSamples))
		{
			int wrappedWritePos=cacheWritePos % cacheSize;
			keyCache.col(wrappedWritePos)=key;
			dataCache.col(wrappedWritePos)=data;
			weightCache(wrappedWritePos)=weight;
			cacheWritePos++;
		}
	}
	if (skipNormalUpdate)
		return;
	updateNormal();
}

void OnlinePiecewiseLinearRegression::updateNormal()
{
	if (nWeightSum[0]!=0 && nWeightSum[1]!=0){
		float total=nWeightSum[0] + nWeightSum[1];
		n=nSum[1]/nWeightSum[1]-nSum[0]/nWeightSum[0];
		n.normalize();

		//			b=-0.5*(n.dot(nSum[0])/nWeightSum[0]+n.dot(nSum[1])/nWeightSum[1]);
		b=-(n.dot(nSum[0])*nWeightSum[1]/(nWeightSum[0]*total)+n.dot(nSum[1])*nWeightSum[0]/(nWeightSum[1]*total));
	}
}

bool OnlinePiecewiseLinearRegression::updateNSums( const Eigen::VectorXf &key, const Eigen::VectorXf &data, float weight, int targetIdx )
{
	float nDotKey=n.dot(key);
	float proj=nDotKey+b;
	bool misclassified=(proj>=0 && targetIdx==0) || (proj<=0 && targetIdx==1);
	nSamples++;
	if (!misclassified)
	{
		weight*=regularization;
		//just update the sum, no regularization neede
		//nSum[targetIdx]+=weight*key;
		//return misclassified;
	}
	nSum[targetIdx]+=weight*key;
	if (normalRegularizationBias+tangentialRegularizationBias>0)
	{
		keyTemp=keyMean[targetIdx]-key;
		float nr=n.dot(keyTemp);
		nSum[targetIdx]+=(normalRegularizationBias*weight*nr)*n;
		keyTemp-=nr*n;  //only tangential component remains
		nSum[targetIdx]+=(tangentialRegularizationBias*weight)*keyTemp;
	}

	/*
	float regularizedProj=n.dot(keyMean[targetIdx])-nDotKey;
	nSum[targetIdx]+=(tangentialRegularizationBias*weight)*(key-(nDotKey/key.squaredNorm())*n);
	*/
	//if (normalRegularizationBias>0)
	//{
	//	//Resist movement along normal
	//	nSum[targetIdx]+=(normalRegularizationBias*weight*regularizedProj)*n;
	//}
	//if (tangentialRegularizationBias)
	//{
	//	float regularizedProj=n.dot(keyMean[targetIdx])-n.dot(key);
	//	nSum[targetIdx]+=(normalRegularizationBias*weight*regularizedProj)*n;
	//}
	//{
	//	//Regularize towards cluster mean
	//	nSum[targetIdx]+=(normalRegularizationBias*weight)*(keyMean[targetIdx]-key);
	//	
	//}

	//update weight
	nWeightSum[targetIdx]+=weight;
	return misclassified;
}

void OnlinePiecewiseLinearRegression::updateNSums2( const Eigen::VectorXf &key, const Eigen::VectorXf &data, float weight, int targetIdx )
{
	float nDotKey=n.dot(key);
	nSamples++;
	nSum[targetIdx]+=weight*key;
	VectorXf r=keyMean[targetIdx]-key;
	float nr=n.dot(r);
	nSum[targetIdx]+=(normalRegularizationBias*weight*nr)*n;
	r-=nr*n;  //only tangential component remains
	nSum[targetIdx]+=(tangentialRegularizationBias*weight)*r;
	//update weight
	nWeightSum[targetIdx]+=weight;
}

void OnlinePiecewiseLinearRegression::getData( const Eigen::VectorXf &key, Eigen::VectorXf &data )
{
	float proj=n.dot(key)+b;
	if (proj<0)
		data=dataMean[0];
	else if (proj>0)
		data=dataMean[1];
	else
	{
		data=dataMean[rand01()];
	}
}

void OnlinePiecewiseLinearRegression::getDataAndClosestKeyMean( const Eigen::VectorXf &key, Eigen::VectorXf &data, Eigen::VectorXf &out_keyMean )
{
	float proj=n.dot(key)+b;
	if (proj<0)
	{
		data=dataMean[0];
		out_keyMean=keyMean[0];
	}
	else if (proj>0)
	{
		data=dataMean[1];
		out_keyMean=keyMean[1];
	}
	else
	{
		int idx=rand01();
		data=dataMean[idx];
		out_keyMean=keyMean[idx];
	}

}


float OnlinePiecewiseLinearRegression::getDistanceToClosestKeyMean( const Eigen::VectorXf &key )
{
	float proj=n.dot(key)+b;
	if (proj<0)
	{
		return (key-keyMean[0]).norm();
	}
	else if (proj>0)
	{
		return (key-keyMean[1]).norm();
	}
	else
	{
		return (key-keyMean[rand01()]).norm();
	}

}


void OnlinePiecewiseLinearRegression::getDataLerped( const Eigen::VectorXf &key, Eigen::VectorXf &data )
{
	float proj=n.dot(key)+b;
	float range=(keyMean[1]-keyMean[0]).norm();
	proj=clipMinMaxf(proj/range*0.5f,-0.5f,0.5f);
	data=0.5f*(dataMean[0]+dataMean[1])+proj*(dataMean[1]-dataMean[0]);

}

float OnlinePiecewiseLinearRegression::getDistanceFromTrainingSet( const Eigen::VectorXf &key )
{
	float proj=n.dot(key)+b;
	int targetIdx;
	if (proj<0)
		targetIdx=0;
	else if (proj>0)
		targetIdx=1;
	else
	{
		targetIdx= rand01();
	}
	keyTemp=keyMean[1]-keyMean[0];
	float var=keyTemp.squaredNorm();
	keyTemp=key-keyMean[targetIdx];
	float sqDev=keyTemp.squaredNorm();
	return sqrtf(sqDev/var);
}

void OnlinePiecewiseLinearRegression::setSplitToBetweenKeyMeans()
{
	n=(keyMean[1]-keyMean[0]).normalized();
	b=-0.5f*n.dot(keyMean[1]+keyMean[0]);
}


} //AaltoGames