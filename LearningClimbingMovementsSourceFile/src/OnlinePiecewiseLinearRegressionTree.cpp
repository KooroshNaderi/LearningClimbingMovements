#include "OnlinePiecewiseLinearRegressionTree.h"

namespace AaltoGames
{

	OnlinePiecewiseLinearRegressionTree::~OnlinePiecewiseLinearRegressionTree(){


		if (hasChildren){
			for (OnlinePiecewiseLinearRegressionTree* tmp : children){
				delete tmp;
			}
		}


	}

	void OnlinePiecewiseLinearRegressionTree::init( OnlinePiecewiseLinearRegressionTree* parent, int keyDim, int dataDim, int maxDepth, int nSamplesPerLeaf, float regularization, int cacheSize)
	{
		this->parent=parent;
		this->maxDepth=maxDepth;
		this->keyDim=keyDim;
		this->dataDim=dataDim;
		this->nSamplesPerLeaf=nSamplesPerLeaf;
		this->regularization=regularization;
		this->cacheSize=cacheSize;
		//alpha^nSamplesPerLeaf=regularization => nSamplesPerLeaf * log(alpha)=log(regularization) => alpha=regularization^(1/nSamplesPerLeaf)
		model.init(keyDim,dataDim,(float)pow(regularization,1.0/(double)nSamplesPerLeaf),cacheSize,0,0);
		//		model.init(keyDim,dataDim,regularization*0.95f,cacheSize,regularization*0.5f,regularization*0.5f);
		depth=1;
		hasChildren=false;
		if (parent!=NULL)
			depth+=parent->depth;
	}
	OnlinePiecewiseLinearRegressionTree *getSibling(OnlinePiecewiseLinearRegressionTree *t)
	{
		if (t==t->parent->children[0])
			return t->parent->children[1];
		else
		{
			return t->parent->children[0];
		}
	}
	void OnlinePiecewiseLinearRegressionTree::addSample( const VectorXf &key, const VectorXf &data, float weight )
	{
		if (hasChildren)
		{
			findLeaf(key)->addSample(key,data,weight);
			return;
		}
		else
		{
			model.addSample(key,data,weight);		//optimize data clustering
			if (depth<maxDepth && model.nSamples>nSamplesPerLeaf)
			{
				//model.setSplitToBetweenKeyMeans();
				children[0]=new OnlinePiecewiseLinearRegressionTree();
				children[1]=new OnlinePiecewiseLinearRegressionTree();
				children[0]->init(this,keyDim,dataDim,maxDepth,nSamplesPerLeaf,regularization,cacheSize);
				children[1]->init(this,keyDim,dataDim,maxDepth,nSamplesPerLeaf,regularization,cacheSize);
				hasChildren=true;
			}
		}
	}

	void OnlinePiecewiseLinearRegressionTree::processAll(const  std::function<void (OnlinePiecewiseLinearRegressionTree *)> &func )
	{
		func(this);
		if (hasChildren)
		{
			children[0]->processAll(func);
			children[1]->processAll(func);
		}
	}

	void OnlinePiecewiseLinearRegressionTree::getData( const VectorXf &key, VectorXf &out_data )
	{
		OnlinePiecewiseLinearRegressionTree *t=findLeaf(key);
		//backtrack one to always get reliable data (not from a model currently being built)
		if (t->parent!=NULL && t->model.nSamples<nSamplesPerLeaf)
			t=t->parent;
		t->model.getData(key,out_data);
	}

	void OnlinePiecewiseLinearRegressionTree::getDataAndClosestKeyMean( const VectorXf &key, VectorXf &out_data, VectorXf &out_keyMean )
	{
		OnlinePiecewiseLinearRegressionTree *t=findLeaf(key);
		//backtrack one to always get reliable data (not from a model currently being built)
		if (t->parent!=NULL && t->model.nSamples<nSamplesPerLeaf)
			t=t->parent;
		t->model.getDataAndClosestKeyMean(key,out_data,out_keyMean);

	}

	float OnlinePiecewiseLinearRegressionTree::getDistanceToClosestKeyMean( const Eigen::VectorXf &key )
	{
		OnlinePiecewiseLinearRegressionTree *t=findLeaf(key);
		//backtrack one to always get reliable data (not from a model currently being built)
		if (t->parent!=NULL && t->model.nSamples<nSamplesPerLeaf)
			t=t->parent;
		return model.getDistanceToClosestKeyMean(key);


	}

	OnlinePiecewiseLinearRegressionTree * OnlinePiecewiseLinearRegressionTree::findLeaf( const VectorXf &key )
	{
		OnlinePiecewiseLinearRegressionTree *t=this;
		while (t->hasChildren)
		{
			float proj=t->model.n.dot(key)+t->model.b;
			if (proj>0)
			{
				t=t->children[1];
			}
			else if (proj<0)
			{
				t=t->children[0];
			}
			else
			{
				t=t->children[rand01()];
			}
		}
		return t;
	}

	float OnlinePiecewiseLinearRegressionTree::getDistanceFromTrainingSet( const Eigen::VectorXf &key )
	{
		OnlinePiecewiseLinearRegressionTree *t=findLeaf(key);
		//backtrack one to always get reliable data (not from a model currently being built)
		if (t->parent!=NULL && t->model.nSamples<nSamplesPerLeaf)
			t=t->parent;
		return t->model.getDistanceFromTrainingSet(key);
	}


}