#include "OnlineForest.h"

static double volumePow=0.0;
static double weightPow=0.0f;
namespace AaltoGames
{



//a class and static instance for evaluating Gaussian pdf definite integrals
class GaussianIntegral
{
	VectorXd cdf;
	static const int nTabledSd=20;
	static const int tableSize=65536;
public:
	GaussianIntegral()
	{
		cdf.resize(tableSize);
		double acc=0;
		for (int i=0; i<tableSize; i++)
		{
			float x=(float)(i*2-tableSize);
			x/=(float)tableSize;  //to -1...1
			x*=(float)nTabledSd;
			acc+=exp(-0.5f*x*x);
			cdf[i]=acc;
		}
		cdf=cdf/acc;

	}
	int toIndex(float x, float mean, float sd, float biasInBins=0)
	{
		x-=mean;
		x/=sd;
		x/=(float)nTabledSd;
		x+=0.5f;
		x*=(float)tableSize;
		x+=biasInBins;
		return clipMinMaxi((int)x,0,tableSize-1);
	}
	double eval(float rangeStart, float rangeEnd, float mean, float sd)
	{
		//the eval range is expanded with 1 bin so that the integral does not evaluate to 0 even if both rangeStart and rangeEnd fall in the same bin
		int startIndex=toIndex(rangeStart,mean,sd,-0.5f);
		int endIndex=toIndex(rangeEnd,mean,sd,0.5f);
		return cdf[endIndex]-cdf[startIndex];
	}
};
static GaussianIntegral gi;

OnlineTree::OnlineTree( OnlineTree *parent/*=NULL*/,PoolAllocator<OnlineTree> *allocator/*=NULL*/ )
{
	this->allocator=allocator;
	children[0]=children[1]=NULL;
	depth=1;
	weight=1;
	this->parent=parent;
	nSamples=0;
	firstToPrune=this;
	importance=1;
	bounded=false;
	volume=1;
}

void OnlineTree::setBounds( const VectorXf &minCorner, const VectorXf &maxCorner )
{
	Debug::throwError("setBounds() not implemented");
	//bounded=true;
	//this->minCorner=minCorner;
	//this->maxCorner=maxCorner;
	//for (int i=0; i<minCorner.rows(); i++)
	//{
	//	//ensure that node volume computations will result in something reasonable
	//	if (this->minCorner[i]==this->maxCorner[i])
	//	{
	//		const float epsilon=0.001f;
	//		this->minCorner[i]-=epsilon;
	//		this->maxCorner[i]+=epsilon;
	//	}
	//}
}

OnlineTree * OnlineTree::findNode( const VectorXf &key )
{
	//drop key to leaf
	OnlineTree *t=this;
	while (t->hasChildren())
	{
		if (key[t->splitVar]<t->splitLoc)
			t=t->children[0];
		else //if (key[t->splitVar]<t->splitLoc)
			t=t->children[1];
		//else
		//{
		//	t=t->children[randInt(0,1)];
		//}
	}
	return t;
}

void OnlineTree::clear()
{
	//		allocator->reset();
	////using the c style callback interface to avoid heap allocs, this method may be called frequently
	processAll_cb([](OnlineTree *tree, void *data){
		AALTO_ASSERT1(tree->allocator!=NULL);
		if (tree!=data)	//free everything but this (on which clear() called)
		{
			tree->allocator->free(tree);
		}
	},this);
	//for root, set nSamples=0 to signal three not initialized
	nSamples=0;
}

void OnlineTree::add( const VectorXf &key, void *data, double weight/*=1*/, int time/*=0*/ )
{
	AALTO_ASSERT1(key.rows()>0);
	//if tree not initialized, init root with the key and data
	if (nSamples==0)
	{
		init(key,data,weight,time);
	}
	else
	{
		//find the node to split
		findNode(key)->addToThis(key,data,weight,time);
	}
}

void OnlineTree::processAll( const std::function<void (OnlineTree *)> &func )
{
	if (nSamples>1)
	{
		children[0]->processAll(func);
		children[1]->processAll(func);
	}
	else
	{
		func(this);
	}
}

void OnlineTree::processAll_cb( void(*callback)(OnlineTree *, void *),void *data/*=NULL*/ )
{
	if (children[0]!=NULL)
		children[0]->processAll_cb(callback,data);
	if (children[1]!=NULL)
		children[1]->processAll_cb(callback,data);
	callback(this,data);
}

int OnlineTree::getLeaves( OnlineTree **out_leaves, int nSoFar/*=0*/ )
{
	if (nSamples>1)
	{
		nSoFar=children[0]->getLeaves(out_leaves,nSoFar);
		nSoFar=children[1]->getLeaves(out_leaves,nSoFar);
	}
	else
	{
		out_leaves[nSoFar] = this;
		nSoFar++;
	}
	return nSoFar;
}

void OnlineTree::pruneTo( int maxSamples )
{
	while (nSamples>maxSamples)
	{
		if (firstToPrune==this)
			nSamples=0;  //don't delete this...
		else
		{
			AALTO_ASSERT1(firstToPrune->nSamples==1);	//must only prune leaf nodes
			prune(firstToPrune);
		}
	}
}

OnlineTree * OnlineTree::prune( OnlineTree *pruned )
{
	if (pruned==this->getRoot() && this->getRoot()->nSamples<=1)
	{
		this->getRoot()->nSamples=0;
		return NULL;
	}
	OnlineTree *parent=pruned->parent;
	AALTO_ASSERT1(pruned->getRoot() == this);	//check that the pruned exists in this tree
	AALTO_ASSERT1(findNode(*pruned->key)==pruned);	//check that the pruned exists in this tree
	AALTO_ASSERT1(parent!=NULL); //can't prune root
	AALTO_ASSERT1(pruned->nSamples==1); //can't prune other than leaves
	int prunedIdx= pruned==parent->children[0] ? 0 : 1;
	OnlineTree *kept=parent->children[1-prunedIdx];
	allocator->free(pruned);
	//Replace parent with the kept subtree or pruned, i.e., replace all parent's properties except its parent
	//Note that we can't just delete the parent, as someone might hold a pointer to it
	//kept->setBounds(parent->minCorner,parent->maxCorner);
	//Replace parent of pruned by the pruned node's sibling.
	//First copy all members except parent
	kept->parent=parent->parent;
	*parent=*kept; 
	allocator->free(kept);
	//Relink children
	if (parent->children[0]!=NULL)
		parent->children[0]->parent=parent;
	if (parent->children[1]!=NULL)
		parent->children[1]->parent=parent;
	if (parent->firstToPrune==kept || parent->firstToPrune==pruned)
		parent->firstToPrune=parent;
	AALTO_ASSERT1(parent->parent==NULL || (parent->parent->children[0]==parent || parent->parent->children[1]==parent)); //check that the structure is not broken
	parent->propagate();
//	if (firstToPrune==pruned)
//		parent->propagate();
	AALTO_ASSERT1(firstToPrune!=pruned);
	return parent;
}

void OnlineTree::setImportance( double importance )
{
	this->importance=importance;
	propagate();
}

void OnlineTree::init( const VectorXf &key, void *data, double weight, int time )
{
	this->key=&key;
	this->data=data;
	this->weight=weight;
	this->subTreeIntegral=weight;
	this->time=time;
	if (parent!=NULL)
		depth=parent->depth+1;
	else
	{
		depth=1;
	}
	nSamples=1;
	children[0]=children[1]=NULL;
	//if (bounded)
	//{
	//	//update bounds in case the sample falls outside them
	//	for (int i=0; i<key.cols(); i++)
	//	{
	//		minCorner[i]=_min(minCorner[i],key[i]);
	//		maxCorner[i]=_max(maxCorner[i],key[i]);
	//	}
	//	double volumeTmp=1;
	//	for (int i=0; i<key.cols(); i++)
	//	{
	//		volumeTmp*=maxCorner[i]-minCorner[i];
	//	}
	//	volume=volumeTmp;
	//	//AALTO_ASSERT1(volumePow==0 || volume>0); //if volumePow>0 can't have zero volume nodes
	//}
	//else
	//{
		volume=pow(0.5,depth);
	//}
	importance=pow(weight,weightPow)*pow(volume,volumePow); 
	firstToPrune=this;
}

void OnlineTree::addToThis( const VectorXf &newKey, void *newData, double newWeight, int newTime )
{
	//if key exactly same, just update data
	if (*key==newKey)
	{
		if (newWeight>=weight)
		{
			data=newData;
			key=&newKey;
			time=newTime;
			weight=newWeight;
			subTreeIntegral=weight;
		}
		propagate();
		return;
	}

	//random split variable, split in the middle between old and new sample.
	//rejection sample until found a variable where the old and new keys differ
	float maxDiff=-1.0f;
	splitVar=-1;
	for (int i=0; i<(*key).rows(); i++)
	{
		float diff=fabs((*key)[i]-newKey[i]);
		if (diff>maxDiff)
		{
			maxDiff=diff;
			splitVar=i;
		}
	}
	//do 
	//{
	//	splitVar=randInt(0,(*key).rows()-1);
	//} while ((*key)[splitVar]==newKey[splitVar]);
	splitLoc=0.5f*((*key)[splitVar]+newKey[splitVar]);
	children[0]=allocator->allocate();
	children[1]=allocator->allocate();
	children[0]->parent=this;
	children[0]->allocator=allocator;
	children[1]->parent=this;
	children[1]->allocator=allocator;
	//if (bounded)
	//{
	//	children[0]->setBounds(minCorner,maxCorner);
	//	children[1]->setBounds(minCorner,maxCorner);
	//	children[0]->maxCorner[splitVar]=splitLoc;
	//	children[1]->minCorner[splitVar]=splitLoc;
	//}
	int newIdx=newKey[splitVar]<splitLoc ? 0 : 1;
	children[newIdx]->init(newKey,newData,newWeight,newTime);
	children[1-newIdx]->init((*key),data,weight,time);
	propagate();
}

OnlineTree * OnlineTree::getRoot()
{
	OnlineTree *result=this;
	while (result->parent!=NULL)
		result=result->parent;
	return result;
}



void OnlineTree::propagate()
{
	OnlineTree *tree=this;
	do 
	{
		if (tree->children[0]!=NULL && tree->children[1]!=NULL)
		{
			tree->subTreeIntegral=tree->children[0]->subTreeIntegral+tree->children[1]->subTreeIntegral;
			tree->nSamples=tree->children[0]->nSamples + tree->children[1]->nSamples;
			//if (tree->children[0]->hasChildren() && !tree->children[1]->hasChildren())
			//	tree->firstToPrune=tree->children[0]->firstToPrune;
			//else if (!tree->children[0]->hasChildren() && tree->children[1]->hasChildren())
			//	tree->firstToPrune=tree->children[1]->firstToPrune;
			//else{
			if (tree->children[0]->firstToPrune->importance < tree->children[1]->firstToPrune->importance)
				tree->firstToPrune=tree->children[0]->firstToPrune;
			else if (tree->children[0]->firstToPrune->importance > tree->children[1]->firstToPrune->importance)
				tree->firstToPrune=tree->children[1]->firstToPrune;
			else
				tree->firstToPrune=tree->children[rand01()]->firstToPrune;
			//				}
			AALTO_ASSERT1(tree->firstToPrune->nSamples==1);	//must only prune leaf nodes
		}
		//if (tree->firstToPrune->nSamples>1)
		//	tree=tree->firstToPrune;  //should never happen
		//else
		AALTO_ASSERT1(tree->parent==NULL || (tree->parent->children[0]==tree || tree->parent->children[1]==tree)); //check that the structure is not broken
		tree=tree->parent;
	} while (tree!=NULL);
}

OnlineTree * OnlineTree::sample()
{
	OnlineTree *tree=this;
	while (tree->hasChildren())
	{
		double p0=tree->children[0]->subTreeIntegral;
		double p1=tree->children[1]->subTreeIntegral;

		double total=random()*(p0+p1);
		if (total<p0)
			tree=tree->children[0];
		else if (total>p0)
			tree=tree->children[1];
		else
		{
			tree=tree->children[rand01()];
		}
	}
	return tree;
}



void OnlineForest::init( int nTrees, int maxSamples/*=100*/, PruningModes pruningMode) 
{
	trees.resize(nTrees+1);
	nSamples=0;
	this->pruningMode=pruningMode;
	this->nTrees=nTrees;
	this->maxSamples=maxSamples;
	minmax.init(maxSamples);
	for (int i=0; i<maxSamples; i++)
	{
		minmax.setValue(i,DBL_MAX);
	}
	samples.resize(maxSamples+1);
	rebuildSpeed=10; //by default, perform this many rebuild sample additions per addSample()
	int poolSize=(nTrees+1)*maxSamples*4;
	if (allocator.poolSize()!=poolSize)
		allocator.init(poolSize);  //TODO: solve what's the correct bound for the memory usage
	else
	{
		allocator.reset();
	}
	//we allocate one extra tree that will be constantly rebuilt and then swapped in place of some other when ready
	for (int i=0; i<nTrees+1; i++)
	{ 
		trees[i]=new OnlineTree(); 
		trees[i]->allocator = &allocator;
	}
	time=0;
	rebuildTreeIdx=0;
	rebuildSampleIdx=0;
	randomizer.init(1);		//rebuilding will be first done with 1 sample
}

void OnlineForest::setBounds( const VectorXf &minCorner, const VectorXf &maxCorner )
{
	for (int i=0; i<nTrees+1; i++)
		trees[i]->setBounds(minCorner,maxCorner);
}

void OnlineForest::addSample( const VectorXf &key, const VectorXf &data, double weight )
{
	if (nSamples==0)
	{
		//allocate buffers
		for (int i=0; i<maxSamples; i++)
		{
			samples[i].data.resize(data.rows());
			samples[i].key.resize(key.rows());
		}
	}
	int storeIdx;
	OnlineTree *tree=NULL;
	if (nSamples>0)
		tree=trees[0]->findNode(key);
	//Check for duplicates
	const float replaceProbability=0;//0.5f;
	if (tree==NULL || (key!=*tree->key && randomf()>=replaceProbability))	
	{
		//if not a duplicate, this is is a new sample
		storeIdx=nSamples;
		nSamples++;
		if (storeIdx==maxSamples){
			//random pruning: if forest full, just replace random sample
			if (pruningMode==pmRandom)
			{
				storeIdx=randInt(0,maxSamples-1); 
			}
			else
			{
				storeIdx=minmax.getMinIdx();
				const int nIter=3;
				//here we update the priorities a few times in case there's been changes due to rebuilding etc.
				for (int i=0; i<nIter; i++)
				{
					Sample &s=samples[storeIdx];
					updateSampleStatistics(s);
					if (pruningMode==pmMinDataDiff)
						minmax.setValue(storeIdx,s.weight*(s.dataVariance+s.keyVariance));
					else
						minmax.setValue(storeIdx,s.weight*s.dataVariance);
					int old=storeIdx;
					storeIdx=minmax.getMinIdx();
					if (old==storeIdx)
						break;
				}
			}
			//prune old one
			//TODO: select pruned based on volumes, update tree 0 volumes as means
			for (int treeIdx=0; treeIdx<=nTrees; treeIdx++)
			{
				OnlineTree *tree=trees[treeIdx];
				OnlineTree *leaf=tree->findNode(samples[storeIdx].key);
				//bool found = samples[storeIdx].key == samples[(int)leaf->data].key;
				bool found = samples[storeIdx].key == *leaf->key;
				//AALTO_ASSERT1(treeIdx==nTrees || found);	//the sample not necessarily yet in the tree being rebuilt. TODO find why this occasionally still asserts
				if (found && !(treeIdx==nTrees && rebuildSampleIdx==0 && storeIdx==0))  //Note the edge case where rebuilding has just finished and we would prune sample 0
					tree->prune(leaf);
			}
			nSamples=maxSamples;
		}
	}
	else
	{
		//Handle duplicates. If new weight smaller than old one, return. Otherwise replace;
		if (weight<tree->weight)
		{
			return;
		}
		storeIdx=(int)tree->data;
	}
	samples[storeIdx].key=key;
	samples[storeIdx].data=data;
	samples[storeIdx].weight=weight;

	//update bounds info (will be applied when rebuilding)
	if (nSamples==1)
	{
		minKey=key;
		maxKey=key;
		//setBounds(minKey,maxKey);
	}
	else
	{
		for (int i=0; i<key.rows(); i++)
		{
			minKey[i]=_min(minKey[i],key[i]);
			maxKey[i]=_max(maxKey[i],key[i]);
		}
	}
	time++;

	//add sample to all trees
	for (int treeIdx=0; treeIdx<=nTrees; treeIdx++)
	{
		OnlineTree &tree=*trees[treeIdx];
		tree.add(samples[storeIdx].key,(void *)storeIdx,weight,time);
		AALTO_ASSERT1(storeIdx==(int)(tree.findNode(samples[storeIdx].key)->data));
	}

	//update pruning priority
	if (pruningMode!=pmRandom)
	{
		Sample &s=samples[storeIdx];
		updateSampleStatistics(s);
		if (pruningMode==pmMinDataDiff)
			minmax.setValue(storeIdx,s.weight*(s.dataVariance+s.keyVariance));
		else
			minmax.setValue(storeIdx,s.weight*s.dataVariance);
	}

	//rebuilding
	{
		//rebuild, i.e., add all samples to a new tree in random order
 		int chunkStart=rebuildSampleIdx;
		int chunkEnd=_min(nSamples,rebuildSampleIdx+rebuildSpeed);
		OnlineTree &tree=*trees[nTrees];
		for (; rebuildSampleIdx<chunkEnd; rebuildSampleIdx++)
		{
			int idx=randomizer.get();
			Sample &sample=samples[idx];
			bool alreadyAdded=false;
			if (tree.nSamples>0)
			{
				OnlineTree *leaf=tree.findNode(samples[idx].key);
				alreadyAdded= samples[idx].key == *leaf->key;
			}
			if (!alreadyAdded)
			{
				tree.add(sample.key,(void *)idx,sample.weight);
				AALTO_ASSERT1(idx==(int)tree.findNode(sample.key)->data);
			}
		}
		//rebuilding done?
		if (rebuildSampleIdx>=nSamples)
		{
			//last step of rebuilding: prune if there were pruning operations that happened while rebuilding
			//(the prune operations must be queued, as the keys to prune might not have been yet added to the rebuilt tree)
			//swap the rebuilt tree in place of the next one to rebuild
			_swap(trees[rebuildTreeIdx],trees[nTrees]);
			rebuildTreeIdx=(rebuildTreeIdx+1) % nTrees;

			OnlineTree &tree=*trees[nTrees];
			tree.clear();
			rebuildSampleIdx=0;
			randomizer.init(nSamples);
		}
	}

	////Prune trees to contain at most maxSamples samples. Note that we use tree 0 as the "master" tree
	////that updates importances based on the average volume occupied by the samples in all trees.
	////The master alone defines the order of pruning samples.
	////This is to avoid a quadratic cost of calling updateSampleImportanceBasedOnMeanVolume() for all trees.
	//{
	//	if (randomPruning)
	//	{
	//		for (int treeIdx=0; treeIdx<nTrees; treeIdx++)
	//		{
	//			OnlineTree &tree=*trees[treeIdx];
	//			while (tree.nSamples>maxSamples)
	//			{
	//				tree.prune(tree.sample());
	//			}
	//		}
	//	}
	//	else
	//	{
	//		OnlineTree &tree=*trees[0];
	//		updateSampleImportanceBasedOnMeanVolume(key,tree);
	//		while (tree.nSamples>maxSamples)
	//		{
	//			Eigen::VectorXf *keyToPrune=NULL;
	//			for (int i=0; i<10; i++)
	//			{
	//				OnlineTree *oldFirstToPrune=tree.firstToPrune;
	//				updateSampleImportanceBasedOnMeanVolume(oldFirstToPrune->key, tree);
	//				if (tree.firstToPrune==oldFirstToPrune)
	//					break;
	//			}
	//			keyToPrune=&tree.firstToPrune->key;				
	//			for (int treeIdx=1; treeIdx<nTrees; treeIdx++)
	//			{
	//				OnlineTree *t=trees[treeIdx]->findNode(*keyToPrune);
	//				//AALTO_ASSERT1(t->key==keyToPrune);
	//				if (t->key==*keyToPrune)
	//					trees[treeIdx]->prune(t);
	//			}
	//			keysToPruneAfterRebuild[keysToPruneQueueIdx++]=*keyToPrune;
	//			tree.prune(tree.firstToPrune);
	//		}
	//	}
	//}
	//printf("Memory remaining: %d\n",allocator.numFree());
}

void OnlineForest::updateSampleImportanceBasedOnMeanVolume( const VectorXf &key, OnlineTree &tree )
{
	if (volumePow==0)
		return;
	//tree 0 acts as the "master" tree, using its volume as the average of other trees
	double meanVolume=0;
	for (int treeIdx=0; treeIdx<nTrees; treeIdx++)
	{
		OnlineTree *target=tree.findNode(key);
//		AALTO_ASSERT1(target->key==key);  //TODO why does this fail
		meanVolume+=target->volume;
	}
	meanVolume/=(double)(nTrees);
	OnlineTree *target=tree.findNode(key);
//	AALTO_ASSERT1(target->key==key); //TODO why does this fail
	target->setImportance(pow(meanVolume,volumePow)*pow(target->weight,weightPow));
}


//old implementation, only samples among nearest
OnlineTree *OnlineForest::sampleWithPriorNearestOnly( const VectorXf &x, const VectorXf &xSd )
{
	double desirabilities[1000];
	OnlineTree *neighbors[1000];
	AALTO_ASSERT1(nTrees*2<1000);
	double total=0;
	int nNeighbors=0;
	for (int i=0; i<nTrees; i++)
	{
		OnlineTree *tree=trees[i]->findNode(x);
		if (tree->parent!=NULL)
		{
			neighbors[nNeighbors++]=tree->parent->children[0];
			neighbors[nNeighbors++]=tree->parent->children[1];
		}
		else
		{
			neighbors[nNeighbors++]=tree;
		}
	}
	for (int i=0; i<nNeighbors; i++)
	{
		OnlineTree *tree=neighbors[i];
		double desirability=tree->weight;
		float acc=0;
		for (int j=0; j<x.rows(); j++)
		{
			float diff=x[j]-(*tree->key)[j];
			acc+=squared(diff)/squared(xSd[j]);
		}
		desirability*=exp(-0.5f*acc);
		desirabilities[i]=desirability;
		total+=desirability;
	}

	int selected=randInt(0,nNeighbors-1);
	if (total>0)
	{
		double r=random()*total;
		double cumulative=desirabilities[0];
		selected=0;
		while (cumulative<r && selected<nNeighbors-1)
		{
			selected++;
			cumulative+=desirabilities[selected];
		}
	}
	return neighbors[selected];
}


OnlineTree * OnlineForest::sample()
{
	int treeIdx=randInt(0,nTrees-1);
	return trees[treeIdx]->sample();
}

OnlineTree *OnlineForest::sampleWithPrior( const VectorXf &x, const VectorXf &xSd, SamplingMethods method, int nIter )
{
	if (method==smOnlyNearest)
		return sampleWithPriorNearestOnly(x,xSd);	
	const int maxKeyDim=512;
	float minCorner[maxKeyDim],maxCorner[maxKeyDim];
	AALTO_ASSERT1(x.rows()<=maxKeyDim);
	AALTO_ASSERT1(x.rows()==minKey.rows());
	int keyDim=minKey.rows();
	for (int i=0; i<keyDim; i++)
	{
		minCorner[i]=minKey[i];
		maxCorner[i]=maxKey[i];
	}
	AALTO_ASSERT1(nIter==1);  //others not implemented yet
	//const int maxIter=32;
	//OnlineTree *iterated[maxIter];
	//double cdf[maxIter];
	//AALTO_ASSERT1(nIter<=maxIter);

	//for (int iter=0; iter<nIter; iter++)
	//{
		int treeIdx=randInt(0,nTrees-1);
		OnlineTree *tree=trees[treeIdx];
		while (tree->hasChildren())
		{
			//approximate integral of products with product of integrals, but ignore the Gaussian prior if we are outside the tabled range
			int d=tree->splitVar;
			double leftPriorIntegral=gi.eval(minCorner[d],tree->splitLoc,x[d],xSd[d]);
			double rightPriorIntegral=gi.eval(tree->splitLoc,maxCorner[d],x[d],xSd[d]);
			if (leftPriorIntegral==0 && rightPriorIntegral==0)  //check for gi computing precision error
			{
				leftPriorIntegral=1.0;
				rightPriorIntegral=1.0;
			}
			double leftIntegral=tree->children[0]->subTreeIntegral;
			double rightIntegral=tree->children[1]->subTreeIntegral;
			leftIntegral*=leftPriorIntegral;
			rightIntegral*=rightPriorIntegral;
			double r=random()*(leftIntegral+rightIntegral);
			if (r<leftIntegral || (r==leftIntegral && rand01()==0))
			{
				maxCorner[d]=tree->splitLoc;
				tree=tree->children[0];
			}
			else if (r>leftIntegral)
			{
				minCorner[d]=tree->splitLoc;
				tree=tree->children[1];
			}
		}
		//if (nIter==1)
		//iterated[nIter]=tree;
		//cdf[nIter]=tree->weight;


//	}
	return tree;
	double total=0;
}



void OnlineForest::getNeighbors( int maxReturned, const VectorXf &key, OnlineTree **out_neighbors, int &out_numNeighbors, int searchDepth )
{
	AALTO_ASSERT1(searchDepth<=2);
	out_numNeighbors=0;
	for (int i=0; i<nTrees; i++)
	{
		OnlineTree *tree=trees[i]->findNode(key);
		if (tree->parent!=NULL && searchDepth>1)
		{
			out_neighbors[out_numNeighbors++]=tree->parent->children[0]->findNode(key);
			if (out_numNeighbors>=maxReturned)
				return;
			out_neighbors[out_numNeighbors++]=tree->parent->children[1]->findNode(key);
			if (out_numNeighbors>=maxReturned)
				return;
		}
		else
		{
			out_neighbors[out_numNeighbors++]=tree;
			if (out_numNeighbors>=maxReturned)
				return;
		}
	}
}



void OnlineForest::getNeighborSamples( int maxReturned, const VectorXf &key, Sample **out_neighbors, int &out_numNeighbors, int searchDepth )
{
	AALTO_ASSERT1(searchDepth<=2);
	out_numNeighbors=0;
	for (int i=0; i<nTrees; i++)
	{
		OnlineTree *tree=trees[i]->findNode(key);
		if (tree->parent!=NULL && searchDepth>1)
		{
			out_neighbors[out_numNeighbors++]=&samples[(int)tree->parent->children[0]->findNode(key)->data];
			if (out_numNeighbors>=maxReturned)
				return;
			out_neighbors[out_numNeighbors++]=&samples[(int)tree->parent->children[1]->findNode(key)->data];
			if (out_numNeighbors>=maxReturned)
				return;
		}
		else
		{
			out_neighbors[out_numNeighbors++]=&samples[(int)tree->data];
			if (out_numNeighbors>=maxReturned)
				return;
		}
	}
}
OnlineForest::OnlineForest()
{
}

void OnlineForest::getMeanData( const VectorXf &key,VectorXf &out_data )
{
	out_data.setZero();
	float wTot=0;
	for (int i=0; i<nTrees; i++)
	{
		Sample &s=samples[(int)trees[i]->findNode(key)->data];
		out_data+=s.data*(float)s.weight;
		wTot+=(float)s.weight;
	}
	out_data/=wTot;
}


void OnlineForest::getMeanDataWithoutOutliers( const VectorXf &key,VectorXf &out_data, float outlierThreshold )
{
	AALTO_ASSERT1(nTrees<=256);
	Sample *found[256];
	double distances[256];
	double meanDist=0;
	for (int i=0; i<nTrees; i++)
	{
		found[i]=&samples[(int)trees[i]->findNode(key)->data];
	}
	for (int i=0; i<nTrees; i++)
	{
		double dist=(found[i]->key-key).squaredNorm();
		distances[i]=dist;
		meanDist+=dist;
	}
	meanDist/=(double)nTrees;
	float wTot=0;
	out_data.setZero();
	for (int i=0; i<nTrees; i++)
	{
		double dist=distances[i];
		if (dist<=meanDist*outlierThreshold)
		{
			out_data+=found[i]->data*(float)found[i]->weight;
			wTot+=(float)found[i]->weight;
		}
	}
	out_data/=wTot;
}

bool OnlineForest::getWeightedMeanData( const VectorXf &key,const VectorXf &keySd, VectorXf &out_data )
{
	int nControlDimensions=out_data.rows();
	AALTO_ASSERT1(nControlDimensions<=1000);
	double result[1000];
	double totalWeight=0;
	memset(result,0,sizeof(double)*nControlDimensions);
	float acc=0;
	for (int i=0; i<nTrees; i++)
	{
		Sample &sample=samples[(int)trees[i]->findNode(key)->data];
		double weight=exp(-0.5*((key-sample.key).cwiseQuotient(keySd)).squaredNorm());
		weight*=sample.weight;
		for (int j=0; j<nControlDimensions; j++)
		{
			result[j]+=weight*sample.data[j];
		}
		totalWeight+=weight;
	}
	bool valid=true;
	for (int j=0; j<nControlDimensions; j++)
	{
		float scaled=(float)(result[j]/totalWeight);
		if (!validFloat(scaled))
		{
			valid=false;
			scaled=0;
		}
		out_data[j]=scaled;
	}
	return valid;
}

OnlineForest::Sample * OnlineForest::getNearest( const VectorXf &key, const VectorXf &keySd, float &out_quadraticForm )
{
	Sample *neighbors[1000];
	int nNeighbors=0;
	getNeighborSamples(1000,key,neighbors,nNeighbors);
	float minqf=FLT_MAX;
	Sample *result=NULL;
	for (int i=0; i<nNeighbors; i++)
	{
		float f=((key-neighbors[i]->key).cwiseQuotient(keySd)).squaredNorm();
		if (f<minqf)
		{
			minqf=f;
			result=neighbors[i];
		}
	}
	out_quadraticForm=minqf;
	return result;
}

void OnlineForest::getNearestData( const VectorXf &key, const VectorXf &keySd, VectorXf &out_data )
{
	float temp;
	Sample *nearest=getNearest(key,keySd,temp);
	out_data=nearest->data;
}

void OnlineForest::updateSampleStatistics( Sample &s )
{
	double keyVariance=0;
	double dataVariance=0;
	for (int i=0; i<nTrees; i++)
	{
		OnlineTree *tree=trees[i]->findNode(s.key);
		if (tree->parent!=NULL)
		{
			Sample &sample0=samples[(int)tree->parent->children[0]->findNode(s.key)->data];
			Sample &sample1=samples[(int)tree->parent->children[1]->findNode(s.key)->data];
			keyVariance+=(sample0.key-sample1.key).squaredNorm();
			dataVariance+=(sample0.data-sample1.data).squaredNorm();
		}
	}
	s.keyVariance=keyVariance;
	s.dataVariance=dataVariance;
}


} //AaltoGames