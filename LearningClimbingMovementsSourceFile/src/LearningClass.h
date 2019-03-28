#pragma once

#include <Eigen/Eigen> 
#include <vector>
#include "DiagonalGMM.h"
#include "TrajectoryOptimization.h"
#include "OnlineForest.h"
#include "OnlinePiecewiseLinearRegressionTree.h"
#include "GenericDensityForest.hpp"
#include "RegressionUtils.hpp"
#include "LearnerControl\ANN.h"
//#include "MixtureTree.h"
#include "MemoryStack.hpp"
#include "ProbUtils.hpp"

#include <future>
#include <iostream>
#include <fstream>
#include <ctime>
#include <deque>
#include <random>
#include <map>
#include <exception>

using namespace AaltoGames;

enum mMachineLearningType
{
	Success = 0, Cost = 1, Policy = 2, _Debugg = 3, PolicyCost = 4
};

class mDataManager
{
public:
	std::deque<std::shared_ptr<mTransitionData>> transition_data_;
	std::deque<std::shared_ptr<mTransitionData>> validation_data_;
	std::mutex copying_transition_data_;

	unsigned learning_budget_;
	unsigned validation_budget_;
	float validation_fraction_;
	SimulationContext* mContext;

	mDataManager(SimulationContext* iContext)
	{
		mContext = iContext;
		validation_fraction_ = 0.1f;
		learning_budget_ = 10000;
	}

	int getDataSize(const std::deque<std::shared_ptr<mTransitionData>>& data)
	{
		return (int)(data.size());
	}

	void addSample(mTransitionData* nTransitionSample, int fixed_index_num)
	{
		validation_budget_ = validation_fraction_ * learning_budget_;
		int validation_fixed_index = validation_fraction_ * fixed_index_num;
		//Form the teaching sample
		std::shared_ptr<mTransitionData> transitionSample = std::shared_ptr<mTransitionData>(nTransitionSample);
		std::lock_guard<std::mutex> lock(copying_transition_data_);

		float randu = mTools::getRandomBetween_01();
		if (randu < validation_fraction_)
		{
			validation_data_.push_back(std::move(transitionSample));
		}
		else
		{
			transition_data_.push_back(std::move(transitionSample));
		}

		if (transition_data_.size() > learning_budget_)
		{
			transition_data_.erase(transition_data_.begin() + fixed_index_num); // pop_front();
		}

		if (validation_data_.size() > validation_budget_)
		{
			validation_data_.erase(validation_data_.begin() + validation_fixed_index);// pop_front();
		}

		return;
	}

	Eigen::VectorXf getOutput(const std::deque<std::shared_ptr<mTransitionData>>& data, int i, mMachineLearningType& outType)
	{
		int data_index = i;

		const std::shared_ptr<mTransitionData>& datum = data[data_index];
		switch (outType)
		{
		case mMachineLearningType::Cost:
			return datum->getCostVal();
			break;
		case mMachineLearningType::Success:
			return datum->getSucceedVal();
			break;
		case mMachineLearningType::Policy:
			return datum->getPolicyVal();
			break;
		case mMachineLearningType::_Debugg:
			return datum->getDebugVal();
			break;
		case mMachineLearningType::PolicyCost:
			return datum->getCostVal();
			break;
		}

		return Eigen::VectorXf();
	}

	Eigen::VectorXf getState(const std::deque<std::shared_ptr<mTransitionData>>& data, int i, mMachineLearningType& outType)
	{
		int data_index = i;

		const std::shared_ptr<mTransitionData>& datum = data[data_index];
		switch (outType)
		{
		case mMachineLearningType::Cost:
			return datum->getCostState();//body_index
			break;
		case mMachineLearningType::Success:
			return datum->getSuccessState();
			break;
		case mMachineLearningType::Policy:
			return datum->getPolicyState();
			break;
		case mMachineLearningType::_Debugg:
			return datum->getDebugState();
			break;
		case mMachineLearningType::PolicyCost:
			return datum->getPolicyState();
			break;
		}

		return Eigen::VectorXf();
	}

	bool getAddState(const std::deque<std::shared_ptr<mTransitionData>>& data, int i, mMachineLearningType& outType)
	{
		int data_index = i;

		const std::shared_ptr<mTransitionData>& datum = data[data_index];
		switch (outType)
		{
		case mMachineLearningType::Cost:
			return datum->_succeed;
			break;
		case mMachineLearningType::Success:
			return true;
			break;
		case mMachineLearningType::Policy:
			return datum->_succeed;
			break;
		case mMachineLearningType::_Debugg:
			return true;
			break;
		case mMachineLearningType::PolicyCost:
			return datum->_succeed;
			break;
		}

		return false;
	}

	void setNormState(const std::deque<std::shared_ptr<mTransitionData>>& data, int i, mMachineLearningType& outType, const VectorXf& normState)
	{
		int data_index = i;
		
		const std::shared_ptr<mTransitionData>& datum = data[data_index];
		switch (outType)
		{
		case mMachineLearningType::Cost:
			datum->norm_cost_state = normState;
			break;
		case mMachineLearningType::Success:
			datum->norm_success_state = normState;
			break;
		case mMachineLearningType::Policy:
			datum->norm_policy_state = normState;
			break;
		case mMachineLearningType::_Debugg:
			datum->norm_debug_state = normState;
			break;
		case mMachineLearningType::PolicyCost:
			datum->norm_policy_state = normState;
			break;
		}

		return;
	}

	void setNormOutput(const std::deque<std::shared_ptr<mTransitionData>>& data, int i, mMachineLearningType& outType, const VectorXf& normOutput)
	{
		int data_index = i;
		
		const std::shared_ptr<mTransitionData>& datum = data[data_index];
		switch (outType)
		{
		case mMachineLearningType::Cost:
			datum->norm_cost_output = normOutput;
			break;
		case mMachineLearningType::Success:
			datum->norm_success_output = normOutput;
			break;
		case mMachineLearningType::Policy:
			datum->norm_policy_output = normOutput;
			break;
		case mMachineLearningType::_Debugg:
			datum->norm_debug_output = normOutput;
			break;
		case mMachineLearningType::PolicyCost:
			datum->norm_cost_output = normOutput;
			break;
		}

		return;
	}

	void setVarianceOutput(const std::deque<std::shared_ptr<mTransitionData>>& data, int i, mMachineLearningType& outType, const VectorXf& varOutput)
	{
		int data_index = i;
		
		const std::shared_ptr<mTransitionData>& datum = data[data_index];
		switch (outType)
		{
		case mMachineLearningType::Cost:
			datum->variance_cost_output = varOutput;
			break;
		case mMachineLearningType::Success:
			datum->variance_success_output = varOutput;
			break;
		case mMachineLearningType::Policy:
			datum->variance_policy_output = varOutput;
			break;
		case mMachineLearningType::_Debugg:
			datum->variance_debug_output = varOutput;
			break;
		case mMachineLearningType::PolicyCost:
			datum->variance_cost_output = varOutput;
			break;
		}

		return;
	}

	void setDerivativeOutput(const std::deque<std::shared_ptr<mTransitionData>>& data, int i, mMachineLearningType& outType, const VectorXf& dOutput)
	{
		int data_index = i;
		
		const std::shared_ptr<mTransitionData>& datum = data[data_index];
		switch (outType)
		{
		case mMachineLearningType::Cost:
			datum->derivative_cost_output = dOutput;
			break;
		case mMachineLearningType::Success:
			datum->derivative_success_output = dOutput;
			break;
		case mMachineLearningType::Policy:
			datum->derivative_policy_output = dOutput;
			break;
		case mMachineLearningType::_Debugg:
			datum->derivative_debug_output = dOutput;
			break;
		case mMachineLearningType::PolicyCost:
			datum->derivative_cost_output = dOutput;
			break;
		}

		return;
	}

	float* getNormState(const std::deque<std::shared_ptr<mTransitionData>>& data, int i, mMachineLearningType& outType)
	{
		int data_index = i;
		
		const std::shared_ptr<mTransitionData>& datum = data[data_index];
		switch (outType)
		{
		case mMachineLearningType::Cost:
			return datum->norm_cost_state.data();
			break;
		case mMachineLearningType::Success:
			return datum->norm_success_state.data();
			break;
		case mMachineLearningType::Policy:
			return datum->norm_policy_state.data();
			break;
		case mMachineLearningType::_Debugg:
			return datum->norm_debug_state.data();
			break;
		case mMachineLearningType::PolicyCost:
			return datum->norm_policy_state.data();
			break;
		}

		return NULL;
	}

	float* getNormOutput(const std::deque<std::shared_ptr<mTransitionData>>& data, int i, mMachineLearningType& outType)
	{
		int data_index = i;

		const std::shared_ptr<mTransitionData>& datum = data[data_index];
		switch (outType)
		{
		case mMachineLearningType::Cost:
			return datum->norm_cost_output.data();
			break;
		case mMachineLearningType::Success:
			return datum->norm_success_output.data();
			break;
		case mMachineLearningType::Policy:
			return datum->norm_policy_output.data();
			break;
		case mMachineLearningType::_Debugg:
			return datum->norm_debug_output.data();
			break;
		case mMachineLearningType::PolicyCost:
			return datum->norm_cost_output.data();
			break;
		}

		return NULL;
	}

	float* getVarianceOutput(const std::deque<std::shared_ptr<mTransitionData>>& data, int i, mMachineLearningType& outType)
	{
		int data_index = i;

		const std::shared_ptr<mTransitionData>& datum = data[data_index];
		switch (outType)
		{
		case mMachineLearningType::Cost:
			return datum->variance_cost_output.data();
			break;
		case mMachineLearningType::Success:
			return datum->variance_success_output.data();
			break;
		case mMachineLearningType::Policy:
			return datum->variance_policy_output.data();
			break;
		case mMachineLearningType::_Debugg:
			return datum->variance_debug_output.data();
			break;
		case mMachineLearningType::PolicyCost:
			return datum->variance_cost_output.data();
			break;
		}

		return NULL;
	}

	float* getDerivativeOutput(const std::deque<std::shared_ptr<mTransitionData>>& data, int i, mMachineLearningType& outType)
	{
		int data_index = i;
		
		const std::shared_ptr<mTransitionData>& datum = data[data_index];
		switch (outType)
		{
		case mMachineLearningType::Cost:
			return datum->derivative_cost_output.data();
			break;
		case mMachineLearningType::Success:
			return datum->derivative_success_output.data();
			break;
		case mMachineLearningType::Policy:
			return datum->derivative_policy_output.data();
			break;
		case mMachineLearningType::_Debugg:
			return datum->derivative_debug_output.data();
			break;
		case mMachineLearningType::PolicyCost:
			return datum->derivative_cost_output.data();
			break;
		}

		return NULL;
	}
};

class mANNBase
{
public:
	enum mNormalizationType
	{
		MeanSTD = 0, MinMax = 1
	};

	mMachineLearningType _MLOutType;
	mDataManager* _DataManager;

	unsigned int bach_size;
	bool use_variance_loss_function;
	bool flag_normalize_output;

	bool isMinMaxValuesSet;

	int curEpochNum;

	mNormalizationType inputNormalizationType;
	mNormalizationType outputNormalizationType;

	std::string toString(const VectorXf& _iVec)
	{
		std::string write_buff;
		char _buff[2000];

		sprintf_s(_buff, "%d,", _iVec.size()); write_buff += _buff;
		for (int d = 0; d < _iVec.size(); d++)
		{
			if (d == _iVec.size() - 1)
			{
				sprintf_s(_buff, "%.3f\n", _iVec[d]); write_buff += _buff;
			}
			else
			{
				sprintf_s(_buff, "%.3f,", _iVec[d]); write_buff += _buff;
			}
		}

		return write_buff;
	}

	void saveToFile(std::string filename)
	{
		std::string filename_minmaxVal = filename + "MinMax.txt";
		if (isMinMaxValuesSet)
		{
			mFileHandler fileWriter(filename_minmaxVal, "w");

			fileWriter.writeLine(toString(inputMax));
			fileWriter.writeLine(toString(inputMin));
			fileWriter.writeLine(toString(inputMean));
			fileWriter.writeLine(toString(inputSD));

			fileWriter.writeLine(toString(outputMax));
			fileWriter.writeLine(toString(outputMin));
			fileWriter.writeLine(toString(outputMean));
			fileWriter.writeLine(toString(outputSD));

			fileWriter.mCloseFile();
		}

		std::string filename1 = filename + ".txt";

		if (NN_in_training_.get())
		{
			NN_in_training_->write_to_file(filename1);
		}

		/*if (use_variance_loss_function)
		{
			if (Var_NN_in_training_.get())
			{
				std::string filename2 = filename + "Var.txt";
				Var_NN_in_training_->write_to_file(filename2);
			}
		}*/
	}

	bool loadFromFile(std::string filename)
	{
		bool isNNloaded = false;

		std::string filename_minmaxVal = filename + "MinMax.txt";
		if (fileExists(filename_minmaxVal.c_str()))
		{
			mFileHandler fileReader(filename_minmaxVal, "r");
			std::vector<std::vector<float>> mVals;
			fileReader.readFile(mVals);

			// nStateDimensions
			nStateDimensions = mVals[0][0];

			inputMax = -FLT_MAX * VectorXf::Ones(nStateDimensions);
			inputMin = FLT_MAX * VectorXf::Ones(nStateDimensions);
			inputMean = VectorXf::Zero(nStateDimensions);
			inputSD = VectorXf::Ones(nStateDimensions);

			for (int d = 0; d < nStateDimensions; d++)
			{
				inputMax[d] = mVals[0][d + 1];
				inputMin[d] = mVals[1][d + 1];
				inputMean[d] = mVals[2][d + 1];
				inputSD[d] = mVals[3][d + 1];
			}

			// nOutputDimensions
			nOutputDimensions = mVals[4][0];

			outputMax = -FLT_MAX * VectorXf::Ones(nOutputDimensions);
			outputMin = FLT_MAX * VectorXf::Ones(nOutputDimensions);
			outputMean = VectorXf::Zero(nOutputDimensions);
			outputSD = VectorXf::Ones(nOutputDimensions);

			for (int d = 0; d < nOutputDimensions; d++)
			{
				outputMax[d] = mVals[4][d + 1];
				outputMin[d] = mVals[5][d + 1];
				outputMean[d] = mVals[6][d + 1];
				outputSD[d] = mVals[7][d + 1];
			}

			fileReader.mCloseFile();

			isMinMaxValuesSet = true;
		}

		std::string filename1 = filename + ".txt";
		if (fileExists(filename1.c_str()))
		{
			NN_in_training_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron);
			NN_in_training_->load_from_file(filename1);

			std::unique_ptr<MultiLayerPerceptron> tmp = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron(*NN_in_training_));
			NN_ = std::move(tmp);

			isNNloaded = true;
		}

		/*if (use_variance_loss_function)
		{
			std::string filename2 = filename + "Var.txt";
			if (fileExists(filename2.c_str()))
			{
				Var_NN_in_training_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron);
				Var_NN_in_training_->load_from_file(filename2);

				std::unique_ptr<MultiLayerPerceptron> tmp = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron(*Var_NN_in_training_));
				Var_NN_ = std::move(tmp);
			}
		}*/

		return isNNloaded;
	}

	mANNBase(mDataManager* iDataManager, const mMachineLearningType& iMLOutType)
	{
		_DataManager = iDataManager;
		_MLOutType = iMLOutType;

		isMinMaxValuesSet = false;

		_learning_rate = 0.0001f;
		_max_gradient_norm = 1000000000.0f;

		bach_size = 1000;
		use_variance_loss_function = false;
		flag_normalize_output = true;
		inputNormalizationType = MeanSTD;
		outputNormalizationType = MeanSTD;

		curEpochNum = 0;

		regularization_noise_ = 0.1f;

		cMSETrain = std::numeric_limits<float>::infinity();
		cMSETest = std::numeric_limits<float>::infinity();

		NN_.reset();
		NN_copy_.reset();
		NN_in_training_.reset();

		//Var_NN_.reset();
		//Var_NN_copy_.reset();
		//Var_NN_in_training_.reset();

		auto dummy = []()
		{
			return;
		};

		long_term_learning_for_output_ = std::async(std::launch::async, dummy);
//		long_term_learning_for_var_output_ = std::async(std::launch::async, dummy);

		nStateDimensions = -1;
		nOutputDimensions = -1;

//		current_update_num = 0;
//		update_var = true;
		counter_randIndexOutput = 0;
//		counter_randIndexVar = 0;
	}

	void updateMinMaxMeanVals()
	{
		if (isMinMaxValuesSet)
		{
			return;
		}

		if (_DataManager == NULL)
		{
			return;
		}

		VectorXf _state0 = _DataManager->getState(_DataManager->transition_data_, 0, _MLOutType);
		if (nStateDimensions != (int)_state0.size())
		{
			nStateDimensions = (int)_state0.size();
			inputMax = -FLT_MAX * VectorXf::Ones(nStateDimensions);
			inputMin = FLT_MAX * VectorXf::Ones(nStateDimensions);

			inputMean = VectorXf::Zero(nStateDimensions);
			inputSD = VectorXf::Ones(nStateDimensions);
		}
		VectorXf _output0 = _DataManager->getOutput(_DataManager->transition_data_, 0, _MLOutType);
		if (nOutputDimensions != (int)_output0.size())
		{
			nOutputDimensions = (int)_output0.size();
			
			outputMax = -FLT_MAX * VectorXf::Ones(nOutputDimensions);
			outputMin = FLT_MAX * VectorXf::Ones(nOutputDimensions);

			outputMean = VectorXf::Zero(nOutputDimensions);
			outputSD = VectorXf::Ones(nOutputDimensions);
		}

		inputMean = VectorXf::Zero(nStateDimensions);
		outputMean = VectorXf::Zero(nOutputDimensions);
		int all_added_data_size = 0;
		int training_data_size = _DataManager->getDataSize(_DataManager->transition_data_);
		for (int i = 0; i < training_data_size; i++)
		{
			bool flag_add = _DataManager->getAddState(_DataManager->transition_data_, i, _MLOutType);
			if (flag_add)
			{
				all_added_data_size++;
				VectorXf _statei = _DataManager->getState(_DataManager->transition_data_, i, _MLOutType);
				VectorXf _outputi = _DataManager->getOutput(_DataManager->transition_data_, i, _MLOutType);
				inputMean += _statei;
				outputMean += _outputi;

				for (int d = 0; d < nOutputDimensions; d++)
				{
					if (outputMin[d] > _outputi[d])
					{
						outputMin[d] = _outputi[d];
					}
					if (outputMax[d] < _outputi[d])
					{
						outputMax[d] = _outputi[d];
					}
				}

				for (int d = 0; d < nStateDimensions; d++)
				{
					if (inputMin[d] > _statei[d])
					{
						inputMin[d] = _statei[d];
					}
					if (inputMax[d] < _statei[d])
					{
						inputMax[d] = _statei[d];
					}
				}
			}
		}

		int validation_data_size = _DataManager->getDataSize(_DataManager->validation_data_);
		for (int i = 0; i < validation_data_size; i++)
		{
			bool flag_add = _DataManager->getAddState(_DataManager->validation_data_, i, _MLOutType);
			if (flag_add)
			{
				all_added_data_size++;
				VectorXf _statei = _DataManager->getState(_DataManager->validation_data_, i, _MLOutType);
				VectorXf _outputi = _DataManager->getOutput(_DataManager->validation_data_, i, _MLOutType);
				inputMean += _statei;
				outputMean += _outputi;

				for (int d = 0; d < nOutputDimensions; d++)
				{
					if (outputMin[d] > _outputi[d])
					{
						outputMin[d] = _outputi[d];
					}
					if (outputMax[d] < _outputi[d])
					{
						outputMax[d] = _outputi[d];
					}
				}

				for (int d = 0; d < nStateDimensions; d++)
				{
					if (inputMin[d] > _statei[d])
					{
						inputMin[d] = _statei[d];
					}
					if (inputMax[d] < _statei[d])
					{
						inputMax[d] = _statei[d];
					}
				}
			}
		}

		inputMean = inputMean / (float)(all_added_data_size);
		outputMean = outputMean / (float)(all_added_data_size);

		inputSD = VectorXf::Zero(nStateDimensions);
		outputSD = VectorXf::Zero(nOutputDimensions);
		for (int i = 0; i < training_data_size; i++)
		{
			bool flag_add = _DataManager->getAddState(_DataManager->transition_data_, i, _MLOutType);
			if (flag_add)
			{
				VectorXf _statei = _DataManager->getState(_DataManager->transition_data_, i, _MLOutType);
				VectorXf _outputi = _DataManager->getOutput(_DataManager->transition_data_, i, _MLOutType);
				inputSD += (_statei - inputMean).cwiseAbs2();
				outputSD += (_outputi - outputMean).cwiseAbs2();
			}
		}
		for (int i = 0; i < validation_data_size; i++)
		{
			bool flag_add = _DataManager->getAddState(_DataManager->validation_data_, i, _MLOutType);
			if (flag_add)
			{
				VectorXf _statei = _DataManager->getState(_DataManager->validation_data_, i, _MLOutType);
				VectorXf _outputi = _DataManager->getOutput(_DataManager->validation_data_, i, _MLOutType);
				inputSD += (_statei - inputMean).cwiseAbs2();
				outputSD += (_outputi - outputMean).cwiseAbs2();
			}
		}

		inputSD = (inputSD / (float)(all_added_data_size)).cwiseSqrt();
		outputSD = (outputSD / (float)(all_added_data_size)).cwiseSqrt();

		isMinMaxValuesSet = true;

		return;
	}

	bool train()
	{
		if (!isMinMaxValuesSet)
			return false;

		if (curEpochNum <= 0)
			return false;

		if (_DataManager == NULL)
		{
			return false;
		}

		/*if (use_variance_loss_function)
		{
			if (counter_randIndexVar >= (int)(mRandIndexVar.size()))
			{
				mRandIndexVar.clear();
				for (int i = 0; i < _DataManager->getDataSize(_DataManager->transition_data_); ++i) mRandIndexVar.push_back(i);
				std::random_shuffle(mRandIndexVar.begin(), mRandIndexVar.end());
				counter_randIndexVar = 0;
			}

			if (current_update_num <= 0)
			{
				update_var = true;
			}
			if (current_update_num >= 5)
			{
				update_var = false;
			}

			if (update_var)
			{
				if (long_term_learning_for_var_output_.wait_for(std::chrono::microseconds(1)) == std::future_status::ready)
				{
					auto train_function_var = [&]()
					{
						trainNNVariance();
						current_update_num++;
					};

					if (Var_NN_copy_.get())
					{
						Var_NN_.swap(Var_NN_copy_);
						Var_NN_copy_.reset();
					}

					long_term_learning_for_var_output_ = std::async(std::launch::async, train_function_var);
				}
			}
		}*/

//		if (!use_variance_loss_function || (use_variance_loss_function && !update_var))
//		{
		if (long_term_learning_for_output_.wait_for(std::chrono::microseconds(1)) == std::future_status::ready)
		{
			auto train_function = [&]()
			{
				trainNNOutput();

//				current_update_num--;
			};

			if (counter_randIndexOutput >= (int)(mRandIndexOutput.size()) && counter_randIndexOutput > 0 && NN_copy_.get())
			{
				NN_.swap(NN_copy_);
				NN_copy_.reset();

				curEpochNum--;
			}

			if (counter_randIndexOutput >= (int)(mRandIndexOutput.size()))
			{
				mRandIndexOutput.clear();
				for (int i = 0; i < _DataManager->getDataSize(_DataManager->transition_data_); ++i) mRandIndexOutput.push_back(i);
				std::random_shuffle(mRandIndexOutput.begin(), mRandIndexOutput.end());
				counter_randIndexOutput = 0;
			}

			long_term_learning_for_output_ = std::async(std::launch::async, train_function);
			return true;
		}
//		}
		return false;
	}

	//float isInRange(const VectorXf& _state)
	//{
	//	VectorXf d = (_state - inputMean);
	//	VectorXf r2 = 2 * inputSD;
	//	VectorXf r3 = 3 * inputSD;
	//	float max_val = (r3 - r2).norm();
	//	VectorXf dis_val = VectorXf::Zero(d.size());
	//	for (int i = 0; i < nStateDimensions; i++)
	//	{
	//		float dis2 = abs(d[i]) - r2[i];
	//		//float dis3 = abs(d[i]) - r3[i];
	//		if (dis2 > 0)
	//		{
	//			dis_val[i] = d[i];
	//		}
	//	}
	//	return (dis_val.norm() / r3.norm()) + 1.0f;
	//}

	bool __stdcall getMachineLearningOutput(float *state, float* out_, float* var_out = nullptr, bool flag_clamp = true, bool use_training_net = false)
	{
		MultiLayerPerceptron* mlp = nullptr;
		if (use_training_net)
		{
			mlp = NN_in_training_.get();
		}
		else
		{
			mlp = NN_.get();
		}

		if (mlp == nullptr)
		{
			return false;
		}

		VectorXf norm_state_;
		norm_state_.resize(nStateDimensions);

		Eigen::Map<const Eigen::VectorXf> out_state = Eigen::Map<const Eigen::VectorXf>(state, nStateDimensions);

		getNormalizedState(norm_state_, out_state);

		mlp->run(norm_state_.data());

		for (int i = 0; i < nOutputDimensions; i++)
		{
			float sd = outputNormalizationType == MeanSTD ? outputSD[i] : (outputMean[i] - outputMin[i] + 0.001f);//(outputMax[i] - outputMin[i] + 0.001f);
			float bias = outputNormalizationType == MeanSTD ? outputMean[i] : outputMin[i];
			float output = (mlp->output_operation_->outputs_[i]);

			if (flag_normalize_output)
			{
				output = output*sd + bias;
			}

			if (output - output != output - output)
			{
				output = 0.0f;
			}

			if (flag_clamp)
			{
				output = (std::min)(output, outputMax[i]);
				output = (std::max)(output, outputMin[i]);
			}

			out_[i] = output;

		}

		if (var_out)
		{
			for (int i = 0; i < nOutputDimensions; i++)
			{
				float sd = outputNormalizationType == MeanSTD ? outputSD[i] : (outputMean[i] - outputMin[i] + 0.001f);//(outputMax[i] - outputMin[i] + 0.001f);
				float output = (mlp->output_operation_->outputs_[nOutputDimensions + i]);

				if (flag_normalize_output)
				{
					output = output*sd;
				}

				if (output - output != output - output)
				{
					output = 0.0f;
				}

				output = abs(output);

				//if (output < std::numeric_limits<float>::epsilon()) {
				//	output = std::numeric_limits<float>::epsilon();
				//}

				var_out[i] = output;

			}
		}
		return true;
	}

	//bool __stdcall getMachineLearningVarOutput(float *state, float* out_)//, bool flag_clamp = true, bool use_training_net = false)
	//{
	//	MultiLayerPerceptron* mlp = nullptr;
	//	//if (use_training_net)
	//	//{
	//	//	mlp = //Var_NN_in_training_.get();
	//	//}
	//	//else
	//	//{
	//	mlp = NN_.get();
	//	//}
	//	if (mlp == nullptr)
	//	{
	//		return false;
	//	}
	//	VectorXf norm_state_;
	//	norm_state_.resize(nStateDimensions);
	//  Eigen::Map<const Eigen::VectorXf> out_state = Eigen::Map<const Eigen::VectorXf>(state, nStateDimensions);
	//	getNormalizedState(norm_state_, out_state);
	//	mlp->run(norm_state_.data());
	//	for (int i = 0; i < nOutputDimensions; i++)
	//	{
	//		float& output = (mlp->output_operation_->outputs_[i]);
	//		if (flag_clamp)
	//		{
	//			if (output - output != output - output)
	//			{
	//				output = 0.0f;
	//			}
	//			//				output = std::min(output, 1.0f);
	//			//				output = std::max(output, 0.001f);
	//		}
	//		out_[i] = output;
	//	}
	//	return true;
	//}

	void waitForLearning()
	{
		long_term_learning_for_output_.wait();

		if (cMSETrain < std::numeric_limits<float>::infinity() && NN_copy_.get())
		{
			NN_.swap(NN_copy_);
			NN_copy_.reset();
		}

		/*if (use_variance_loss_function)
		{
			long_term_learning_for_var_output_.wait();

			if (Var_NN_copy_.get())
			{
				Var_NN_.swap(Var_NN_copy_);
				Var_NN_copy_.reset();
			}
		}*/
	}

	float getMSETest()
	{
		return cMSETest;
	}

	float getMSETrain()
	{
		return cMSETrain;
	}

	void setLearningRate(float _rate)
	{
		_learning_rate = _rate;
		if (NN_in_training_.get())
		{
			NN_in_training_.get()->learning_rate_ = _rate;
		}

		/*if (Var_NN_in_training_.get())
		{
			Var_NN_in_training_.get()->learning_rate_ = _rate;
		}*/
	}

	void setMaxGradientNorm(float _val)
	{
		_max_gradient_norm = _val;
		if (NN_in_training_.get())
		{
			NN_in_training_.get()->max_gradient_norm_ = _val;
		}

		/*if (Var_NN_in_training_.get())
		{
			Var_NN_in_training_.get()->max_gradient_norm_ = _val;
		}*/
	}

	/*float getMaxOutPut()
	{
		return outputMax[0];
	}*/

	void calculateMSEOnTrainAndTestSet()
	{
		std::deque<std::shared_ptr<mTransitionData>> transition_data;
		std::deque<std::shared_ptr<mTransitionData>> validation_data;
		{
			std::lock_guard<std::mutex> lock(_DataManager->copying_transition_data_);
			transition_data = _DataManager->transition_data_;
			validation_data = _DataManager->validation_data_;
		}

		if (use_variance_loss_function)
			cMSETrain = getLoss(transition_data);
		else
			cMSETrain = getMSE(transition_data);

		if (use_variance_loss_function)
			cMSETest = getLoss(validation_data);
		else
			cMSETest = getMSE(validation_data);
	}

private:
	float _learning_rate;
	float _max_gradient_norm;

	float regularization_noise_;

//	int current_update_num;
//	bool update_var;

	float cMSETrain;
	float cMSETest;

	std::vector<int> mRandIndexOutput;
	int counter_randIndexOutput;
//	std::vector<int> mRandIndexVar;
//	int counter_randIndexVar;

	// neural networks for output (calculate mean of output for given same input state)
	std::unique_ptr<MultiLayerPerceptron> NN_;
	std::unique_ptr<MultiLayerPerceptron> NN_copy_;
	std::unique_ptr<MultiLayerPerceptron> NN_in_training_;

	//// neural networks for variance of output (for applying loss function of deep )
	//std::unique_ptr<MultiLayerPerceptron> Var_NN_;
	//std::unique_ptr<MultiLayerPerceptron> Var_NN_copy_;
	//std::unique_ptr<MultiLayerPerceptron> Var_NN_in_training_;

	// data for training and validation of neural network
	std::future<void> long_term_learning_for_output_;
//	std::future<void> long_term_learning_for_var_output_;

	int nStateDimensions;
	int nOutputDimensions;

	// should be taken out from data
	VectorXf outputMin;
	VectorXf outputMax;
	VectorXf outputMean;
	VectorXf outputSD;

	VectorXf inputMin;
	VectorXf inputMax;
	VectorXf inputMean;
	VectorXf inputSD;

	void init_neural_net(int input_dim, int output_dim, MultiLayerPerceptron& net)
	{
		unsigned seed = (unsigned)time(nullptr);
		srand(seed);

		int layer_width = 40;

		/*if (input_dim == output_dim)
		{
			layer_width = std::max(output_dim, layer_width);
			layer_width = std::max(input_dim, layer_width);
		}*/

		if (use_variance_loss_function)
		{
			output_dim = 2 * output_dim;
		}

		std::vector<unsigned> layers;
		if (_MLOutType == mMachineLearningType::Policy)
		{
			layers.push_back(input_dim);
			layers.push_back(100);
			layers.push_back(100);
			layers.push_back(100);
			layers.push_back(output_dim);
		}
		else if (_MLOutType == mMachineLearningType::Success)
		{
			layers.push_back(input_dim);
			layers.push_back(30);
			layers.push_back(30);
			layers.push_back(30);
			layers.push_back(output_dim);
		}
		else if (_MLOutType == mMachineLearningType::_Debugg)
		{
			layers.push_back(input_dim);
			layers.push_back(20);
			layers.push_back(25);
			layers.push_back(15);
			layers.push_back(output_dim);
		}
		else if (_MLOutType == mMachineLearningType::PolicyCost)
		{
			layers.push_back(input_dim);
			layers.push_back(100);
			layers.push_back(100);
			layers.push_back(100);
			layers.push_back(output_dim);
		}
		else
		{
			layers.push_back(input_dim);
			layers.push_back(30);
			layers.push_back(30);
			layers.push_back(30);
			layers.push_back(output_dim);
		}

		const int max_amount_of_params = 51224;
		int amount_params = 0;

		unsigned hidden_layer_width = 73;

		net.build_network(layers);
		amount_params = net.get_amount_parameters();

		net.max_gradient_norm_ = _max_gradient_norm;
		net.learning_rate_ = _learning_rate;
		net.min_weight_ = std::numeric_limits<float>::lowest();
		net.max_weight_ = (std::numeric_limits<float>::max)();
		net.adam_first_moment_smoothing_ = 0.9f;
		net.adam_second_moment_smoothing_ = 0.99f;

		net.drop_out_stdev_ = regularization_noise_;

		net.error_drop_out_prob_ = 0.0f;
		net.weight_decay_ = 0.01f;

		float weight_min = -0.1f;
		float weight_max = 0.1f;
		float bias_val = 0.0f;
		net.randomize_weights(weight_min, weight_max, bias_val);
	}

	void getNormalizedState(VectorXf& normalized_input, const VectorXf& inState)
	{
		normalized_input = inState;

		/*for (unsigned d = 0; d < (unsigned)inState.size(); d++)
		{
			if (normalized_input[d] < inputMin[d])
			{
				normalized_input[d] = inputMin[d];
			}

			if (normalized_input[d] > inputMax[d])
			{
				normalized_input[d] = inputMax[d];
			}
		}*/

		/*for (unsigned d = 0; d < (unsigned)inState.size(); d++)
		{
			float sd = inputMax[d] - inputMin[d] + 0.001f;
			normalized_input[d] = (inState[d] - inputMin[d]) / (sd);
		}*/

		for (unsigned d = 0; d < (unsigned)inState.size(); d++)
		{
			float sd = inputSD[d];//inputMax[d] - inputMin[d] + 0.001f;
			if (sd > 0)
			{
				normalized_input[d] = (inState[d] - inputMean[d]) / (sd);
			}
			else
			{
				normalized_input[d] = (inState[d] - inputMean[d]) / (sd + 0.0001f);
			}
		}
	}

	void getNormalizedOutput(VectorXf& normalized_output, const VectorXf& iOutput)
	{
		normalized_output = iOutput;

		if (flag_normalize_output)
		{
			for (unsigned d = 0; d < (unsigned)iOutput.size(); d++)
			{
				if (outputNormalizationType == MeanSTD)
				{
					float sd = outputSD[d];
					if (sd > 0)
						normalized_output[d] = (iOutput[d] - outputMean[d]) / (sd);
					else
						normalized_output[d] = (iOutput[d] - outputMean[d]) / (sd + 0.0001f);
				}
				else
				{
					float sd = outputMean[d] - outputMin[d] + 0.001f;//outputMax[d] - outputMin[d] + 0.001f;
					normalized_output[d] = (iOutput[d] - outputMin[d]) / (sd);
				}
			}
		}
	}

	void trainNNOutput()
	{
		if (!_DataManager)
			return;

		unsigned int limit_size = bach_size;

		std::deque<std::shared_ptr<mTransitionData>> transition_data;
		std::deque<std::shared_ptr<mTransitionData>> validation_data;
		{
			std::lock_guard<std::mutex> lock(_DataManager->copying_transition_data_);
			transition_data = _DataManager->transition_data_;
			validation_data = _DataManager->validation_data_;
		}

		if (transition_data.size() == 0)
			return;

		unsigned input_dim = nStateDimensions;
		unsigned output_dim = nOutputDimensions;

		if (!NN_in_training_.get())
		{
			NN_in_training_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());

			init_neural_net(input_dim, output_dim, *NN_in_training_);
		}

		std::vector<float*> inputs;
		std::vector<float*> outputs;

		inputs.reserve(transition_data.size());
		outputs.reserve(transition_data.size());

		auto form_training_data = [&](std::deque<std::shared_ptr<mTransitionData>>& data_set)
		{
			inputs.clear();
			outputs.clear();

			int counter_data = 0;
			bool flag_continue = true;
			while (flag_continue)
			{
				int index_data = -1;
				if (counter_randIndexOutput < (int)(mRandIndexOutput.size()))
					index_data = mRandIndexOutput[counter_randIndexOutput];
				else
					flag_continue = false;

				if (!(index_data >= 0 && index_data < _DataManager->getDataSize(data_set)))
					flag_continue = false;

				if (flag_continue)
				{
					VectorXf _state = _DataManager->getState(data_set, index_data, _MLOutType);
					VectorXf norm_state(nStateDimensions);
					getNormalizedState(norm_state, _state);
					_DataManager->setNormState(data_set, index_data, _MLOutType, norm_state);

					bool flag_add = _DataManager->getAddState(data_set, index_data, _MLOutType);

					if (flag_add)
					{
						counter_data++;
						inputs.push_back(_DataManager->getNormState(data_set, index_data, _MLOutType));
					}

					VectorXf _output = _DataManager->getOutput(data_set, index_data, _MLOutType);

					VectorXf norm_output(nOutputDimensions);
					getNormalizedOutput(norm_output, _output);
					_DataManager->setNormOutput(data_set, index_data, _MLOutType, norm_output);

					if (flag_add)
						outputs.push_back(_DataManager->getNormOutput(data_set, index_data, _MLOutType));
				}

				if (counter_data > (int)(limit_size))
				{
					flag_continue = false;
				}

				counter_randIndexOutput++;
			}

		};

		form_training_data(transition_data);

		// do not train on batch that is not having same size as others
		if (inputs.size() < limit_size)
			return;

		int max_epochs = 1;
		int epoch = 0;
		bool is_gradient = use_variance_loss_function;
		while (epoch < max_epochs)
		{
			//NN_in_training_->train_back_prop((const float**)inputs.data(), (const float**)outputs.data(), inputs.size(), inputs.size(), is_gradient);
			if (!use_variance_loss_function)
			{
				NN_in_training_->train_adam((const float**)inputs.data(), (const float**)outputs.data(), inputs.size(), inputs.size(), is_gradient);
			}
			else
			{
				NN_in_training_->bayesian_optimization((const float**)inputs.data(), (const float**)outputs.data(), inputs.size(), inputs.size());
			}

			epoch++;
		}

		if (use_variance_loss_function)
			cMSETrain = getLoss(transition_data);
		else
			cMSETrain = getMSE(transition_data);

		if (cMSETrain - cMSETrain != cMSETrain - cMSETrain)
		{
			int input_size = NN_in_training_->input_operation_->size_;
			int output_size = NN_in_training_->output_operation_->size_;

			NN_in_training_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron);
			init_neural_net(input_size, output_size, *NN_in_training_);
			cMSETrain = std::numeric_limits<float>::infinity();
		}

		if (NN_in_training_.get())
		{
			std::unique_ptr<MultiLayerPerceptron> tmp = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron(*NN_in_training_));
			NN_copy_ = std::move(tmp);
		}
		if (use_variance_loss_function)
			cMSETest = getLoss(validation_data);
		else
			cMSETest = getMSE(validation_data);

		return;
	}

	float getMSE(const std::deque<std::shared_ptr<mTransitionData>>& val_data)
	{
		if (!_DataManager)
			return FLT_MAX;

		if (val_data.size() < 1)
		{
			return FLT_MAX;
		}

		if (NN_in_training_.get())
		{
			VectorXf _state = _DataManager->getState(val_data, 0, _MLOutType);
			if (NN_in_training_->input_operation_->size_ != _state.size())
			{
				return FLT_MAX;
			}
			VectorXf _output = _DataManager->getOutput(val_data, 0, _MLOutType);

			int out_size = _output.size();
			if (use_variance_loss_function)
			{
				out_size *= 2;
			}

			if (NN_in_training_->output_operation_->size_ != out_size)
			{
				return FLT_MAX;
			}

			VectorXf mse_dimension = VectorXf::Zero(nOutputDimensions);
			int data_size = _DataManager->getDataSize(val_data);
			float data_size_added = 0;
			for (int i = 0; i < data_size; i++)
			{
				VectorXf _out(nOutputDimensions);

				VectorXf _state_i = _DataManager->getState(val_data, i, _MLOutType);
				VectorXf _output_i = _DataManager->getOutput(val_data, i, _MLOutType);
				bool flag_add = _DataManager->getAddState(val_data, i, _MLOutType);

				if (flag_add)
				{
					getMachineLearningOutput(_state_i.data(), _out.data(), NULL, false, true);

					if (flag_normalize_output)
					{
						for (int d = 0; d < nOutputDimensions; d++)
						{
							mse_dimension[d] += squared((_out[d] - _output_i[d]) / outputSD[d]);
						}
					}
					else
					{
						for (int d = 0; d < nOutputDimensions; d++)
						{
							mse_dimension[d] += squared(_out[d] - _output_i[d]);
						}
					}
					data_size_added++;
				}
			}

			float max_val = -FLT_MAX;
			float min_val = FLT_MAX;

			int max_error_dim = -1;
			int min_error_dim = -1;
			for (int d = 0; d < nOutputDimensions; d++)
			{
				mse_dimension[d] = mse_dimension[d] / data_size_added;
				if (max_val < mse_dimension[d]) {
					max_val = mse_dimension[d];
					max_error_dim = d;
				}

				if (min_val > mse_dimension[d]) {
					min_val = mse_dimension[d];
					min_error_dim = d;
				}
				min_val = min(min_val, mse_dimension[d]);
			}

			//std::cout << max_error_dim << " " << max_val << " " << min_error_dim << " " << min_val << "\n";
			return mse_dimension.norm();
		}

		return FLT_MAX;
	}

	float getLoss(const std::deque<std::shared_ptr<mTransitionData>>& val_data)
	{
		return getMSE(val_data);

		/*float _loss = FLT_MAX;
		if (NN_in_training_.get() && Var_NN_in_training_.get())
		{
			_loss = 0.0f;
			int data_size = _DataManager->getDataSize(val_data, _MLOutType);
			float data_size_added = 0;
			for (int i = 0; i < data_size; i++)
			{
				const std::shared_ptr<mTransitionData>& datum = val_data[i];

				VectorXf _mean_val = outputMean;
				VectorXf _sd_val = outputSD.cwiseAbs2();

				VectorXf _state_i = _DataManager->getState(val_data, i, _MLOutType);
				VectorXf _output_i = _DataManager->getOutput(val_data, i, _MLOutType);

				bool flag_add = _DataManager->getAddState(val_data, i, _MLOutType);

				if (flag_add)
				{
					getMachineLearningOutput(_state_i.data(), _mean_val.data(), false, true);
					getMachineLearningVarOutput(_state_i.data(), _sd_val.data(), false, true);

					VectorXf mse_output(nOutputDimensions);

					if (!use_variance_loss_function)
					{
						getNormalizedOutput(mse_output, _output_i);
					}
					else
					{
						mse_output = _output_i;
					}

					_loss += ((_mean_val - mse_output).norm() * (exp(-_sd_val[0]) / 2.0f) + _sd_val[0]);
					data_size_added++;
				}
			}

			_loss /= data_size_added;
		}

		return _loss;*/
	}

};