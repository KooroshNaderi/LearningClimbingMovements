/*

Part of Aalto University Game Tools. See LICENSE.txt for licensing info. 

*/




#ifndef SMOOTHING_CONTROL_H
#define SMOOTHING_CONTROL_H
#include <Eigen/Eigen> 
#include <vector>
#include "DiagonalGMM.h"
#include "TrajectoryOptimization.h"
#include "OnlineForest.h"
#include "OnlinePiecewiseLinearRegressionTree.h"
#include "GenericDensityForest.hpp"
#include "RegressionUtils.hpp"
#include "ANN.h"
#include "MixtureTree.h"
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

#ifdef SWIG
#define __stdcall  //SWIG doesn't understand __stdcall, but fortunately c# assumes it for virtual calls
#endif


#define MAX_NEAREST_NEIGHBORS 1000

namespace AaltoGames
{

#define physicsBrokenCost 1e20	

	//Note: although we use Eigen vectors internally, the public interface uses only plain float arrays for maximal portability
	class SmoothingControl //: public ITrajectoryOptimization
	{

	public:

		float cMSE;

#ifndef SWIG
		class TeachingSample{

		public:

			typedef float Scalar;

			Eigen::Matrix<Scalar,Eigen::Dynamic,1> state_;
			Eigen::Matrix<Scalar,Eigen::Dynamic,1> future_state_;
			Eigen::Matrix<Scalar,Eigen::Dynamic,1> control_;

			Eigen::VectorXf state_control_;
			Eigen::VectorXf control_state_;
			Eigen::VectorXf state_future_state_;
			
			Scalar cost_to_go_;
			Scalar instantaneous_cost_;

			Eigen::VectorXf input_for_learner_;
			Eigen::VectorXf output_for_learner_;


			bool operator==(const TeachingSample& other){

				float diff = 0.0f;
				for (int i = 0; i < state_.size(); i++){
					diff = state_[i] - other.state_[i];
					if (std::abs(diff) > 0.0f){
						return false;
					}
				}

				for (int i = 0; i < future_state_.size(); i++){
					diff = future_state_[i] - other.future_state_[i];
					if (std::abs(diff) > 0.0f){
						return false;
					}
				}

				for (int i = 0; i < control_.size(); i++){
					diff = control_[i] - other.control_[i];
					if (std::abs(diff) > 0.0f){
						return false;
					}
				}

				return true;

			}

			bool operator!=(TeachingSample& other){

				return !(*this == other);

			}


			//float sortingDistance; //mainly for debug
			//float predictedTransitionCost;
			TeachingSample()
			{
				state_ = Eigen::VectorXf::Zero(0);
				future_state_ = state_;
				control_ = state_;
				state_control_ = state_;
				control_state_ = state_;
				state_future_state_ = state_;

				cost_to_go_ = std::numeric_limits<float>::infinity();
				instantaneous_cost_ = std::numeric_limits<float>::infinity();
			}

			TeachingSample(const Eigen::VectorXf& state, const Eigen::VectorXf& control, const Eigen::VectorXf& future_state)
			{
				state_ = state;
				future_state_ = future_state;
				control_ = control;

				int state_dim = state_.size();
				int control_dim = control_.size();
				int joint_dim = state_dim + control_dim;

				state_control_.resize(joint_dim);
				state_control_.head(state_dim) = state;
				state_control_.tail(control_dim) = control;

				control_state_.resize(joint_dim);
				control_state_.head(control_dim) = control;
				control_state_.tail(state_dim) = state;

				state_future_state_.resize(state_dim*2);
				state_future_state_.head(state_dim) = state;
				state_future_state_.tail(state_dim) = future_state;

				cost_to_go_ = std::numeric_limits<float>::infinity();
				instantaneous_cost_ = std::numeric_limits<float>::infinity();
			}

			TeachingSample(const MarginalSample& marginal_sample);

			~TeachingSample()
			{
				state_ = Eigen::VectorXf::Zero(0);
				future_state_ = state_;
				control_ = state_;
				state_control_ = state_;
				control_state_ = state_;
				state_future_state_ = state_;

				cost_to_go_ = std::numeric_limits<float>::infinity();
				instantaneous_cost_ = std::numeric_limits<float>::infinity();
			}
		};
#endif
		int nStateDimensions;
		int nControlDimensions;



		int number_of_nearest_neighbor_trees_;
		int number_of_data_in_leaf_;
		int number_of_hyperplane_tries_;


		unsigned learning_budget_;
		int nn_trajectories_;


		float noPriorTrajectoryPortion;
		float storedSamplesPercentage;
		int amount_recent_;
		bool use_forests_;

#ifndef SWIG
		typedef TeachingSample::Scalar (*teaching_sample_distance)(TeachingSample&,TeachingSample&);
		std::vector<std::pair<TeachingSample*,float> > k_nearest_neighbor_linear_search(int k, std::vector<TeachingSample*>& data, TeachingSample& key_vector, teaching_sample_distance distance_function);


		
		std::deque<std::unique_ptr<TeachingSample> > recent_samples_;
		GenericDensityForest<TeachingSample> ann_forest_;
		GenericDensityTree<TeachingSample> tree_under_construction_;
		std::deque<TeachingSample> adding_buffer_;
		//This vector may be resizes only in the <init> method.
		std::vector<TeachingSample*> learned_samples_;
		int lookahead_;

		std::mutex copying_transition_data_;
		std::mutex copying_dynamics_data_;
		std::mutex memory_training_mutex_;
		std::mutex mixture_training_mutex_;
		std::mutex using_mixture_;
		std::mutex copying_dynamics_model_for_actor_;
		std::mutex using_critic_;
		std::mutex using_actor_in_training_;

		std::future<void> long_term_learning_;
		std::future<void> building_model_;
		std::future<void> training_autoencoder_;
		std::future<void> training_discriminator_;
		std::future<void> training_critic_;
		std::future<void> training_dynamics_model_;

#endif
		int amount_data_in_tree_;

		SmoothingControl();
		~SmoothingControl();

		//minValues and maxValues contain first the bounds for state variables, then for control variables
		//stateKernelStd==NULL corresponds to the special case of Q=0
		//Note that instead of specifying the Q and sigmas of the paper, the values are provided as float standard deviations corresponding to the diagonal elements, 
		//controlPriorStd=sqrt(diag. of \sigma_{0}^2 C_u), controlPriorDiffStd = sqrt(diag. of \sigma_{1}^2 C_u), controlPriorDiffDiffStd = sqrt(diag. of \sigma_{2}^2 C_u)
		void init(int nSampledTrajectories, int nSteps, int nStateDimensions, int nControlDimensions, const float *controlMinValues, const float *controlMaxValues, const float *controlMean, const float *controlPriorStd, const float *controlDiffPriorStd, const  float controlMutationStdScale, bool useMirroring);
		virtual double __stdcall getBestTrajectoryCost();
		virtual void __stdcall getBestControl(int timeStep, float *out_control);
		virtual void __stdcall getControlToUse(float* state, float *out_control);
		virtual const float* __stdcall getRecentControl(float* state, float *out_control, int thread);
		int get_up_to_k_nearest(float* state, int k, int thread, TeachingSample** samples);
		virtual void __stdcall getMachineLearningControl(float *state, float* out_control, int sampled_idx = 0, bool variation = false);
		virtual void __stdcall getMachineLearningControlStdev(float *state, float* out_control_stdev, int sampled_idx = 0);
		virtual void __stdcall getQLearningControl(float *state, float* out_control, int sampled_idx = 0);
//		virtual void __stdcall performQLearning(float *state, float* out_control, float* end_state, float instantaneous_cost);
		virtual void __stdcall getBestControlState( int timeStep, float *out_state );
		//Returns the original state cost for the best trajectory passed from the client to LearningControlPBP for the given timestep. This is mainly for debug and testing purposes.
		virtual double __stdcall getBestTrajectoryOriginalStateCost( int timeStep);
		virtual void __stdcall setSamplingParams(const float *controlPriorStd, const float *controlDiffPriorStd, float controlMutationStdScale);
//		void shiftWithAutoencoder(float* state, float* control, int sample_idx);
//		void shiftWithCritic(float* state, float* control, int sample_idx);

		//returns the prior GMM for the given time step
#ifndef SWIG
		void getConditionalControlGMM(int timeStep, const Eigen::VectorXf &state, DiagonalGMM &dst);


		void init_neural_net(int input_dim, int output_dim, MultiLayerPerceptron& net, bool sigmoid_network = false);

		enum ControlScheme {
			L1, KL
		};

		ControlScheme control_scheme_;
		int since_running_on_neural_network_;
		int forced_sampling_;
		int sampling_counter_;
		bool use_sampling_;
		double policy_cost_;
		int sampling_interval_;
		int initial_learning_iterations_;

		std::vector<std::vector<unsigned>> state_groupings_;
		std::vector<std::vector<unsigned>> control_groupings_;
		
		std::vector<std::unique_ptr<MultiLayerPerceptron> > actor_;
		std::vector<std::unique_ptr<MultiLayerPerceptron> > actor_copy_;
		std::unique_ptr<MultiLayerPerceptron > actor_in_training_;
		std::unique_ptr<MultiLayerPerceptron > actor_for_critic_;
		std::unique_ptr<MultiLayerPerceptron > competing_actor_in_training_;

		std::unique_ptr<MultiLayerPerceptron > actor_stdev_in_training_;
		std::vector<std::unique_ptr<MultiLayerPerceptron> > actor_stdev_;
		std::vector<std::unique_ptr<MultiLayerPerceptron> > actor_stdev_copy_;

//		bool use_dynamics_model_;
		//std::vector<std::unique_ptr<MultiLayerPerceptron> > dynamics_model_;
		//std::vector<std::unique_ptr<MultiLayerPerceptron> > dynamics_model_copy_;
		std::unique_ptr<MultiLayerPerceptron > dynamics_model_in_training_;
		std::unique_ptr<MultiLayerPerceptron > dynamics_model_for_actor_;
		

		std::vector<std::unique_ptr<MultiLayerPerceptron> > rl_actor_;
		std::vector<std::unique_ptr<MultiLayerPerceptron> > rl_actor_copy_;

		
		
		struct EvolutionData {
			int num_evaluations_;
			float cost_;
			std::vector<float> fenotype_;
		};

		int min_evolution_population_;
		int evaluations_for_evolution_;

//		int evolutionary_nets_amount_;
//		int evolution_pool_size_;
		float evolution_rate_;
		float evolution_mutation_stdev_;
		std::vector<std::unique_ptr<EvolutionData>> evolutionary_pool_;
		std::map<int, int> evolution_usage_to_pool_idx_;
		std::vector<std::unique_ptr<MultiLayerPerceptron> > evolutionary_nets_;



		std::unique_ptr<MultiLayerPerceptron> critic_in_training_;
		std::unique_ptr<MultiLayerPerceptron> critic_target_;
		std::unique_ptr<MultiLayerPerceptron> critic_;
		std::unique_ptr<MultiLayerPerceptron> critic_copy_;

		std::deque<std::shared_ptr<TeachingSample> > critic_data_;

		int critic_freeze_interval_;
		int since_critic_freeze_;
//		bool q_learning_;


//		bool use_autoencoder_;

		std::vector<std::unique_ptr<MultiLayerPerceptron> > autoencoder_;
		std::vector<std::unique_ptr<MultiLayerPerceptron> > autoencoder_copy_;
		std::unique_ptr<MultiLayerPerceptron > autoencoder_in_training_;

		float familiarity_threshold_;
//		bool use_discriminator_;
		std::unique_ptr<MultiLayerPerceptron > discriminator_;
		std::unique_ptr<MultiLayerPerceptron > discriminator_copy_;
		std::unique_ptr<MultiLayerPerceptron > discriminator_in_training_;
		std::unique_ptr<MultiLayerPerceptron > discriminator_competing_;


		std::deque<std::string> get_settings();

		void train_actor();
		void train_actor_with_stdev();
		void train_actor_using_dynamics_model();
		void train_autoencoder();
		void train_discriminator();
		void train_critic();
		void train_critic_greedy();
		void train_critic_cacla();
		void train_critic_greedy_cacla();
		void train_dynamics_model();
		float get_trajectory_cost(int end_idx);
		float best_trajectory_match_metric(int end_idx);

		float regularization_noise_;
		float validation_fraction_;
		int amount_discriminator_data_;

		std::deque<std::shared_ptr<TeachingSample> > validation_data_;
		std::deque<std::shared_ptr<TeachingSample> > transition_data_;

		std::deque<std::shared_ptr<TeachingSample> > contractive_data_;

		int amount_dynamics_data_;
		std::deque<std::shared_ptr<TeachingSample> > dynamics_buffer_;
		std::deque<std::shared_ptr<TeachingSample> > dynamics_data_;
		
		std::deque<std::shared_ptr<TeachingSample> > discriminator_data_good_;
		std::deque<std::shared_ptr<TeachingSample> > discriminator_buffer_good_;
		std::deque<std::shared_ptr<TeachingSample> > discriminator_data_bad_;
		std::deque<std::shared_ptr<TeachingSample> > discriminator_buffer_bad_;



		typedef std::pair<Eigen::VectorXf, Eigen::VectorXf> DiagonalGaussian;
		std::vector<DiagonalGaussian> sampling_distributions_;
		std::vector<std::vector<DiagonalGaussian> > gaussian_distributions_;

		Eigen::VectorXf controlMin, controlMax, controlMean, controlPriorStd, controlDiffPriorStd, controlMutationStd;
		DiagonalGMM staticPrior;

#endif
		int previous_search_window_;
		bool use_machine_learning_;
		int machine_learning_samples_;
		int noisy_machine_learning_samples_;

		/*
		Below, an interface for operation without callbacks. This is convenient for C# integration and custom multithreading. See InvPendulumTest.cpp for the usage.
		*/
		virtual void __stdcall startIteration(bool advanceTime, const float *initialState, const float *mirroredInitialState);
		virtual void __stdcall startPlanningStep(int stepIdx);
		//typically, this just returns sampleIdx. However, if there's been a resampling operation, multiple new samples may link to the same previous sample (and corresponding state)
		virtual int __stdcall getPreviousSampleIdx(int sampleIdx, int timeStep = -1);
		//samples a new control, considering an optional gaussian prior with diagonal covariance (this corresponds to the \mu_p, \sigma_p, C_u in the paper, although the C_u is computed on Unity side and raw stdev and mean arrays are passed to the optimizer)
		virtual void __stdcall getControl(int sampleIdx, float *out_control, const float *priorMean=0, const float *priorStd=0);
		virtual void __stdcall getControlL1(int sampleIdx, float *out_control, const float *priorMean = 0, const float *priorStd = 0);
		virtual void __stdcall getControlKL(int sampleIdx, float *out_control, const float *priorMean = 0, const float *priorStd = 0);

		//For the sample <sampleIdx>, get the state that the simulator thinks the sample is in. If <out_state> is nullptr the this checks that the <marginals> link together correctly regarding the state.
		virtual void __stdcall getAssumedStartingState(int sampleIdx, float *out_state);
		//t=0 => initial segment state, t=1 => resulting state after applying control
		virtual void __stdcall updateResults(int sampleIdx, const float* starting_state, const float *used_control, const float *end_state, double stateCost, double controlCost = 0.0f);
		virtual void __stdcall endPlanningStep(int stepIdx);
		virtual void __stdcall endIteration();
		virtual float __stdcall decideSampling(float* state, float* control, float* future_state);
		//uniformBias: portion of no-prior samples in the paper (0..1)
		//resampleThreshold: resampling threshold, same as in the paper. Default 0.5
		//useGaussianBackPropagation: if true, the Gaussian local refinement (Algorithm 2 of the paper) is used. 
		//gbpRegularization: the regularization of Algorithm 2. Default 0.001 
		virtual void __stdcall setParams(float resampleThreshold, bool learning, int nTrajectories)
		{
			this->resampleThreshold=resampleThreshold;
			this->learning_=learning;
			this->nSamples=nTrajectories;
		}
		virtual void __stdcall setResamplingThreshold(float new_threashold){
			new_threashold = std::min(1.0f,new_threashold);
			new_threashold = std::max(0.0f,new_threashold);
			resampleThreshold = new_threashold;
		}
		virtual int __stdcall getBestSampleLastIdx();
		virtual int __stdcall getIterationIdx() {
			return iterationIdx;
		}
		//Eigen::MatrixXf marginalDataToMatrix(void);
		//Eigen::MatrixXf LearningControlPBP::stateAndControlToMatrix(void);

		//A vector of MarginalSamples for each graph node, representing a GMM of joint state and control 
		//This is made public for easier debug visualizations, but you should treat this as read-only data.
#ifndef SWIG
		std::vector<std::vector<MarginalSample> > marginals;
		std::deque<std::vector<MarginalSample> > previousMarginals;

		
#endif
		int getNumTrajectories();
		int getNumSteps();
		void restart()
		{
			iterationIdx=0;
		}
	protected:
		bool learning_;
		std::vector<Eigen::VectorXf > fullPolicyTrajectories;

		MarginalSample performed_transition_;
		bool old_best_valid_;
		bool smoothed_valid_;
		std::deque<MarginalSample> oldBest;
		double bestCost;
	

		float previous_time_prob_;
		float previous_frame_prob_;
		float nearest_neighbor_prob_;
		float machine_learning_prob_;

		Eigen::VectorXf previous_frame_stdev_;
		Eigen::VectorXf nearest_neighbor_stdev_;
		Eigen::VectorXf machine_learning_stdev_;


		int nSteps,maxSteps;
		int nSamples;
		int maxSamples;
		int iterationIdx;
		bool resample;
		int currentStep;
		int nextStep;
		int bestFullSampleIdx;
		bool timeAdvanced;
		DynamicPdfSampler *selector;
		double resampleThreshold;  //0 = never resample, >1 = always resample
		float learningCostThreshold;
		std::vector<std::vector<TeachingSample* > > policySamples;
	public:
		bool learnedInLastIteration;
		int getCurrentStep(void);
	private:

		std::map<int,std::unique_ptr<DiagonalGMM> > proposals_;
		std::map<int,std::unique_ptr<DiagonalGMM> > priors_;

		std::vector<TeachingSample> keys_;

		SmoothingControl operator=(const SmoothingControl& other);
		SmoothingControl(const SmoothingControl& other);
		void resize_marginal(void);


	};

} //namespace AaltoGames


#endif //OnlineForestLearningControl_H


