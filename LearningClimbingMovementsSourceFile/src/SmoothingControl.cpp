

//#define EIGEN_RUNTIME_NO_MALLOC // Define this symbol to enable runtime tests for allocations
#include "SmoothingControl.h"
#include "DynamicPdfSampler.h"
#define ENABLE_DEBUG_OUTPUT
#include "Debug.h"
#include "EigenMathUtils.h"
#include <iostream> 
#include <time.h>
#include "ClippedGaussianSampling.h"
//#include "RegressionUtils.hpp"
using namespace Eigen;

namespace AaltoGames
{

	void mean_and_stdev(const std::deque<std::unique_ptr<SmoothingControl::TeachingSample> >& samples, Eigen::VectorXf& mean, Eigen::VectorXf& stdev) {

		if (samples.size() <= 1) {
			mean.resize(0);
			stdev.resize(0);
		}

		int dimensions = samples.front()->state_control_.size();
		mean.resize(dimensions);
		mean.setZero();
		stdev.resize(dimensions);
		stdev.setZero();

		for (const std::unique_ptr<SmoothingControl::TeachingSample>& sample : samples) {
			mean += sample->state_control_;
		}
		mean /= (float)samples.size();

		for (const std::unique_ptr<SmoothingControl::TeachingSample>& sample : samples) {
			stdev += (sample->state_control_ - mean).cwiseAbs2();
		}
		stdev /= (float)samples.size();
		stdev = stdev.cwiseSqrt();

	}

	void mean_and_stdev(const std::deque<std::shared_ptr<SmoothingControl::TeachingSample> >& samples, Eigen::VectorXf& mean, Eigen::VectorXf& stdev) {

		if (samples.size() <= 1) {
			mean.resize(0);
			stdev.resize(0);
		}

		int dimensions = samples.front()->state_control_.size();
		mean.resize(dimensions);
		mean.setZero();
		stdev.resize(dimensions);
		stdev.setZero();

		for (const std::shared_ptr<SmoothingControl::TeachingSample>& sample : samples) {
			mean += sample->state_control_;
		}
		mean /= (float)samples.size();

		for (const std::shared_ptr<SmoothingControl::TeachingSample>& sample : samples) {
			stdev += (sample->state_control_ - mean).cwiseAbs2();
		}
		stdev /= (float)samples.size();
		stdev = stdev.cwiseSqrt();

	}


	float compare_only_state(const float* memory_vector, const float* state, void* dim) {

		int dimension = *((int*)dim);

		float result = 0.0f;

		for (int i = 0; i < dimension; i++) {
			result += std::abs(memory_vector[i] - state[i]);
		}

		return result;

	}

	static inline bool valid_float(const float& num) {
		return (num - num) == (num - num);
	}

	static bool determinism_debug = false;

	class IncrementalWeighedAverage
	{
	public:
		double average;
		double sum;
		double wSum;
		IncrementalWeighedAverage()
		{
			average = 0;
			sum = 0;
			wSum = 0;
		}
		void update(double val, double weight)
		{
			wSum += weight;
			sum += weight*val;
			average = sum / wSum;
			if (!validFloat(average))
				average = 0;
		}
		void update2(double val, double power)
		{
			sum += pow(val, power);
			wSum += 1.0;
			average = pow(sum / wSum, 1 / power);
			if (!validFloat(average))
				average = 0;
		}
	};

	template<typename first_type>
	static bool second_is_smaller(const std::pair<first_type, float>& first, const std::pair<first_type, float>& second) {
		return first.second < second.second;
	}

	static bool smaller_cost_to_go(const std::pair<SmoothingControl::TeachingSample*, float>& first, const std::pair<SmoothingControl::TeachingSample*, float>& second) {
		return first.first->cost_to_go_ < second.first->cost_to_go_;
	}

	static bool smaller_cost_to_go_sample(const SmoothingControl::TeachingSample& first, const SmoothingControl::TeachingSample& second) {
		return first.cost_to_go_ < second.cost_to_go_;
	}

	static bool smaller_cost_to_go_ptr(const SmoothingControl::TeachingSample* first, const SmoothingControl::TeachingSample* second) {
		return first->cost_to_go_ < second->cost_to_go_;
	}


	static double evalDiagGaussianQuadraticForm(const VectorXf &x, const VectorXf &mean, const VectorXf &sd)
	{
		//Eigen::internal::set_is_malloc_allowed(false);
		return ((x - mean).cwiseQuotient(sd)).squaredNorm();
		//	Eigen::internal::set_is_malloc_allowed(true);
	}


	void SmoothingControl::init_neural_net(int input_dim, int output_dim, MultiLayerPerceptron& net, bool sigmoid_network) {

		unsigned seed = (unsigned)time(nullptr);
		srand(seed);

		int layer_width = 100;

		if (input_dim == output_dim) {
			layer_width = std::max(output_dim, layer_width);
			layer_width = std::max(input_dim, layer_width);
		}

		std::vector<unsigned> layers;
		layers.push_back(input_dim);
		layers.push_back(layer_width);
		layers.push_back(layer_width);
		layers.push_back(layer_width);
		layers.push_back(layer_width);
		layers.push_back(layer_width);
		layers.push_back(output_dim);

		const int max_amount_of_params = 51224;
		int amount_params = 0;

		unsigned hidden_layer_width = 73;

		if (sigmoid_network) {
			//net.build_sigmoid_network(layers);
		}
		else {

			net.build_network(layers);
			amount_params = net.get_amount_parameters();

		}




		Eigen::VectorXf range = (controlMax - controlMin).cwiseAbs();
		float min_range = range.minCoeff();

		net.max_gradient_norm_ = 0.5f;
		net.learning_rate_ = 0.0001f;
		net.min_weight_ = std::numeric_limits<float>::lowest();
		net.max_weight_ = std::numeric_limits<float>::max();
		net.adam_first_moment_smoothing_ = 0.9f;
		net.adam_second_moment_smoothing_ = 0.99f;

		net.drop_out_stdev_ = regularization_noise_;

		float weight_min = -0.2f;
		float weight_max = 0.2f;
		float bias_val = 0.1f;
		net.randomize_weights(weight_min, weight_max, bias_val);
		//		net.error_path_ = MultiLayerPerceptron::ErrorPath::NORMAL;
		//		net.use_input_scaling_ = true;

				//If the input is over input_familiarity_threshold_ standard deviations from the mean input, it will be dropped out.
		//		net.input_familiarity_threshold_ = 10.0f;

	}


	int SmoothingControl::getCurrentStep(void) {
		return currentStep;
	}

	std::vector<std::pair<SmoothingControl::TeachingSample*, float> > SmoothingControl::k_nearest_neighbor_linear_search(int k, std::vector<TeachingSample*>& data, TeachingSample& key_vector, teaching_sample_distance distance_function) {

		float current_dist = 0;
		std::vector<std::pair<TeachingSample*, float> > closest;

		for (TeachingSample* datum : data) {

			current_dist = distance_function(*datum, key_vector);

			if ((int)closest.size() < k) {
				closest.push_back(std::make_pair(datum, current_dist));
				std::sort(closest.begin(), closest.end(), smaller_cost_to_go);
				std::stable_sort(closest.begin(), closest.end(), second_is_smaller<TeachingSample*>);
			}
			else {
				if (current_dist < closest[closest.size() - 1].second) {
					closest[closest.size() - 1] = std::make_pair(datum, current_dist);
					std::sort(closest.begin(), closest.end(), smaller_cost_to_go);
					std::stable_sort(closest.begin(), closest.end(), second_is_smaller<TeachingSample*>);
				}
			}

		}

		return closest;

	}

	void SmoothingControl::init(int nSamples, int nSteps, int nStateDimensions, int nControlDimensions, const float *controlMinValues, const float *controlMaxValues, const float *controlMean, const float *controlPriorStd, const float *controlDiffPriorStd, const float controlMutationStdScale, bool _useMirroring)
	{
		this->controlMin.resize(nControlDimensions);
		this->controlMax.resize(nControlDimensions);
		memcpy(&this->controlMin[0], controlMinValues, sizeof(float)*nControlDimensions);
		memcpy(&this->controlMax[0], controlMaxValues, sizeof(float)*nControlDimensions);

		unsigned seed = (unsigned)time(nullptr);
		srand(seed);

		resampleThreshold = 0;
		Eigen::initParallel();
		time_t timer;
		struct tm y2k = { 0 };
		double seconds;

		y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
		y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

		time(&timer);

		actor_in_training_.reset();
		actor_.clear();
		actor_copy_.clear();

		autoencoder_in_training_.reset();
		autoencoder_.clear();
		autoencoder_copy_.clear();

		discriminator_.reset();
		discriminator_copy_.reset();
		discriminator_in_training_.reset();
		discriminator_competing_.reset();

		critic_in_training_.reset();


		seconds = difftime(timer, mktime(&y2k));
		//srand((int)seconds);	//set random seed
		iterationIdx = 0;
		marginals.resize(nSteps + 1);
		previousMarginals.resize(nSteps + 1);
		this->nSamples = nSamples;
		this->maxSamples = nSamples;
		this->nSteps = nSteps;
		this->maxSteps = nSteps;

		this->nStateDimensions = nStateDimensions;
		this->nControlDimensions = nControlDimensions;
		this->controlMean.resize(nControlDimensions);
		this->controlPriorStd.resize(nControlDimensions);
		this->controlDiffPriorStd.resize(nControlDimensions);


		policySamples.resize(nSteps + 1);
		for (int i = 0; i <= nSteps; i++) {
			policySamples[i].resize(nSamples);
			for (int j = 0; j < nSamples; j++) {
				policySamples[i][j] = nullptr;
			}
		}
		oldBest.resize(nSteps + 1);
		if (controlMean != NULL)
		{
			memcpy(&this->controlMean[0], controlMean, sizeof(float)*nControlDimensions);
		}
		else
		{
			this->controlMean.setZero();
		}
		for (int step = 0; step <= nSteps; step++)
		{
			oldBest[step].init(nStateDimensions, nControlDimensions);
			marginals[step].resize(nSamples);

			for (int i = 0; i < nSamples; i++)
			{
				marginals[step][i].init(nStateDimensions, nControlDimensions);
			}
		}
		selector = new DynamicPdfSampler(nSamples);
		bestCost = FLT_MAX;  //not yet found any solution
		timeAdvanced = false;
		Eigen::VectorXf fullMin(nControlDimensions*nSteps), fullMax(nControlDimensions*nSteps);
		for (int step = 0; step < nSteps; step++)
		{
			for (int d = 0; d < nControlDimensions; d++)
			{
				fullMin[step*nControlDimensions + d] = controlMin[d];
				fullMax[step*nControlDimensions + d] = controlMax[d];
			}
		}
		//		samplingTree.init(nControlDimensions*nSteps,0,fullMin.data(),fullMax.data(),nSamples*3);
		setSamplingParams(controlPriorStd, controlDiffPriorStd, controlMutationStdScale);
		bestFullSampleIdx = 0;

		learning_ = true;



		//		evolutionary_nets_amount_ = 0;
		//		evolution_pool_size_ = 100;
		//		if (evolutionary_nets_amount_ <= 0) {
		//			evolution_pool_size_ = 0;
		//		}
		min_evolution_population_ = 0;//evolution_pool_size_ / 2;
		evaluations_for_evolution_ = 20;
		evolution_rate_ = 0.1f;
		evolution_mutation_stdev_ = 0.01f;
		evolutionary_pool_.clear();
		evolutionary_nets_.clear();

		//		for (int i = 0; i < evolutionary_nets_amount_; i++) {
		//			std::unique_ptr<MultiLayerPerceptron> tmp_actor = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron);
		//			init_neural_net(nStateDimensions, nControlDimensions, *tmp_actor);
		//			evolutionary_nets_.push_back(std::move(tmp_actor));
		//		}

		const float min_weight = -0.2f;
		const float max_weight = 0.2f;

		//		for (int i = 0; i < evolution_pool_size_; i++) {
		//
		//			MultiLayerPerceptron* net = evolutionary_nets_[0].get();
		//			net->randomize_weights(min_weight, max_weight);
		//
		//			std::unique_ptr<EvolutionData> individual = std::unique_ptr<EvolutionData>(new EvolutionData);
		//			individual->fenotype_ = net->get_parameters_as_one_vector();
		//			individual->cost_ = 0.0f;
		//			individual->num_evaluations_ = 0;
		//
		//			evolutionary_pool_.push_back(std::move(individual));
		//
		//		}

		//		for (int i = 0; i < evolutionary_nets_amount_; i++) {
		//			int rand_idx = rand() % evolutionary_pool_.size();
		//
		//			evolution_usage_to_pool_idx_[i] = rand_idx;
		//			evolutionary_nets_[i]->set_parameters(evolutionary_pool_[rand_idx]->fenotype_);
		//		}

	}

	const generic_density_forest_vector* teaching_sample_to_key_vector(const SmoothingControl::TeachingSample& sample)
	{

		if (sample.state_.size() == 0)
		{
			std::cout << "Key vector size zero. Msg from key vect fun.";
		}

		const generic_density_forest_vector* return_value = &(sample.state_);

		return return_value;

	}

	SmoothingControl::SmoothingControl()
	{
		cMSE = std::numeric_limits<float>::infinity();
		auto dummy = []() {
			return;
		};

		long_term_learning_ = std::async(std::launch::async, dummy);
		building_model_ = std::async(std::launch::async, dummy);
		training_autoencoder_ = std::async(std::launch::async, dummy);
		training_discriminator_ = std::async(std::launch::async, dummy);
		training_critic_ = std::async(std::launch::async, dummy);
		training_dynamics_model_ = std::async(std::launch::async, dummy);


		previous_frame_stdev_ = Eigen::VectorXf::Zero(0);
		nearest_neighbor_stdev_ = Eigen::VectorXf::Zero(0);
		machine_learning_stdev_ = Eigen::VectorXf::Zero(0);

		sampling_counter_ = 300;
		forced_sampling_ = 1;
		use_sampling_ = true;

		use_forests_ = true;
		amount_recent_ = 100;

		previous_time_prob_ = 1.0f;
		previous_frame_prob_ = 1.0f;
		nearest_neighbor_prob_ = 1.0f;
		machine_learning_prob_ = 1.0f;

		//setParams(0.1,0.1,false);
		amount_data_in_tree_ = 2500;

		nn_trajectories_ = 3;
		machine_learning_samples_ = 3;
		noisy_machine_learning_samples_ = 3;

		use_machine_learning_ = true;
		previous_search_window_ = 0;


		storedSamplesPercentage = 25;
		noPriorTrajectoryPortion = 0.25f;


		setParams(0.5f, true, 0);

		validation_fraction_ = 0.2f;

		learning_budget_ = 2000;
		number_of_data_in_leaf_ = 10;
		number_of_hyperplane_tries_ = 25;
		number_of_nearest_neighbor_trees_ = 5;

		lookahead_ = 1;

		ann_forest_ = GenericDensityForest<TeachingSample>(number_of_nearest_neighbor_trees_);
		ann_forest_.set_key_vector_function(teaching_sample_to_key_vector);
		tree_under_construction_ = GenericDensityTree<TeachingSample>();
		tree_under_construction_.root_->get_key_vector_ = teaching_sample_to_key_vector;

		old_best_valid_ = false;
		smoothed_valid_ = false;

		//		use_autoencoder_ = false;

		//		use_discriminator_ = false;
		amount_discriminator_data_ = 50 * learning_budget_;
		familiarity_threshold_ = 0.6f;


		since_running_on_neural_network_ = learning_budget_;

		critic_freeze_interval_ = 100;
		since_critic_freeze_ = 0;
		//		q_learning_ = false;
		competing_actor_in_training_.reset();

		state_groupings_.clear();
		control_groupings_.clear();

		regularization_noise_ = 0.1f;

		sampling_interval_ = 1;

		//		use_dynamics_model_ = false;
		//		if (q_learning_) {
		//			use_dynamics_model_ = false;
		//		}
		amount_dynamics_data_ = 20000;

		initial_learning_iterations_ = INT_MAX;

		control_scheme_ = ControlScheme::KL;


	}

	SmoothingControl::~SmoothingControl()
	{
		delete selector;

		building_model_.wait();
		long_term_learning_.wait();
		training_autoencoder_.wait();
		training_discriminator_.wait();
		training_critic_.wait();
		training_dynamics_model_.wait();

	}



	void __stdcall SmoothingControl::setSamplingParams(const float *controlPriorStd, const float *controlDiffPriorStd, float controlMutationStdScale)
	{
		memcpy(&this->controlPriorStd[0], controlPriorStd, sizeof(float)*nControlDimensions);
		memcpy(&this->controlDiffPriorStd[0], controlDiffPriorStd, sizeof(float)*nControlDimensions);
		controlMutationStd = controlMutationStdScale*this->controlPriorStd;
		staticPrior.resize(1, nControlDimensions);
		staticPrior.mean[0] = controlMean;
		staticPrior.setStds(this->controlPriorStd);
		staticPrior.weights[0] = 1;
		staticPrior.weightsUpdated();

		previous_frame_stdev_ = controlMutationStd;
		nearest_neighbor_stdev_ = controlMutationStd;
		machine_learning_stdev_ = controlMutationStd;

	}

	//void SmoothingControl::shiftWithAutoencoder(float * state, float * control, int sample_idx)
	//{
	//
	//	if (!use_autoencoder_) {
	//		return;
	//	}
	//
	//	if (autoencoder_.size() == 0) {
	//		return;
	//	}
	//
	//	float tmp_array[500];
	//	Eigen::Map<Eigen::VectorXf> control_state(tmp_array, nStateDimensions + nControlDimensions);
	//
	//	Eigen::Map<Eigen::VectorXf> state_map(state, nStateDimensions);
	//	Eigen::Map<Eigen::VectorXf> control_map(control, nControlDimensions);
	//
	//	control_state.head(nControlDimensions) = control_map;
	//	control_state.tail(nStateDimensions) = state_map;
	//
	//	MultiLayerPerceptron* autoencoder = (autoencoder_)[sample_idx].get();
	//
	//	autoencoder->run(control_state.data(), control_state.data());
	//	control_map = control_state.head(nControlDimensions);
	//}

//	void SmoothingControl::shiftWithCritic(float * state, float * control, int sample_idx)
//	{
//
//		return;
//
//		if (!q_learning_) {
//			return;
//		}
//
//		if (!critic_.get()) {
//			return;
//		}
//
//		MultiLayerPerceptron& critic = *critic_;
//
//		float tmp[500];
//		int idx = 0;
//		for (int i = 0; i < nControlDimensions; i++) {
//			tmp[idx] = control[i];
//			idx++;
//		}
//		for (int i = 0; i < nStateDimensions; i++) {
//			tmp[idx] = state[i];
//			idx++;
//		}
//
//		float orig_gradient_max = critic.max_gradient_norm_;
//		critic.max_gradient_norm_ = 1000000.0f;
//
//		std::lock_guard<std::mutex> lock(using_critic_);
//
//		float delta = 1.0f;
//		const bool is_gradient = true;
//		critic.backpropagate_deltas(tmp, &delta, is_gradient);
//		critic.max_gradient_norm_ = orig_gradient_max;
//
//		const Eigen::VectorXf& gradient = critic.input_operation_->deltas_;
//
//		float shift_amount = 0.1f;
//
//		for (int i = 0; i < nControlDimensions; i++) {
//			control[i] -= shift_amount*gradient[i];
//		}
//
//	}

	void costsToWeights(VectorXd &costs, bool weighedAverage, float maxSqCost)
	{
		VectorXd::Index minIndex;
		costs.minCoeff(&minIndex);

		if (weighedAverage)
		{
			double scale = _min(1.0, maxSqCost / costs[minIndex]);
			costs *= -0.5*scale;
			for (int i = 0; i < costs.rows(); i++)
			{
				costs[i] = exp(costs[i]);
			}
			costs /= costs.sum();
			return;
		}
		else
		{
			costs.setZero();
			costs[minIndex] = 1;

		}
	}

	void SmoothingControl::resize_marginal(void) {
		while ((int)marginals.size() > nSteps + 1) {
			marginals.pop_back();
		}

		while ((int)marginals.size() < nSteps + 1) {
			marginals.push_back(std::vector<MarginalSample>());
		}


		for (int step = 0; step < (int)marginals.size(); step++) {
			std::vector<MarginalSample>& marginal = marginals[step];
			while ((int)marginal.size() > nSamples) {
				marginal.pop_back();
			}

			while ((int)marginal.size() < nSamples) {
				marginal.push_back(MarginalSample());
			}

		}

		for (int step = 0; step < (int)marginals.size(); step++) {
			for (int particle = 0; particle < (int)marginals[0].size(); particle++) {
				MarginalSample& sample = marginals[step][particle];
				sample.init(nStateDimensions, nControlDimensions);
			}
		}
	}


	void __stdcall SmoothingControl::startIteration(bool advanceTime, const float *initialState, const float *mirroredInitialState)
	{

		nSamples = std::min(nSamples, maxSamples);

		DiagonalGaussian sample_dist;
		sample_dist.first = Eigen::VectorXf::Zero(nControlDimensions);
		sample_dist.second = Eigen::VectorXf::Zero(nControlDimensions);

		const int max_distributions = 20;

		std::vector<DiagonalGaussian> dists_tmp(max_distributions, sample_dist);
		gaussian_distributions_ = std::vector<std::vector<DiagonalGaussian> >(nSamples, dists_tmp);
		sampling_distributions_ = std::vector<DiagonalGaussian>(nSamples, sample_dist);


		resize_marginal();


		//int nUniform=(int)(naiveTrajectoryPortion*(float)nSamples);
		for (int step = 0; step <= nSteps; step++)
		{
			for (int i = 0; i < nSamples; i++)
			{
				marginals[step][i].targetStateValid = false;
				policySamples[step][i] = nullptr;
			}
		}

		Eigen::Map<const Eigen::VectorXf> init_state_map(initialState, nStateDimensions);

		for (int i = 0; i < nSamples; i++)
		{
			MarginalSample &sample = marginals[0][i];
			sample.state = init_state_map;
			sample.control.setZero();
			sample.previousControl.setZero();
			sample.previousPreviousControl.setZero();
			sample.forwardBelief = 1;
			sample.fwdMessage = 1;
			sample.belief = 1;
			sample.fullCost = 0;
			sample.stateCost = 0;
			sample.controlCost = 0;
			sample.previousMarginalSampleIdx = i;
			sample.bestStateCost = FLT_MAX;
			sample.costSoFar = 0;
			sample.nForks = 0;
		}

		old_best_valid_ = false;
		smoothed_valid_ = false;

		if (iterationIdx > 0) {
			old_best_valid_ = true;
			smoothed_valid_ = true;
		}

		if (advanceTime)
		{
			if (iterationIdx > 0) {

				old_best_valid_ = false;
				smoothed_valid_ = false;

				float dist = 0.0f;
				if (performed_transition_.state.size() > 0) {
					//					for (int i = 0; i < nStateDimensions; i++) {
					//						float diff = performed_transition_.state[i] - initialState[i];
					//						dist += std::abs(diff);
					//					}
				}

				const float tolerance = 0.00001f;

				if (dist <= tolerance) {
					old_best_valid_ = true;
					smoothed_valid_ = true;
				}

				const bool force_old_best_valid = false;

				if (force_old_best_valid) {
					old_best_valid_ = true;
					smoothed_valid_ = true;
				}

			}
			else {
				old_best_valid_ = false;
				smoothed_valid_ = false;
			}


			if (old_best_valid_) {
				if (oldBest.size() > 0) {
					oldBest.pop_front();
				}
				if (previousMarginals.size() > 0) {
					previousMarginals.pop_front();
				}

				for (int i = 0; i < nSamples; i++)
				{
					marginals[0][i].control = performed_transition_.control;
					marginals[0][i].previousControl = performed_transition_.previousControl;
					marginals[0][i].previousPreviousControl = performed_transition_.previousPreviousControl;
				}
			}
			else {
				oldBest.clear();
				previousMarginals.clear();

				for (int i = 0; i < nSamples; i++)
				{
					marginals[0][i].control.setZero();
					marginals[0][i].previousControl.setZero();
					marginals[0][i].previousPreviousControl.setZero();
				}
			}

		}

		timeAdvanced = advanceTime;
	}

	void __stdcall SmoothingControl::startPlanningStep(int step)
	{



		if (keys_.size() != nSamples) {
			keys_.resize(nSamples);
		}

		if (priors_.size() != nSamples) {

			priors_.clear();

			DiagonalGMM prior;
			prior.resize(1, nControlDimensions);
			prior.weights[0] = 1;	//only need to set once, as the prior will always have just a single component
			prior.weightsUpdated();
			for (int i = 0; i < nSamples; i++) {
				priors_[i] = std::unique_ptr<DiagonalGMM>(new DiagonalGMM(prior));
			}

		}

		if (proposals_.size() != nSamples) {

			proposals_.clear();

			DiagonalGMM proposal = DiagonalGMM();
			proposal.resize(1, nControlDimensions);

			for (int i = 0; i < nSamples; i++) {
				proposals_[i] = std::unique_ptr<DiagonalGMM>(new DiagonalGMM(proposal));
			}

		}


		currentStep = step;
		nextStep = currentStep + 1;


		int particle_idx = 0;

		if (old_best_valid_) {
			if (particle_idx < (int)marginals[nextStep].size()) {
				marginals[nextStep][particle_idx].particleRole = ParticleRole::OLD_BEST;
				particle_idx++;
				marginals[nextStep][particle_idx].priorSampleIdx = 0;
				marginals[nextStep][particle_idx].previous_frame_prior_ = false;
				marginals[nextStep][particle_idx].nearest_neighbor_prior_ = false;
				marginals[nextStep][particle_idx].machine_learning_prior_ = false;
			}

			if (smoothed_valid_) {
				if (particle_idx < (int)marginals[nextStep].size()) {
					marginals[nextStep][particle_idx].particleRole = ParticleRole::SMOOTHED;
					particle_idx++;
					marginals[nextStep][particle_idx].priorSampleIdx = 0;
					marginals[nextStep][particle_idx].previous_frame_prior_ = false;
					marginals[nextStep][particle_idx].nearest_neighbor_prior_ = false;
					marginals[nextStep][particle_idx].machine_learning_prior_ = false;
				}
			}
		}



		//		if (determinism_debug) {
		//			if (particle_idx < (int)marginals[nextStep].size() && use_machine_learning_ && actor_.size() > 0) {
		//				marginals[nextStep][particle_idx].particleRole = ParticleRole::DEBUG_ML;
		//				marginals[nextStep][particle_idx].previous_frame_prior_ = false;
		//				marginals[nextStep][particle_idx].nearest_neighbor_prior_ = false;
		//				marginals[nextStep][particle_idx].machine_learning_prior_ = false;
		//				particle_idx++;
		//			}
		//		}
		//		for (int i = 0; i < evolutionary_nets_amount_; i++)
		//		{
		//			if (particle_idx < (int)marginals[nextStep].size()) {
		//				marginals[nextStep][particle_idx].particleRole = ParticleRole::EVOLUTIONARY;
		//				marginals[nextStep][particle_idx].priorSampleIdx = i;
		//				marginals[nextStep][particle_idx].previous_frame_prior_ = false;
		//				marginals[nextStep][particle_idx].nearest_neighbor_prior_ = false;
		//				marginals[nextStep][particle_idx].machine_learning_prior_ = false;
		//				particle_idx++;
		//			}
		//
		//		}
		//		if (false && use_machine_learning_ && actor_.size() > 0) {
		//			if (particle_idx < (int)marginals[nextStep].size()) {
		//				marginals[nextStep][particle_idx].particleRole = ParticleRole::MACHINE_LEARNING_NO_RESAMPLING;
		//				marginals[nextStep][particle_idx].previous_frame_prior_ = false;
		//				marginals[nextStep][particle_idx].nearest_neighbor_prior_ = false;
		//				marginals[nextStep][particle_idx].machine_learning_prior_ = true;
		//				particle_idx++;
		//			}
		//		}



		if (use_machine_learning_ && actor_.size() > 0) {
			for (int i = 0; i < machine_learning_samples_; i++)
			{
				if (particle_idx < (int)marginals[nextStep].size()) {
					marginals[nextStep][particle_idx].particleRole = ParticleRole::MACHINE_LEARNING_NO_VARIATION;
					marginals[nextStep][particle_idx].previous_frame_prior_ = false;
					marginals[nextStep][particle_idx].nearest_neighbor_prior_ = false;
					marginals[nextStep][particle_idx].machine_learning_prior_ = true;
					particle_idx++;
				}

			}
		}

		//		if (false && q_learning_ && rl_actor_.size() > 0) {
		//			for (int i = 0; i < machine_learning_samples_; i++)
		//			{
		//				if (particle_idx < (int)marginals[nextStep].size()) {
		//					marginals[nextStep][particle_idx].particleRole = ParticleRole::REINFORCEMENT_LEARNING;
		//					marginals[nextStep][particle_idx].previous_frame_prior_ = false;
		//					marginals[nextStep][particle_idx].nearest_neighbor_prior_ = false;
		//					marginals[nextStep][particle_idx].machine_learning_prior_ = true;
		//					particle_idx++;
		//				}
		//
		//			}
		//		}


		if (use_machine_learning_ && actor_.size() > 0) {
			for (int i = 0; i < noisy_machine_learning_samples_; i++)
			{
				if (particle_idx < (int)marginals[nextStep].size()) {
					marginals[nextStep][particle_idx].particleRole = ParticleRole::MACHINE_LEARNING;

					switch (control_scheme_)
					{
					case AaltoGames::SmoothingControl::L1:
						marginals[nextStep][particle_idx].previous_frame_prior_ = true;
						marginals[nextStep][particle_idx].nearest_neighbor_prior_ = false;
						marginals[nextStep][particle_idx].machine_learning_prior_ = true;
						break;
					case AaltoGames::SmoothingControl::KL:
						marginals[nextStep][particle_idx].previous_frame_prior_ = true;
						marginals[nextStep][particle_idx].nearest_neighbor_prior_ = false;
						marginals[nextStep][particle_idx].machine_learning_prior_ = true;
						break;
					default:
						marginals[nextStep][particle_idx].previous_frame_prior_ = true;
						marginals[nextStep][particle_idx].nearest_neighbor_prior_ = false;
						marginals[nextStep][particle_idx].machine_learning_prior_ = true;
						break;
					}

					particle_idx++;
				}

			}
		}


		int nFree = (int)(noPriorTrajectoryPortion*(float)nSamples);
		for (int i = 0; i < nFree; i++) {
			if (particle_idx < (int)marginals[nextStep].size()) {
				marginals[nextStep][particle_idx].particleRole = ParticleRole::FREE;
				marginals[nextStep][particle_idx].previous_frame_prior_ = false;
				marginals[nextStep][particle_idx].nearest_neighbor_prior_ = false;
				marginals[nextStep][particle_idx].machine_learning_prior_ = false;
				particle_idx++;
			}
		}


		if (iterationIdx > 0) //&& learning
		{
			for (int i = 0; i < nn_trajectories_; i++)
			{
				if (particle_idx < (int)marginals[nextStep].size()) {
					marginals[nextStep][particle_idx].particleRole = ParticleRole::NEAREST_NEIGHBOR;
					marginals[nextStep][particle_idx].previous_frame_prior_ = true;
					marginals[nextStep][particle_idx].nearest_neighbor_prior_ = true;
					marginals[nextStep][particle_idx].machine_learning_prior_ = false;
					particle_idx++;
				}
			}
		}




		if (iterationIdx > 0) {
			for (; particle_idx < (int)marginals[nextStep].size(); particle_idx++) {
				if (particle_idx < (int)marginals[nextStep].size()) {
					marginals[nextStep][particle_idx].particleRole = ParticleRole::PREVIOUS_FRAME_PRIOR;
					marginals[nextStep][particle_idx].previous_frame_prior_ = true;
					marginals[nextStep][particle_idx].nearest_neighbor_prior_ = false;
					marginals[nextStep][particle_idx].machine_learning_prior_ = false;
				}
			}
		}


		if (!old_best_valid_) {
			for (int i = 0; i < (int)marginals[nextStep].size(); i++) {
				marginals[nextStep][i].previous_frame_prior_ = false;
			}
		}

		if (actor_.size() == 0) {
			for (int i = 0; i < (int)marginals[nextStep].size(); i++) {
				marginals[nextStep][i].machine_learning_prior_ = false;
			}
		}

		//Resampling, the new and simple version (prune everything with cost larger than best trajectory cost * resamplingThreshold
		if (step > 0)
		{
			//find best trajectory so far
			float bestCost = FLT_MAX;
			int best_idx = 0;
			float worstCost = -FLT_MAX;
			int worst_idx = 0;
			for (int sampleIdx = 0; sampleIdx < nSamples; sampleIdx++)
			{
				const MarginalSample &sample = marginals[step][sampleIdx];
				float full_cost = (float)sample.fullCost;
				if (full_cost < bestCost) {
					//					if (sample.particleRole != ParticleRole::MACHINE_LEARNING_NO_RESAMPLING
					//						&& sample.particleRole != ParticleRole::POLICY_SEARCH) {
					bestCost = full_cost;
					best_idx = sampleIdx;
					//					}
				}

				if (full_cost > worstCost) {

					if (sample.particleRole != ParticleRole::OLD_BEST) {
						//						&& sample.particleRole != ParticleRole::MACHINE_LEARNING_NO_RESAMPLING
						//						&& sample.particleRole != ParticleRole::POLICY_SEARCH
						//						&& sample.particleRole != EVOLUTIONARY) {
						worstCost = full_cost;
						worst_idx = sampleIdx;
					}


				}
			}

			//mark which trajectories are the continued ones
			std::vector<int> forkedTrajectories;
			forkedTrajectories.reserve(nSamples);
			std::vector<int> prunedTrajectories;
			prunedTrajectories.reserve(nSamples);

			float costThreshold = bestCost + (float)resampleThreshold;
			costThreshold += 0.01f;


			for (int sampleIdx = 0; sampleIdx < nSamples; sampleIdx++)
			{

				MarginalSample &nextSample = marginals[nextStep][sampleIdx];
				nextSample.previousMarginalSampleIdx = sampleIdx;

				const MarginalSample& sample = marginals[step][sampleIdx];
				//All trajectories with cost greater than costThreshold and no special role are pruned
				if (sample.fullCost > costThreshold)
				{
					//					if (sample.particleRole != ParticleRole::MACHINE_LEARNING_NO_RESAMPLING
					//						&& sample.particleRole != ParticleRole::POLICY_SEARCH) {
					prunedTrajectories.push_back(sampleIdx);
					//					}
				}
				else
				{
					//					if (sample.particleRole != ParticleRole::MACHINE_LEARNING_NO_RESAMPLING
					//						&& sample.particleRole != ParticleRole::POLICY_SEARCH) {
					forkedTrajectories.push_back(sampleIdx);
					//					}
				}

			}

			const bool support_best = true;
			if (support_best) {

				if (marginals[nextStep][worst_idx].particleRole != ParticleRole::OLD_BEST) {
					marginals[nextStep][worst_idx].previousMarginalSampleIdx = marginals[nextStep][best_idx].previousMarginalSampleIdx;
					marginals[nextStep][worst_idx].particleRole = ParticleRole::MACHINE_LEARNING;
					marginals[nextStep][worst_idx].previous_frame_prior_ = false;
					marginals[nextStep][worst_idx].nearest_neighbor_prior_ = false;
					marginals[nextStep][worst_idx].machine_learning_prior_ = true;
				}

				//std::shared_ptr<TeachingSample> false_state_sample = std::shared_ptr<TeachingSample>(new TeachingSample(marginals[step][worst_idx]));
				//discriminator_buffer_.push_back(false_state_sample);
			}





			//prune others and select a random continued one
			for (int sampleIdx : prunedTrajectories)
			{
				MarginalSample &nextSample = marginals[nextStep][sampleIdx];

				int choises = forkedTrajectories.size();
				if (choises <= 0) {
					choises = nSamples;
				}

				int forking_idx = rand() % choises;
				if (forkedTrajectories.size() > 0) {
					forking_idx = forkedTrajectories[forking_idx];
				}

				nextSample.previousMarginalSampleIdx = forking_idx;
				marginals[step][nextSample.previousMarginalSampleIdx].nForks++;
			}


			for (int sampleIdx = 0; sampleIdx < nSamples; sampleIdx++)
			{
				MarginalSample &nextSample = marginals[nextStep][sampleIdx];

				if (nextSample.particleRole == ParticleRole::OLD_BEST) {
					for (int connecting_idx = 0; connecting_idx < nSamples; connecting_idx++)
					{
						if (marginals[step][connecting_idx].particleRole == ParticleRole::OLD_BEST) {
							nextSample.previousMarginalSampleIdx = connecting_idx;
							break;
						}

						//if (marginals[step][connecting_idx].particleRole != ParticleRole::OLD_BEST) {
						//	old_best_valid_ = false;
						//	if (actor_.get()) {
						//		nextSample.particleRole = ParticleRole::MACHINE_LEARNING;
						//	}
						//	else {
						//		nextSample.particleRole = ParticleRole::PREVIOUS_FRAME_PRIOR;
						//	}
						//}
					}
				}

				//				if (nextSample.particleRole == ParticleRole::EVOLUTIONARY) {
				//					for (int connecting_idx = 0; connecting_idx < nSamples; connecting_idx++)
				//					{
				//						MarginalSample* current_sample = &marginals[step][connecting_idx];
				//						if (current_sample->particleRole == ParticleRole::EVOLUTIONARY) {
				//							if (current_sample->priorSampleIdx == nextSample.priorSampleIdx) {
				//								nextSample.previousMarginalSampleIdx = connecting_idx;
				//								break;
				//							}
				//						}
				//
				//
				//					}
				//				}

				if (nextSample.particleRole == ParticleRole::SMOOTHED) {
					for (int connecting_idx = 0; connecting_idx < nSamples; connecting_idx++)
					{
						if (marginals[step][connecting_idx].particleRole == ParticleRole::SMOOTHED) {
							nextSample.previousMarginalSampleIdx = connecting_idx;
							break;
						}

					}
				}

				//				if (nextSample.particleRole == ParticleRole::MACHINE_LEARNING_NO_RESAMPLING) {
				//					for (int connecting_idx = 0; connecting_idx < nSamples; connecting_idx++)
				//					{
				//						if (marginals[step][connecting_idx].particleRole == ParticleRole::MACHINE_LEARNING_NO_RESAMPLING) {
				//							nextSample.previousMarginalSampleIdx = connecting_idx;
				//							break;
				//						}
				//					}
				//				}
				//
				//				if (nextSample.particleRole == ParticleRole::POLICY_SEARCH) 
				//				{
				//					for (int connecting_idx = 0; connecting_idx < nSamples; connecting_idx++)
				//					{
				//						if (marginals[step][connecting_idx].particleRole == ParticleRole::POLICY_SEARCH) {
				//							nextSample.previousMarginalSampleIdx = connecting_idx;
				//							break;
				//						}
				//					}
				//				}
			}
		}

		//remember the number of forks
		for (int sampleIdx = 0; sampleIdx < nSamples; sampleIdx++)
		{

			MarginalSample &nextSample = marginals[nextStep][sampleIdx];
			MarginalSample &sample = marginals[step][nextSample.previousMarginalSampleIdx];
			nextSample.nForks = sample.nForks;
		}


	}




	double SmoothingControl::getBestTrajectoryCost()
	{
		return bestCost;
	}


	void SmoothingControl::getConditionalControlGMM(int timeStep, const Eigen::VectorXf &state, DiagonalGMM &dst)
	{
		Debug::throwError("Not impl.");
	}



	void SmoothingControl::train_actor()
	{
		std::deque<std::shared_ptr<TeachingSample>> transition_data;
		std::deque<std::shared_ptr<TeachingSample>> validation_data;
		{
			std::lock_guard<std::mutex> lock(copying_transition_data_);
			transition_data = transition_data_;
			validation_data = validation_data_;
		}
		std::deque<std::shared_ptr<TeachingSample>> dynamics_data;
		{
			std::lock_guard<std::mutex> lock(copying_dynamics_data_);
			dynamics_data = dynamics_data_;
		}

		const unsigned minibatch_size = transition_data.size() / 10;
		if (minibatch_size == 0) {
			return;
		}


		const unsigned input_dim = nStateDimensions;
		unsigned output_dim = nControlDimensions;


		if (!actor_in_training_.get()) {
			actor_in_training_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());

			
			init_neural_net(input_dim, output_dim, *actor_in_training_);

		}

		std::vector<float*> inputs;
		std::vector<float*> outputs;

		inputs.reserve(transition_data.size());
		outputs.reserve(transition_data.size());


		Eigen::VectorXf output_mean = Eigen::VectorXf::Zero(0);
		Eigen::VectorXf output_stdev = Eigen::VectorXf::Zero(0);
		Eigen::VectorXf training_output_stdev = output_stdev;


		bool unsupervised = false;

		auto form_training_data = [&](std::deque<std::shared_ptr<TeachingSample>>& data_set) {

			inputs.clear();
			outputs.clear();


			for (const std::shared_ptr<TeachingSample>& datum : data_set) {

				inputs.push_back(datum->state_.data());
				outputs.push_back(datum->control_state_.data());

			}

			training_output_stdev = output_stdev;

		};

		form_training_data(transition_data);


		output_mean = Eigen::VectorXf::Zero(output_dim);
		output_stdev = Eigen::VectorXf::Zero(output_dim);

		for (const float* datum : outputs) {

			Eigen::Map<const Eigen::VectorXf> out_map = Eigen::Map<const Eigen::VectorXf>(datum, output_dim);
			output_mean += out_map;

		}
		output_mean /= (float)outputs.size();

		for (const float* datum : outputs) {

			Eigen::Map<const Eigen::VectorXf> out_map = Eigen::Map<const Eigen::VectorXf>(datum, output_dim);
			output_stdev += (out_map - output_mean).cwiseAbs2();

		}
		output_stdev /= (float)outputs.size();
		output_stdev = output_stdev.cwiseSqrt();


		if (training_output_stdev.size() == 0) {
			training_output_stdev = output_stdev;
		}

		int max_epochs = 5;
		int epoch = 0;
		bool is_gradient = false;
		while (epoch < max_epochs) {

			const bool use_competing_actor = false;
			if (use_competing_actor && !competing_actor_in_training_.get()) {
				competing_actor_in_training_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());


				init_neural_net(input_dim, output_dim, *competing_actor_in_training_);


			}

			actor_in_training_->train_adam((const float**)inputs.data(), (const float**)outputs.data(), inputs.size(), minibatch_size, is_gradient, training_output_stdev.data());
			if (competing_actor_in_training_.get()) {
				competing_actor_in_training_->train_adam((const float**)inputs.data(), (const float**)outputs.data(), inputs.size(), minibatch_size, is_gradient, training_output_stdev.data());
			}

			epoch++;
		}


		form_training_data(validation_data);


		float mse = std::numeric_limits<float>::infinity();
		if (actor_in_training_.get()) {
			mse = actor_in_training_->mse((const float**)inputs.data(), (const float**)outputs.data(), inputs.size());
		}

		if (mse - mse != mse - mse) {

			int input_size = actor_in_training_->input_operation_->size_;
			int output_size = actor_in_training_->output_operation_->size_;

			actor_in_training_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron);
			init_neural_net(input_size, output_size, *actor_in_training_);
		}

		float competing_mse = std::numeric_limits<float>::infinity();
		if (competing_actor_in_training_.get()) {
			competing_mse = competing_actor_in_training_->mse((const float**)inputs.data(), (const float**)outputs.data(), inputs.size());
		}

		std::cout << "Trained actor. MSE: " << mse << " Data: " << transition_data.size() << std::endl;

		cMSE = mse;

		//actor_copy_ = std::unique_ptr<std::vector<MultiLayerPerceptron>>(new std::vector<MultiLayerPerceptron>(nSamples, *actor_in_training_));
		actor_copy_.clear();
		for (int i = 0; i < maxSamples; i++) {
			std::unique_ptr<MultiLayerPerceptron> tmp = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron(*actor_in_training_));
			actor_copy_.push_back(std::move(tmp));
		}


		if (competing_mse < mse) {
			actor_in_training_.swap(competing_actor_in_training_);
			competing_actor_in_training_.reset();
			std::cout << "Swapping better actor." << std::endl;
		}



	}

	void SmoothingControl::train_actor_with_stdev()
	{

		std::deque<std::shared_ptr<TeachingSample>> transition_data;
		std::deque<std::shared_ptr<TeachingSample>> validation_data;
		{
			std::lock_guard<std::mutex> lock(copying_transition_data_);
			transition_data = transition_data_;
			validation_data = validation_data_;
		}


		const unsigned minibatch_size = transition_data.size() / 10;
		if (minibatch_size == 0) {
			return;
		}


		const unsigned input_dim = nStateDimensions;
		unsigned output_dim = nControlDimensions;


		if (!actor_in_training_.get()) {
			actor_in_training_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());
			init_neural_net(input_dim, output_dim, *actor_in_training_);
		}

		if (!actor_stdev_in_training_.get()) {
			actor_stdev_in_training_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());
			init_neural_net(input_dim, output_dim, *actor_stdev_in_training_);
		}



		std::vector<float*> inputs;
		std::vector<float*> outputs;

		inputs.reserve(transition_data.size());
		outputs.reserve(transition_data.size());


		Eigen::VectorXf output_mean = Eigen::VectorXf::Zero(0);
		Eigen::VectorXf output_stdev = Eigen::VectorXf::Zero(0);
		Eigen::VectorXf training_output_stdev = output_stdev;


		auto form_training_data = [&](std::deque<std::shared_ptr<TeachingSample>>& data_set) {

			inputs.clear();
			outputs.clear();


			for (const std::shared_ptr<TeachingSample>& datum : data_set) {

				inputs.push_back(datum->state_.data());
				outputs.push_back(datum->control_state_.data());

			}

			training_output_stdev = output_stdev;

		};

		form_training_data(transition_data);


		output_mean = Eigen::VectorXf::Zero(output_dim);
		output_stdev = Eigen::VectorXf::Zero(output_dim);

		for (const float* datum : outputs) {

			Eigen::Map<const Eigen::VectorXf> out_map = Eigen::Map<const Eigen::VectorXf>(datum, output_dim);
			output_mean += out_map;

		}
		output_mean /= (float)outputs.size();

		for (const float* datum : outputs) {

			Eigen::Map<const Eigen::VectorXf> out_map = Eigen::Map<const Eigen::VectorXf>(datum, output_dim);
			output_stdev += (out_map - output_mean).cwiseAbs2();

		}
		output_stdev /= (float)outputs.size();
		output_stdev = output_stdev.cwiseSqrt();


		if (training_output_stdev.size() == 0) {
			training_output_stdev = output_stdev;
		}

		////////Train mean
		int max_epochs = 5;
		int epoch = 0;
		bool is_gradient = false;
		while (epoch < max_epochs) {
			actor_in_training_->train_adam((const float**)inputs.data(), (const float**)outputs.data(), inputs.size(), minibatch_size, is_gradient, training_output_stdev.data());
			epoch++;
		}

		///////Train stdev

		inputs.clear();
		outputs.clear();

		for (std::shared_ptr<TeachingSample> sample : transition_data) {

			sample->output_for_learner_ = sample->control_;
			actor_in_training_->run(sample->state_.data(), sample->output_for_learner_.data());

			sample->output_for_learner_ -= sample->control_;
			sample->output_for_learner_ = sample->output_for_learner_.cwiseAbs();

			inputs.push_back(sample->state_.data());
			outputs.push_back(sample->output_for_learner_.data());

		}


		epoch = 0;
		while (epoch < max_epochs) {
			actor_stdev_in_training_->train_adam((const float**)inputs.data(), (const float**)outputs.data(), inputs.size(), minibatch_size);
			epoch++;
		}



		///////Validation


		form_training_data(validation_data);


		float mse = std::numeric_limits<float>::infinity();
		if (actor_in_training_.get()) {
			mse = actor_in_training_->mse((const float**)inputs.data(), (const float**)outputs.data(), inputs.size());
		}

		if (mse - mse != mse - mse) {

			int input_size = actor_in_training_->input_operation_->size_;
			int output_size = actor_in_training_->output_operation_->size_;

			actor_in_training_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron);
			init_neural_net(input_size, output_size, *actor_in_training_);
		}

		float competing_mse = std::numeric_limits<float>::infinity();
		if (competing_actor_in_training_.get()) {
			competing_mse = competing_actor_in_training_->mse((const float**)inputs.data(), (const float**)outputs.data(), inputs.size());
		}

		std::cout << "Trained actor. MSE: " << mse << " Data: " << transition_data.size() << std::endl;

		//actor_copy_ = std::unique_ptr<std::vector<MultiLayerPerceptron>>(new std::vector<MultiLayerPerceptron>(nSamples, *actor_in_training_));
		actor_copy_.clear();
		for (int i = 0; i < maxSamples; i++) {
			std::unique_ptr<MultiLayerPerceptron> tmp = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron(*actor_in_training_));
			actor_copy_.push_back(std::move(tmp));
		}

		actor_stdev_copy_.clear();
		for (int i = 0; i < maxSamples; i++) {
			std::unique_ptr<MultiLayerPerceptron> tmp = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron(*actor_stdev_in_training_));
			actor_stdev_copy_.push_back(std::move(tmp));
		}


		if (competing_mse < mse) {
			actor_in_training_.swap(competing_actor_in_training_);
			competing_actor_in_training_.reset();
			std::cout << "Swapping better actor." << std::endl;
		}

	}

	void SmoothingControl::train_actor_using_dynamics_model()
	{


		std::deque<std::shared_ptr<TeachingSample>> transition_data;
		std::deque<std::shared_ptr<TeachingSample>> validation_data;
		{
			std::lock_guard<std::mutex> lock(copying_transition_data_);
			transition_data = transition_data_;
			validation_data = validation_data_;
		}
		std::deque<std::shared_ptr<TeachingSample>> dynamics_data;
		{
			std::lock_guard<std::mutex> lock(copying_dynamics_data_);
			dynamics_data = dynamics_data_;
		}

		std::lock_guard<std::mutex> dynamics_lock(copying_dynamics_model_for_actor_);

		if (!dynamics_model_for_actor_.get()) {
			return;
		}

		const unsigned minibatch_size = transition_data.size() / 10;
		if (minibatch_size == 0) {
			return;
		}


		const unsigned input_dim = nStateDimensions;
		unsigned output_dim = nControlDimensions;


		if (!actor_in_training_.get()) {
			actor_in_training_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());
			init_neural_net(input_dim, output_dim, *actor_in_training_);
		}

		std::vector<float*> inputs;
		std::vector<float*> outputs;

		inputs.reserve(transition_data.size());
		outputs.reserve(transition_data.size());



		auto form_training_data = [&](std::deque<std::shared_ptr<TeachingSample>>& data_set) {

			inputs.clear();
			outputs.clear();


			for (const std::shared_ptr<TeachingSample>& datum : data_set) {

				inputs.push_back(datum->state_.data());
				outputs.push_back(datum->output_for_learner_.data());

			}

		};

		auto form_validation_data = [&](std::deque<std::shared_ptr<TeachingSample>>& data_set) {

			inputs.clear();
			outputs.clear();

			for (const std::shared_ptr<TeachingSample>& datum : data_set) {

				inputs.push_back(datum->state_.data());
				outputs.push_back(datum->control_.data());

			}

		};


		for (std::shared_ptr<TeachingSample> sample : transition_data) {

			sample->output_for_learner_ = sample->control_;
			dynamics_model_for_actor_->run(sample->state_future_state_.data(), sample->output_for_learner_.data());

		}


		std::deque<std::shared_ptr<TeachingSample>> training_data = transition_data;

		Eigen::VectorXf state_future_state;

		for (std::shared_ptr<TeachingSample> teaching_sample : dynamics_data) {

			float min_dist = std::numeric_limits<float>::max();
			TeachingSample* nearest = nullptr;

			for (std::shared_ptr<TeachingSample> reference_sample : transition_data) {

				float dist = (reference_sample->future_state_ - teaching_sample->future_state_).squaredNorm();
				if (dist < min_dist) {
					min_dist = dist;
					nearest = reference_sample.get();
				}

			}

			if (nearest) {

				teaching_sample->output_for_learner_ = teaching_sample->control_;
				state_future_state = teaching_sample->state_future_state_;

				state_future_state.tail(nStateDimensions) = nearest->future_state_;

				dynamics_model_for_actor_->run(state_future_state.data(), teaching_sample->output_for_learner_.data());

				training_data.push_back(teaching_sample);

			}

		}

		form_training_data(training_data);


		int max_epochs = 5;
		int epoch = 0;
		bool is_gradient = false;
		while (epoch < max_epochs) {


			actor_in_training_->train_adam((const float**)inputs.data(), (const float**)outputs.data(), inputs.size(), minibatch_size, is_gradient);

			epoch++;
		}


		form_validation_data(validation_data);


		float mse = std::numeric_limits<float>::infinity();
		if (actor_in_training_.get()) {
			mse = actor_in_training_->mse((const float**)inputs.data(), (const float**)outputs.data(), inputs.size());
		}

		if (mse - mse != mse - mse) {

			int input_size = actor_in_training_->input_operation_->size_;
			int output_size = actor_in_training_->output_operation_->size_;

			actor_in_training_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron);
			init_neural_net(input_size, output_size, *actor_in_training_);
		}


		std::cout << "Trained actor. MSE: " << mse << " Data: " << transition_data.size() << std::endl;

		//actor_copy_ = std::unique_ptr<std::vector<MultiLayerPerceptron>>(new std::vector<MultiLayerPerceptron>(nSamples, *actor_in_training_));
		actor_copy_.clear();
		for (int i = 0; i < maxSamples; i++) {
			std::unique_ptr<MultiLayerPerceptron> tmp = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron(*actor_in_training_));
			actor_copy_.push_back(std::move(tmp));
		}


	}

	void SmoothingControl::train_autoencoder()
	{

		std::deque<std::shared_ptr<TeachingSample>> transition_data;
		std::deque<std::shared_ptr<TeachingSample>> validation_data;
		{
			std::lock_guard<std::mutex> lock(copying_transition_data_);
			transition_data = transition_data_;
			validation_data = validation_data_;
		}

		if (transition_data.size() < 2) {
			return;
		}

		const unsigned minibatch_size = transition_data.size() / 10;

		if (minibatch_size == 0) {
			return;
		}

		int data_dim = transition_data[0]->control_state_.size();


		if (!autoencoder_in_training_.get()) {
			autoencoder_in_training_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());
			init_neural_net(data_dim, data_dim, *autoencoder_in_training_);
		}


		std::vector<float*> inputs;
		inputs.reserve(transition_data.size());


		for (const std::shared_ptr<TeachingSample>& datum : transition_data) {
			inputs.push_back(datum->control_state_.data());
		}


		Eigen::VectorXf output_mean = Eigen::VectorXf::Zero(data_dim);
		Eigen::VectorXf output_stdev = Eigen::VectorXf::Zero(data_dim);

		for (const float* datum : inputs) {

			Eigen::Map<const Eigen::VectorXf> out_map = Eigen::Map<const Eigen::VectorXf>(datum, data_dim);
			output_mean += out_map;

		}
		output_mean /= (float)inputs.size();

		for (const float* datum : inputs) {

			Eigen::Map<const Eigen::VectorXf> out_map = Eigen::Map<const Eigen::VectorXf>(datum, data_dim);
			output_stdev += (out_map - output_mean).cwiseAbs2();

		}
		output_stdev /= (float)inputs.size();
		output_stdev = output_stdev.cwiseSqrt();


		bool is_gradient = false;
		int max_epochs = 3;
		int epoch = 0;
		while (epoch < max_epochs) {

			//actor_2_->train_rprop((const float**)inputs.data(), (const float**)outputs.data(), inputs.size());
			autoencoder_in_training_->train_adam((const float**)inputs.data(), (const float**)inputs.data(), inputs.size(), minibatch_size, is_gradient, output_stdev.data());
			//actor_in_training_->train_adamax((const float**)inputs.data(), (const float**)outputs.data(), inputs.size(), minibatch_size);


			epoch++;
		}

		inputs.clear();
		for (const std::shared_ptr<TeachingSample>& datum : transition_data) {
			inputs.push_back(datum->control_state_.data());
		}

		float mse = autoencoder_in_training_->mse((const float**)inputs.data(), (const float**)inputs.data(), inputs.size());

		if (mse - mse != mse - mse) {
			init_neural_net(data_dim, data_dim, *autoencoder_in_training_);
		}

		std::cout << "Trained autoencoder. MSE: " << mse << " Data: " << transition_data.size() << std::endl;


		autoencoder_copy_.clear();
		for (int i = 0; i < maxSamples; i++) {
			std::unique_ptr<MultiLayerPerceptron> tmp = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron(*autoencoder_in_training_));
			autoencoder_copy_.push_back(std::move(tmp));
		}


	}


	void classify(MultiLayerPerceptron& net, std::deque<std::shared_ptr<SmoothingControl::TeachingSample>> data, int& positives, int& negatives, float& log_prob) {

		positives = 0;
		negatives = 0;
		log_prob = 0.0f;

		for (std::shared_ptr<SmoothingControl::TeachingSample>& datum : data) {
			float result = 0.0f;

			net.run(datum->future_state_.data(), &result);

			log_prob += std::log(std::max(result, std::numeric_limits<float>::epsilon()));

			if (result > 0.5f) {
				positives++;
			}
			else {
				negatives++;
			}

		}

	}

	void SmoothingControl::train_discriminator()
	{

		if (!discriminator_in_training_.get()) {

			const TeachingSample* training_sample = nullptr;

			if (discriminator_data_bad_.size() > 0) {
				training_sample = discriminator_data_bad_.front().get();
			}

			if (discriminator_data_good_.size() > 0) {
				training_sample = discriminator_data_good_.front().get();
			}

			if (!training_sample) {
				return;
			}

			discriminator_in_training_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());

			unsigned input_dim = training_sample->future_state_.size();
			unsigned output_dim = 1;
			bool sigmoid = true;

			init_neural_net(input_dim, output_dim, *discriminator_in_training_, sigmoid);

		}

#if 0
		int epochs = 10;
		int batch_size = discriminator_data_bad_.size() + discriminator_data_good_.size();
		batch_size /= 10;

		std::vector<float*> input_ptrs(batch_size, nullptr);
		std::vector<float*> output_ptrs(batch_size, nullptr);
		std::vector<float> outputs(batch_size, 0.0f);

		const bool is_gradient = true;

		for (int i = 0; i < epochs; i++) {

			if (rand() % 2 == 0) {
				if (discriminator_data_good_.size() == 0) {
					continue;
				}
				for (int datum = 0; datum < batch_size; datum++) {
					int rand_idx = rand() % discriminator_data_good_.size();
					input_ptrs[datum] = discriminator_data_good_[rand_idx]->future_state_.data();

					discriminator_in_training_->run(input_ptrs[datum], &outputs[datum]);
					float& prop = outputs[datum];

					prop = std::max(prop, std::numeric_limits<float>::epsilon());
					//Negative derivative of log(prop)
					prop = -1.0f / prop;

					output_ptrs[datum] = &outputs[datum];
				}
			}
			else {
				if (discriminator_data_bad_.size() == 0) {
					continue;
				}
				for (int datum = 0; datum < batch_size; datum++) {
					int rand_idx = rand() % discriminator_data_bad_.size();

					input_ptrs[datum] = discriminator_data_bad_[rand_idx]->future_state_.data();
					discriminator_in_training_->run(input_ptrs[datum], &outputs[datum]);
					float& prop = outputs[datum];

					prop = std::max(prop, std::numeric_limits<float>::epsilon());
					//Negative derivative of -log(prop)
					prop = 1.0f / prop;

					output_ptrs[datum] = &outputs[datum];
				}
			}

			int minibatch_size = input_ptrs.size();

			//actor_2_->train_rprop((const float**)inputs.data(), (const float**)outputs.data(), inputs.size());
			discriminator_in_training_->train_adam((const float**)input_ptrs.data(), (const float**)output_ptrs.data(), input_ptrs.size(), minibatch_size, is_gradient);
			//actor_in_training_->train_adamax((const float**)inputs.data(), (const float**)outputs.data(), inputs.size(), minibatch_size);
		}
#endif

#if 1

		int batch_size = discriminator_data_bad_.size() + discriminator_data_good_.size();
		int minibatch_size = batch_size / 10;
		if (minibatch_size == 0) {
			return;
		}

		std::vector<float*> input_ptrs(minibatch_size, nullptr);
		std::vector<float*> output_ptrs(minibatch_size, nullptr);

		int amount_good = discriminator_data_good_.size();
		int amount_bad = discriminator_data_bad_.size();

		if (!discriminator_competing_.get()) {
			discriminator_competing_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());
			const bool sigmoid = true;
			init_neural_net(discriminator_in_training_->input_operation_->size_, discriminator_in_training_->output_operation_->size_, *discriminator_competing_, sigmoid);
		}

		int epochs = 10;
		for (int epoch = 0; epoch < epochs; epoch++) {

			float is_true = 1.0f;
			float is_false = 0.0f;

			auto shuffle_data = [](std::deque<std::shared_ptr<TeachingSample>>& data) {
				unsigned amount = data.size();
				if (amount <= 1) {
					return;
				}

				for (unsigned j = 0; j < amount; j++) {
					unsigned rand_idx = rand() % amount;
					data[j].swap(data[rand_idx]);
				}

			};

			std::deque<std::shared_ptr<TeachingSample>> tmp_data_bad = discriminator_data_bad_;
			std::deque<std::shared_ptr<TeachingSample>> tmp_data_good = discriminator_data_good_;

			shuffle_data(tmp_data_bad);
			shuffle_data(tmp_data_good);

			do {
				if (rand() % 2 == 0) {

					int this_batch_size = std::min(tmp_data_good.size(), (unsigned)minibatch_size);

					if (this_batch_size == 0) {
						continue;
					}

					input_ptrs.resize(this_batch_size);
					output_ptrs.resize(this_batch_size);

					for (int idx = 0; idx < this_batch_size; idx++) {
						input_ptrs[idx] = tmp_data_good.front()->future_state_.data();
						tmp_data_good.pop_front();
						output_ptrs[idx] = &is_true;
					}

					discriminator_in_training_->train_adamax((const float**)input_ptrs.data(), (const float**)output_ptrs.data(), input_ptrs.size(), input_ptrs.size());
					if (discriminator_competing_.get()) {
						discriminator_competing_->train_adamax((const float**)input_ptrs.data(), (const float**)output_ptrs.data(), input_ptrs.size(), input_ptrs.size());
					}
				}
				else {
					int this_batch_size = std::min(tmp_data_bad.size(), (unsigned)minibatch_size);

					if (this_batch_size == 0) {
						continue;
					}

					input_ptrs.resize(this_batch_size);
					output_ptrs.resize(this_batch_size);

					for (int idx = 0; idx < this_batch_size; idx++) {
						input_ptrs[idx] = tmp_data_bad.front()->future_state_.data();
						output_ptrs[idx] = &is_false;
						tmp_data_bad.pop_front();
					}

					discriminator_in_training_->train_adamax((const float**)input_ptrs.data(), (const float**)output_ptrs.data(), input_ptrs.size(), input_ptrs.size());
					if (discriminator_competing_.get()) {
						discriminator_competing_->train_adamax((const float**)input_ptrs.data(), (const float**)output_ptrs.data(), input_ptrs.size(), input_ptrs.size());
					}
				}
			} while (tmp_data_bad.size() > 0 && tmp_data_good.size() > 0);

		}
#endif

		int positives = 0;
		int false_positives = 0;
		int negatives = 0;
		int false_negatives = 0;
		float true_prob = 0.0f;
		float false_prob = 0.0f;

		classify(*discriminator_in_training_, discriminator_data_good_, positives, false_negatives, true_prob);
		classify(*discriminator_in_training_, discriminator_data_bad_, false_positives, negatives, false_prob);

		std::cout << "Discriminator: pos " << positives << " false neg " << false_negatives << " neg " << negatives << " false pos " << false_positives << std::endl;
		std::cout << "True prob " << true_prob << " false prob " << false_prob << std::endl;

		int correct = positives + negatives;
		int not_correct = false_positives + false_negatives;
		float orig_accuracy = (float)correct / (float)(correct + not_correct);

		if (discriminator_competing_.get()) {
			classify(*discriminator_competing_, discriminator_data_good_, positives, false_negatives, true_prob);
			classify(*discriminator_competing_, discriminator_data_bad_, false_positives, negatives, false_prob);

			correct = positives + negatives;
			not_correct = false_positives + false_negatives;
			float competing_accuracy = (float)correct / (float)(correct + not_correct);

			if (competing_accuracy > orig_accuracy) {
				discriminator_competing_.swap(discriminator_in_training_);
				const bool sigmoid = true;
				init_neural_net(discriminator_in_training_->input_operation_->size_, discriminator_in_training_->output_operation_->size_, *discriminator_competing_, sigmoid);
				std::cout << "Competing discriminator was better." << std::endl;
			}

		}


		//const float false_positive_threshold = 0.8f;
		//if (false_positive_threshold > 0.0f) {
		//	for (int i = discriminator_data_bad_.size()-1; i >= 0; i--) {
		//		TeachingSample* datum = discriminator_data_bad_[i].get();
		//		float result = 0.0f;

		//		discriminator_in_training_->run(datum->state_future_state_.data(), &result);

		//		if (result > false_positive_threshold) {
		//			discriminator_data_bad_.erase(discriminator_data_bad_.begin()+i);
		//		}

		//	}
		//}

		if (!discriminator_copy_.get()) {
			discriminator_copy_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());
		}

		*discriminator_copy_ = *discriminator_in_training_;
	}


	int __stdcall SmoothingControl::getBestSampleLastIdx()
	{
		return bestFullSampleIdx;
	}


	void __stdcall SmoothingControl::updateResults(int sampleIdx, const float* starting_state, const float *used_control, const float *end_state, double stateCost, double controlCost)
	{
		MarginalSample &nextSample = marginals[nextStep][sampleIdx];

		if (starting_state) {
			Eigen::Map<const Eigen::VectorXf> state_map(starting_state, nStateDimensions);
			float discrepancy = (state_map - nextSample.previousState).cwiseAbs().sum();
			assert(discrepancy <= 0.0f);
			nextSample.previousState = state_map;
		}

		Eigen::Map<const Eigen::VectorXf> new_state_map(end_state, nStateDimensions);
		Eigen::Map<const Eigen::VectorXf> control_map(used_control, nControlDimensions);

		nextSample.control = control_map;
		nextSample.state = new_state_map;

		nextSample.stateCost = stateCost;
		nextSample.originalStateCostFromClient = stateCost;
		nextSample.controlCost = controlCost;

		MarginalSample &previousSample = marginals[currentStep][nextSample.previousMarginalSampleIdx];
		nextSample.fullCost = previousSample.fullCost + nextSample.stateCost + nextSample.controlCost;//+nextSample.controlCost+controlContinuityCost;


	}


	void __stdcall SmoothingControl::endPlanningStep(int stepIdx)
	{

		/*if (determinism_debug) {

			int next_Step = stepIdx + 1;
			for (unsigned i = 0; i < marginals[next_Step].size(); i++) {
				for (unsigned j = 0; j < marginals[next_Step].size(); j++) {


					float discrepancy = 0.0f;

					if ((marginals[next_Step][i].previousState - marginals[next_Step][j].previousState).squaredNorm() <= 0.0f) {

						if ((marginals[next_Step][i].control - marginals[next_Step][j].control).squaredNorm() <= 0.0f) {

							discrepancy = (marginals[next_Step][i].state - marginals[next_Step][j].state).squaredNorm();

						}

					}

					AALTO_ASSERT1(discrepancy <= std::numeric_limits<float>::epsilon());

				}
			}
		}*/

	}


	std::deque<std::string> SmoothingControl::get_settings()
	{
		std::deque<std::string> settings;

		settings.push_back("Number of trajectories: " + std::to_string(nSamples));
		settings.push_back("Number of time steps: " + std::to_string(nSteps));
		settings.push_back("Learning: " + std::to_string(learning_));

		settings.push_back("Number nearest neighbor trajectories: " + std::to_string(nn_trajectories_));
		settings.push_back("Number machine learning trajectories: " + std::to_string(machine_learning_samples_));
		settings.push_back("Number of noisy machine learning trajectories: " + std::to_string(noisy_machine_learning_samples_));

		settings.push_back("Use machine learning: " + std::to_string(use_machine_learning_));
		//		settings.push_back("Use autoencoder: " + std::to_string(use_autoencoder_));
		//		settings.push_back("Use critic: " + std::to_string(q_learning_));
		//		settings.push_back("Use use discriminator: " + std::to_string(use_discriminator_));

		if (state_groupings_.size() > 0 || control_groupings_.size() > 0) {
			settings.push_back("Neural network using grouping.");
		}
		else {
			settings.push_back("No grouping in neural network.");
		}


		settings.push_back("Regularization noise: " + std::to_string(regularization_noise_));
		settings.push_back("Validation fraction: " + std::to_string(validation_fraction_));
		settings.push_back("Learning budget: " + std::to_string(learning_budget_));

		settings.push_back("Amount recently used linear search samples: " + std::to_string(amount_recent_));
		settings.push_back("Use nearest neighbor forests: " + std::to_string(use_forests_));
		settings.push_back("Amount of data in nearest neighbor trees: " + std::to_string(amount_data_in_tree_));
		settings.push_back("Number of trees in forest: " + std::to_string(ann_forest_.forest_.size()));

		settings.push_back("Stored previous frame sample percentage: " + std::to_string(storedSamplesPercentage));
		settings.push_back("Free particle amount: " + std::to_string(noPriorTrajectoryPortion));
		settings.push_back("Resample threshold: " + std::to_string(resampleThreshold));

		if (control_scheme_ == ControlScheme::L1) {
			settings.push_back("Control Scheme: L1");
		}

		if (control_scheme_ == ControlScheme::KL) {
			settings.push_back("Control Scheme: KL");
		}

		return settings;
	}





	void SmoothingControl::train_critic()
	{
		std::deque<std::shared_ptr<TeachingSample>> critic_data;
		{
			std::lock_guard<std::mutex> lock(copying_dynamics_data_);
			critic_data = critic_data_;
		}

		if (critic_data.size() < 200) {
			return;
		}


		if (!critic_in_training_.get()) {
			critic_in_training_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());

			unsigned input_dim = critic_data.front()->control_state_.size();
			unsigned output_dim = 1;

			init_neural_net(input_dim, output_dim, *critic_in_training_);

		}


		bool target_exists = false;
		if (critic_target_.get()) {
			target_exists = true;
		}

		{
			std::lock_guard<std::mutex> lock(using_actor_in_training_);
			if (!actor_for_critic_.get()) {

				unsigned input_dim = critic_data.front()->state_.size();
				unsigned output_dim = critic_data.front()->control_.size();

				actor_for_critic_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());
				init_neural_net(input_dim, output_dim, *actor_for_critic_);
				actor_for_critic_->learning_rate_ = 0.0001f;
			}
		}

		std::vector<float*> input_ptrs;
		input_ptrs.reserve(critic_data.size());
		std::vector<float*> output_ptrs;
		output_ptrs.reserve(critic_data.size());

		std::vector<float> outputs(critic_data.size(), 0.0f);

		Eigen::VectorXf control = Eigen::VectorXf::Zero(nControlDimensions);
		Eigen::VectorXf control_state = Eigen::VectorXf::Zero(nControlDimensions + nStateDimensions);

		int minibatch_size = critic_data.size() / 10;
		minibatch_size = std::min(minibatch_size, 100);
		if (minibatch_size <= 0) {
			return;
		}

		const float discount_factor = 0.9f;

		const float cost_upper_bound = 10.0f;
		const float cost_lower_bound = 0.0f;

		const float value_upper_bound = 25.0f;
		const float value_lower_bound = 0.0f;

		int epochs = 10;
		for (int epoch = 0; epoch < epochs; epoch++) {
			if (since_critic_freeze_ > critic_freeze_interval_) {
				if (!critic_target_.get()) {
					critic_target_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());
				}

				*critic_target_ = *critic_in_training_;


				since_critic_freeze_ = 0;
				std::cout << "Critic freeze" << std::endl;
			}




			input_ptrs.clear();
			output_ptrs.clear();

			for (unsigned i = 0; i < critic_data.size(); i++) {
				TeachingSample* datum = critic_data[i].get();
				if (target_exists) {
					std::lock_guard<std::mutex> lock(using_actor_in_training_);
					actor_for_critic_->run(datum->future_state_.data(), control.data());
					control_state.tail(nStateDimensions) = datum->future_state_;
					control_state.head(nControlDimensions) = control;

					critic_target_->run(control_state.data(), &outputs[i]);
					outputs[i] *= discount_factor;

					float instantaneous_cost = datum->instantaneous_cost_;

					if (instantaneous_cost < cost_lower_bound) {
						instantaneous_cost = cost_lower_bound;
					}

					if (instantaneous_cost > cost_upper_bound) {
						instantaneous_cost = cost_upper_bound;
					}

					outputs[i] += instantaneous_cost;
				}
				else {
					float instantaneous_cost = datum->instantaneous_cost_;

					if (instantaneous_cost < cost_lower_bound) {
						instantaneous_cost = cost_lower_bound;
					}

					if (instantaneous_cost > cost_upper_bound) {
						instantaneous_cost = cost_upper_bound;
					}

					outputs[i] = instantaneous_cost;
				}

				float& value = outputs[i];

				if (value < value_lower_bound) {
					value = value_lower_bound;
				}

				if (value > value_upper_bound) {
					value = value_upper_bound;
				}

				input_ptrs.push_back(datum->control_state_.data());
				output_ptrs.push_back(&outputs[i]);
			}
			critic_in_training_->train_adam((const float**)input_ptrs.data(), (const float**)output_ptrs.data(), input_ptrs.size(), minibatch_size);

			since_critic_freeze_++;
		}

		critic_in_training_->mse_ = critic_in_training_->mse((const float**)input_ptrs.data(), (const float**)output_ptrs.data(), input_ptrs.size());

		std::cout << "Critic MSE: " << critic_in_training_->mse_ << std::endl;



		int actor_epochs = 2;
		const float delta = 1.0f;
		const bool is_gradient = true;
		for (int epoch = 0; epoch < actor_epochs; epoch++) {

			std::deque<std::shared_ptr<TeachingSample>> training_data = critic_data;


			//Shuffle
			for (unsigned int i = 0; i < training_data.size(); i++) {
				int rand_idx = rand() % training_data.size();
				training_data[i].swap(training_data[rand_idx]);
			}

			while (training_data.size() > 0) {

				input_ptrs.clear();
				output_ptrs.clear();

				for (int i = 0; i < minibatch_size; i++) {

					if (training_data.size() == 0) {
						break;
					}

					std::shared_ptr<TeachingSample> sample = training_data.back();
					training_data.pop_back();

					sample->output_for_learner_ = sample->control_state_;
					actor_for_critic_->run(sample->state_.data(), sample->output_for_learner_.data());

					MultiLayerPerceptron* critic_to_use = nullptr;

					critic_to_use = critic_in_training_.get();


					critic_to_use->backpropagate_deltas(sample->output_for_learner_.data(), &delta, is_gradient);
					sample->output_for_learner_ = critic_to_use->input_operation_->deltas_.array() / critic_to_use->input_stdev_.array();

					input_ptrs.push_back(sample->state_.data());
					output_ptrs.push_back(sample->output_for_learner_.data());

				}

				actor_for_critic_->train_adamax((const float**)input_ptrs.data(), (const float**)output_ptrs.data(), input_ptrs.size(), minibatch_size, is_gradient);

			}

		}

		rl_actor_copy_.clear();
		for (int i = 0; i < maxSamples; i++) {
			std::unique_ptr<MultiLayerPerceptron> tmp = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron(*actor_for_critic_));
			rl_actor_copy_.push_back(std::move(tmp));
		}


		//if (!critic_copy_.get()) {
		//	critic_copy_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());
		//}
		//*critic_copy_ = *critic_in_training_;

	}

	void SmoothingControl::train_critic_greedy()
	{
		std::deque<std::shared_ptr<TeachingSample>> critic_data;
		{
			std::lock_guard<std::mutex> lock(copying_dynamics_data_);
			critic_data = critic_data_;
		}

		unsigned start_training_at = 1000;
		if (critic_data.size() < start_training_at) {
			return;
		}


		if (!critic_in_training_.get()) {
			critic_in_training_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());

			unsigned input_dim = critic_data.front()->control_state_.size();
			unsigned output_dim = 1;

			init_neural_net(input_dim, output_dim, *critic_in_training_);

		}

		{
			std::lock_guard<std::mutex> lock(using_actor_in_training_);
			if (!actor_for_critic_.get()) {

				unsigned input_dim = critic_data.front()->state_.size();
				unsigned output_dim = critic_data.front()->control_.size();

				actor_for_critic_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());
				init_neural_net(input_dim, output_dim, *actor_for_critic_);
				actor_for_critic_->learning_rate_ = 0.0001f;
			}
		}

		std::vector<float*> input_ptrs;
		input_ptrs.reserve(critic_data.size());
		std::vector<float*> output_ptrs;
		output_ptrs.reserve(critic_data.size());

		std::vector<float> outputs(critic_data.size(), 0.0f);

		int minibatch_size = critic_data.size() / 10;
		minibatch_size = std::min(minibatch_size, 100);
		if (minibatch_size <= 0) {
			return;
		}


		const float cost_upper_bound = 100.0f;
		const float cost_lower_bound = 0.0f;


		int epochs = 10;
		for (int epoch = 0; epoch < epochs; epoch++) {

			input_ptrs.clear();
			output_ptrs.clear();

			for (unsigned i = 0; i < critic_data.size(); i++) {
				TeachingSample* datum = critic_data[i].get();

				float instantaneous_cost = datum->instantaneous_cost_;

				if (instantaneous_cost < cost_lower_bound) {
					instantaneous_cost = cost_lower_bound;
				}

				if (instantaneous_cost > cost_upper_bound) {
					instantaneous_cost = cost_upper_bound;
				}

				outputs[i] = instantaneous_cost;

				input_ptrs.push_back(datum->control_state_.data());
				output_ptrs.push_back(&outputs[i]);
			}
			critic_in_training_->train_adam((const float**)input_ptrs.data(), (const float**)output_ptrs.data(), input_ptrs.size(), minibatch_size);
		}

		critic_in_training_->mse_ = critic_in_training_->mse((const float**)input_ptrs.data(), (const float**)output_ptrs.data(), input_ptrs.size());

		std::cout << "Critic MSE: " << critic_in_training_->mse_ << std::endl;

		int actor_epochs = 2;
		const float delta = 1.0f;
		const bool is_gradient = true;
		for (int epoch = 0; epoch < actor_epochs; epoch++) {

			std::deque<std::shared_ptr<TeachingSample>> training_data;
			training_data = critic_data;

			//Shuffle
			for (unsigned int i = 0; i < training_data.size(); i++) {
				int rand_idx = rand() % training_data.size();
				training_data[i].swap(training_data[rand_idx]);
			}

			while (training_data.size() > 0) {

				input_ptrs.clear();
				output_ptrs.clear();

				for (int i = 0; i < minibatch_size; i++) {

					if (training_data.size() == 0) {
						break;
					}

					std::shared_ptr<TeachingSample> sample = training_data.back();
					training_data.pop_back();

					sample->output_for_learner_ = sample->control_state_;
					actor_for_critic_->run(sample->state_.data(), sample->output_for_learner_.data());

					MultiLayerPerceptron* critic_to_use = critic_in_training_.get();


					critic_to_use->backpropagate_deltas(sample->output_for_learner_.data(), &delta, is_gradient);
					sample->output_for_learner_ = critic_to_use->input_operation_->deltas_.array() / critic_to_use->input_stdev_.array();

					input_ptrs.push_back(sample->state_.data());
					output_ptrs.push_back(sample->output_for_learner_.data());

				}

				actor_for_critic_->train_adamax((const float**)input_ptrs.data(), (const float**)output_ptrs.data(), input_ptrs.size(), minibatch_size, is_gradient);

			}

		}

		rl_actor_copy_.clear();
		for (int i = 0; i < maxSamples; i++) {
			std::unique_ptr<MultiLayerPerceptron> tmp = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron(*actor_for_critic_));
			rl_actor_copy_.push_back(std::move(tmp));
		}
	}

	void SmoothingControl::train_critic_cacla()
	{


		std::deque<std::shared_ptr<TeachingSample>> critic_data;
		{
			std::lock_guard<std::mutex> lock(copying_dynamics_data_);
			critic_data = critic_data_;
		}

		if (critic_data.size() < 200) {
			return;
		}


		if (!critic_in_training_.get()) {
			critic_in_training_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());

			unsigned input_dim = critic_data.front()->control_state_.size();
			unsigned output_dim = 1;

			init_neural_net(input_dim, output_dim, *critic_in_training_);

		}


		bool target_exists = false;
		if (critic_target_.get()) {
			target_exists = true;
		}

		{
			std::lock_guard<std::mutex> lock(using_actor_in_training_);
			if (!actor_for_critic_.get()) {

				unsigned input_dim = critic_data.front()->state_.size();
				unsigned output_dim = critic_data.front()->control_.size();

				actor_for_critic_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());
				init_neural_net(input_dim, output_dim, *actor_for_critic_);
				actor_for_critic_->learning_rate_ = 0.0001f;
			}
		}

		std::vector<float*> input_ptrs;
		input_ptrs.reserve(critic_data.size());
		std::vector<float*> output_ptrs;
		output_ptrs.reserve(critic_data.size());

		std::vector<float> outputs(critic_data.size(), 0.0f);

		Eigen::VectorXf control = Eigen::VectorXf::Zero(nControlDimensions);
		Eigen::VectorXf control_state = Eigen::VectorXf::Zero(nControlDimensions + nStateDimensions);

		int minibatch_size = critic_data.size() / 10;
		minibatch_size = std::min(minibatch_size, 100);
		if (minibatch_size <= 0) {
			return;
		}

		const float discount_factor = 0.9f;

		const float cost_upper_bound = 200.0f;
		const float cost_lower_bound = 0.0f;

		const float value_upper_bound = cost_upper_bound / (1.0f - discount_factor);
		const float value_lower_bound = cost_lower_bound / (1.0f - discount_factor);

		int epochs = 10;
		for (int epoch = 0; epoch < epochs; epoch++) {
			if (since_critic_freeze_ > critic_freeze_interval_) {
				if (!critic_target_.get()) {
					critic_target_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());
				}

				*critic_target_ = *critic_in_training_;


				since_critic_freeze_ = 0;
				std::cout << "Critic freeze" << std::endl;
			}




			input_ptrs.clear();
			output_ptrs.clear();

			for (unsigned i = 0; i < critic_data.size(); i++) {
				TeachingSample* datum = critic_data[i].get();
				if (target_exists) {
					std::lock_guard<std::mutex> lock(using_actor_in_training_);
					actor_for_critic_->run(datum->future_state_.data(), control.data());
					control_state.tail(nStateDimensions) = datum->future_state_;
					control_state.head(nControlDimensions) = control;

					critic_target_->run(control_state.data(), &outputs[i]);
					outputs[i] *= discount_factor;

					float instantaneous_cost = datum->instantaneous_cost_;

					if (instantaneous_cost < cost_lower_bound) {
						instantaneous_cost = cost_lower_bound;
					}

					if (instantaneous_cost > cost_upper_bound) {
						instantaneous_cost = cost_upper_bound;
					}

					outputs[i] += instantaneous_cost;
				}
				else {
					float instantaneous_cost = datum->instantaneous_cost_;

					if (instantaneous_cost < cost_lower_bound) {
						instantaneous_cost = cost_lower_bound;
					}

					if (instantaneous_cost > cost_upper_bound) {
						instantaneous_cost = cost_upper_bound;
					}

					outputs[i] = instantaneous_cost;
				}

				float& value = outputs[i];

				if (value < value_lower_bound) {
					value = value_lower_bound;
				}

				if (value > value_upper_bound) {
					value = value_upper_bound;
				}

				input_ptrs.push_back(datum->control_state_.data());
				output_ptrs.push_back(&outputs[i]);
			}
			critic_in_training_->train_adam((const float**)input_ptrs.data(), (const float**)output_ptrs.data(), input_ptrs.size(), minibatch_size);

			since_critic_freeze_++;
		}

		critic_in_training_->mse_ = critic_in_training_->mse((const float**)input_ptrs.data(), (const float**)output_ptrs.data(), input_ptrs.size());

		std::cout << "Critic MSE: " << critic_in_training_->mse_ << std::endl;


		int actor_epochs = 2;
		const bool is_gradient = false;
		for (int epoch = 0; epoch < actor_epochs; epoch++) {

			std::deque<std::shared_ptr<TeachingSample>> training_data;
			Eigen::VectorXf control_state;

			for (std::shared_ptr<TeachingSample> sample : critic_data) {


				float sample_cost = sample->instantaneous_cost_;
				float future_cost = 0.0f;

				if (critic_target_.get()) {
					control_state = sample->control_state_;
					control_state.tail(nStateDimensions) = sample->future_state_;

					actor_for_critic_->run(sample->future_state_.data(), control_state.data());
					critic_in_training_->run(control_state.data(), &future_cost);

					sample_cost += discount_factor * future_cost;
				}


				float predicted_cost = 0.0f;
				control_state = sample->control_state_;
				actor_for_critic_->run(sample->state_.data(), control_state.data());
				critic_in_training_->run(control_state.data(), &predicted_cost);

				if (sample_cost < predicted_cost) {
					training_data.push_back(sample);
				}

			}

			//Shuffle
			for (unsigned int i = 0; i < training_data.size(); i++) {
				int rand_idx = rand() % training_data.size();
				training_data[i].swap(training_data[rand_idx]);
			}



			input_ptrs.clear();
			output_ptrs.clear();

			for (std::shared_ptr<TeachingSample> sample : training_data) {
				input_ptrs.push_back(sample->state_.data());
				output_ptrs.push_back(sample->control_.data());
			}


			actor_for_critic_->train_adam((const float**)input_ptrs.data(), (const float**)output_ptrs.data(), input_ptrs.size(), minibatch_size, is_gradient);



		}

		rl_actor_copy_.clear();
		for (int i = 0; i < maxSamples; i++) {
			std::unique_ptr<MultiLayerPerceptron> tmp = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron(*actor_for_critic_));
			rl_actor_copy_.push_back(std::move(tmp));
		}

	}



	void SmoothingControl::train_critic_greedy_cacla()
	{
		std::deque<std::shared_ptr<TeachingSample>> critic_data;
		{
			std::lock_guard<std::mutex> lock(copying_dynamics_data_);
			critic_data = critic_data_;
		}

		unsigned start_training_at = 1000;
		if (critic_data.size() < start_training_at) {
			return;
		}


		if (!critic_in_training_.get()) {
			critic_in_training_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());

			unsigned input_dim = critic_data.front()->control_state_.size();
			unsigned output_dim = 1;

			init_neural_net(input_dim, output_dim, *critic_in_training_);

		}

		{
			std::lock_guard<std::mutex> lock(using_actor_in_training_);
			if (!actor_for_critic_.get()) {

				unsigned input_dim = critic_data.front()->state_.size();
				unsigned output_dim = critic_data.front()->control_.size();

				actor_for_critic_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());
				init_neural_net(input_dim, output_dim, *actor_for_critic_);
				actor_for_critic_->learning_rate_ = 0.0001f;
			}
		}

		std::vector<float*> input_ptrs;
		input_ptrs.reserve(critic_data.size());
		std::vector<float*> output_ptrs;
		output_ptrs.reserve(critic_data.size());

		std::vector<float> outputs(critic_data.size(), 0.0f);

		int minibatch_size = critic_data.size() / 10;
		minibatch_size = std::min(minibatch_size, 100);
		if (minibatch_size <= 0) {
			return;
		}


		const float cost_upper_bound = 100.0f;
		const float cost_lower_bound = 0.0f;


		int epochs = 10;
		for (int epoch = 0; epoch < epochs; epoch++) {

			input_ptrs.clear();
			output_ptrs.clear();

			for (unsigned i = 0; i < critic_data.size(); i++) {
				TeachingSample* datum = critic_data[i].get();

				float instantaneous_cost = datum->instantaneous_cost_;

				if (instantaneous_cost < cost_lower_bound) {
					instantaneous_cost = cost_lower_bound;
				}

				if (instantaneous_cost > cost_upper_bound) {
					instantaneous_cost = cost_upper_bound;
				}

				outputs[i] = instantaneous_cost;

				input_ptrs.push_back(datum->control_state_.data());
				output_ptrs.push_back(&outputs[i]);
			}
			critic_in_training_->train_adam((const float**)input_ptrs.data(), (const float**)output_ptrs.data(), input_ptrs.size(), minibatch_size);
		}

		critic_in_training_->mse_ = critic_in_training_->mse((const float**)input_ptrs.data(), (const float**)output_ptrs.data(), input_ptrs.size());

		std::cout << "Critic MSE: " << critic_in_training_->mse_ << std::endl;

		int actor_epochs = 2;
		const bool is_gradient = false;
		for (int epoch = 0; epoch < actor_epochs; epoch++) {

			std::deque<std::shared_ptr<TeachingSample>> training_data;
			Eigen::VectorXf control_state;

			for (std::shared_ptr<TeachingSample> sample : critic_data) {


				float sample_cost = sample->instantaneous_cost_;

				float predicted_cost = 0.0f;
				control_state = sample->control_state_;
				actor_for_critic_->run(sample->state_.data(), control_state.data());
				critic_in_training_->run(control_state.data(), &predicted_cost);

				if (sample_cost <= predicted_cost) {
					training_data.push_back(sample);
				}

			}

			//Shuffle
			for (unsigned int i = 0; i < training_data.size(); i++) {
				int rand_idx = rand() % training_data.size();
				training_data[i].swap(training_data[rand_idx]);
			}



			input_ptrs.clear();
			output_ptrs.clear();

			for (std::shared_ptr<TeachingSample> sample : training_data) {
				input_ptrs.push_back(sample->state_.data());
				output_ptrs.push_back(sample->control_.data());
			}


			actor_for_critic_->train_adam((const float**)input_ptrs.data(), (const float**)output_ptrs.data(), input_ptrs.size(), minibatch_size, is_gradient);



		}

		rl_actor_copy_.clear();
		for (int i = 0; i < maxSamples; i++) {
			std::unique_ptr<MultiLayerPerceptron> tmp = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron(*actor_for_critic_));
			rl_actor_copy_.push_back(std::move(tmp));
		}
	}


	void SmoothingControl::train_dynamics_model()
	{


		std::deque<std::shared_ptr<TeachingSample>> dynamics_data;
		{
			std::lock_guard<std::mutex> lock(copying_dynamics_data_);
			dynamics_data = dynamics_data_;
		}

		const unsigned minibatch_size = dynamics_data.size() / 10;
		if (minibatch_size == 0) {
			return;
		}

		int start_training_at = 200;
		if (dynamics_data.size() < (unsigned int)start_training_at) {
			return;
		}


		std::deque<std::shared_ptr<TeachingSample>> validation_data;
		{
			std::lock_guard<std::mutex> lock(copying_transition_data_);
			validation_data = validation_data_;
		}




		const unsigned input_dim = 2 * nStateDimensions;
		const unsigned output_dim = nControlDimensions;

		if (!dynamics_model_in_training_.get()) {
			dynamics_model_in_training_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());
			init_neural_net(input_dim, output_dim, *dynamics_model_in_training_);
		}

		std::vector<float*> inputs;
		std::vector<float*> outputs;

		auto form_data_set = [&]() {
			inputs.clear();
			outputs.clear();

			inputs.reserve(dynamics_data.size());
			outputs.reserve(dynamics_data.size());

			if (dynamics_data.size() == 0) {
				return;
			}

			for (const std::shared_ptr<TeachingSample>& datum : dynamics_data) {
				inputs.push_back(datum->state_future_state_.data());
				outputs.push_back(datum->control_.data());
			}
		};

		form_data_set();

		const bool use_output_scaling = false;
		if (use_output_scaling)
		{
			Eigen::VectorXf output_mean = Eigen::VectorXf::Zero(output_dim);
			Eigen::VectorXf output_stdev = Eigen::VectorXf::Zero(output_dim);

			for (const float* datum : outputs) {

				Eigen::Map<const Eigen::VectorXf> out_map = Eigen::Map<const Eigen::VectorXf>(datum, output_dim);
				output_mean += out_map;

			}
			output_mean /= (float)outputs.size();

			for (const float* datum : outputs) {

				Eigen::Map<const Eigen::VectorXf> out_map = Eigen::Map<const Eigen::VectorXf>(datum, output_dim);
				output_stdev += (out_map - output_mean).cwiseAbs2();

			}
			output_stdev /= (float)outputs.size();
			output_stdev = output_stdev.cwiseSqrt();
		}

		bool is_gradient = false;
		int max_epochs = 3;
		int epoch = 0;
		while (epoch < max_epochs) {
			dynamics_model_in_training_->train_adam((const float**)inputs.data(), (const float**)outputs.data(), inputs.size(), minibatch_size);
			epoch++;
		}


		float mse = std::numeric_limits<float>::infinity();
		if (dynamics_model_in_training_.get()) {
			mse = dynamics_model_in_training_->mse((const float**)inputs.data(), (const float**)outputs.data(), inputs.size());
		}

		if (mse - mse != mse - mse) {
			dynamics_model_in_training_.reset();
		}

		if (dynamics_model_in_training_.get()) {
			std::lock_guard<std::mutex> lock(copying_dynamics_model_for_actor_);
			dynamics_model_for_actor_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron);
			*dynamics_model_for_actor_ = *dynamics_model_in_training_;
		}

		std::cout << "Trained dynamics model. MSE: " << mse << " Data: " << dynamics_data.size() << std::endl;


	}

	float SmoothingControl::get_trajectory_cost(int end_idx)
	{
		float cost = 0.0f;

		int time_step = marginals.size() - 1;

		while (time_step > 0) {

			const MarginalSample& sample = marginals[time_step][end_idx];
			cost += float(sample.controlCost + sample.stateCost);

			end_idx = sample.previousMarginalSampleIdx;
			time_step--;
		}

		return cost;
	}

	float SmoothingControl::best_trajectory_match_metric(int end_idx)
	{
		float match_metric = 0.0f;

		int time_step = marginals.size() - 1;

		while (time_step > 0) {

			const MarginalSample& sample = marginals[time_step][end_idx];

			float current_match = (oldBest[time_step].state - sample.state).norm();
			current_match = std::max(current_match, std::numeric_limits<float>::epsilon());
			current_match = 1.0f / current_match;

			match_metric += current_match;

			end_idx = sample.previousMarginalSampleIdx;
			time_step--;
		}


		return match_metric;
	}

	bool is_valid_sample(const SmoothingControl::TeachingSample& sample, int state_dim, int control_dim) {
		bool is_valid = true;

		if (sample.state_.size() != state_dim) {
			is_valid = false;
		}
		if (sample.future_state_.size() != state_dim) {
			is_valid = false;
		}
		if (sample.control_.size() != control_dim) {
			is_valid = false;
		}

		for (int i = 0; i < state_dim; i++) {
			float num = sample.state_[i];
			if (!(num > std::numeric_limits<float>::lowest() && num < std::numeric_limits<float>::max())) {
				is_valid = false;
			}

			num = sample.future_state_[i];
			if (!(num > std::numeric_limits<float>::lowest() && num < std::numeric_limits<float>::max())) {
				is_valid = false;
			}
		}

		for (int i = 0; i < control_dim; i++) {
			const float num = sample.control_[i];
			if (!(num > std::numeric_limits<float>::lowest() && num < std::numeric_limits<float>::max())) {
				is_valid = false;
			}
		}

		return is_valid;
	}

	void __stdcall SmoothingControl::endIteration()
	{
		learnedInLastIteration = false;

		policy_cost_ = std::numeric_limits<float>::max();
		float varied_policy_cost = std::numeric_limits<float>::max();
		//Backpropagating the costs and finding out the best trajectory
		{

			//find best trajectory and init costToGo. Also init fullSampleIdx
			bestCost = DBL_MAX;
			int bestIdx = 0;
			for (int i = 0; i < nSamples; i++)
			{
				MarginalSample &sample = marginals[nSteps][i];
				sample.costToGo = 0;//sample.stateCost+sample.controlCost; 
				if (sample.fullCost < bestCost)
				{
					bestCost = sample.fullCost;
					bestIdx = i;
				}

				//				if (sample.particleRole == ParticleRole::MACHINE_LEARNING_NO_RESAMPLING) {
				//					policy_cost_ = (float)sample.fullCost;
				//				}
				//
				//				if (sample.particleRole == ParticleRole::POLICY_SEARCH) {
				//					varied_policy_cost = (float)sample.fullCost;
				//				}

				sample.fullSampleIdx = i;
			}

			double maxCost = FLT_MAX / 100.0f / (float)(nSteps + 1); //can't use DBL_MAX, as the costs might get summed and result in INF
			for (int step = nSteps - 1; step >= 0; step--)
			{
				//backward propagation
				for (int i = 0; i < nSamples; i++)
				{
					MarginalSample &sample = marginals[step][i];
					sample.costToGo = maxCost;
					sample.fullSampleIdx = -1;  //denotes that the trajectory has been pruned
				}
				for (int nextIdx = 0; nextIdx < nSamples; nextIdx++)
				{
					MarginalSample &nextSample = marginals[step + 1][nextIdx];
					MarginalSample &sample = marginals[step][nextSample.previousMarginalSampleIdx];
					//sample.costToGo=_min(sample.costToGo,sample.stateCost + sample.controlCost + nextSample.costToGo);

					//fullCost + costToGo should always sum to the full cost at the step==nSteps
					sample.costToGo = _min(sample.costToGo, nextSample.costToGo + nextSample.stateCost + nextSample.controlCost);
					sample.fullSampleIdx = nextSample.fullSampleIdx;
				}
			}


			oldBest.resize(nSteps + 1);
			//recover the best trajectory, store for using at next iteration
			bestFullSampleIdx = bestIdx;	//index of the last marginal sample of the best full sample		
			for (int step = nSteps; step >= 0; step--)
			{
				oldBest[step] = marginals[step][bestIdx];
				bestIdx = marginals[step][bestIdx].previousMarginalSampleIdx;
			}





			//			const bool use_evolution = false;
			//			if (use_evolution) 
			//			{
			//				for (int i = 0; i < nSamples; i++)
			//				{
			//
			//
			//					const MarginalSample &sample = marginals[nSteps][i];
			//					if (sample.particleRole == ParticleRole::EVOLUTIONARY) {
			//
			//						int net_idx = sample.priorSampleIdx;
			//
			//						//const float net_cost = get_trajectory_cost(i);
			//						const float net_cost = -best_trajectory_match_metric(i);
			//
			//						int pool_idx = evolution_usage_to_pool_idx_[net_idx];
			//
			//						EvolutionData& data = *evolutionary_pool_[pool_idx];
			//						data.num_evaluations_++;
			//						data.cost_ += (net_cost - data.cost_) / (float)data.num_evaluations_;
			//
			//					}
			//
			//				}
			//
			//
			//				int evolution_population = 0;
			//
			//				for (unsigned int i = 0; i < evolutionary_pool_.size(); i++) {
			//					if (evolutionary_pool_[i]->num_evaluations_ > evaluations_for_evolution_) {
			//						evolution_population++;
			//					}
			//				}
			//
			//				//evolution_population = 0;
			//				//std::vector<float> supervised = evolutionary_pool_[0];
			//				//if (actor_.size() > 0) {
			//				//	supervised = actor_[0]->get_parameters_as_one_vector();
			//				//	for (int i = 0; i < evolution_pool_size_; i++) {
			//				//		evolutionary_pool_[i] = supervised;
			//				//	}
			//				//}
			//
			//				if (evolution_population > min_evolution_population_) 
			//				{
			//
			//					std::vector<float> supervised = actor_[0]->get_parameters_as_one_vector();
			//
			//					auto mutate = [&](std::vector<float>& fenotype) {
			//
			//						float gaussian_noise = 0.0f;
			//
			//						float small_variation = 0.0001f;
			//						float big_variation = 1.0f;
			//						float big_variation_prob = 0.001f;
			//
			//						for (float& gene : fenotype) {
			//
			//							gaussian_noise = 0.0f;
			//							BoxMuller<float>(&gaussian_noise, 1);
			//
			//							if (sampleUniform<float>() < big_variation_prob) {
			//								gaussian_noise *= big_variation;
			//							}
			//							else {
			//								gaussian_noise *= small_variation;
			//							}
			//
			//							gene += gaussian_noise;
			//
			//						}
			//
			//					};
			//
			//					auto cross_breed = [](std::vector<float>& child, const std::vector<float>& parent1, const std::vector<float>& parent2) {
			//
			//						child.resize(parent1.size());
			//						for (unsigned int idx = 0; idx < child.size(); idx++) {
			//
			//							float cross_over = sampleUniform<float>();
			//
			//							child[idx] = parent1[idx] * cross_over + parent2[idx] * (1.0f - cross_over);
			//
			//						}
			//
			//					};
			//
			//
			//
			//					int evolution_population_frac = 2;
			//					for (int i = 0; i < evolution_population / evolution_population_frac; i++) {
			//
			//						int mutate_idx = 0;
			//						float max_cost = std::numeric_limits<float>::lowest();
			//
			//						int best_idx = 0;
			//						float min_cost = std::numeric_limits<float>::max();
			//
			//
			//						for (unsigned int j = 0; j < evolutionary_pool_.size(); j++) {
			//							if (evolutionary_pool_[j]->num_evaluations_ > evaluations_for_evolution_) {
			//
			//								if (evolutionary_pool_[j]->cost_ > max_cost) {
			//									max_cost = evolutionary_pool_[j]->cost_;
			//									mutate_idx = j;
			//								}
			//
			//								if (evolutionary_pool_[j]->cost_ < min_cost) {
			//									min_cost = evolutionary_pool_[j]->cost_;
			//									best_idx = j;
			//								}
			//
			//
			//							}
			//						}
			//
			//						evolutionary_pool_[mutate_idx]->cost_ = 0.0f;
			//						evolutionary_pool_[mutate_idx]->num_evaluations_ = 0;
			//
			//
			//
			//						if (false && i == 0) {
			//
			//							evolutionary_pool_[mutate_idx]->fenotype_ = supervised;
			//							mutate(evolutionary_pool_[mutate_idx]->fenotype_);
			//
			//						}
			//						else {
			//
			//							evolutionary_pool_[mutate_idx]->fenotype_ = evolutionary_pool_[best_idx]->fenotype_;
			//							mutate(evolutionary_pool_[mutate_idx]->fenotype_);
			//
			//						}
			//
			//					}
			//
			//
			//				}
			//
			//
			//				for (int i = 0; i < evolutionary_nets_amount_; i++) {
			//
			//					int rand_idx = rand() % evolutionary_pool_.size();
			//					evolution_usage_to_pool_idx_[i] = rand_idx;
			//					evolutionary_nets_[i]->set_parameters(evolutionary_pool_[rand_idx]->fenotype_);
			//
			//				}
			//
			//
			//			}


		}


		///////////////////////////////////////////////////////////////////////////


		//Store the storedSamplesPercentage samples to guide sampling in the next frame
		{
			previousMarginals.resize(nSteps + 1);
			for (int step = 0; step <= nSteps; step++)
			{
				int nPruned = ((int)(marginals[step].size()*storedSamplesPercentage)) / 100;
				if (nPruned != (int)previousMarginals[step].size())
					previousMarginals[step].resize(nPruned);
			}


			std::vector<int> best_index(marginals[0].size(), 0);
			for (unsigned i = 0; i < best_index.size(); i++) {
				best_index[i] = i;
			}

			auto cost_sorting_lambda = [&](int i, int j) {

				int last_index = marginals.size() - 1;

				if (marginals[last_index][i].fullCost < marginals[last_index][j].fullCost) {
					return true;
				}
				else {
					return false;
				}

			};

			std::sort(best_index.begin(), best_index.end(), cost_sorting_lambda);

			for (unsigned sample = 0; sample < previousMarginals[0].size(); sample++) {

				int back_track_sample = best_index[sample];

				for (int step = marginals.size() - 1; step >= 0; step--) {

					previousMarginals[step][sample] = marginals[step][back_track_sample];
					previousMarginals[step][sample].previousMarginalSampleIdx = sample;
					back_track_sample = marginals[step][back_track_sample].previousMarginalSampleIdx;

				}

			}


		}

		///////////////////////////////////////////////////////////////////////////

		//Learning
		if (learning_ && timeAdvanced)
		{
			//		if (bestCost<learningCostThreshold && bestCost/expectedPolicyCost<learningCostImprovementThreshold){

			AALTO_ASSERT1(lookahead_ == 1);
			//Form the teaching sample
			std::shared_ptr<TeachingSample> teachingSample = std::shared_ptr<TeachingSample>(new TeachingSample(oldBest[1]));
			teachingSample->cost_to_go_ = (TeachingSample::Scalar)bestCost;

			if (use_forests_) {
				bool add_samples_to_forest = true;
				if (add_samples_to_forest) {
					ann_forest_.add_sample(number_of_hyperplane_tries_, number_of_data_in_leaf_, 12, *teachingSample);
					adding_buffer_.push_back(*teachingSample);
				}
			}



			recent_samples_.push_back(std::unique_ptr<TeachingSample>(new TeachingSample(*teachingSample)));
			while ((int)recent_samples_.size() > amount_recent_) {
				recent_samples_.pop_front();
			}


			//if tree  rebuilder thread finished, launch another one
			const bool rebuild_tree_switch = true;
			if (rebuild_tree_switch && use_forests_)
			{

				if (building_model_.wait_for(std::chrono::microseconds(1)) == std::future_status::ready) {
					//write_teaching_data_to_file("density_discriminating_field_data_"+std::to_string(iterationIdx)+".txt",discriminating_field_.data_);

					for (TeachingSample& sample : adding_buffer_) {
						tree_under_construction_.add_sample(number_of_hyperplane_tries_, number_of_data_in_leaf_, 40, sample);
					}
					adding_buffer_.clear();

					int swap_idx = rand() % ann_forest_.forest_.size();
					ann_forest_.forest_[swap_idx].root_.swap(tree_under_construction_.root_);


					auto rebuild_tree = [&]() {

						std::clock_t building_start_time = std::clock();

						tree_under_construction_.rebuild_tree(number_of_hyperplane_tries_, number_of_data_in_leaf_, amount_data_in_tree_);

						std::clock_t build_time = std::clock() - building_start_time;
						float build_time_sec = (float)build_time / (float)CLOCKS_PER_SEC;
						//std::cout << "Tree built in [sec]: " << std::to_string(build_time_sec) <<"\n";

					};

					building_model_ = std::async(std::launch::async, rebuild_tree);



				}
			}


			{
				std::lock_guard<std::mutex> lock(copying_transition_data_);
				float randu = (float)rand() / (float)RAND_MAX;
				if (randu < validation_fraction_) {
					validation_data_.push_back(std::move(teachingSample));
				}
				else {
					transition_data_.push_back(std::move(teachingSample));
				}

				int amount_validation_data = (int)(validation_fraction_*(float)learning_budget_);
				while (validation_data_.size() > (unsigned int)amount_validation_data) {
					validation_data_.pop_front();
				}
				while (transition_data_.size() > learning_budget_) {
					transition_data_.pop_front();
				}
			}



			//float extra_training_decision = (float)rand() / (float)RAND_MAX;
			//if (extra_training_decision < policy_cost_ / (policy_cost_ + bestCost)) {
			//	transition_buffer_.push_back(std::shared_ptr<TeachingSample>(new TeachingSample(oldBest[1])));
			//}




			for (int dynamics_samples = 10; dynamics_samples > 0; dynamics_samples--) {

				if (marginals.size() == 0) {
					continue;
				}

				int random_time = rand() % (int)marginals.size();
				while (random_time < 1) {
					random_time = rand() % (int)marginals.size();
				}

				const std::vector<MarginalSample>& samples_at_time = marginals[random_time];
				if (samples_at_time.size() == 0) {
					continue;
				}
				int random_idx = rand() % (int)samples_at_time.size();

				const MarginalSample& sample = marginals[random_time][random_idx];
				std::shared_ptr<TeachingSample> dynamics_sample = std::shared_ptr<TeachingSample>(new TeachingSample(sample));
				dynamics_buffer_.push_back(dynamics_sample);

			}


			{
				std::lock_guard <std::mutex> lock(copying_dynamics_data_);
				for (std::shared_ptr<TeachingSample> dynamics_sample : dynamics_buffer_) {
					if (dynamics_data_.size() >= (unsigned int)amount_dynamics_data_) {
						int rand_idx = rand() % dynamics_data_.size();
						dynamics_data_[rand_idx] = dynamics_sample;
					}
					else {
						dynamics_data_.push_back(dynamics_sample);
					}
				}


				while (dynamics_data_.size() > (unsigned int)amount_dynamics_data_) {
					dynamics_data_.pop_front();
				}
			}


			{
				std::lock_guard <std::mutex> lock(copying_dynamics_data_);
				for (std::shared_ptr<TeachingSample> dynamics_sample : dynamics_buffer_) {

					if (critic_data_.size() < (unsigned int)amount_dynamics_data_) {
						critic_data_.push_back(dynamics_sample);
					}
					else {
						int rand_idx = rand() % amount_dynamics_data_;
						critic_data_[rand_idx] = dynamics_sample;
					}

				}

				while (critic_data_.size() > (unsigned int)amount_dynamics_data_) {
					critic_data_.pop_front();
				}

			}


			dynamics_buffer_.clear();

			//			if (q_learning_) {
			//				if (training_critic_.wait_for(std::chrono::microseconds(1)) == std::future_status::ready) {
			//
			//					auto train_function = [&]() {
			//						train_critic();
			//						//train_critic_greedy();
			//						//train_critic_greedy_cacla();
			//						//train_critic_cacla();
			//					};
			//
			//
			//					rl_actor_copy_.swap(rl_actor_);
			//
			//
			//
			//					//std::cout << "Memory samples: " << memory_samples_.size() << std::endl;
			//					training_critic_ = std::async(std::launch::async, train_function);
			//				}
			//			}


			if (use_machine_learning_) {

				if (long_term_learning_.wait_for(std::chrono::microseconds(1)) == std::future_status::ready) {


					auto train_function = [&]() {
						/*if (use_dynamics_model_) {
							train_actor_using_dynamics_model();
						}
						else {*/
						train_actor();
						//train_actor_with_stdev();
					//}
					};

					actor_.swap(actor_copy_);
					actor_stdev_.swap(actor_stdev_copy_);

					//std::cout << "Memory samples: " << memory_samples_.size() << std::endl;
					long_term_learning_ = std::async(std::launch::async, train_function);
				}

			}

			//if (use_autoencoder_) {
//
//
			//	if (training_autoencoder_.wait_for(std::chrono::microseconds(1)) == std::future_status::ready) {
//
			//		auto train_function = [&]() {
			//			train_autoencoder();
			//		};
//
			//		autoencoder_.swap(autoencoder_copy_);
//
			//		//std::cout << "Memory samples: " << memory_samples_.size() << std::endl;
			//		training_autoencoder_ = std::async(std::launch::async, train_function);
			//	}
			//}


			/*if (use_dynamics_model_) {

				if (training_dynamics_model_.wait_for(std::chrono::microseconds(1)) == std::future_status::ready) {


					auto train_function = [&]() {
						train_dynamics_model();
					};

					training_dynamics_model_ = std::async(std::launch::async, train_function);

				}
			}*/



		}

		performed_transition_ = oldBest[1];
		iterationIdx++;
	}

	float SmoothingControl::decideSampling(float* state, float* control, float* future_state)
	{

		if (policy_cost_ < bestCost) {
			sampling_counter_ = nSteps / 2;
		}

		if (!use_sampling_) {
			policy_cost_ = std::numeric_limits<float>::max();
		}


		use_sampling_ = true;
		if (sampling_counter_ > 0) {
			use_sampling_ = false;
		}
		sampling_counter_--;

		return 0.0f;

		//		if (!use_discriminator_) {
		//			use_sampling_ = true;
		//			return 0.0f;
		//		}


		Eigen::Map<Eigen::VectorXf> state_map(state, nStateDimensions);
		Eigen::Map<Eigen::VectorXf> future_state_map(future_state, nStateDimensions);
		Eigen::Map<Eigen::VectorXf> control_map(control, nControlDimensions);

		std::shared_ptr<TeachingSample> sample = std::shared_ptr<TeachingSample>(new TeachingSample(state_map, control_map, future_state_map));
		if (!use_sampling_) {
			discriminator_buffer_bad_.push_back(sample);
		}
		else {
			discriminator_buffer_good_.push_back(sample);
		}

		float familiarity = 0.0f;
		if (discriminator_.get()) {
			discriminator_->run(sample->future_state_.data(), &familiarity);
		}

		bool was_using_sampling = use_sampling_;


#if 1
		use_sampling_ = true;
		if (familiarity > familiarity_threshold_ && sampling_counter_ < 0) {
			use_sampling_ = false;
		}

		if (familiarity < familiarity_threshold_) {
			sampling_counter_ = forced_sampling_;
		}

		sampling_counter_--;
#else
		use_sampling_ = true;
		if (familiarity > familiarity_threshold_) {
			use_sampling_ = false;
		}
#endif



		if (use_sampling_) {
			since_running_on_neural_network_++;
		}
		else {
			since_running_on_neural_network_ = 0;
		}

		if (training_discriminator_.wait_for(std::chrono::microseconds(1)) == std::future_status::ready) {

			for (std::shared_ptr<TeachingSample> discrimnator_sample : discriminator_buffer_bad_) {
				discriminator_data_bad_.push_back(discrimnator_sample);
			}
			discriminator_buffer_bad_.clear();

			while ((int)discriminator_data_bad_.size() > amount_discriminator_data_) {
				discriminator_data_bad_.pop_front();
			}


			for (std::shared_ptr<TeachingSample> discrimnator_sample : discriminator_buffer_good_) {
				discriminator_data_good_.push_back(discrimnator_sample);
			}
			discriminator_buffer_good_.clear();

			while ((int)discriminator_data_good_.size() > amount_discriminator_data_) {
				discriminator_data_good_.pop_front();
			}


			discriminator_.swap(discriminator_copy_);

			auto train_function = [&]() {
				train_discriminator();
			};

			//std::cout << "Memory samples: " << memory_samples_.size() << std::endl;
			training_discriminator_ = std::async(std::launch::async, train_function);
		}



		return familiarity;

	}



	void __stdcall SmoothingControl::getBestControl(int timeStep, float *out_control)
	{
		//the +1 because oldBest stores marginal states which store "incoming" controls instead of "outgoing"
		memcpy(out_control, oldBest[timeStep + 1].control.data(), nControlDimensions * sizeof(float));
	}

	void SmoothingControl::getControlToUse(float* state, float * out_control)
	{
		if (!use_sampling_) {
			//if (q_learning_) {
			//	getQLearningControl(state, out_control);
			//}
			//else {
			getMachineLearningControl(state, out_control);
			//}
			performed_transition_.state.resize(0);
		}
		else {
			Eigen::Map<Eigen::VectorXf> control_map(out_control, nControlDimensions);
			control_map = performed_transition_.control;
		}
	}


	const float* SmoothingControl::getRecentControl(float * state, float * out_control, int thread)
	{

		Eigen::Map<Eigen::VectorXf> state_vec(state, nStateDimensions);
		Eigen::Map<Eigen::VectorXf> control_out_vec(out_control, nControlDimensions);
		control_out_vec.setZero();


		TeachingSample* teaching_sample = nullptr;
		if (use_forests_) {
			TeachingSample& key = keys_[thread];
			key.state_ = Eigen::Map<Eigen::VectorXf>(state, nStateDimensions);
			teaching_sample = ann_forest_.get_approximate_nearest(key);
		}


		float dist = std::numeric_limits<float>::infinity();
		if (teaching_sample) {
			dist = (teaching_sample->state_ - state_vec).cwiseAbs().sum();
		}


		for (std::unique_ptr<TeachingSample>& sample : recent_samples_) {

			float current_dist = (sample->state_ - state_vec).cwiseAbs().sum();

			if (current_dist < dist) {
				dist = current_dist;
				teaching_sample = sample.get();
			}

		}

		if (teaching_sample) {
			control_out_vec = teaching_sample->control_;
			return teaching_sample->state_.data();
		}

		return nullptr;
	}

	int SmoothingControl::get_up_to_k_nearest(float * state, int k, int thread, SmoothingControl::TeachingSample** samples)
	{
		Eigen::Map<Eigen::VectorXf> state_vec(state, nStateDimensions);
		for (int i = 0; i < k; i++) {
			samples[i] = nullptr;
		}


		int found = 0;
		if (use_forests_) {
			TeachingSample& key = keys_[thread];
			key.state_ = Eigen::Map<Eigen::VectorXf>(state, nStateDimensions);
			found = ann_forest_.get_up_to_k_nearest(key.state_, samples, k);
		}

		auto is_closer = [&](TeachingSample* first, TeachingSample* second) {


			if (!first && !second) {
				return false;
			}
			if (first && !second) {
				return true;
			}
			if (!first && second) {
				return false;
			}


			float dist_first = (first->state_ - state_vec).norm();
			float dist_second = (second->state_ - state_vec).norm();

			if (dist_first < dist_second) {
				return true;
			}

			return false;

		};

		auto remove_duplicates = [&]() {
			for (int i = 0; i < k; i++) {
				for (int j = i + 1; j < k; j++) {
					if (samples[i] && samples[j]) {
						if (*samples[i] == *samples[j]) {
							samples[j] = nullptr;
						}
					}
				}
			}
		};

		for (std::unique_ptr<TeachingSample>& sample : recent_samples_) {

			if (is_closer(sample.get(), samples[k - 1])) {

				remove_duplicates();

				for (int i = 0; i < k; i++) {
					if (!samples[i]) {
						samples[i] = sample.get();
						break;
					}
				}

				std::sort(&samples[0], &samples[k], is_closer);
			}

		}

		remove_duplicates();
		std::sort(&samples[0], &samples[k], is_closer);

		int num_samples = 0;
		while (samples[num_samples] != nullptr && num_samples < k) {
			num_samples++;
		}

		return num_samples;
	}

	void __stdcall SmoothingControl::getBestControlState(int timeStep, float *out_state)
	{
		//the +1 because oldBest stores marginal states which store "incoming" controls instead of "outgoing"
		memcpy(out_state, oldBest[timeStep + 1].state.data(), nStateDimensions * sizeof(float));
	}

	double __stdcall SmoothingControl::getBestTrajectoryOriginalStateCost(int timeStep)
	{
		//the +1 because oldBest stores marginal states which store "incoming" controls instead of "outgoing"
		return oldBest[timeStep + 1].originalStateCostFromClient;
	}

	int __stdcall SmoothingControl::getPreviousSampleIdx(int sampleIdx, int timeStep)
	{

		if (timeStep < 0) {
			timeStep = nextStep;
		}

		if ((int)marginals.size() > timeStep && (int)marginals[timeStep].size() > sampleIdx) {
			MarginalSample &nextSample = marginals[timeStep][sampleIdx];
			return nextSample.previousMarginalSampleIdx;
		}
		else {
			return 0; //Needed for simulating main thread in Unity.
		}
	}

	void __stdcall SmoothingControl::getAssumedStartingState(int sampleIdx, float *out_state) {
		MarginalSample &nextSample = marginals[nextStep][sampleIdx];
		MarginalSample& currentSample = marginals[currentStep][nextSample.previousMarginalSampleIdx];

		if (out_state) {
			for (int i = 0; i < currentSample.state.size(); i++) {
				out_state[i] = currentSample.state[i];
			}
		}
	}

	void __stdcall SmoothingControl::getMachineLearningControl(float *state, float* out_control, int sample_idx, bool variation) {

		if (actor_.size() == 0) {
			return;
		}

		MultiLayerPerceptron* mlp = nullptr;
		mlp = actor_[sample_idx].get();

		if (variation) {
			mlp->training_ = true;
		}

		mlp->run(state);

		if (variation) {
			mlp->training_ = false;
		}


		for (int i = 0; i < nControlDimensions; i++) {

			float& output = mlp->output_operation_->outputs_[i];

			if (output - output != output - output) {
				output = 0.0f;
			}

			output = std::min(output, controlMax[i]);
			output = std::max(output, controlMin[i]);

			out_control[i] = output;
		}

	}

	void SmoothingControl::getMachineLearningControlStdev(float * state, float * out_control_stdev, int sampled_idx)
	{

		if (actor_stdev_.size() == 0) {

			//for (int i = 0; i < nControlDimensions; i++) {
			//	out_control_stdev[i] = controlMax[i] - controlMin[i];
			//}

			return;
		}

		MultiLayerPerceptron* mlp = nullptr;
		mlp = actor_stdev_[sampled_idx].get();
		mlp->run(state);



		for (int i = 0; i < nControlDimensions; i++) {
			float& output = mlp->output_operation_->outputs_[i];
			out_control_stdev[i] = output;
		}

	}

	void SmoothingControl::getQLearningControl(float * state, float * out_control, int sampled_idx)
	{

		if (rl_actor_.size() == 0) {
			return;
		}


		MultiLayerPerceptron* mlp = rl_actor_[sampled_idx].get();
		mlp->run(state);


		for (int i = 0; i < nControlDimensions; i++) {

			float& output = mlp->output_operation_->outputs_[i];

			if (output - output != output - output) {
				output = 0.0f;
			}

			output = std::min(output, controlMax[i]);
			output = std::max(output, controlMin[i]);

			out_control[i] = output;
		}

	}

	//	void SmoothingControl::performQLearning(float * state, float * out_control, float * end_state, float instantaneous_cost)
	//	{
	//
	////		if (!q_learning_) {
	//			return;
	////		}
	//
	//		Eigen::Map<Eigen::VectorXf> state_map(state, nStateDimensions);
	//		Eigen::Map<Eigen::VectorXf> end_state_map(end_state, nStateDimensions);
	//		Eigen::Map<Eigen::VectorXf> control_map(out_control, nControlDimensions);
	//
	//		std::shared_ptr<TeachingSample> new_experience = std::shared_ptr<TeachingSample>(new TeachingSample(state_map, control_map, end_state_map));
	//		new_experience->instantaneous_cost_ = instantaneous_cost;
	//		new_experience->cost_to_go_ = 0.0f;
	//
	//		dynamics_buffer_.push_back(new_experience);
	//
	//		if (training_critic_.wait_for(std::chrono::microseconds(1)) == std::future_status::ready) {
	//
	//			{
	//				std::lock_guard <std::mutex> lock(copying_dynamics_data_);
	//				for (std::shared_ptr<TeachingSample> dynamics_sample : dynamics_buffer_) {
	//					if (dynamics_data_.size() >= (unsigned int)amount_dynamics_data_) {
	//						int rand_idx = rand() % dynamics_data_.size();
	//						dynamics_data_[rand_idx] = dynamics_sample;
	//					}
	//					else {
	//						dynamics_data_.push_back(dynamics_sample);
	//					}
	//				}
	//			}
	//
	//			dynamics_buffer_.clear();
	//
	//			while (dynamics_data_.size() > (unsigned int)amount_dynamics_data_) {
	//				dynamics_data_.pop_front();
	//			}
	//
	//
	//
	//			auto train_function = [&]() {
	//				train_critic();
	//			};
	//
	//
	//			rl_actor_copy_.swap(rl_actor_);
	//
	//			//std::cout << "Memory samples: " << memory_samples_.size() << std::endl;
	//			training_critic_ = std::async(std::launch::async, train_function);
	//		}
	//
	//
	//	}

	static void form_kl_distribution(SmoothingControl::DiagonalGaussian& compromise, std::vector<SmoothingControl::DiagonalGaussian>& distributions, int amount_dists, float* probs = nullptr) {

		compromise.first.setZero();
		compromise.second.setZero();

		const int max_distributions_ever = 100;
		float tmp_array[max_distributions_ever];


		float weight_tmp = 1.0f / (float)amount_dists;
		if (probs) {
			float wSum = 0.0f;

			for (int i = 0; i < amount_dists; i++) {
				wSum += probs[i];
			}

			bool all_clear = true;
			for (int i = 0; i < amount_dists; i++) {
				probs[i] /= wSum;
				all_clear &= valid_float(probs[i]);
			}

			if (!all_clear) {
				for (int i = 0; i < amount_dists; i++) {
					probs[i] = weight_tmp;
				}
			}


		}
		else {
			probs = tmp_array;
			for (int i = 0; i < amount_dists; i++) {
				probs[i] = weight_tmp;
			}
		}

		//Mean
		for (int i = 0; i < amount_dists; i++) {
			const float& weight = probs[i];
			compromise.first += weight*distributions[i].first;
		}

		//Stdev
		for (int i = 0; i < amount_dists; i++) {
			const float& weight = probs[i];
			Eigen::VectorXf& stdev = compromise.second;
			int dims = stdev.size();
			for (int dim = 0; dim < dims; dim++) {

				const float& sigma = distributions[i].second[dim];
				const float& mu = distributions[i].first[dim];

				float diff = compromise.first[dim] - mu;

				stdev[dim] += weight*(diff*diff + sigma*sigma);

			}
		}

		Eigen::VectorXf& stdev = compromise.second;
		int dims = stdev.size();
		for (int dim = 0; dim < dims; dim++) {
			stdev[dim] = std::sqrt(stdev[dim]);
		}

	}


	void __stdcall SmoothingControl::getControl(int sampleIdx, float *out_control, const float *priorMean, const float *priorStd)
	{

		switch (control_scheme_)
		{
		case AaltoGames::SmoothingControl::L1:
			getControlL1(sampleIdx, out_control, priorMean, priorStd);
			break;
		case AaltoGames::SmoothingControl::KL:
			getControlKL(sampleIdx, out_control, priorMean, priorStd);
			break;
		default:
			getControlKL(sampleIdx, out_control, priorMean, priorStd);
			break;
		}

	}



	void SmoothingControl::getControlKL(int sampleIdx, float * out_control, const float * priorMean, const float * priorStd)
	{
		//int nUniform=(int)(naiveTrajectoryPortion*(double)nSamples);
		//link the marginal samples to each other so that full-dimensional samples can be recovered
		MarginalSample &nextSample = marginals[nextStep][sampleIdx];
		MarginalSample &currentSample = marginals[currentStep][nextSample.previousMarginalSampleIdx];
		Eigen::VectorXf& control = nextSample.control;


		nextSample.previousState = currentSample.state;
		nextSample.previousPreviousState = currentSample.previousState;

		nextSample.previousControl = currentSample.control;
		nextSample.previousPreviousControl = currentSample.previousControl;



		bool processed = false;
		//special processing for old best trajectory
		if (nextSample.particleRole == ParticleRole::OLD_BEST)
		{
			if (iterationIdx > 0 && nextStep < (int)oldBest.size()) {
				//const float old_best_noise = 0.01f;

				//BoxMuller<float>(control);
				//control *= old_best_noise;

				//the old best solution
				control = oldBest[nextStep].control;	//nextStep as index because the control is always stored to the next sample ("control that brought me to this state")

				processed = true;
			}
			else {
				nextSample.particleRole = ParticleRole::MACHINE_LEARNING_NO_VARIATION;
			}
		}

		if (!processed)
		{
			if (nextSample.particleRole == ParticleRole::SMOOTHED && nextStep + 1 < (int)oldBest.size())
			{

				float prev_weight = 0.1f;
				float current_weight = 0.8f;
				float future_weight = 0.1f;

				control = prev_weight*oldBest[nextStep].previousControl;
				control += current_weight*oldBest[nextStep].control;
				control += future_weight*oldBest[nextStep + 1].control;

				//control = oldBest[nextStep].control;
				//shiftWithAutoencoder(currentSample.state.data(), control.data(), sampleIdx);

			}
			else if (nextSample.particleRole == ParticleRole::MACHINE_LEARNING_NO_VARIATION)
			{
				getMachineLearningControl(currentSample.state.data(), control.data(), sampleIdx);
				//shiftWithAutoencoder(currentSample.state.data(), control.data(), sampleIdx);

			}
			//			else if (nextSample.particleRole == ParticleRole::EVOLUTIONARY)
			//			{
			//
			//				MultiLayerPerceptron* net = evolutionary_nets_[nextSample.priorSampleIdx].get();
			//				net->run(currentSample.state.data(), control.data());
			//
			//			}
			//			else if (nextSample.particleRole == ParticleRole::POLICY_SEARCH)
			//			{
			//
			//				getMachineLearningControl(currentSample.state.data(), control.data(), sampleIdx);
			//
			//				if (currentStep == 0) {
			//					//Gaussian noise
			//					BoxMuller<float>(out_control, nControlDimensions);
			//					Eigen::Map<Eigen::VectorXf> noise_map(out_control, nControlDimensions);
			//
			//					float stdev = 0.1f;
			//					noise_map *= stdev;
			//
			//					control += noise_map;
			//				}
			//
			//				//shiftWithAutoencoder(currentSample.state.data(), control.data(), sampleIdx);
			//			}
			//			else if (nextSample.particleRole == ParticleRole::MACHINE_LEARNING_NO_RESAMPLING)
			//			{
			//				getMachineLearningControl(currentSample.state.data(), control.data(), sampleIdx);
			//				//shiftWithAutoencoder(currentSample.state.data(), control.data(), sampleIdx);
			//			}
			//			else if (nextSample.particleRole == ParticleRole::REINFORCEMENT_LEARNING)
			//			{
			//				getQLearningControl(currentSample.state.data(), control.data(), sampleIdx);
			//				//shiftWithAutoencoder(currentSample.state.data(), control.data(), sampleIdx);
			//			}
			//			else if (nextSample.particleRole == ParticleRole::DEBUG_ML) {
			//				getMachineLearningControl(currentSample.state.data(), control.data(), sampleIdx);
			//			}
			else
			{

				const bool zero_variances = true;


				float probs[1000];
				int distribution_idx = 0;



				//multiply the difference prior and static prior to yield the proposal
				if (!old_best_valid_)	//prior for difference not available at first step, except in online optimization after first iteration 
				{
					gaussian_distributions_[sampleIdx][distribution_idx].first = staticPrior.mean[0];
					gaussian_distributions_[sampleIdx][distribution_idx].second = staticPrior.std[0];
					probs[distribution_idx] = 1.0f;
					distribution_idx++;
				}
				else {
					//first the difference prior
					gaussian_distributions_[sampleIdx][distribution_idx].first = currentSample.control;
					gaussian_distributions_[sampleIdx][distribution_idx].second = controlMutationStd;
					probs[distribution_idx] = 1.0f;
					distribution_idx++;


					////second difference prior
					//gaussian_distributions_[sampleIdx][distribution_idx].first = currentSample.previousControl;
					//gaussian_distributions_[sampleIdx][distribution_idx].second = controlMutationStd;
					//distribution_idx++;
				}
				//replaceable_idx = -1;

				//the optional prior passed as argument
				if (priorMean != NULL && priorStd != NULL)
				{
					Eigen::VectorXf& mean = gaussian_distributions_[sampleIdx][distribution_idx].first;
					Eigen::VectorXf& std = gaussian_distributions_[sampleIdx][distribution_idx].second;
					probs[distribution_idx] = 1.0f;
					distribution_idx++;

					for (int dim = 0; dim < nControlDimensions; dim++) {
						mean[dim] = priorMean[dim];
						std[dim] = priorStd[dim];
					}

					//if (zero_variances) {
					//	std.setZero();
					//}
				}

				const MarginalSample* nearest = nullptr;
				float nearest_dist = std::numeric_limits<float>::infinity();
				//use previous frame trajectories as prior. 
				if (nextSample.previous_frame_prior_ && previous_frame_stdev_.size() > 0) {

					if (previousMarginals.size() > (unsigned int)nextStep) {
						const std::vector<MarginalSample>& marginal = previousMarginals[nextStep];
						float current_dist = 0.0f;

						for (unsigned i = 0; i < marginal.size(); i++) {
							current_dist = (marginal[i].previousState - currentSample.state).cwiseAbs().sum();
							//current_dist = marginal[i].costToGo;

							if (current_dist < nearest_dist) {
								nearest_dist = current_dist;
								nearest = &marginal[i];
							}

						}



						if (nearest) {
							gaussian_distributions_[sampleIdx][distribution_idx].first = nearest->control;
							gaussian_distributions_[sampleIdx][distribution_idx].second = previous_frame_stdev_;
							probs[distribution_idx] = 1.0f;

							if (zero_variances) {
								gaussian_distributions_[sampleIdx][distribution_idx].second.setZero();
							}

							distribution_idx++;



						}
					}
				}


				if (nextSample.nearest_neighbor_prior_ && nearest_neighbor_stdev_.size() > 0) {

					const int k = 5;
					TeachingSample* samples[k + 1];
					int num_samples = get_up_to_k_nearest(currentSample.state.data(), k, sampleIdx, samples);




					if (num_samples > 0) {

						for (int idx = 0; idx < num_samples; idx++) {
							const TeachingSample* sample = samples[idx];
							gaussian_distributions_[sampleIdx][distribution_idx].first = sample->control_;
							gaussian_distributions_[sampleIdx][distribution_idx].second = nearest_neighbor_stdev_;
							probs[distribution_idx] = 1.0f;

							if (zero_variances) {
								gaussian_distributions_[sampleIdx][distribution_idx].second.setZero();
							}

							distribution_idx++;
						}


					}
				}


				if (nextSample.machine_learning_prior_ && machine_learning_stdev_.size() > 0)
				{
					getMachineLearningControl(currentSample.state.data(), control.data());


					gaussian_distributions_[sampleIdx][distribution_idx].first = control;
					gaussian_distributions_[sampleIdx][distribution_idx].second = machine_learning_stdev_;

					getMachineLearningControlStdev(currentSample.state.data(), gaussian_distributions_[sampleIdx][distribution_idx].second.data(), sampleIdx);

					if (zero_variances) {
						gaussian_distributions_[sampleIdx][distribution_idx].second.setZero();
					}

					probs[distribution_idx] = 1.0f;
					distribution_idx++;




#if 0
					getQLearningControl(currentSample.state.data(), control.data(), sampleIdx);

					gaussian_distributions_[sampleIdx][distribution_idx].first = control;
					gaussian_distributions_[sampleIdx][distribution_idx].second = machine_learning_stdev_;

					if (zero_variances) {
						gaussian_distributions_[sampleIdx][distribution_idx].second.setZero();
					}

					probs[distribution_idx] = 1.0f;
					distribution_idx++;
#endif

				}


				form_kl_distribution(sampling_distributions_[sampleIdx], gaussian_distributions_[sampleIdx], distribution_idx, probs);


				for (int i = 0; i < control.size(); i++) {
					control[i] = (float)sample_clipped_gaussian(sampling_distributions_[sampleIdx].first[i], sampling_distributions_[sampleIdx].second[i], controlMin(i), controlMax(i)); //Opposed to proposal's sampleWithLimits this doesn't have the granularity of rare areas.
				}

				//if (nextSample.machine_learning_prior_ || nextSample.nearest_neighbor_prior_) {
				//	shiftWithAutoencoder(currentSample.state.data(), control.data(), sampleIdx);
				//}

			}
		}

		//Clamping the control to the bounds
		for (int i = 0; i < control.size(); i++) {

			float& control_val = control[i];

			if (!(control_val > std::numeric_limits<float>::lowest() && control_val < std::numeric_limits<float>::max())) {
				control_val = 0.0f;
			}

			control[i] = std::min(std::max(control[i], controlMin[i]), controlMax[i]);
			out_control[i] = control[i];
		}
	}



	void SmoothingControl::getControlL1(int sampleIdx, float * out_control, const float * priorMean, const float * priorStd)
	{

		//int nUniform=(int)(naiveTrajectoryPortion*(double)nSamples);
		//link the marginal samples to each other so that full-dimensional samples can be recovered
		MarginalSample &nextSample = marginals[nextStep][sampleIdx];
		MarginalSample &currentSample = marginals[currentStep][nextSample.previousMarginalSampleIdx];
		Eigen::VectorXf& control = nextSample.control;


		nextSample.previousState = currentSample.state;
		nextSample.previousPreviousState = currentSample.previousState;

		nextSample.previousControl = currentSample.control;
		nextSample.previousPreviousControl = currentSample.previousControl;


		bool processed = false;
		//special processing for old best trajectory
		if (nextSample.particleRole == ParticleRole::OLD_BEST)
		{
			if (iterationIdx > 0 && nextStep < (int)oldBest.size()) {
				//the old best solution
				control = oldBest[nextStep].control;	//nextStep as index because the control is always stored to the next sample ("control that brought me to this state")
				processed = true;
			}
			else {
				nextSample.particleRole = ParticleRole::MACHINE_LEARNING_NO_VARIATION;
			}
		}

		if (!processed)
		{
			if (nextSample.particleRole == ParticleRole::SMOOTHED && nextStep + 1 < (int)oldBest.size())
			{

				float prev_weight = 0.1f;
				float current_weight = 0.8f;
				float future_weight = 0.1f;

				control = prev_weight*oldBest[nextStep].previousControl;
				control += current_weight*oldBest[nextStep].control;
				control += future_weight*oldBest[nextStep + 1].control;

				//control = oldBest[nextStep].control;
				//shiftWithAutoencoder(currentSample.state.data(), control.data(), sampleIdx);

			}
			else if (nextSample.particleRole == ParticleRole::MACHINE_LEARNING_NO_VARIATION)
			{
				getMachineLearningControl(currentSample.state.data(), control.data(), sampleIdx);
				//				shiftWithAutoencoder(currentSample.state.data(), control.data(), sampleIdx);

			}
			//			else if (nextSample.particleRole == ParticleRole::MACHINE_LEARNING_NO_RESAMPLING)
			//			{
			//				getMachineLearningControl(currentSample.state.data(), control.data(), sampleIdx);
			//				//shiftWithAutoencoder(currentSample.state.data(), control.data(), sampleIdx);
			//			}
			//			else if (nextSample.particleRole == ParticleRole::DEBUG_ML) {
			//				getMachineLearningControl(currentSample.state.data(), control.data(), sampleIdx);
			//			}
			else
			{


				//When no kernels used, we sample from the product of the static control prior, the difference priors given the previous sampled controls,
				//and the "mutation prior" of the sample from previous frame
				DiagonalGMM *proposal = proposals_[sampleIdx].get();
				DiagonalGMM *prior = priors_[sampleIdx].get();

				//first the difference prior
				prior->mean[0] = currentSample.control;
				prior->std[0] = controlDiffPriorStd;

				//multiply the difference prior and static prior to yield the proposal
				if (currentStep == 0 && (!timeAdvanced || iterationIdx == 0))	//prior for difference not available at first step, except in online optimization after first iteration 
				{
					proposal->copyFrom(staticPrior);
				}
				else
				{
					DiagonalGMM::multiply(staticPrior, *prior, *proposal);
					//proposal->copyFrom(*prior);
				}



				//second difference prior
				//prior->mean[0]=currentSample.previousControl;
				//prior->std[0]=controlDiffDiffPriorStd;
				//DiagonalGMM::multiply(*prior,*proposal,*proposal);	//ok to have same source and destination if both have only 1 gaussian

				//the optional prior passed as argument
				if (priorMean != NULL && priorStd != NULL)
				{
					memcpy(&prior->mean[0][0], priorMean, sizeof(float)*nControlDimensions);
					memcpy(&prior->std[0][0], priorStd, sizeof(float)*nControlDimensions);

					for (int dim = 0; dim < nControlDimensions; dim++) {
						float& num = prior->mean[0][dim];

						if (num < controlMin[dim]) {
							num = controlMin[dim];
						}


						if (num > controlMax[dim]) {
							num = controlMax[dim];
						}

					}

					DiagonalGMM::multiply(*prior, *proposal, *proposal);	//ok to have same source and destination if both have only 1 gaussian
				}




				const MarginalSample* nearest = nullptr;
				float nearest_dist = std::numeric_limits<float>::infinity();

				//use previous frame trajectories as prior. 
				if (nextSample.previous_frame_prior_) {
					if (nextStep < (int)previousMarginals.size()) {
						const std::vector<MarginalSample>& marginal = previousMarginals[nextStep];


						float current_dist = 0.0f;

						for (unsigned i = 0; i < marginal.size(); i++) {
							current_dist = (marginal[i].previousState - currentSample.state).cwiseAbs().sum();
							//current_dist = marginal[i].costToGo;

							if (current_dist < nearest_dist) {
								nearest_dist = current_dist;
								nearest = &marginal[i];
							}

						}
					}
				}





				const float* recent_state = nullptr;

				if (nearest && nextSample.nearest_neighbor_prior_) {
					recent_state = getRecentControl(currentSample.state.data(), control.data(), sampleIdx);
				}

				if (recent_state) {


					const bool force_multiply = false;


					if (force_multiply) {
						for (int i = 0; i < control.size(); i++) {
							productNormalDist(nearest->control[i], controlMutationStd[i], proposal->mean[0](i), proposal->std[0](i), proposal->mean[0](i), proposal->std[0](i));
							productNormalDist(control[i], controlMutationStd[i], proposal->mean[0](i), proposal->std[0](i), proposal->mean[0](i), proposal->std[0](i));
						}
					}
					else {
						Eigen::Map<const Eigen::VectorXf> state_vec(recent_state, nStateDimensions);

						float current_dist = (state_vec - currentSample.state).cwiseAbs().sum();
						if (current_dist < nearest_dist) {

							for (int i = 0; i < control.size(); i++) {
								productNormalDist(control[i], controlMutationStd[i], proposal->mean[0](i), proposal->std[0](i), proposal->mean[0](i), proposal->std[0](i));
							}

						}
						else {
							if (nearest) {
								for (int i = 0; i < nearest->control.size(); i++) {
									productNormalDist(nearest->control[i], controlMutationStd[i], proposal->mean[0](i), proposal->std[0](i), proposal->mean[0](i), proposal->std[0](i));
								}
							}
						}
					}

				}
				else {
					if (nearest) {
						for (int i = 0; i < nearest->control.size(); i++) {
							productNormalDist(nearest->control[i], controlMutationStd[i], proposal->mean[0](i), proposal->std[0](i), proposal->mean[0](i), proposal->std[0](i));
						}
					}
				}



				if (nextSample.machine_learning_prior_)
				{

					getMachineLearningControl(currentSample.state.data(), control.data(), sampleIdx);

					for (int i = 0; i < control.size(); i++) {
						productNormalDist(control[i], machine_learning_stdev_[i], proposal->mean[0](i), proposal->std[0](i), proposal->mean[0](i), proposal->std[0](i));
					}

				}


				for (int i = 0; i < control.size(); i++) {
					float& mean = proposal->mean[0](i);
					float& stdev = proposal->std[0](i);
					control[i] = (float)sample_clipped_gaussian(mean, stdev, controlMin(i), controlMax(i)); //Opposed to proposal's sampleWithLimits this doesn't have the granularity of rare areas.
				}

				//				if (nextSample.machine_learning_prior_) {
				//					shiftWithAutoencoder(currentSample.state.data(), control.data(), sampleIdx);
				//					shiftWithCritic(currentSample.state.data(), control.data(), sampleIdx);
				//				}

			}
		}

		//Clamping the control to the bounds
		for (int i = 0; i < control.size(); i++) {
			control[i] = std::min(std::max(control[i], controlMin[i]), controlMax[i]);
			out_control[i] = control[i];
		}
	}


	int SmoothingControl::getNumTrajectories()
	{
		return nSamples;
	}
	int SmoothingControl::getNumSteps()
	{
		return nSteps;
	}





	SmoothingControl::TeachingSample::TeachingSample(const MarginalSample & marginal_sample)
	{
		state_ = marginal_sample.previousState;
		future_state_ = marginal_sample.state;
		control_ = marginal_sample.control;

		cost_to_go_ = 0.0f;
		instantaneous_cost_ = (float)marginal_sample.originalStateCostFromClient + (float)marginal_sample.controlCost;

		state_control_ = Eigen::VectorXf::Zero(state_.size() + control_.size());
		control_state_ = state_control_;

		state_control_.head(state_.size()) = state_;
		state_control_.tail(control_.size()) = control_;

		control_state_.head(control_.size()) = control_;
		control_state_.tail(state_.size()) = state_;

		state_future_state_.resize(state_.size() * 2);
		state_future_state_.head(state_.size()) = state_;
		state_future_state_.tail(state_.size()) = future_state_;

		input_for_learner_ = state_;
	}

} //AaltoGames;
