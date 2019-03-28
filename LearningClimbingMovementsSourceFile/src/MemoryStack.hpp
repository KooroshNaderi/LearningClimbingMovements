
#ifndef MEMORY_STACK_HPP
#define MEMORY_STACK_HPP

#include <vector>
#include <deque>
#include <memory>


template<typename Scalar>
class Memory {

public:

	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> MemoryVector;
	MemoryVector memory_;
	Scalar freshness_;
	Scalar forget_rate_;
	Scalar forget_threshold_;
	Scalar learning_rate_;

	std::vector<std::unique_ptr<Memory<Scalar> > > associated_memories_;
	Memory* parent_;

	void default_params() {
		forget_rate_ = (Scalar)0.99;
		forget_threshold_ = (Scalar)0.001;
		learning_rate_ = (Scalar)0.01;
		freshness_ = (Scalar)0;
	}

	Memory(unsigned dimension) {
		memory_.resize(dimension);
		memory_.setZero();
		default_params();
		associated_memories_.clear();
	}

	Memory(unsigned dimension, unsigned branching_factor, unsigned depth) {
		memory_.resize(dimension);
		memory_.setZero();
		default_params();
		associated_memories_.clear();
		parent_ = nullptr;

		if (depth == 0) {
			return;
		}
		else {
			while (associated_memories_.size() < branching_factor) {
				associated_memories_.push_back(std::unique_ptr<Memory<Scalar> >(new Memory<Scalar>(dimension, branching_factor, depth - 1)));
				associated_memories_.back()->parent_ = this;
			}
		}
	}

	Memory(const Memory<Scalar>& other) {
		parent_ = nullptr;
		associated_memories_.clear();

		memory_ = other.memory_;
		freshness_ = other.freshness_;
		forget_rate_ = other.forget_rate_;
		forget_threshold_ = other.forget_threshold_;
		learning_rate_ = other.learning_rate_;

		for (const std::unique_ptr<Memory<Scalar> >& other_mem : other.associated_memories_) {
			associated_memories_.push_back(std::unique_ptr<Memory<Scalar> >(new Memory<Scalar>(*other_mem)));
			associated_memories_.back()->parent_ = this;
		}

	}

	Memory operator=(const Memory<Scalar>& other) {
		parent_ = nullptr;
		associated_memories_.clear();

		memory_ = other.memory_;
		freshness_ = other.freshness_;
		forget_rate_ = other.forget_rate_;
		forget_threshold_ = other.forget_threshold_;
		learning_rate_ = other.learning_rate_;

		for (const std::unique_ptr<Memory<Scalar> >& other_mem : other.associated_memories_) {
			associated_memories_.push_back(std::unique_ptr<Memory<Scalar> >(new Memory<Scalar>(*other_mem)));
			associated_memories_.back()->parent_ = this;
		}

		return *this;
	}

	void set_to(const Scalar* observation) {
		unsigned observation_dim = memory_.size();
		Eigen::Map<const MemoryVector> obs_vec = Eigen::Map<const MemoryVector>(observation, observation_dim);

		freshness_ = 1;
		memory_ = obs_vec;

		for (std::unique_ptr<Memory<Scalar> >& memory : associated_memories_) {
			memory->set_to(observation);
		}

	}

	std::vector<Memory*> get_memories() {
		std::vector<Memory*> result;

		if (associated_memories_.size() == 0) {
			result.push_back(this);
		}

		for (std::unique_ptr<Memory<Scalar> >& memory : associated_memories_) {
			std::vector<Memory*> children = memory->get_memories();

			for (Memory* mem : children) {
				result.push_back(mem);
			}

		}

		return result;
	}

	Memory* get_closest(const Scalar* observation) {

		unsigned observation_dim = memory_.size();
		Eigen::Map<const MemoryVector> obs_vec = Eigen::Map<const MemoryVector>(observation, observation_dim);

		Memory* best_memory = this;
		std::vector<std::unique_ptr<Memory> >* candidates = &associated_memories_;

		while (candidates) {
			Scalar nearest_dist = std::numeric_limits<Scalar>::infinity();
			Memory* nearest = nullptr;

			for (std::unique_ptr<Memory>& memory_candidate : *candidates) {
				Scalar current_dist = (memory_candidate->memory_ - obs_vec).norm();

				if (current_dist < nearest_dist) {
					nearest_dist = current_dist;
					nearest = memory_candidate.get();
				}

			}

			if (nearest) {
				best_memory = nearest;
				candidates = &best_memory->associated_memories_;
			}
			else {
				candidates = nullptr;
			}
		}

		return best_memory;
	}

	void add_new_observation(const Scalar* observation) {

		unsigned observation_dim = memory_.size();
		Eigen::Map<const MemoryVector> obs_vec = Eigen::Map<const MemoryVector>(observation, observation_dim);

		if (associated_memories_.size() == 0) {
			return;
		}


		std::deque<Memory*> all_memories_;
		all_memories_.push_back(this);
		for (unsigned i = 0; i < all_memories_.size(); i++) {
			for (std::unique_ptr<Memory<Scalar> >& memory : all_memories_[i]->associated_memories_) {
				memory->freshness_ *= forget_rate_;
				all_memories_.push_back(memory.get());
			}
		}

		Memory<Scalar>* best_memory = get_closest(observation);

		Memory<Scalar>* search_memory = best_memory->parent_;
		Memory<Scalar>* worst_memory = best_memory;
		while (search_memory) {
			for (std::unique_ptr<Memory<Scalar> >& memory : search_memory->associated_memories_) {
				if (memory->freshness_ < worst_memory->freshness_) {
					worst_memory = memory.get();
				}
			}
			search_memory = search_memory->parent_;
		}

		if (worst_memory->freshness_ < forget_threshold_) {
			worst_memory->set_to(observation);
		}

		search_memory = best_memory;
		while (search_memory) {
			search_memory->freshness_ = 1;
			search_memory->memory_ += learning_rate_*(obs_vec - search_memory->memory_);
			search_memory = search_memory->parent_;
		}


	}


	const Memory* get_memory(const Scalar* key, void* helping_structure,  Scalar(*distance_metric)(const Scalar* memory_vector, const Scalar* key_vector, void* helping_structure)) {

		Memory* best_memory = this;
		std::vector<std::unique_ptr<Memory> >* candidates = &associated_memories_;

		while (candidates) {
			Scalar nearest_dist = std::numeric_limits<Scalar>::infinity();
			Memory* nearest = nullptr;

			for (std::unique_ptr<Memory>& memory_candidate : *candidates) {
				Scalar current_dist = distance_metric(memory_candidate->memory_.data(),key, helping_structure);

				if (current_dist < nearest_dist) {
					nearest_dist = current_dist;
					nearest = memory_candidate.get();
				}

			}

			if (nearest) {
				best_memory = nearest;
				candidates = &best_memory->associated_memories_;
			}
			else {
				candidates = nullptr;
			}
		}

		return best_memory;

	}

};

#endif
