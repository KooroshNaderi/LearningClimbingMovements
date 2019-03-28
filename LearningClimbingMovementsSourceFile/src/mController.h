#include <deque>
#include <future>
#include "ProbUtils.hpp"
//#include "LearnerControl\\SmoothingControl.h"
#include "CPBPLearning\\ControlPBP.h"

#include "DynamicMinMax.h"
#include "CMAES.hpp"
#include "ClippedGaussianSampling.h"
#include "Spline\Spline.h"
#include "LearningClass.h"

static const int nTrajectories = contextNUM - 1;
static const int nTimeSteps = useOfflinePlanning ? int(cTime*1.5001f) : int(cTime/2);

static const float poseAngleSd=deg2rad*60.0f;
static const float maxSpeedRelToRange= 4.0f; //joint with PI rotation range will at max rotate at this times PI radians per second
static const float controlDiffSdScale = useOfflinePlanning ? 0.5f : 0.2f;
// for online method it works best if nPhysicsPerStep == 1. Also, Putting nPhysicsPerStep to 3 does not help in online method and makes simulation awful (bad posture, not reaching a stance)
static const int nPhysicsPerStep = optimizerType == otCMAES ? 1 : (useOfflinePlanning ? 3 : 1);

static const float noCostImprovementThreshold = 50.0f;

//CMAES and CPBP params
static const float torsoMinFMax=20.0f;
static const bool optimizeFMax = false;//optimizerType == otLearnerCPBP ? false : true;
static const int fmCount = optimizeFMax ? fmEnd : 0;

class outSavedData 
{
public:
	outSavedData(const BipedState& _c)
	{
		_s = _c;
		_s.bodyStates = _c.bodyStates;
		_s.control_cost = _c.control_cost;
		_s.counter_let_go = _c.counter_let_go;
		_s.forces = _c.forces;
		_s.hold_bodies_ids = _c.hold_bodies_ids;
		_s.hold_bodies_info = _c.hold_bodies_info;
		_s.saving_slot_state = _c.saving_slot_state;
		_s._body_control_cost = _c._body_control_cost;
	}
	outSavedData(VectorXf _LS, VectorXf _LFS, VectorXf _LA, float _cost)
	{
		_learningS = _LS;
		_learningFS = _LFS;
		_learningA = _LA;
		_costForLearningS = _cost;
	}

	BipedState _s;

	VectorXf _learningS;
	VectorXf _learningFS;
	VectorXf _learningA;
	float _costForLearningS;
};

class mController 
{
public:
	enum ControlledPoses { MiddleTrunk = 0, LeftLeg = 1, RightLeg = 2, LeftHand = 3, RightHand = 4, Posture = 5, TorsoDir = 6};
	enum StanceBodies {sbLeftLeg=0,sbRightLeg,sbLeftHand,sbRightHand};
	enum CostComponentNames {VioateDis, Velocity, ViolateVel, Angles, cLeftLeg, cRightLeg, cLeftHand, cRightHand, cMiddleTrunkP, cMiddleTrunkD, cHead, cNaN};

	OptimizerTypes mControllerType;

	class mForceSequence
	{
	private:
		Vector3 resllForce;
		Vector3 resrlForce;
		Vector3 reslhForce;
		Vector3 resrhForce;

		std::vector<int> counter_let_go;
		unsigned int max_store_forces;

		void push_force(std::vector<Vector3>& _forces, Vector3 _cFroce)
		{
			if (_forces.size() >= max_store_forces)
			{
				_forces.erase(_forces.begin());
			}
			_forces.push_back(_cFroce);
			return;
		}

		Vector3 calculateAvgForce(std::vector<Vector3>& _forces)
		{
			Vector3 sum_forces(0.0f,0.0f,0.0f);
			for (unsigned int i = 0; i < _forces.size(); i++)
			{
				sum_forces += _forces[i];
			}
			if (_forces.size() > 0)
			{
				sum_forces /= (float)_forces.size();
			}

			return sum_forces;
		}

	public:

		void setMaxStorageForces(unsigned int _newNumStorage)
		{
			max_store_forces = _newNumStorage;
		}

		mForceSequence()
		{
			counter_let_go = std::vector<int>(4,0);
			max_store_forces = 30;

			resllForce = Vector3(0,0,0);
			resrlForce = Vector3(0,0,0);
			reslhForce = Vector3(0,0,0);
			resrhForce = Vector3(0,0,0);
		}

		unsigned int getMaxLetGo()
		{
			return max_store_forces;
		}

		int getCountLetGo(StanceBodies _bodyName)
		{
			return counter_let_go[int(_bodyName)];
		}

		void updateCounterLetGo(StanceBodies _bodyName, float _diff) // _diff = fi.norm() - f_max
		{
			float _threshold = 100.0f; // hands have grasping while feet do not
			if (_bodyName == StanceBodies::sbLeftLeg || _bodyName == StanceBodies::sbRightLeg)
			{
				_threshold = 50.0f;
			}

			int update_count = int(_diff / _threshold);

			if (_diff <= 0)
			{
				update_count = min(update_count , -1);
			}

			counter_let_go[int(_bodyName)] += update_count;
			if (counter_let_go[int(_bodyName)] < 0)
				counter_let_go[int(_bodyName)] = 0;
		}

		void makeZeroCounter(StanceBodies _bodyName)
		{
			counter_let_go[int(_bodyName)] = 0;
		}

		void makeZeroForce(StanceBodies _bodyName)
		{
			switch (_bodyName)
			{
			case StanceBodies::sbLeftLeg:
				resllForce = Vector3(0,0,0);
				//				llForces.clear();
				break;
			case StanceBodies::sbRightLeg:
				resrlForce = Vector3(0,0,0);
				//				rlForces.clear();
				break;
			case StanceBodies::sbLeftHand:
				reslhForce = Vector3(0,0,0);
				//				lhForces.clear();
				break;
			case StanceBodies::sbRightHand:
				resrhForce = Vector3(0,0,0);
				//				rhForces.clear();
				break;
			}
		}

		void updateForces(StanceBodies _bodyName, Vector3 _cFroce)
		{
			float max_force = maximumForce;
			Vector3 _dis;
			float update_ratio = 1/(float)max_store_forces;
			switch (_bodyName)
			{
			case StanceBodies::sbLeftLeg:
				_dis = (_cFroce - resllForce);
				resllForce += update_ratio * _dis;

				if (resllForce.norm() > max_force)
					resllForce = resllForce.normalized() * max_force;

				break;
			case StanceBodies::sbRightLeg:
				_dis = (_cFroce - resrlForce);
				resrlForce += update_ratio * _dis;

				if (resrlForce.norm() > max_force)
					resrlForce = resrlForce.normalized() * max_force;

				break;
			case StanceBodies::sbLeftHand:
				_dis = (_cFroce - reslhForce);
				reslhForce += update_ratio * _dis;

				if (reslhForce.norm() > max_force)
					reslhForce = reslhForce.normalized() * max_force;

				break;
			case StanceBodies::sbRightHand:
				_dis = (_cFroce - resrhForce);
				resrhForce += update_ratio * _dis;

				if (resrhForce.norm() > max_force)
					resrhForce = resrhForce.normalized() * max_force;

				break;
			}

			return;
		}

		Vector3 getAvgForce(StanceBodies _bodyName)
		{
			switch (_bodyName)
			{
			case StanceBodies::sbLeftLeg:
				return resllForce;//
				//return calculateAvgForce(llForces);
				break;
			case StanceBodies::sbRightLeg:
				return resrlForce; //
				//return calculateAvgForce(rlForces);
				break;
			case StanceBodies::sbLeftHand:
				return reslhForce; //
				//return calculateAvgForce(lhForces);
				break;
			case StanceBodies::sbRightHand:
				return resrhForce; //
				//return calculateAvgForce(rhForces);
				break;
			}
			return Vector3(0.0f,0.0f,0.0f);
		}

		void setAvgForce(StanceBodies _bodyName, Vector3 _cForce)
		{
			switch (_bodyName)
			{
			case StanceBodies::sbLeftLeg:
				resllForce = _cForce;//
				break;
			case StanceBodies::sbRightLeg:
				resrlForce = _cForce; //
				break;
			case StanceBodies::sbLeftHand:
				reslhForce = _cForce; //
				break;
			case StanceBodies::sbRightHand:
				resrhForce = _cForce; //
				break;
			}
			return;
		}

		void setCounterLetGo(StanceBodies _bodyName, int _cCounterLetGo)
		{
			counter_let_go[int(_bodyName)] = _cCounterLetGo;
			return;
		}
	};

	mController()
	{
		mControllerType = optimizerType;
		using_sampling = true;
		_forceOnHandsFeet = std::vector<mForceSequence>(contextNUM, mForceSequence());
		_body_parts_control_cost = std::vector<std::vector<float>>(contextNUM, std::vector<float>(4, 10));
		
		current_cost_state = FLT_MAX;
		current_cost_control = FLT_MAX;
		best_trajectory_cost = FLT_MAX;
		isReachedToTargetStance = false;

		consider_force_opt = false;
	}

	std::vector<std::vector<BipedState>> states_trajectory_steps;

	int masterContextID;

	BipedState startState;
	BipedState resetState;

	float current_cost_state;
	float current_cost_control;
	float best_trajectory_cost;
	bool isReachedToTargetStance;

	virtual void optimize_the_cost(bool advance_time, std::vector<ControlledPoses>& sourcePos, std::vector<Vector3>& targetPos
		, std::vector<int>& targetHoldIDs, bool showDebugInfo, bool allowOnGround) = 0;

	virtual void syncMasterContextWithStartState(bool loadAnyWay) = 0;

	virtual bool simulateBestTrajectory(bool flagSaveSlots, std::vector<int>& dHoldIDs, std::vector<outSavedData>& outStates) = 0;

	virtual void reset() = 0;

	float getCost(std::vector<ControlledPoses>& sourcePos, std::vector<Vector3>& targetPos, std::vector<int>& targetHoldIDs, bool allowOnGround)
	{
		return computeStateCost(mContextController, sourcePos, targetPos, targetHoldIDs, false, masterContextID, false, allowOnGround);
	}

	void visualizeForceDirections(bool printDebugInfo = true)
	{
		debugVisulizeForceOnHnadsFeet(masterContextID, printDebugInfo);
		return;
	}

	void setCurrentForce(StanceBodies _bodyName, Vector3 _cForce)
	{
		_forceOnHandsFeet[masterContextID].setAvgForce(_bodyName, _cForce);
	}

	Vector3 getCurrentForce(StanceBodies _bodyName)
	{
		return _forceOnHandsFeet[masterContextID].getAvgForce(_bodyName);
	}

	void setCurrentCounterLetGo(StanceBodies _bodyName, int _cCounterLetGo)
	{
		_forceOnHandsFeet[masterContextID].setCounterLetGo(_bodyName, _cCounterLetGo);
	}

	void getCurrentCounterLetGo(StanceBodies _bodyName)
	{
		_forceOnHandsFeet[masterContextID].getCountLetGo(_bodyName);
	}

	virtual void unInit() = 0;

	bool using_sampling;

	VectorXf getControlMin()
	{
		return controlMin;
	}

	VectorXf getControlMax()
	{
		return controlMax;
	}

	virtual Eigen::VectorXf getBestControl(int cTimeStep = 0) = 0;

	int getStateDim()
	{
		return stateFeatures[masterContextID].size();
	}

	int getControlDim()
	{
		return controlMax.size();
	}

protected:
	Eigen::VectorXf control_init_tmp;
	Eigen::VectorXf controlMin;
	Eigen::VectorXf controlMax;
	Eigen::VectorXf controlMean;
	Eigen::VectorXf controlSd;
	Eigen::VectorXf poseMin;
	Eigen::VectorXf poseMax;
	Eigen::VectorXf defaultPose;

	bool consider_force_opt;

	std::vector<Eigen::VectorXf> stateFeatures;

	std::vector<VectorXf> posePriorSd, posePriorMean, threadControls;

	SimulationContext* mContextController;

	std::vector<mForceSequence> _forceOnHandsFeet;
	std::vector<std::vector<float>> _body_parts_control_cost;
	// same functions

	// assumes we are in the "trajectory_idx" context, and save or restore should be done after this function based on returned "physicsBroken"
	virtual bool advance_simulation_context(Eigen::VectorXf& cControl, int trajectory_idx, float& controlCost, std::vector<int>& dHoldIDs,
		bool debugPrint, bool flagSaveSlots, std::vector<BipedState>& nStates) = 0;

	// assumes you are in correct simulation context and state
	void saveBipedStateSimulationTrajectory(int trajectory_idx, int cStep, float cControlCost)
	{
		saveOdeState(states_trajectory_steps[trajectory_idx][cStep].saving_slot_state, trajectory_idx);
		states_trajectory_steps[trajectory_idx][cStep].hold_bodies_ids = mContextController->holdPosIndex[trajectory_idx];

		for (int i = 0; i < 4; i++)
		{
			int hold_id_i = mContextController->holdPosIndex[trajectory_idx][i];
			if (hold_id_i != -1)
			{
				states_trajectory_steps[trajectory_idx][cStep].hold_bodies_info[i] = mContextController->holds_body[hold_id_i];
			}
			states_trajectory_steps[trajectory_idx][cStep].forces[i] = _forceOnHandsFeet[trajectory_idx].getAvgForce(StanceBodies(sbLeftLeg + i));
			states_trajectory_steps[trajectory_idx][cStep].counter_let_go[i] = _forceOnHandsFeet[trajectory_idx].getCountLetGo(StanceBodies(sbLeftLeg + i));
			states_trajectory_steps[trajectory_idx][cStep]._body_control_cost[i] = _body_parts_control_cost[trajectory_idx][i];
		}
		states_trajectory_steps[trajectory_idx][cStep].control_cost = cControlCost;
		return;
	}

	float getLinearForceModel(SimulationContext* iContextCPBP, int bIndex, Vector3& cFi, int targetContext)
	{
		float f_max = 0.0f;
		if (iContextCPBP->getHoldBodyIDs(bIndex, targetContext) >= 0)
		{
			f_max = iContextCPBP->holds_body[iContextCPBP->getHoldBodyIDs(bIndex, targetContext)].calculateLinearForceModel(iContextCPBP, bIndex, cFi);
		}
		return f_max;
	}

	bool modelLetGoBodyPart(int targetContext, int b_i)
	{
		SimulationContext::ContactPoints mContactPoint = (SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + b_i);

		Vector3 fi = _forceOnHandsFeet[targetContext].getAvgForce(StanceBodies(StanceBodies::sbLeftLeg + b_i));
		float f_max = getLinearForceModel(mContextController, b_i, fi, targetContext);

		// considering the effect of direction of the force
		_forceOnHandsFeet[targetContext].updateCounterLetGo(StanceBodies(StanceBodies::sbLeftLeg + b_i), fi.norm() - f_max);


		int max_leg_go = this->_forceOnHandsFeet[targetContext].getMaxLetGo(); // if after 15 * 1/30 (half of second) force is out of bound then let go
		if (this->_forceOnHandsFeet[targetContext].getCountLetGo(StanceBodies(StanceBodies::sbLeftLeg + b_i)) >= max_leg_go)
		{
			mContextController->detachContactPoint(mContactPoint, targetContext);
			_forceOnHandsFeet[targetContext].makeZeroCounter(StanceBodies(StanceBodies::sbLeftLeg + b_i));
			return true;
		}
		return false;
	}

	//////////// connect or disconnect hands and feet of the climber ///////////////////
	void mConnectDisconnectContactPoint(std::vector<int>& desired_holds_ids, int targetContext, std::vector<bool>& _allowRelease)
	{
		//float min_reject_angle = (PI / 2) - (0.3f * PI);
		for (unsigned int i = 0; i < desired_holds_ids.size(); i++)
		{
			SimulationContext::ContactPoints mContactPoint = (SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i);

			if (desired_holds_ids[i] != -1)
			{
				Vector3 hold_pos_i = mContextController->getHoldPos(desired_holds_ids[i]);
				Vector3 contact_pos_i = mContextController->getEndPointPosBones(SimulationContext::ContactPoints::LeftLeg + i);
				float dis_i = (hold_pos_i - contact_pos_i).norm();

				float _connectionThreshold = 0.5f * mContextController->getHoldSize(desired_holds_ids[i]);

				if (dis_i <= _connectionThreshold)
				{
	//				if (i <= 1) // left leg and right leg
	//				{
	//					if (desired_holds_ids[0] != desired_holds_ids[1])
	//					{
	//					mContextController->attachContactPointToHold(mContactPoint, desired_holds_ids[i], targetContext);
	//					}
	//					else if (mContextController->getHoldBodyIDs(0, targetContext) != desired_holds_ids[1] && mContextController->getHoldBodyIDs(1, targetContext) != desired_holds_ids[0])
	//					{
	//						mContextController->attachContactPointToHold(mContactPoint, desired_holds_ids[i], targetContext);
	//					}
	//				}
	//				else
	//				{
					mContextController->attachContactPointToHold(mContactPoint, desired_holds_ids[i], targetContext);
	//				}
				}
				else if ((dis_i > _connectionThreshold + 0.1f || desired_holds_ids[i] != mContextController->getHoldBodyIDs(i, targetContext)) && _allowRelease[i])
				{
					mContextController->detachContactPoint(mContactPoint, targetContext);
				}
			}
			else
			{
				if (_allowRelease[i])
				{
					mContextController->detachContactPoint(mContactPoint, targetContext);
				}
			}
		}

		return;
	}

	// we want to copy from fcontext to tcontext
	bool disconnectHandsFeetJointsFromTo(int fContext, int tContext)
	{
		bool flag_set_state = false;
		for (int i = 0; i < mContextController->getHoldBodyIDsSize(); i++)
		{
			if (mContextController->getHoldBodyIDs(i, fContext) != mContextController->getHoldBodyIDs(i, tContext))
			{
				mContextController->detachContactPoint((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i), tContext);
				flag_set_state = true;
			}
		}
		return flag_set_state;
	}
	// we want to copy from fcontext to tcontext
	void connectHandsFeetJointsFromTo(int fContext, int tContext)
	{
		for (unsigned int i = 0; i < startState.hold_bodies_ids.size(); i++)
		{
			int attachHoldId = mContextController->getHoldBodyIDs(i, fContext);
			if (attachHoldId != mContextController->getHoldBodyIDs(i, tContext))
			{
				mContextController->detachContactPoint((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i), tContext);
				if (attachHoldId >= 0)
				{
					mContextController->attachContactPointToHold((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i), attachHoldId, tContext);
				}
			}
		}
		return;
	}

	bool disconnectHandsFeetJoints(int targetContext)
	{
		bool flag_set_state = false;
		for (int i = 0; i < mContextController->getHoldBodyIDsSize(); i++)
		{
			if (targetContext == ALLTHREADS)
			{
				for (int c = 0; c < mContextController->maxNumContexts; c++)
				{
					if (startState.hold_bodies_ids[i] != mContextController->getHoldBodyIDs(i, c))
					{
						mContextController->detachContactPoint((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i), c);
						flag_set_state = true;
					}
				}
			}
			else
			{
				if (startState.hold_bodies_ids[i] != mContextController->getHoldBodyIDs(i, targetContext))
				{
					mContextController->detachContactPoint((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i), targetContext);
					flag_set_state = true;
				}
			}
		}
		return flag_set_state;
	}

	void connectHandsFeetJoints(int targetContext)
	{
		for (unsigned int i = 0; i < startState.hold_bodies_ids.size(); i++)
		{
			if (targetContext == ALLTHREADS)
			{
				for (int c = 0; c < mContextController->maxNumContexts; c++)
				{
					if (startState.hold_bodies_ids[i] != mContextController->getHoldBodyIDs(i, c))
					{
						mContextController->detachContactPoint((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i), c);
						if (startState.hold_bodies_ids[i] >= 0)
						{
							mContextController->attachContactPointToHold((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i), startState.hold_bodies_ids[i], c);
						}
					}
				}
			}
			else
			{
				if (startState.hold_bodies_ids[i] != mContextController->getHoldBodyIDs(i, targetContext))
				{
					mContextController->detachContactPoint((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i), targetContext);
					if (startState.hold_bodies_ids[i] >= 0)
					{
						mContextController->attachContactPointToHold((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i), startState.hold_bodies_ids[i], targetContext);
					}
				}
			}
		}
		return;
	}

	////////////////////////////////////////////// debug visualize for climber*s state ////////////////////////////////////////////////
	void debug_visualize(int trajectory_idx)
	{
		Vector3 red = Vector3(0, 0, 255.0f) / 255.0f;
		Vector3 green = Vector3(0, 255.0f, 0) / 255.0f;
		Vector3 cyan = Vector3(255.0f, 255.0f, 0) / 255.0f;

//		for (int t = temp_nTrajectories - 1; t >= 0; t--)
//		{
		int idx = trajectory_idx;

		//Vector3 color = Vector3(0.5f, 0.5f,0.5f);
		//if (idx ==0)
		//	color=green;

		//rcSetColor(color.x(), color.y(), color.z());
		SimulationContext::drawLine(mContextController->initialPosition[idx], mContextController->resultPosition[idx]);
//		}

		return;
	}

	// assumes the correct context is resored
	void debugVisulizeForceOnHnadsFeet(int targetContext, bool printDebugInfo = true)
	{
		rcSetColor(1,0,0);
		float threshold = 50.0f;

		Vector3 fHand = _forceOnHandsFeet[targetContext].getAvgForce(StanceBodies::sbLeftLeg);//startState.forces[0];
		if (fHand.norm() < threshold)
			fHand = Vector3::Zero();
		if (printDebugInfo) rcPrintString("force on left leg, x:%f, y:%f, z:%f", fHand.x(), fHand.y(), fHand.z());
		Vector3 p1 = mContextController->getBonePosition(SimulationContext::BodyName::BodyLeftFoot);
		Vector3 p2 = p1;
		if (fHand.norm() != 0)
			p2 += fHand.normalized();
		mContextController->drawLine(p1, p2);

		fHand = _forceOnHandsFeet[targetContext].getAvgForce(StanceBodies::sbRightLeg);//startState.forces[1];
		if (fHand.norm() < threshold)
			fHand = Vector3::Zero();
		if (printDebugInfo) rcPrintString("force on right leg, x:%f, y:%f, z:%f", fHand.x(), fHand.y(), fHand.z());
		p1 = mContextController->getBonePosition(SimulationContext::BodyName::BodyRightFoot);
		p2 = p1;
		if (fHand.norm() != 0)
			p2 += fHand.normalized();
		mContextController->drawLine(p1, p2);

		fHand = _forceOnHandsFeet[targetContext].getAvgForce(StanceBodies::sbLeftHand);//startState.forces[2];
		if (fHand.norm() < threshold)
			fHand = Vector3::Zero();
		if (printDebugInfo) rcPrintString("force on left hand, x:%f, y:%f, z:%f", fHand.x(), fHand.y(), fHand.z());
		p1 = mContextController->getBonePosition(SimulationContext::BodyName::BodyLeftHand);
		p2 = p1;
		if (fHand.norm() != 0)
			p2 += fHand.normalized();
		mContextController->drawLine(p1, p2);

		fHand = _forceOnHandsFeet[targetContext].getAvgForce(StanceBodies::sbRightHand);//startState.forces[3];
		if (fHand.norm() < threshold)
			fHand = Vector3::Zero();
		if (printDebugInfo) rcPrintString("force on right hand, x:%f, y:%f, z:%f", fHand.x(), fHand.y(), fHand.z());
		p1 = mContextController->getBonePosition(SimulationContext::BodyName::BodyRightHand);
		p2 = p1;
		if (fHand.norm() != 0)
			p2 += fHand.normalized();
		mContextController->drawLine(p1, p2);

		return;
	}

	////////////////////////////////////////////// compute control and state costs ////////////////////////////////////////////////////
	float compute_control_cost(SimulationContext* iContextCPBP, int targetContext)
	{
		float result=0;

		for (int i = 0; i < 4; i++)
			_body_parts_control_cost[targetContext][i] = 0.0f;

		float elbowTorqueSd = 16.0f * 9.81f * iContextCPBP->boneSize[SimulationContext::BodyName::BodyLeftArm];
		float legTorqueSd = 70.0f * 9.81f * iContextCPBP->boneSize[SimulationContext::BodyName::BodyLeftThigh];

		for (int j = 0; j < iContextCPBP->getJointSize(); j++)
		{
			SimulationContext::BodyName b_i = iContextCPBP->getBodyFromJointIndex(j);

			float torqueSD = forceCostSd;
			float tourque_value = iContextCPBP->getMotorAppliedSqTorque(j);
			int body_index;
			switch (b_i)
			{
			case SimulationContext::BodyName::BodyLeftShoulder:
			case SimulationContext::BodyName::BodyRightShoulder:
				torqueSD = elbowTorqueSd;
				body_index = b_i - SimulationContext::BodyName::BodyLeftShoulder + 2;
				_body_parts_control_cost[targetContext][body_index] += tourque_value;
				break;
			case SimulationContext::BodyName::BodyLeftArm:
			case SimulationContext::BodyName::BodyRightArm:
				torqueSD = elbowTorqueSd;
				body_index = b_i - SimulationContext::BodyName::BodyLeftArm + 2;
				_body_parts_control_cost[targetContext][body_index] += tourque_value;
				break;
			case SimulationContext::BodyName::BodyLeftHand:
			case SimulationContext::BodyName::BodyRightHand:
				torqueSD = elbowTorqueSd;
				body_index = b_i - SimulationContext::BodyName::BodyLeftHand + 2;
				_body_parts_control_cost[targetContext][body_index] += tourque_value;
				break;
			case SimulationContext::BodyName::BodyTrunk:
			case SimulationContext::BodyName::BodySpine:
				break;
			case SimulationContext::BodyName::BodyLeftThigh:
			case SimulationContext::BodyName::BodyRightThigh:
				torqueSD = legTorqueSd;
				body_index = b_i - SimulationContext::BodyName::BodyLeftThigh;
				_body_parts_control_cost[targetContext][body_index] += tourque_value;
				break;
			case SimulationContext::BodyName::BodyLeftLeg:
			case SimulationContext::BodyName::BodyRightLeg:
				torqueSD = legTorqueSd;
				body_index = b_i - SimulationContext::BodyName::BodyLeftLeg;
				_body_parts_control_cost[targetContext][body_index] += tourque_value;
				break;
			case SimulationContext::BodyName::BodyLeftFoot:
			case SimulationContext::BodyName::BodyRightFoot:
				torqueSD = 100.0f * 9.81f * iContextCPBP->boneSize[SimulationContext::BodyName::BodyLeftFoot];
				body_index = b_i - SimulationContext::BodyName::BodyLeftFoot;
				_body_parts_control_cost[targetContext][body_index] += tourque_value;
				break;
			default:
				break;
			}

			result += (tourque_value)/squared(torqueSD);
		}

		float handForceSd = 70.0f * 9.81f;

		result += (iContextCPBP->getSqForceOnFingers(targetContext) / squared(handForceSd));

		//result /= (5.0f); // some normalization is needed to allow the optimization find a solution

		return result;
	}

	float computeStateCost(SimulationContext* iContextCPBP, std::vector<ControlledPoses>& sourcePos, std::vector<Vector3>& targetPos, std::vector<int>& targetHoldIDs
		,  bool printDebug, int targetContext, bool calculateOnlyForce, bool allowBodyOnGround)
	{
		// we assume that we are in the correct simulation context using restoreOdeState
		float trunkDistSd = 20.0f; 
		float angleSd = poseAngleSd;  //loose prior, seems to make more natural movement
		float tPoseAngleSd = deg2rad*5.0f;

		float endEffectorDistSd = optimizerType == otCMAES ? 0.0025f : 0.0025f;
		float velSd = optimizerType == otCMAES ? 100.0f : 0.25f;  //velSd only needed for C-PBP to reduce noise
		float chestDirSd = optimizerType == otCMAES ? 0.05f : 0.05f;

		float forceSd = 0.5f;

		float stateCost = 0;
		float preStateCost = 0;

		if (!allowBodyOnGround)
		{
			for (unsigned int k = 0; k < BodyNUM; k++)
			{
				float z_bone = iContextCPBP->getBonePosition(k).z();
				if (z_bone < 0.0f)
				{
					stateCost += squared(0.1f - z_bone) / (0.01f * 0.01f);
				}
			}

			/*float left_foot_z = iContextCPBP->getBonePosition(SimulationContext::BodyName::BodyLeftFoot).z();
			if (iContextCPBP->getHoldBodyIDs((int)(sbLeftLeg), targetContext) == -1 && left_foot_z < 0.25f)
			{
				stateCost += squared(left_foot_z - 0.25f) / (endEffectorDistSd * endEffectorDistSd);
			}
			float right_foot_z = iContextCPBP->getBonePosition(SimulationContext::BodyName::BodyRightFoot).z();
			if (iContextCPBP->getHoldBodyIDs((int)(sbRightLeg), targetContext) == -1 && right_foot_z < 0.25f)
			{
				stateCost += squared(right_foot_z - 0.25f) / (endEffectorDistSd * endEffectorDistSd);
			}*/
		}

		if (consider_force_opt)
		{
			// this part + letting of the body part makes the cost function so ill-mannered, so CMA-ES cannot find any good solution!
			for (unsigned int i = 0; i < iContextCPBP->holdPosIndex[targetContext].size(); i++)
			{
				Vector3 fi = _forceOnHandsFeet[targetContext].getAvgForce((StanceBodies)(StanceBodies::sbLeftLeg + i));
				int hold_id = iContextCPBP->holdPosIndex[targetContext][i];
				if (hold_id != -1)
				{
					float dis = iContextCPBP->holds_body[hold_id].disFromIdeal(fi);
					stateCost += squared(dis / forceSd);
				}
				else
				{
					stateCost += 4 * 4;
				}
				//float f_max = getLinearForceModel(iContextCPBP, i, fi, targetContext);
				//int _hold_id_i = iContextCPBP->getHoldBodyIDs(i, targetContext);
				//stateCost += squared(fi.norm() / (forceSd * (f_max + 0.1f)));

				//			if (_hold_id_i != targetHoldIDs[i]) // cost will be added for reaching to target, but should not move unless stable
				//			{
				////				if (_hold_id_i >= 0)
				////				{
				////					// it will change to a non-linear function (like a k-nearest directions)
				////					// if it is non-linear dir finding targetDir could be costly, better removed from cost function
				////					Vector3 targetDir = iContextCPBP->holds_body[_hold_id_i].getDIdealBodyPart(iContextCPBP, i, fi); 
				////					stateCost += squared(fi.norm() / (forceSd * f_max + 20.0f - 3 * diffAngle(targetDir,fi)));
				////				}
				//			}
			}
		}
		if (calculateOnlyForce)
		{
			return stateCost;
		}

		/*int num_connected = 0;
		for (unsigned int i = 0; i < targetHoldIDs.size(); i++)
		{
			if (iContextCPBP->holdPosIndex[targetContext][i] != -1)
			{
				num_connected++;
			}
		}*/

		/*if (printDebug)
		{
			rcPrintString("Force cost: %f", stateCost - preStateCost);
			preStateCost = stateCost;
		}*/

		////////////////////////////////////////////// Breaking Ode
		if (iContextCPBP->checkViolatingRelativeDis())
		{
			stateCost += 1e20;
		}

		if (printDebug)
		{
			rcPrintString("Distance violation cost %f",stateCost - preStateCost);
			preStateCost = stateCost;
		}

		/////////////////////////////////////////////// Velocity

		for (unsigned int k = 0; k < BodyNUM; k++)
		{
			stateCost += iContextCPBP->getBoneLinearVelocity(k).squaredNorm() / (velSd*velSd);
		}

		if (printDebug)
		{
			rcPrintString("Velocity cost %f",stateCost - preStateCost);
			preStateCost = stateCost;
		}

		///////////////////////////////////////////// Posture
		Vector3 lFootPos = iContextCPBP->getEndPointPosBones(SimulationContext::BodyName::BodyLeftFoot);
		Vector3 rFootPos = iContextCPBP->getEndPointPosBones(SimulationContext::BodyName::BodyRightFoot);

		//if (lFootPos.z() < 0.1f && rFootPos.z() < 0.1f)
		//{
		//	if (optimizerType == otCMAES)  //in C-PBP, pose included in the sampling proposal. 
		//	{
		//		for (unsigned int k = 0; k < iContextCPBP->bodyIDs.size(); k++)
		//		{
		//			if (k != SimulationContext::BodyLeftShoulder
		//				&& k != SimulationContext::BodyLeftArm
		//				&& k != SimulationContext::BodyLeftHand
		//				&& k != SimulationContext::BodyRightShoulder
		//				&& k != SimulationContext::BodyRightArm
		//				&& k != SimulationContext::BodyRightHand)
		//			{
		//				Eigen::Quaternionf q = ode2eigenq(odeBodyGetQuaternion(iContextCPBP->bodyIDs[k]));
		//				float diff = q.angularDistance(initialRotations[k]);
		//				stateCost += squared(diff) / squared(tPoseAngleSd);
		//			}
		//		}
		//	}
		//}
		//else
		//{
		if (optimizerType == otCMAES)  //in C-PBP, pose included in the sampling proposal. 
		{
			for (int k = 0; k < iContextCPBP->getJointSize(); k++)
			{
				float diff_ang = iContextCPBP->getDesMotorAngleFromID(k) - iContextCPBP->getJointAngle(k);

				stateCost += (squared(diff_ang) / (angleSd*angleSd));
			}
		}
		//} 

		if (printDebug)
		{
			rcPrintString("Pose cost %f",stateCost - preStateCost);
			preStateCost = stateCost;
		}

		Vector3 posTrunk = iContextCPBP->getBonePosition(SimulationContext::BodyName::BodyTrunk);
		Vector3 minGeomPos = posTrunk;
		if (DemoID == mDemoTestClimber::DemoPillar)
		{
			float minDis = FLT_MAX;
			for (int w = 0; w < iContextCPBP->startingHoldGeomsIndex; w++)
			{
				Vector3 posPillar = iContextCPBP->getGeomPosition(w);
				posPillar[2] = posTrunk[2];
				float cDis = (posPillar - posTrunk).norm();
				if (cDis < minDis)
				{
					minDis = cDis;
					minGeomPos = posPillar;
				}
			}
		}

		bool isTorsoDirAdded = false;
		for (unsigned int i = 0; i < sourcePos.size(); i++)
		{
			ControlledPoses posID = sourcePos[i];
			Vector3 dPos = targetPos[i];

			float cWeight = 1.0f;

			Vector3 cPos(0.0f, 0.0f, 0.0f);

			if ((int)posID <= ControlledPoses::RightHand && (int)posID >= ControlledPoses::LeftLeg)
			{
				cPos = iContextCPBP->getEndPointPosBones(SimulationContext::ContactPoints::LeftLeg + posID - ControlledPoses::LeftLeg);
				cWeight = 1.0f / endEffectorDistSd;

				int cHoldID = iContextCPBP->getHoldBodyIDs((int)(posID - ControlledPoses::LeftLeg), targetContext);
				if (cHoldID == targetHoldIDs[(int)(posID - ControlledPoses::LeftLeg)])
				{
					cWeight = 0;
				}
			}
			else if (posID == ControlledPoses::MiddleTrunk)
			{
				cPos = iContextCPBP->computeCOM();
				dPos[0] = cPos[0];//only distance from wall matters
				dPos[2] = cPos[2];

				if (DemoID == mDemoTestClimber::DemoPillar)
				{
					dPos = minGeomPos;
					dPos[2] = cPos[2];
				}

				cWeight = 1 / trunkDistSd; //weight_average_important;
			}
			else if (posID == ControlledPoses::TorsoDir)
			{
				// user defined direction
				Vector3 dirTrunk = iContextCPBP->getBodyDirectionY(SimulationContext::BodyName::BodyTrunk);
				cWeight = 1 / chestDirSd;
				cPos = dirTrunk;
				
				if (DemoID == mDemoTestClimber::DemoPillar)
				{
					dPos = (minGeomPos - posTrunk).normalized();
				}

				isTorsoDirAdded = true;
			}

			stateCost += (cWeight * (cPos - dPos)).squaredNorm();

			if (printDebug)
			{
				if ((int)posID == ControlledPoses::LeftLeg)
				{
					rcPrintString("Left leg cost %f", stateCost - preStateCost);
				}
				else if ((int)posID == ControlledPoses::RightLeg)
				{
					rcPrintString("Right leg cost %f", stateCost - preStateCost);
				}
				else if ((int)posID == ControlledPoses::LeftHand)
				{
					rcPrintString("Left hand cost %f", stateCost - preStateCost);
				}
				else if ((int)posID == ControlledPoses::RightHand)
				{
					rcPrintString("Right hand cost %f", stateCost - preStateCost);
				}
				else if ((int)posID == ControlledPoses::MiddleTrunk)
				{
					rcPrintString("Torso pos cost %f", stateCost - preStateCost);
				}
				else if ((int)posID == ControlledPoses::TorsoDir)
				{
					rcPrintString("Chest direction cost %f", stateCost - preStateCost);
				}
				preStateCost = stateCost;
			}
		}

		// some default direction for torso
		if (!isTorsoDirAdded)
		{
			Vector3 dirTrunk = iContextCPBP->getBodyDirectionY(SimulationContext::BodyName::BodyTrunk);
			Vector3 desDir(0, 1, 0);
			if (DemoID == mDemoTestClimber::DemoPillar)
			{
				desDir = (minGeomPos - posTrunk).normalized();
			}

			stateCost += ((dirTrunk - desDir)/chestDirSd).squaredNorm(); // chest toward the wall
			if (printDebug)
			{
				rcPrintString("Chest direction cost %f",stateCost - preStateCost);
				preStateCost = stateCost;
			}
		}

		if (stateCost != stateCost)
		{
			stateCost = 1e20;
		}

		return stateCost;
	} 

	Vector3 getMidPoint(std::vector<int>& _HoldIDs, int& count)
	{
		Vector3 midPoint(0, 0, 0);
		count = 0;
		for (unsigned int i = 0; i < _HoldIDs.size(); i++)
		{
			if (_HoldIDs[i] != -1)
			{
				Vector3 hold_pos = mContextController->getHoldPos(_HoldIDs[i]);
				midPoint += hold_pos;
				count++;
			}
		}

		if (count != 0)
		{
			midPoint /= (float)count;
		}

		return midPoint;
	}

	Vector3 getToDirection(Vector3 _fromPoint, std::vector<int>& targetHoldIDs)
	{
		Vector3 _dir(0.0f, 0.0f, 1.0f);
		int _count = 0;
		Vector3 midTragetPoint = getMidPoint(targetHoldIDs, _count);

		if (_count != 0)
		{
			_dir = (midTragetPoint - _fromPoint).normalized();
		}

		return _dir;
	}

	/////////////////////////////////// computing feature for climber's state /////////////////////////////////////////////////////////
	static void pushStateFeature(int &featureIdx, float *stateFeatures, const Vector3& v)
	{
		stateFeatures[featureIdx++] = v.x();
		stateFeatures[featureIdx++] = v.y();
		stateFeatures[featureIdx++] = v.z();
	}

	static void pushStateFeature(int &featureIdx, float *stateFeatures, const Vector4& v)
	{
		stateFeatures[featureIdx++] = v.x();
		stateFeatures[featureIdx++] = v.y();
		stateFeatures[featureIdx++] = v.z();
		stateFeatures[featureIdx++] = v.w();
	}

	static void pushStateFeature(int &featureIdx, float *stateFeatures, const float& f)
	{
		stateFeatures[featureIdx++] = f;
	}

	virtual int computeStateFeatures(SimulationContext* iContextCPBP, float *stateFeatures, int targetContext) = 0;
} *mOptimizer;

// Original C-PBP (both Online and Offline) used to do the transition between stance to stance in the stance graph
class mOptCPBP : public mController
{
private:
	class CPBPTrajectoryResults
	{
	public:
		std::vector<VectorXf> control;
		int nSteps;
		float cost;
	};

	ControlPBP flc;
	Eigen::VectorXf controlDiffSd;
	Eigen::VectorXf controlDiffDiffSd;
	int currentIndexOfflineState;
	bool isOnlineOpt;

	float bestCostOffLineCPBP;
	CPBPTrajectoryResults CPBPOfflineControls;
public:
	int control_size;
	mOptCPBP(SimulationContext* iContexts, BipedState& iStartState)
	{
		consider_force_opt = false;

		int maxTimeSteps = nTimeSteps + 10;
		states_trajectory_steps = std::vector<std::vector<BipedState>>(nTrajectories + 1, std::vector<BipedState>(maxTimeSteps, iStartState));

		for (int i = 0; i <= nTrajectories; i++)
		{
			for (int j = 0; j < maxTimeSteps; j++)
			{
				states_trajectory_steps[i][j] = iStartState.getNewCopy(iContexts->getNextFreeSavingSlot(), iContexts->getMasterContextID());
			}
		}

		startState = iStartState;
		iContexts->saveContextIn(startState);

		startState.getNewCopy(iContexts->getNextFreeSavingSlot(), iContexts->getMasterContextID());

		current_cost_state = 0.0f;
		current_cost_control = 0.0f;

		mContextController = iContexts;
		masterContextID = mContextController->getMasterContextID();

		control_size = mContextController->getJointSize() + fmCount + 5 + 1;

		control_init_tmp = Eigen::VectorXf::Zero(control_size);
		//controller init (motor target velocities are the controlled variables, one per joint)
		controlMin = control_init_tmp;
		controlMax = control_init_tmp;
		controlMean = control_init_tmp;
		controlSd = control_init_tmp;
		controlDiffSd = control_init_tmp;
		controlDiffDiffSd = control_init_tmp;
		poseMin=control_init_tmp;
		poseMax=control_init_tmp;
		defaultPose=control_init_tmp;

		for (int i = 0; i < mContextController->getJointSize(); i++)
		{
			//we are making everything relative to the rotation ranges. getBodyi() could also return the controlled body (corresponding to a BodyName enum)
			defaultPose[i]=iContexts->getDesMotorAngleFromID(i);
			poseMin[i]=iContexts->getJointAngleMin(i);
			poseMax[i]=iContexts->getJointAngleMax(i);
			float angleRange=poseMax[i]-poseMin[i];
			controlMin[i] = -maxSpeedRelToRange*angleRange;
			controlMax[i] = maxSpeedRelToRange*angleRange;

			controlMean[i] = 0;
			controlSd[i] = 0.5f * controlMax[i];
			controlDiffSd[i] = controlDiffSdScale * controlMax[i];		//favor small accelerations
			controlDiffDiffSd[i] = 1000.0f; //NOP
		}
		//fmax control
		for (int i = mContextController->getJointSize(); i < mContextController->getJointSize()+fmCount; i++)
		{
			//we are making everything relative to the rotation ranges. getBodyi() could also return the controlled body (corresponding to a BodyName enum)
			controlMin[i] = minimumForce;
			controlMax[i] = maximumForce;

			controlMean[i] = 0;
			controlSd[i] = controlMax[i];
			controlDiffSd[i] = controlDiffSdScale * controlMax[i];		//favor small accelerations
			controlDiffDiffSd[i] = 1000.0f; //NOP
		}
		//letting go control
		for (int i = mContextController->getJointSize()+fmCount; i < control_size; i++)
		{
			controlMin[i] = 0;
			controlMax[i] = 1.0f;

			controlMean[i] = (controlMin[i] + controlMax[i]) / 2.0f;
			controlSd[i] = controlMax[i];
			controlDiffSd[i] = useOfflinePlanning ? 0.2f * controlDiffSdScale * controlMax[i] : 0.6f * controlDiffSdScale * controlMax[i];		//favor small accelerations
			controlDiffDiffSd[i] = 1000.0f; //NOP
		}

		setCurrentOdeContext(masterContextID);

		float temp[1000];
		int stateDim = computeStateFeatures(mContextController, temp, masterContextID); // compute state features of startState or loaded master state
		Eigen::VectorXf stateSd(stateDim);
		for (int i = 0; i < stateDim; i++)
			stateSd[i] = 0.25f;
		float control_variation = 0.1f;

		flc.init(nTrajectories, nTimeSteps / nPhysicsPerStep, stateDim, control_size, controlMin.data()
			, controlMax.data(), controlMean.data(), controlSd.data(), controlDiffSd.data(), controlDiffDiffSd.data(), control_variation, NULL);
		flc.setParams(0.25f, 0.5f, false, 0.001f);

		int nPoseParams = mContextController->getJointSize();
		for (int i = 0; i <= nTrajectories; i++)
		{
			stateFeatures.push_back(Eigen::VectorXf::Zero(stateDim));
		}

		posePriorSd = std::vector<VectorXf>(nTrajectories, VectorXf(control_size));
		posePriorMean = std::vector<VectorXf>(nTrajectories, VectorXf(control_size));
		threadControls = std::vector<VectorXf>(nTrajectories, VectorXf(control_size));

		currentIndexOfflineState = 0;
		isOnlineOpt = !useOfflinePlanning;

		bestCostOffLineCPBP = FLT_MAX;
		CPBPOfflineControls.control = std::vector<VectorXf>(int(nTimeSteps / nPhysicsPerStep), VectorXf(control_size));
	}

	void reset()
	{
		flc.reset();

		bestCostOffLineCPBP = FLT_MAX;
		return;
	}

	void optimize_the_cost(bool advance_time, std::vector<ControlledPoses>& sourcePos, std::vector<Vector3>& targetPos, std::vector<int>& targetHoldIDs, bool showDebugInfo, bool allowOnGround)
	{
		//Vector3 midPoint(0, 0, 0);
		//int _count = 0;
		//for (unsigned int i = 0; i < targetHoldIDs.size(); i++)
		//{
		//	if (targetHoldIDs[i] != -1) // && iContextCPBP->getHoldBodyIDs(i, targetContext) != targetHoldIDs[i]
		//	{
		//		Vector3 hold_pos = mContextController->getHoldPos(targetHoldIDs[i]);
		//		midPoint += hold_pos;
		//		_count++;
		//	}
		//}

		setCurrentOdeContext(masterContextID);

		int cContext = getCurrentOdeContext();

		restoreOdeState(masterContextID); // we have loaded master context state

		//Update the current state and pass it to the optimizer
		Eigen::VectorXf &stateFeatureMaster = stateFeatures[masterContextID];
		computeStateFeatures(mContextController, &stateFeatureMaster[0], masterContextID);

		if (advance_time)
		{
			//debug-print current state cost components
			float cStateCost = computeStateCost(mContextController, sourcePos, targetPos, targetHoldIDs, showDebugInfo, masterContextID, false, allowOnGround); // true
			rcPrintString("Traj. cost for controller: %f", cStateCost);

			if (showDebugInfo) debugVisulizeForceOnHnadsFeet(masterContextID);
		}

		flc.startIteration(advance_time, &stateFeatureMaster[0]);
		bool standing = mContextController->getHoldBodyIDs((int)(sbLeftLeg), masterContextID)==-1 && mContextController->getHoldBodyIDs((int)(sbRightLeg), masterContextID)==-1;

		std::vector<int> nSteps(nTrajectories, 0);
		std::vector<std::vector<int>> bTrajectoryEachStep(nTimeSteps/nPhysicsPerStep, std::vector<int>(nTrajectories,-1));

		for (int step = 0; step < nTimeSteps/nPhysicsPerStep; step++)
		{
			flc.startPlanningStep(step);

			for (int i = 0; i < nTrajectories; i++)
			{
				if (step == 0)
				{
					//save the physics state: at first step, the master context is copied to every other context
					saveOdeState(i, masterContextID);

					_forceOnHandsFeet[i] = _forceOnHandsFeet[masterContextID];

				}
				else
				{
					//at others than the first step, just save each context so that the resampling can branch the paths
					saveOdeState(i, i);
					bTrajectoryEachStep[step][i] = flc.getPreviousSampleIdx(i);

					_forceOnHandsFeet[i] = _forceOnHandsFeet[flc.getPreviousSampleIdx(i)];
				}
			}

			std::deque<std::future<bool>> worker_queue;
			SimulationContext::BodyName targetDrawnLines = SimulationContext::BodyName::BodyTrunk;
			std::vector<BipedState> nStates;
			for (int t = nTrajectories - 1; t >= 0; t--)
			{
				//lambda to be executed in the thread of the simulation context
				auto simulate_one_step = [&](int trajectory_idx)
				{
					int previousStateIdx = flc.getPreviousSampleIdx(trajectory_idx);
					setCurrentOdeContext(trajectory_idx);

					int cContext = getCurrentOdeContext();

					disconnectHandsFeetJointsFromTo(previousStateIdx, trajectory_idx);
					restoreOdeState(previousStateIdx);
					connectHandsFeetJointsFromTo(previousStateIdx, trajectory_idx);

					//compute pose prior, needed for getControl()
					int nPoseParams = mContextController->getJointSize();

					float dt=timeStep*(float)nPhysicsPerStep;
					posePriorMean[trajectory_idx].setZero();
					if (!standing)
					{
						for (int i = 0; i < nPoseParams; i++)
						{
							posePriorMean[trajectory_idx][i] = (defaultPose[i]-mContextController->getJointAngle(i))/dt;
						}
					}
					else
					{
						for (int i = 0; i < nPoseParams; i++)
						{
							posePriorMean[trajectory_idx][i] = (0-mContextController->getJointAngle(i))/dt; //t-pose: zero angles
						}
					}
					posePriorSd[trajectory_idx].head(nPoseParams) = (poseMax.head(nPoseParams)-poseMin.head(nPoseParams))*(poseAngleSd/dt);
					posePriorSd[trajectory_idx].tail(control_size - nPoseParams).setConstant(1000.0f); //no additional prior on FMax //
					Eigen::VectorXf &control = threadControls[trajectory_idx];

					flc.getControl(trajectory_idx, control.data(),posePriorMean[trajectory_idx].data(),posePriorSd[trajectory_idx].data());

					std::vector<ControlledPoses> _lsourcePos = sourcePos;
					std::vector<Vector3> _ltargetPos = targetPos;

				//	if (_count != 0)
				//	{
				//		midPoint /= (float)_count;
				//		Vector3 dVelocity = (midPoint - mContextController->getBonePosition(SimulationContext::BodyName::BodyTrunk)).normalized();

				//		_lsourcePos.push_back(ControlledPoses::LeanBack);
				////		if (control[control.size() - 1] > 0.5f)
				////			_ltargetPos.push_back(-dVelocity);
				////		else
				//		_ltargetPos.push_back(dVelocity);
				//	}

					//step physics
					mContextController->initialPosition[trajectory_idx] = mContextController->getEndPointPosBones(targetDrawnLines);

					//apply the random control and step forward and evaluate control cost 
					float controlCost = 0.0f;
					float stateCost = 0;
					bool physicsBroken = false;
					for (int k = 0; k < nPhysicsPerStep && !physicsBroken; k++)
					{
						float cControlCost = 0.0f;
						physicsBroken = !advance_simulation_context(control, trajectory_idx, cControlCost, targetHoldIDs, false, false, nStates);
						controlCost += cControlCost;

						saveBipedStateSimulationTrajectory(trajectory_idx, nSteps[trajectory_idx], cControlCost);

						nSteps[trajectory_idx] = nSteps[trajectory_idx] + 1;

						if (physicsBroken)
						{
							stateCost = 1e20;
							disconnectHandsFeetJointsFromTo(previousStateIdx, trajectory_idx);
							restoreOdeState(previousStateIdx);
							connectHandsFeetJointsFromTo(previousStateIdx, trajectory_idx);
						}
						else
						{
							//compute state cost, only including the hold costs at the last step
							stateCost += computeStateCost(mContextController, _lsourcePos, _ltargetPos, targetHoldIDs, false, trajectory_idx, false, allowOnGround) / nPhysicsPerStep;
						}
					}

					mContextController->resultPosition[trajectory_idx] = mContextController->getEndPointPosBones(targetDrawnLines);

					Eigen::VectorXf &stateFeatureOthers = stateFeatures[trajectory_idx];
					computeStateFeatures(mContextController, stateFeatureOthers.data(), trajectory_idx);
					//we can add control cost to state cost, as state cost is actually uniquely associated with a tuple [previous state, control, next state]
					flc.updateResults(trajectory_idx, control.data(), stateFeatureOthers.data(), stateCost+controlCost,posePriorMean[trajectory_idx].data(),posePriorSd[trajectory_idx].data());

					return true;
				};

				worker_queue.push_back(std::async(std::launch::async,simulate_one_step,t));
			}

			for (std::future<bool>& is_ready : worker_queue)
			{
				is_ready.wait();
			}

			flc.endPlanningStep(step);

			//debug visualization
			debug_visualize(step);
		}
		flc.endIteration();
		cContext = getCurrentOdeContext();

		current_cost_state = flc.getBestTrajectoryCost();

		// for offline mode
		if (!advance_time)
		{
			current_cost_state = flc.getBestTrajectoryCost();
			if (current_cost_state < bestCostOffLineCPBP)
			{
				bestCostOffLineCPBP = current_cost_state;
				CPBPOfflineControls.cost = bestCostOffLineCPBP;


				int cBestIdx = flc.getBestSampleLastIdx();

				CPBPOfflineControls.nSteps = nSteps[cBestIdx];

				int cStep = CPBPOfflineControls.nSteps - 1;
				for (int step = nTimeSteps/nPhysicsPerStep - 1; step >= 0; step--)
				{
					setCurrentOdeContext(cBestIdx);
					for (int n = 0; n < nPhysicsPerStep; n++)
					{
						if (cStep >= 0)
						{
							restoreOdeState(states_trajectory_steps[cBestIdx][cStep].saving_slot_state);
							saveOdeState(states_trajectory_steps[nTrajectories][cStep].saving_slot_state, cBestIdx);

							states_trajectory_steps[nTrajectories][cStep].control_cost = states_trajectory_steps[cBestIdx][cStep].control_cost;
							states_trajectory_steps[nTrajectories][cStep].counter_let_go = states_trajectory_steps[cBestIdx][cStep].counter_let_go;
							states_trajectory_steps[nTrajectories][cStep].forces = states_trajectory_steps[cBestIdx][cStep].forces;
							states_trajectory_steps[nTrajectories][cStep].hold_bodies_ids = states_trajectory_steps[cBestIdx][cStep].hold_bodies_ids;
							states_trajectory_steps[nTrajectories][cStep].hold_bodies_info = states_trajectory_steps[cBestIdx][cStep].hold_bodies_info;
						}
						cStep--;
					}
					cBestIdx = bTrajectoryEachStep[step][cBestIdx];
				}

				for (int step = 0; step < nTimeSteps/nPhysicsPerStep; step++)
				{
					Eigen::VectorXf control = control_init_tmp;
					flc.getBestControl(step, control.data());
					CPBPOfflineControls.control[step] = control;
				}
			}

			//visualize
			setCurrentOdeContext(flc.getBestSampleLastIdx());
			mContextController->mDrawStuff(-1,-1,flc.getBestSampleLastIdx(),true,false);
			float stateCost = computeStateCost(mContextController, sourcePos, targetPos, targetHoldIDs, showDebugInfo, flc.getBestSampleLastIdx(), false, allowOnGround);
			rcPrintString("Traj. cost for controller: %f", current_cost_state);
			if (showDebugInfo) debugVisulizeForceOnHnadsFeet(flc.getBestSampleLastIdx());

			setCurrentOdeContext(masterContextID);
		}

		return;
	}

	// return true if simulation of the best trajectory is done from zero until maxTimeSteps otherwise false
	bool simulateBestTrajectory(bool flagSaveSlots, std::vector<int>& dHoldIDs, std::vector<outSavedData>& outStates)
	{
		if (isOnlineOpt)
		{
			syncMasterContextWithStartState(true);

			setCurrentOdeContext(masterContextID);
			restoreOdeState(masterContextID);

			current_cost_control = 0.0f;
			bool physicsBroken = false;

			for (int k = 0; k < nPhysicsPerStep && !physicsBroken; k++)
			{
				std::vector<BipedState> nStates;
				float cControl = 0.0f;
				physicsBroken = !advance_simulation_context(getBestControl(0), // current best control
					masterContextID, // apply to master context
					cControl, // out current cost
					dHoldIDs, // desired hold ids
					false, // show debud info
					flagSaveSlots, // flag for saving immediate states
					nStates); // output states

		//		apply_connection(getBestControl(0), masterContextID, dHoldIDs);

				current_cost_control += cControl;

				mContextController->saveContextIn(startState);

				saveOdeState(masterContextID, masterContextID);

				for (unsigned int i = 0; i < nStates.size(); i++)
				{
					outStates.push_back(nStates[i]);
				}
			}

			// start state is alwayse updated after something happens to master context
			startState.hold_bodies_ids = mContextController->holdPosIndex[masterContextID];
			for (int i = 0; i < 4; i++)
			{
				if (startState.hold_bodies_ids[i] != -1)
				{
					startState.hold_bodies_info[i] = mContextController->holds_body[startState.hold_bodies_ids[i]];
				}
				startState.forces[i] = _forceOnHandsFeet[masterContextID].getAvgForce(StanceBodies(StanceBodies::sbLeftLeg + i));
				startState.counter_let_go[i] = _forceOnHandsFeet[masterContextID].getCountLetGo(StanceBodies(StanceBodies::sbLeftLeg + i));
			}
			startState.control_cost = current_cost_control;

			if (!physicsBroken)
			{
				return true;
			}

			restoreOdeState(masterContextID);
			return false;
		}
		else
		{
			setCurrentOdeContext(masterContextID);

			int maxTimeSteps = CPBPOfflineControls.nSteps;
			if (currentIndexOfflineState < maxTimeSteps)
			{
				startState = states_trajectory_steps[nTrajectories][currentIndexOfflineState];

				syncMasterContextWithStartState(true);

				saveOdeState(masterContextID, masterContextID);

				if (flagSaveSlots)
				{
					BipedState nState = this->startState.getNewCopy(mContextController->getNextFreeSavingSlot(), masterContextID);
					outStates.push_back(nState);
				}

				currentIndexOfflineState++;

				return false;
			}
			else
			{
				saveOdeState(masterContextID, masterContextID);

				currentIndexOfflineState = 0;

				return true;
			}
		}
	}

	// sync all contexts with the beginning state of the optimization
	void syncMasterContextWithStartState(bool loadAnyWay)
	{
		int cOdeContext = getCurrentOdeContext();

		setCurrentOdeContext(ALLTHREADS);

		bool flag_set_state = disconnectHandsFeetJoints(ALLTHREADS);

		bool flag_is_state_sync = false;
		// startState.saving_slot_state determines the current state that we want to be in
		if (masterContextID != startState.saving_slot_state || loadAnyWay)
		{
			restoreOdeState(startState.saving_slot_state, false);

			for (int i = 0; i < 4; i++)
			{
				setCurrentForce((StanceBodies)(StanceBodies::sbLeftLeg + i), startState.forces[i]);
				setCurrentCounterLetGo((StanceBodies)(StanceBodies::sbLeftLeg + i), startState.counter_let_go[i]);
			}

			saveOdeState(masterContextID,0);
			mContextController->saveContextIn(startState);
			startState.saving_slot_state = masterContextID;
			flag_is_state_sync = true;
		}

		if (flag_set_state && !flag_is_state_sync)
		{
			restoreOdeState(masterContextID, false);

			for (int i = 0; i < 4; i++)
			{
				setCurrentForce((StanceBodies)(StanceBodies::sbLeftLeg + i), startState.forces[i]);
				setCurrentCounterLetGo((StanceBodies)(StanceBodies::sbLeftLeg + i), startState.counter_let_go[i]);
			}

			saveOdeState(masterContextID,0);
			mContextController->saveContextIn(startState);
			flag_is_state_sync = true;
		}

		connectHandsFeetJoints(ALLTHREADS);

		setCurrentOdeContext(cOdeContext);

		return;
	}

	void unInit() 
	{
		return;
	}

	Eigen::VectorXf getBestControl(int cTimeStep)
	{
		if (isOnlineOpt)
		{
			Eigen::VectorXf control = control_init_tmp;
			flc.getBestControl(cTimeStep, control.data());
			return control;
		}
		else
		{
			return CPBPOfflineControls.control[cTimeStep];
		}
	}

private:

	int computeStateFeatures(SimulationContext* iContextCPBP, float *stateFeatures, int targetContext) // BipedState state
	{

		int featureIdx = 0;
		const int nStateBones=6;
		SimulationContext::BodyName stateBones[6]={
			SimulationContext::BodySpine,
			SimulationContext::BodyTrunk,
			SimulationContext::BodyRightArm,
			SimulationContext::BodyRightLeg,
			SimulationContext::BodyLeftArm,
			SimulationContext::BodyLeftLeg};


		for (int i = 0; i < nStateBones; i++)
		{
			pushStateFeature(featureIdx, stateFeatures, iContextCPBP->getBonePosition(stateBones[i]));
			pushStateFeature(featureIdx, stateFeatures, iContextCPBP->getBoneLinearVelocity(stateBones[i]));
			pushStateFeature(featureIdx, stateFeatures, iContextCPBP->getBoneAngle(stateBones[i]));
			pushStateFeature(featureIdx, stateFeatures, iContextCPBP->getBoneAngularVelocity(stateBones[i]));
		}

		for (int i = 0; i < iContextCPBP->getHoldBodyIDsSize(); i++)
		{
			if (iContextCPBP->getHoldBodyIDs(i, targetContext) != -1)
			{
				pushStateFeature(featureIdx, stateFeatures, 1.0f);
			}
			else
			{
				pushStateFeature(featureIdx, stateFeatures, 0.0f);
			}
		}

		/*for (int i = 0; i < iContextCPBP->getHoldBodyIDsSize(); i++)
		{
		Vector3 fi = iContextCPBP->getForceVectorOnEndBody((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i), targetContext);
		if (fi.norm() > 0)
		fi = fi.normalized();
		pushStateFeature(featureIdx, stateFeatures, fi);
		}*/

		return featureIdx;
	}

	// advance one step simulation assuming we are in the correct context then apply control and step, calculate the control cost, also return next simulated state if asked
	bool advance_simulation_context(Eigen::VectorXf& cControl, int trajectory_idx, float& controlCost, std::vector<int>& dHoldIds,
		bool debugPrint, bool flagSaveSlots, std::vector<BipedState>& nStates)
	{
		bool physicsBroken = false;
	
		for (int _m = 0; _m < 4; _m++)
		{
			modelLetGoBodyPart(trajectory_idx, _m);
		}

		apply_connection(cControl, trajectory_idx, dHoldIds);
		apply_control(mContextController, cControl, trajectory_idx);

		physicsBroken = !stepOde(timeStep,false);

		if (flagSaveSlots && !physicsBroken)
		{
			BipedState nState = this->startState.getNewCopy(mContextController->getNextFreeSavingSlot(), trajectory_idx);
			for (int i = 0; i < 4; i++)
				nState.forces[i] = _forceOnHandsFeet[trajectory_idx].getAvgForce((StanceBodies)(StanceBodies::sbLeftLeg + i));

			nStates.push_back(nState);
		}

		if (!physicsBroken)
		{
			controlCost += compute_control_cost(mContextController, trajectory_idx);
		}

		if (debugPrint) rcPrintString("Control cost for controller: %f",controlCost);

		if (!physicsBroken)
		{
			return true;
		}

		return false;
	}

	void apply_control(SimulationContext* iContextCPBP, const Eigen::VectorXf& control, int trajectory_idx)
	{
		for (int j = 0; j < iContextCPBP->getJointSize(); j++)
		{
			float c_j = control[j];
			iContextCPBP->setMotorSpeed(j, c_j, nPhysicsPerStep);
		}

		if (fmCount != 0)
		{
			//fmax values for each group are after the joint motor speeds
			iContextCPBP->setMotorGroupFmaxes(&control[iContextCPBP->getJointSize()], torsoMinFMax);
		}

		for (int i = 0; i < 4; i++)
		{
			if (mContextController->getHoldBodyIDs(i, trajectory_idx) != -1)
			{
				_forceOnHandsFeet[trajectory_idx].updateForces(StanceBodies(StanceBodies::sbLeftLeg + i)
					, mContextController->getForceVectorOnEndBody((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i), trajectory_idx));
			}
			else
			{
				_forceOnHandsFeet[trajectory_idx].makeZeroForce(StanceBodies(StanceBodies::sbLeftLeg + i));
				_forceOnHandsFeet[trajectory_idx].makeZeroCounter(StanceBodies(StanceBodies::sbLeftLeg + i));
			}
		}

		return;
	}

	void apply_connection(Eigen::VectorXf& cControl, int trajectory_idx, std::vector<int>& dHoldIds)
	{
		Vector3 midPoint(0, 0, 0);
		int _count = 0;

		for (unsigned int i = 0; i < dHoldIds.size(); i++)
		{
			if (dHoldIds[i] != -1)//&& mContextController->getHoldBodyIDs(i, trajectory_idx) != dHoldIds[i]
			{
				Vector3 hold_pos = mContextController->getHoldPos(dHoldIds[i]);
				midPoint += hold_pos;
				_count++;
			}
		}

		int nPoseParams = mContextController->getJointSize();
		std::vector<bool> mAllowRelease;
		mAllowRelease.reserve(5);

		if (_count != 0)
		{
			midPoint /= (float)_count;

			Vector3 dVelocity = (midPoint - mContextController->getBonePosition(SimulationContext::BodyName::BodyTrunk)).normalized();
			Vector3 cVel = mContextController->getBoneLinearVelocity(SimulationContext::BodyName::BodyTrunk);
			float vel_dir_diff = (dVelocity - cVel.normalized()).norm();
			float vel_norm = cVel.norm();
			for (int _m = 0; _m < 5; _m++)
			{
				mAllowRelease.push_back(vel_dir_diff < 0.1f && vel_norm > cControl[nPoseParams + fmCount + _m]);//vel_mag <= cControl[nPoseParams + fmCount + _m]);
			}
			//if (vel_diff < 0.1f && cVel.norm() > 1.0f)
			//{
			//	for (int _m = 0; _m < 5; _m++)
			//	{
			//		mAllowRelease.push_back(0.5f > cControl[nPoseParams + fmCount + _m]);//vel_mag <= cControl[nPoseParams + fmCount + _m]);
			//	}
			//}
			//else
			//{
			//	for (int _m = 0; _m < 5; _m++)
			//	{
			//		mAllowRelease.push_back(false);
			//	}
			//}
		}
		else
		{
			for (int _m = 0; _m < 5; _m++)
			{
				mAllowRelease.push_back(0.5f > cControl[nPoseParams + fmCount + _m]);
			}
		}

		if (mAllowRelease.size() == 5)
		{
			if (mAllowRelease[4])
			{
				for (unsigned int i = 0; i < dHoldIds.size(); i++)
				{
					mAllowRelease[i] = true;
				}
			}
		}
		mConnectDisconnectContactPoint(dHoldIds, trajectory_idx, mAllowRelease);

		return;
	}

};

//CMAES params

static const bool cmaesLinearInterpolation = false;

static const float minSegmentDuration = 0.1f;
static const float maxSegmentDuration = 0.5f;
static const float maxFullTrajectoryDuration = 1.5f;
static const float motorPoseInterpolationTime = timeStep *2.0f;

// Original CMA-ES used to do the transition between stance to stance in the stance graph
class mOptCMAES : public mController
{
private:
	class CMAESTrajectoryResults
	{
	public:
		int nSteps;
		float state_cost;
		float control_cost;
		VectorXf controlPoints;
		bool isReached;
	};

	CMAES<float> cmaes;

	std::vector<std::vector<RecursiveTCBSpline>> splines;  //vector for each simulation context, each vector with the number of control params (joint velocities/poses and fmax)

	CMAESTrajectoryResults cmaesResults[nTrajectories];
	CMAESTrajectoryResults bestCmaesTrajectory;

	std::vector<CMAESTrajectoryResults> bestCmaesSamples;// keep best CmaesSamples of all iterations
	std::vector<int> cTargetIDHolds;

	int currentIndexOfflineState;

	static const int nCMAESSegments = 4;
	static const int id_size_control = 4;
public:
	int control_size;
	int nAngles;

	Eigen::VectorXf machineLearningInput;

	mOptCMAES(SimulationContext* iContexts, BipedState& iStartState)
	{
		consider_force_opt = false;

		int maxTimeSteps = int(maxFullTrajectoryDuration / timeStep) + 10;
		states_trajectory_steps = std::vector<std::vector<BipedState>>(nTrajectories + 1, std::vector<BipedState>(maxTimeSteps, iStartState));

		for (int i = 0; i <= nTrajectories; i++)
		{
			_forceOnHandsFeet[i].setMaxStorageForces(1);
			for (int j = 0; j < maxTimeSteps; j++)
			{
				states_trajectory_steps[i][j] = iStartState.getNewCopy(iContexts->getNextFreeSavingSlot(), iContexts->getMasterContextID());
			}
		}

		startState = iStartState;
		iContexts->saveContextIn(startState);

		startState.getNewCopy(iContexts->getNextFreeSavingSlot(), iContexts->getMasterContextID());

		current_cost_state = 0.0f;
		current_cost_control = 0.0f;

		mContextController = iContexts;
		masterContextID = mContextController->getMasterContextID();

		nAngles = mContextController->getJointSize() - 9;
		control_size = nAngles + fmCount;

		control_init_tmp = Eigen::VectorXf::Zero(control_size);
		//controller init (motor target velocities are the controlled variables, one per joint)
		controlMin = control_init_tmp;
		controlMax = control_init_tmp;
		controlMean = control_init_tmp;
		controlSd = control_init_tmp;
		poseMin = control_init_tmp;
		poseMax = control_init_tmp;
		defaultPose = control_init_tmp;

		int _count = 0;
		for (int i = 0; i < mContextController->getJointSize(); i++)
		{
			if (!((mContextController->jointIDIndex[i] + 1) == SimulationContext::BodyName::BodyHead ||
				(mContextController->jointIDIndex[i] + 1) == SimulationContext::BodyName::BodyLeftHand ||
				(mContextController->jointIDIndex[i] + 1) == SimulationContext::BodyName::BodyRightHand))
			{
				//we are making everything relative to the rotation ranges. getBodyi() could also return the controlled body (corresponding to a BodyName enum)
				defaultPose[_count] = iContexts->getDesMotorAngleFromID(i);
				poseMin[_count] = iContexts->getJointAngleMin(i);
				poseMax[_count] = iContexts->getJointAngleMax(i);
				float angleRange = poseMax[_count] - poseMin[_count];
				controlMin[_count] = -maxSpeedRelToRange*angleRange;
				controlMax[_count] = maxSpeedRelToRange*angleRange;
				controlMean[_count] = 0;
				controlSd[_count] = 0.5f * controlMax[_count];
				_count++;
			}
		}

		//fmax control
		for (int i = nAngles; i < nAngles + fmCount; i++)
		{
			//we are making everything relative to the rotation ranges. getBodyi() could also return the controlled body (corresponding to a BodyName enum)
			controlMin[i] = minimumForce;
			controlMax[i] = maximumForce;
			controlMean[i] = 0;
			controlSd[i] = controlMax[i];
		}

		setCurrentOdeContext(masterContextID);

		float temp[1000];
		int stateDim = computeStateFeatures(mContextController, temp, masterContextID); // compute state features of startState or loaded master state
		Eigen::VectorXf stateSd(stateDim);
		for (int i = 0; i < stateDim; i++)
			stateSd[i] = 0.25f;

		//for each segment, the control vector has a duration value and a control point (motor speeds and fmaxes)
		splines.resize(contextNUM);
		for (int i = 0; i<contextNUM; i++)
		{
			splines[i].resize(control_size);
		}

		for (int i = 0; i <= nTrajectories; i++)
		{
			stateFeatures.push_back(Eigen::VectorXf::Zero(stateDim));
		}

		threadControls = std::vector<VectorXf>(nTrajectories, VectorXf(control_size));

		for (int i = 0; i < nTrajectories; i++)
		{
			if (cmaesResults[i].controlPoints.size() == 0)
			{
				cmaesResults[i].controlPoints.resize(nCMAESSegments*(1 + control_size) + id_size_control);
			}
		}
		if (bestCmaesTrajectory.controlPoints.size() == 0)
		{
			bestCmaesTrajectory.controlPoints.resize(nCMAESSegments*(1 + control_size) + id_size_control);
		}

		currentIndexOfflineState = 0;
	}

	// return true if simulation of the best trajectory is done from zero until maxTimeSteps otherwise false
	bool simulateBestTrajectory(bool flagSaveSlots, std::vector<int>& dHoldIDs, std::vector<outSavedData>& outStates)
	{
		setCurrentOdeContext(masterContextID);

		int maxTimeSteps = getNSteps();
		if (currentIndexOfflineState < maxTimeSteps)
		{
			startState = states_trajectory_steps[nTrajectories][currentIndexOfflineState];

			syncMasterContextWithStartState(true);

			saveOdeState(masterContextID, masterContextID);

			if (flagSaveSlots)
			{
				BipedState nState = this->startState.getNewCopy(mContextController->getNextFreeSavingSlot(), masterContextID);
				outStates.push_back(nState);
			}

			currentIndexOfflineState++;
			return false;
		}
		else
		{
			saveOdeState(masterContextID, masterContextID);

			currentIndexOfflineState = 0;

			return true;
		}

	}

	void optimize_the_cost(bool firstIter, std::vector<ControlledPoses>& sourcePos, std::vector<Vector3>& targetPos, std::vector<int>& targetHoldIDs, bool showDebugInfo, bool allowOnGround)
	{
		if (!mTools::isSetAEqualsSetB(cTargetIDHolds, targetHoldIDs))
		{
			bestCmaesSamples.clear();
			cTargetIDHolds = targetHoldIDs;
		}

		setCurrentOdeContext(masterContextID);

		restoreOdeState(masterContextID); // we have loaded master context state

		int nValuesPerSegment = 1 + control_size;
		int nCurrentTrajectories = nTrajectories; //  firstIter ? nTrajectories : nTrajectories / 2;

		std::vector<std::pair<VectorXf, float>> nSamples;

		float min_effect_time = minSegmentDuration;
		float min_time_letgo = 0.0f;
		int max_let_go_segment = 1; //nCMAESSegments - 2;
		//at first iteration, sample the population from the control prior
		if (firstIter)
		{
			cmaes.init_with_dimension(nCMAESSegments*nValuesPerSegment + id_size_control);
			cmaes.setMean(machineLearningInput);
			//cmaes.selected_samples_fraction_ = 0.5;
			//cmaes.use_step_size_control_ = true;
			//cmaes.minimum_exploration_variance_ = 0.01f;
			nSamples.resize(nCurrentTrajectories);

			best_trajectory_cost = FLT_MAX;
			current_cost_control = FLT_MAX;
			current_cost_state = FLT_MAX;
			isReachedToTargetStance = false;

			bestCmaesTrajectory.state_cost = FLT_MAX;
			bestCmaesTrajectory.control_cost = FLT_MAX;
			bestCmaesTrajectory.isReached = false;
			bestCmaesTrajectory.nSteps = 0;

			for (int sampleIdx = 0; sampleIdx < nCurrentTrajectories; sampleIdx++)
			{
				VectorXf &sample = nSamples[sampleIdx].first;

				sample.resize(nCMAESSegments*nValuesPerSegment + id_size_control);

				if (!using_sampling && sampleIdx < nCurrentTrajectories / 2)
				{
					sample = machineLearningInput;

					if (sampleIdx == 0) // inputing exact machine learning output
						continue;

					float total_time_sample = 0.0f;
					for (int segmentIdx = 0; segmentIdx < nCMAESSegments; segmentIdx++)
					{
						int segmentStart = segmentIdx*nValuesPerSegment;

						sample[segmentStart] = sample_clipped_gaussian(sample[segmentStart], 0.1f, minSegmentDuration, maxSegmentDuration); //duration in 0.2...1 seconds
						if (segmentIdx <= max_let_go_segment)
							total_time_sample += sample[segmentStart];

						for (int velIdx = 0; velIdx < nAngles; velIdx++)
						{
							float sd = poseAngleSd / 6.0f;
							if (segmentIdx == nCMAESSegments - 1)
								sd = poseAngleSd;
							sample[segmentStart + 1 + velIdx] = sample_clipped_gaussian(sample[segmentStart + 1 + velIdx], sd, poseMin[velIdx], poseMax[velIdx]);//defaultPose[velIdx]
						}

						for (int fmaxIdx = 0; fmaxIdx < fmCount; fmaxIdx++)
						{
							//safest to start with high fmax and let the optimizer decrease them later, thus the mean at maximumForce
							sample[segmentStart + 1 + nAngles + fmaxIdx] = sample_clipped_gaussian(sample[segmentStart + 1 + nAngles + fmaxIdx], (maximumForce - minimumForce) / 10.0f, minimumForce, maximumForce);
						}
					}

					for (int id = 0; id < id_size_control; id++)
					{
						sample[nCMAESSegments*nValuesPerSegment + id] = sample_clipped_gaussian(sample[nCMAESSegments*nValuesPerSegment + id], 0.1f, min_time_letgo, total_time_sample - min_effect_time);
					}
				}
				else
				{
					float total_time_sample = 0.0f;
					for (int segmentIdx = 0; segmentIdx < nCMAESSegments; segmentIdx++)
					{
						int segmentStart = segmentIdx*nValuesPerSegment;

						sample[segmentStart] = minSegmentDuration + (maxSegmentDuration - minSegmentDuration)*randomf(); //duration in 0.2...1 seconds
						if (segmentIdx <= max_let_go_segment)
							total_time_sample += sample[segmentStart];
						for (int velIdx = 0; velIdx < nAngles; velIdx++)
						{
							sample[segmentStart + 1 + velIdx] = sample_clipped_gaussian(mContextController->getJointAngle(velIdx), poseAngleSd, poseMin[velIdx], poseMax[velIdx]);//defaultPose[velIdx]
						}

						for (int fmaxIdx = 0; fmaxIdx < fmCount; fmaxIdx++)
						{
							//safest to start with high fmax and let the optimizer decrease them later, thus the mean at maximumForce
							sample[segmentStart + 1 + nAngles + fmaxIdx] = sample_clipped_gaussian((maximumForce + minimumForce) / 2.0f, (maximumForce - minimumForce) / 2.0f, minimumForce, maximumForce);
						}
					}

					for (int id = 0; id < id_size_control; id++)
					{
						sample[nCMAESSegments*nValuesPerSegment + id] = min_time_letgo + (total_time_sample - min_effect_time - min_time_letgo) * randomf();
					}
				}
			}

		}
		//at subsequent iterations, CMAES does the sampling. Note that we clamp the samples to bounds  
		else
		{
			nSamples.resize(nCurrentTrajectories);
			auto points = cmaes.sample(nCurrentTrajectories);

			for (int sampleIdx = 0; sampleIdx<nCurrentTrajectories; sampleIdx++)
			{
				VectorXf &sample = nSamples[sampleIdx].first;

				sample = points[sampleIdx];

				//clip the values
				float total_time_sample = 0.0f;
				for (int segmentIdx = 0; segmentIdx < nCMAESSegments; segmentIdx++)
				{
					int segmentStart = segmentIdx*nValuesPerSegment;
					clampCMAESSegmentControls(segmentIdx, true, nAngles, nValuesPerSegment, sample);
					if (segmentIdx <= max_let_go_segment)
						total_time_sample += sample[segmentStart];
				}

				for (int id = 0; id < id_size_control; id++)
				{
					sample[nCMAESSegments*nValuesPerSegment + id] = clipMinMaxf(sample[nCMAESSegments*nValuesPerSegment + id], min_time_letgo, total_time_sample - min_effect_time);
				}
			}
		}

		//evaluate the samples, i.e., simulate in parallel and accumulate cost
		std::deque<std::future<void>> worker_queue;
		std::vector<BipedState> nStates;
		bool reachedBestState = false;
		for (int sampleIdx = 0; sampleIdx < nCurrentTrajectories; sampleIdx++)
		{
			auto simulate_sample = [&](int trajectory_idx)
			{
				Vector3 color = Vector3(0.5f, 0.5f, 0.5f);
				if (showDebugInfo)
				{
					rcSetColor(color.x(), color.y(), color.z());
				}
				VectorXf &sample = nSamples[trajectory_idx].first;
				float &control_cost = cmaesResults[trajectory_idx].control_cost;
				float &stateCost = cmaesResults[trajectory_idx].state_cost;

				stateCost = 0;
				control_cost = 0;
				cmaesResults[trajectory_idx].nSteps = 0;

				//restore physics state from master (all simulated trajectories start from current state)
				setCurrentOdeContext(trajectory_idx);

				disconnectHandsFeetJoints(trajectory_idx);
				restoreOdeState(masterContextID);
				connectHandsFeetJoints(trajectory_idx);

				//setup spline interpolation initial state
				std::vector<RecursiveTCBSpline> &spl = splines[trajectory_idx]; //this context's spline interpolators
				for (int i = 0; i<nAngles; i++)
				{

					spl[i].setValueAndTangent(mContextController->getJointAngle(i), mContextController->getJointAngleRate(i));
					spl[i].linearMix = cmaesLinearInterpolation ? 1 : 0;
					
				}
				for (int i = nAngles; i < nAngles + fmCount; i++)
				{
					spl[i].setValueAndTangent(mContextController->getJointFMax(i - nAngles), 0);
					spl[i].linearMix = 1;
				}

				//setup spline interpolation control points: we start at segment 0, and keep the times of the next two control points in t1 and t2
				int segmentIdx = 0;

				float t1 = sample[segmentIdx*nValuesPerSegment];

				int lastSegment = nCMAESSegments - 2;// the last control point only defines the tangent
				float totalTime = 0;
				bool physicsBroken = false;
				while (segmentIdx <= lastSegment && totalTime<maxFullTrajectoryDuration)
				{
					mContextController->initialPosition[trajectory_idx] = mContextController->getEndPointPosBones(SimulationContext::BodyName::BodyTrunk);
					//interpolate
					const VectorXf &controlPoint1 = sample.segment(segmentIdx*nValuesPerSegment + 1, control_size);
					const VectorXf &controlPoint2 = sample.segment((segmentIdx+1)*nValuesPerSegment + 1, control_size);
					float t2 = t1 + sample[(segmentIdx + 1)*nValuesPerSegment];
					
					VectorXf &interpolatedControl = threadControls[trajectory_idx];

					for (int i = 0; i < control_size; i++)
					{
						float p1 = controlPoint1[i], p2 = controlPoint2[i];
						spl[i].step(timeStep, p1, t1, p2, t2);
						interpolatedControl[i] = spl[i].getValue();
					}
					//clamp for safety
					clampCMAESSegmentControls(0, false, nAngles, nValuesPerSegment, interpolatedControl);

					//apply the interpolated control and step forward and evaluate control cost 
					float controlCost = 0.0f;
					apply_connection(sample, trajectory_idx, targetHoldIDs, totalTime);
					physicsBroken = !advance_simulation_context(interpolatedControl, trajectory_idx, controlCost, targetHoldIDs, false, false, nStates);

					//float stateCost = 0;
					if (physicsBroken)
					{
						//stateCost = 1e20;
						disconnectHandsFeetJoints(trajectory_idx);
						restoreOdeState(masterContextID);
						connectHandsFeetJoints(trajectory_idx);
					}
					else
					{
						stateCost += computeStateCost(mContextController, sourcePos, targetPos, targetHoldIDs, false, trajectory_idx, true, allowOnGround);
					}

					// this is the cost of all trajectory
					//state_cost += stateCost;
					
					if (physicsBroken)
					{
						break;
					}

					control_cost += controlCost;
					// check whether we are at the end of a segment
					t1 -= timeStep;
					if (t1 < 0)
					{
						segmentIdx++;
						t1 = t2-timeStep;
					}
					totalTime += timeStep;

					saveBipedStateSimulationTrajectory(trajectory_idx, cmaesResults[trajectory_idx].nSteps, controlCost);
					cmaesResults[trajectory_idx].nSteps++;
					
					mContextController->resultPosition[trajectory_idx] = mContextController->getEndPointPosBones(SimulationContext::BodyName::BodyTrunk);

					if (showDebugInfo) debug_visualize(trajectory_idx);

				} //for each step in segment

				if (showDebugInfo)
				{
					rcSetColor(color.x(), color.y(), color.z());
				}

				
				if (physicsBroken)
				{
					stateCost = 1e20;
				}
				else
				{
					//compute state cost, only including the hold costs at the last step
					stateCost += computeStateCost(mContextController, sourcePos, targetPos, targetHoldIDs, false, trajectory_idx, false, allowOnGround);
				}

				nSamples[trajectory_idx].second = stateCost;
				cmaesResults[trajectory_idx].controlPoints = sample;
				//cmaesResults[trajectory_idx].state_cost = stateCost;
				cmaesResults[trajectory_idx].isReached = mTools::isSetAEqualsSetB(targetHoldIDs, mContextController->holdPosIndex[trajectory_idx]);
				if (cmaesResults[trajectory_idx].isReached)
				{
					reachedBestState = true;
				}
			};  //lambda for simulating sample

			worker_queue.push_back(std::async(std::launch::async, simulate_sample, sampleIdx));


		} //for each cample

		for (std::future<void>& is_ready : worker_queue)
		{
			is_ready.wait();
		}

		//find min cost, convert costs to goodnesses through negating, and update cmaes
		float minCost = FLT_MAX;
		int bestIdx = 0;

		for (int sampleIdx = 0; sampleIdx < nCurrentTrajectories; sampleIdx++)
		{
			float reduce_cost = 0.0f;
			if (reachedBestState)
			{
				reduce_cost = cmaesResults[sampleIdx].state_cost + cmaesResults[sampleIdx].control_cost;
			}
			else
			{
				reduce_cost = cmaesResults[sampleIdx].state_cost;
			}

			if (reduce_cost < minCost)
			{
				bestIdx = sampleIdx;
				minCost = reduce_cost;
			}

			nSamples[sampleIdx].second = -1.0f * (reduce_cost);
		}

		current_cost_state = minCost;

		//Remember if this iteration produced the new best one. This is needed just in case CMAES loses it in the next iteration.
		//For example, at the end of the optimization, a reaching hand might fluctuate in and out of the hold, and the results will be rejected
		//if the hand is not reaching the hold when iteration stopped

		if ((cmaesResults[bestIdx].state_cost < bestCmaesTrajectory.state_cost && !bestCmaesTrajectory.isReached)
			|| (cmaesResults[bestIdx].isReached && minCost < best_trajectory_cost))
		{
			bestCmaesTrajectory = cmaesResults[bestIdx];
			isReachedToTargetStance = bestCmaesTrajectory.isReached;
			best_trajectory_cost = bestCmaesTrajectory.state_cost + bestCmaesTrajectory.control_cost;
			current_cost_control = cmaesResults[bestIdx].control_cost;
			for (int step = 0; step < bestCmaesTrajectory.nSteps; step++)
			{
				setCurrentOdeContext(bestIdx);
				restoreOdeState(states_trajectory_steps[bestIdx][step].saving_slot_state);
				saveOdeState(states_trajectory_steps[nTrajectories][step].saving_slot_state, bestIdx);

				states_trajectory_steps[nTrajectories][step].control_cost = states_trajectory_steps[bestIdx][step].control_cost;
				states_trajectory_steps[nTrajectories][step].counter_let_go = states_trajectory_steps[bestIdx][step].counter_let_go;
				states_trajectory_steps[nTrajectories][step].forces = states_trajectory_steps[bestIdx][step].forces;
				states_trajectory_steps[nTrajectories][step].hold_bodies_ids = states_trajectory_steps[bestIdx][step].hold_bodies_ids;
				states_trajectory_steps[nTrajectories][step].hold_bodies_info = states_trajectory_steps[bestIdx][step].hold_bodies_info;
				states_trajectory_steps[nTrajectories][step]._body_control_cost = states_trajectory_steps[bestIdx][step]._body_control_cost;
			}
		}

		auto has_higher_score_ptr = [&](const std::pair<VectorXf, float>& ptr_1, const std::pair<VectorXf, float>& ptr_2) {
			return ptr_1.second > ptr_2.second;
		};

		auto it_begin = nSamples.begin();
		auto it_end = nSamples.end();
		std::sort(it_begin, it_end, has_higher_score_ptr);

		int n_elit_samples = 0;
		if ((int)bestCmaesSamples.size() > n_elit_samples)
		{
			int s_index = (int)(nCurrentTrajectories * cmaes.selected_samples_fraction_) - n_elit_samples; // 50;//
			for (int i = 0; i < n_elit_samples; i++)
			{
				nSamples[s_index + i].first = bestCmaesSamples[i].controlPoints;
			}
		}

		if (bestCmaesSamples.size() == 0)
		{
			bestCmaesSamples.resize(n_elit_samples);
			for (int i = 0; i < n_elit_samples; i++)
			{
				bestCmaesSamples[i].controlPoints = nSamples[i].first;
			}
		}
		else
		{
			for (int i = 0; i < n_elit_samples; i++)
			{
				if (bestCmaesSamples[i].state_cost > nSamples[i].second)
				{
					bestCmaesSamples[i].controlPoints = nSamples[i].first;
				}
			}
		}

		cmaes.update(nSamples, false);

		//visualize
		setCurrentOdeContext(bestIdx);

		float stateCost = computeStateCost(mContextController, sourcePos, targetPos, targetHoldIDs, showDebugInfo, bestIdx, false, allowOnGround); // true
		
	//	if (bestCmaesTrajectory.isReached)
	//	{
		float sum_torque = 0.0001f;
		for (unsigned int i = 0; i < 4; i++)
		{
			sum_torque 
				+= states_trajectory_steps[nTrajectories][bestCmaesTrajectory.nSteps-1]._body_control_cost[i];
		}
		for (unsigned int i = 0; i < 4; i++)
		{
			mContextController->mColorBodies[i] 
				= states_trajectory_steps[nTrajectories][bestCmaesTrajectory.nSteps - 1]._body_control_cost[i] / sum_torque;
		}
	//	}
		mContextController->mDrawStuff(-1, -1, bestIdx, true, false);
		if (showDebugInfo) debugVisulizeForceOnHnadsFeet(bestIdx);

		setCurrentOdeContext(masterContextID);
		return;
	}

	// sync all contexts with the beginning state of the optimization
	void syncMasterContextWithStartState(bool loadAnyWay = true)
	{
		int cOdeContext = getCurrentOdeContext();

		setCurrentOdeContext(ALLTHREADS);

		bool flag_set_state = disconnectHandsFeetJoints(ALLTHREADS);

		restoreOdeState(startState.saving_slot_state, false);

		for (int i = 0; i < 4; i++)
		{
			setCurrentForce((StanceBodies)(StanceBodies::sbLeftLeg + i), startState.forces[i]);
			setCurrentCounterLetGo((StanceBodies)(StanceBodies::sbLeftLeg + i), startState.counter_let_go[i]);
		}

		saveOdeState(masterContextID, 0);
		mContextController->saveContextIn(startState);
		startState.saving_slot_state = masterContextID;

		float sum_torque = 0.0001f;
		for (unsigned int i = 0; i < 4; i++)
		{
			sum_torque += startState._body_control_cost[i];
		}
		for (unsigned int i = 0; i < 4; i++)
		{
			mContextController->mColorBodies[i] = startState._body_control_cost[i] / sum_torque;
		}

		connectHandsFeetJoints(ALLTHREADS);

		setCurrentOdeContext(cOdeContext);

		return;
	}

	void reset()
	{
		return;
	}

	void unInit() {}

	Eigen::VectorXf getBestControl(int cTimeStep = 0)
	{
		return bestCmaesTrajectory.controlPoints;
	}

private:

	int computeStateFeatures(SimulationContext* iContextCPBP, float *stateFeatures, int targetContext) // BipedState state
	{
		int featureIdx = 0;
		const int nStateBones = 5;
		SimulationContext::BodyName stateBones[nStateBones] =
		{
			SimulationContext::BodyTrunk,
			SimulationContext::BodyRightArm,
			SimulationContext::BodyRightLeg,
			SimulationContext::BodyLeftArm,
			SimulationContext::BodyLeftLeg
		};

		const Quaternionf root_rotation = ode2eigenq(odeBodyGetQuaternion(iContextCPBP->bodyIDs[SimulationContext::BodySpine]));
		const Quaternionf root_inverse = root_rotation.inverse();

		Vector3f root_pos(odeBodyGetPosition(iContextCPBP->bodyIDs[SimulationContext::BodySpine]));
		Vector3f root_vel(odeBodyGetLinearVel(iContextCPBP->bodyIDs[SimulationContext::BodySpine]));

		Vector3f bone_pos;
		Vector3f bone_vel;

		Vector3f pos_relative_root;
		Vector3f vel_relative_root;

		for (int i = 0; i < nStateBones; i++)
		{
			pos_relative_root.noalias() = iContextCPBP->getBonePosition(stateBones[i]) - root_pos;
			vel_relative_root.noalias() = iContextCPBP->getBoneLinearVelocity(stateBones[i]) - root_vel;

			bone_pos.noalias() = root_inverse*pos_relative_root;
			bone_vel.noalias() = root_inverse*vel_relative_root;

			pushStateFeature(featureIdx, stateFeatures, bone_pos);
			pushStateFeature(featureIdx, stateFeatures, bone_vel);
		}


		/*int featureIdx = 0;
		const int nStateBones = 6;
		SimulationContext::BodyName stateBones[6] = {
			SimulationContext::BodySpine,
			SimulationContext::BodyTrunk,
			SimulationContext::BodyRightArm,
			SimulationContext::BodyRightLeg,
			SimulationContext::BodyLeftArm,
			SimulationContext::BodyLeftLeg };


		for (int i = 0; i < nStateBones; i++)
		{
			pushStateFeature(featureIdx, stateFeatures, iContextCPBP->getBonePosition(stateBones[i]));
			pushStateFeature(featureIdx, stateFeatures, iContextCPBP->getBoneLinearVelocity(stateBones[i]));
			pushStateFeature(featureIdx, stateFeatures, iContextCPBP->getBoneAngle(stateBones[i]));
			pushStateFeature(featureIdx, stateFeatures, iContextCPBP->getBoneAngularVelocity(stateBones[i]));
		}*/

		for (int i = 0; i < iContextCPBP->getHoldBodyIDsSize(); i++)
		{
			if (iContextCPBP->getHoldBodyIDs(i, targetContext) != -1)
			{
				pushStateFeature(featureIdx, stateFeatures, 1.0f);
			}
			else
			{
				pushStateFeature(featureIdx, stateFeatures, 0.0f);
			}
		}

		return featureIdx;
	}

	void clampCMAESSegmentControls(int segmentIdx, bool clampDuration, int _nAngles, int nValuesPerSegment, VectorXf &sample)
	{
		int segmentStart = segmentIdx*nValuesPerSegment;

		if (clampDuration)
		{
			sample[segmentStart] = clipMinMaxf(sample[segmentStart], minSegmentDuration, maxSegmentDuration);
			segmentStart++;
		}

		for (int velIdx = 0; velIdx<_nAngles; velIdx++)
		{
			sample[segmentStart + velIdx] = clipMinMaxf(sample[segmentStart + velIdx], poseMin[velIdx], poseMax[velIdx]);
		}
		
		for (int fmaxIdx = 0; fmaxIdx < fmCount; fmaxIdx++)
		{
			//safest to start with high fmax and let the optimizer decrease them later, thus the mean at maximumForce
			sample[segmentStart + _nAngles + fmaxIdx] = clipMinMaxf(sample[segmentStart + _nAngles + fmaxIdx], minimumForce, maximumForce);
		}
		//for (int id = 0; id < id_size_control; id++)
		//{
		//	//safest to start with high fmax and let the optimizer decrease them later, thus the mean at maximumForce
		//	sample[segmentStart + nAngles + fmCount + id] = clipMinMaxf(sample[segmentStart + nAngles + fmCount + id], 0.0f, 1.0f);
		//}
	}

	bool advance_simulation_context(Eigen::VectorXf& cControl, int trajectory_idx, float& controlCost, std::vector<int>& dHoldIDs,
		bool debugPrint, bool flagSaveSlots, std::vector<BipedState>& nStates)
	{
		bool physicsBroken = false;

		if (consider_force_opt || 
			(mContextController->holdPosIndex[trajectory_idx][2] == -1 && mContextController->holdPosIndex[trajectory_idx][3] == -1 
				&& TestID == mEnumTestCaseClimber::RunLearnerRandomly))
		{
			for (int _m = 0; _m < 4; _m++)
			{
				modelLetGoBodyPart(trajectory_idx, _m);
			}
		}
		
		apply_control(mContextController, cControl, trajectory_idx);
		
		physicsBroken = !stepOde(timeStep, false);

		for (int i = 0; i < 4 && !physicsBroken; i++)
		{
			if (mContextController->getHoldBodyIDs((int)(sbLeftLeg)+i, trajectory_idx) != -1)
			{
				_forceOnHandsFeet[trajectory_idx].updateForces(StanceBodies(StanceBodies::sbLeftLeg + i)
					, mContextController->getForceVectorOnEndBody((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i), trajectory_idx));
			}
			else
			{
				_forceOnHandsFeet[trajectory_idx].makeZeroForce(StanceBodies(StanceBodies::sbLeftLeg + i));
				_forceOnHandsFeet[trajectory_idx].makeZeroCounter(StanceBodies(StanceBodies::sbLeftLeg + i));
			}
		}

		if (!physicsBroken)
		{
			controlCost += compute_control_cost(mContextController, trajectory_idx);
		}
		else
		{
			controlCost = 1e20;
		}
		if (flagSaveSlots && !physicsBroken)
		{
			BipedState nState = this->startState.getNewCopy(mContextController->getNextFreeSavingSlot(), trajectory_idx);
			for (int i = 0; i < 4; i++)
			{
				nState.forces[i] = _forceOnHandsFeet[trajectory_idx].getAvgForce((StanceBodies)(StanceBodies::sbLeftLeg + i));
				nState.counter_let_go[i] = _forceOnHandsFeet[trajectory_idx].getCountLetGo((StanceBodies)(StanceBodies::sbLeftLeg + i));
			}
			nStates.push_back(nState);
		}
		
		if (debugPrint) rcPrintString("Control cost for controller: %f", controlCost);

		if (!physicsBroken)
		{
			return true;
		}

		return false;
	}

	int getNSteps()
	{
		return bestCmaesTrajectory.nSteps;
	}

	void apply_control(SimulationContext* iContextCPBP, const Eigen::VectorXf& control, int trajectory_idx)
	{
		int _count = 0;
		for (int j = 0; j < iContextCPBP->getJointSize(); j++)
		{
			if ((iContextCPBP->jointIDIndex[j] + 1) == SimulationContext::BodyName::BodyHead ||
				(iContextCPBP->jointIDIndex[j] + 1) == SimulationContext::BodyName::BodyLeftHand ||
				(iContextCPBP->jointIDIndex[j] + 1) == SimulationContext::BodyName::BodyRightHand)
			{
				iContextCPBP->driveMotorToPose(j, 0, motorPoseInterpolationTime);
			}
			else
			{
				float c_j = control[_count];
				iContextCPBP->driveMotorToPose(j, c_j, motorPoseInterpolationTime);
				_count++;
			}
			
		}
		if (fmCount != 0)
		{
			//fmax values for each group are after the joint motor speeds
			iContextCPBP->setMotorGroupFmaxes(&control[nAngles], torsoMinFMax);
		}
	}

	void apply_connection(const Eigen::VectorXf& sample_control, int trajectory_idx, std::vector<int>& targetHoldIDs, float time)
	{
		bool temp[id_size_control];
		std::vector<bool> mAllowRelease(id_size_control,temp);

		for (int _m = 0; _m < id_size_control; _m++)
		{
			mAllowRelease[_m] = time > sample_control[sample_control.size() -1 - _m];
		}

		if (id_size_control == 5)
		{
			if (mAllowRelease[4])
			{
				for (int _m = 0; _m < id_size_control - 1; _m++)
				{
					mAllowRelease[_m] = true;
				}
			}
		}
		mConnectDisconnectContactPoint(targetHoldIDs, trajectory_idx, mAllowRelease);
	}
};
