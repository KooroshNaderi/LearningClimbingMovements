#pragma once

class mSampleStructure
{
public:
	void initialization()
	{
		cOptimizationCost = FLT_MAX;
		numItrFixedCost = 0;

		isOdeConstraintsViolated = false;
		isReached = false;
		isRejected = false;

		// variables for recovery from playing animation
		restartSampleStartState = false;

		control_cost = 0.0f;

		starting_time = 0.0f;

		desired_hold_ids = std::vector<int>(4, -1);
		initial_hold_ids = std::vector<int>(4, -1);

		desired_hold_prototypeIDs = std::vector<int>(4, -1);
		initial_hold_prototypeIDs = std::vector<int>(4, -1);
	}

	mSampleStructure()
	{
		initialization();
		isSet = false;

		//		closest_node_index = -1;
	}

	mSampleStructure(std::vector<mOptCPBP::ControlledPoses>& iSourceP, std::vector<Vector3>& iDestinationP,
		std::vector<int>& iInitialHoldIDs, std::vector<int>& iDesiredHoldIDs)//, int iClosestIndexNode)
																			 // std::vector<Vector3>& iWorkSpaceHolds, std::vector<Vector3>& iWorkSpaceColor, std::vector<Vector3>& iDPoint)
	{
		initialization();
		isSet = true;

		for (unsigned int i = 0; i < iSourceP.size(); i++)
		{
			sourceP.push_back(iSourceP[i]);
			destinationP.push_back(iDestinationP[i]);
		}

		initial_hold_ids = iInitialHoldIDs;
		desired_hold_ids = iDesiredHoldIDs;
	}

	void drawDesiredTorsoDir(Vector3& _from) // if the last element of source point is torsoDir we have a desired direction
	{
		if (sourceP.size() > 0)
		{
			if (sourceP[sourceP.size() - 1] == mOptCPBP::ControlledPoses::TorsoDir)
			{
				rcSetColor(1.0f, 0.0f, 0.0f);
				Vector3 _to = _from + destinationP[destinationP.size() - 1];
				SimulationContext::drawLine(_from, _to);
			}
		}
	}

	std::vector<mOptCPBP::ControlledPoses> sourceP; // contains head, trunk and contact points sources
	std::vector<Vector3> destinationP; // contains head's desired angle, and trunk's and contact points's desired positions (contact points's desired positions have the most value to us)

	std::vector<int> desired_hold_ids; // desired holds's ids to reach
	std::vector<int> desired_hold_prototypeIDs; // desired holds's ids to reach

	std::vector<int> initial_hold_ids; // connected joints to (ll,rl,lh,rh); -1 means it is disconnected, otherwise it is connected to the hold with the same id
	std::vector<int> initial_hold_prototypeIDs; // connected joints to (ll,rl,lh,rh); -1 means it is disconnected, otherwise it is connected to the hold with the same id

	float starting_time;
	// check cost change
	float cOptimizationCost;
	int numItrFixedCost;

	//	int closest_node_index;
	std::vector<BipedState> statesFromTo;

	bool isOdeConstraintsViolated;
	bool isReached;
	bool isRejected;
	bool isSet;

	// variables for recovery from playing animation
	bool restartSampleStartState;

	// handling energy cost - control cost
	float control_cost;
	float target_cost;
	float success_rate;
	float cost_var;

	VectorXf controlPoints;

	BipedState initBodyState;

//	VectorXf initFeatureState;
//	Vector3 spinePos;

	bool isUsingMachineLearning;
	bool isAllowedOnGround;
};

class mSampler
{
public:
	float climberRadius;
	float climberLegLegDis;
	float climberHandHandDis;

	const float angle_limit_wall = 0.75f * PI;
	// helping sampling
	std::vector<std::vector<int>> indices_higher_than;
	std::vector<std::vector<int>> indices_lower_than;
	std::vector<std::vector<int>> indices_around_hand;

	SimulationContext* mContext;

	mSampler(SimulationContext* iContextRRT)
	{
		mContext = iContextRRT;

		climberRadius = iContextRRT->getClimberRadius();
		climberLegLegDis = iContextRRT->getClimberLegLegDis();
		climberHandHandDis = iContextRRT->getClimberHandHandDis();

		init();
	}

	void init()
	{
		indices_higher_than.clear();
		indices_lower_than.clear();
		indices_around_hand.clear();

		fillAroundHandHoldIndices();
		fillInLowerHigherHoldIndices();
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	std::vector<std::vector<int>> getListOfStanceSamples(std::vector<int>& from_hold_ids, std::vector<int>& to_hold_ids, bool isInitialStance)
	{
		std::vector<std::vector<int>> list_samples;

		std::vector<int> initial_holds_ids = from_hold_ids;

		std::vector<int> diff_hold_index;
		std::vector<int> same_hold_index;

		for (unsigned int i = 0; i < to_hold_ids.size(); i++)
		{
			if (initial_holds_ids[i] != to_hold_ids[i] || initial_holds_ids[i] == -1)
			{
				diff_hold_index.push_back(i);
			}
			else
			{
				same_hold_index.push_back(i);
			}
		}

		std::vector<std::vector<int>> possible_hold_index_diff;
		std::vector<unsigned int> itr_index_diff;
		for (unsigned int i = 0; i < diff_hold_index.size(); i++)
		{
			std::vector<int> possible_hold_diff_i;
			int index_diff_i = diff_hold_index[i];

			mTools::addToSetIDs(-1, possible_hold_diff_i);
			mTools::addToSetIDs(to_hold_ids[index_diff_i], possible_hold_diff_i);

			mTools::addToSetIDs(initial_holds_ids[index_diff_i], possible_hold_diff_i);

			itr_index_diff.push_back(0);
			possible_hold_index_diff.push_back(possible_hold_diff_i);
		}

		bool flag_continue = true;

		if (diff_hold_index.size() == 0)
		{
			flag_continue = false;
			list_samples.push_back(to_hold_ids);
		}

		while (flag_continue)
		{
			// create sample n
			std::vector<int> sample_n;
			for (int i = to_hold_ids.size() - 1; i >= 0; i--)
			{
				sample_n.push_back(-1);
			}
			for (unsigned int i = 0; i < diff_hold_index.size(); i++)
			{
				int index_diff_i = diff_hold_index[i];
				int itr_index_diff_i = itr_index_diff[i];
				std::vector<int> possible_hold_diff_i = possible_hold_index_diff[i];
				sample_n[index_diff_i] = possible_hold_diff_i[itr_index_diff_i];
			}
			for (unsigned int i = 0; i < same_hold_index.size(); i++)
			{
				int index_same_i = same_hold_index[i];
				sample_n[index_same_i] = initial_holds_ids[index_same_i];
			}

			// increase itr num
			for (unsigned int i = 0; i < diff_hold_index.size(); i++)
			{
				itr_index_diff[i] = itr_index_diff[i] + 1;
				if (itr_index_diff[i] >= possible_hold_index_diff[i].size())
				{
					if (i == diff_hold_index.size() - 1)
					{
						flag_continue = false;
					}
					itr_index_diff[i] = 0;
				}
				else
				{
					break;
				}
			}

			///////////////////////////////////////////////////////////////
			// prior for adding sample_n to the list of possible samples //
			///////////////////////////////////////////////////////////////
			if (!isAllowedHandsLegsInDSample(initial_holds_ids, sample_n, isInitialStance)) // are letting go of hands and legs allowed
			{
				continue;
			}

			std::vector<Vector3> sample_n_hold_points; float size_n = 0;
			Vector3 midPointN = mContext->getHoldStancePosFrom(sample_n, sample_n_hold_points, size_n);
			if (size_n == 0)
			{
				continue;
			}

			if (!acceptDirectionLegsAndHands(midPointN, sample_n, sample_n_hold_points))
			{
				continue;
			}

			if (!isFromStanceCloseEnough(initial_holds_ids, sample_n))
			{
				continue;
			}

			if (!earlyAcceptOfSample(sample_n, isInitialStance)) // is it kinematically reachable
			{
				continue;
			}

			list_samples.push_back(sample_n);
		}

		return list_samples;
	}

	void getListOfStanceSamplesAround(int to_hold_id, std::vector<int>& from_hold_ids, std::vector<std::vector<int>> &out_list_samples)
	{
		out_list_samples.clear();
		static std::vector<std::vector<int>> possible_hold_index_diff;
		if (possible_hold_index_diff.size() != 4)
			possible_hold_index_diff.resize(4);
		const int itr_index_diff_size = 4;
		int itr_index_diff[itr_index_diff_size] = { 0,0,0,0 };

		for (unsigned int i = 0; i < 4; i++)
		{
			std::vector<int> &possible_hold_diff_i = possible_hold_index_diff[i];
			possible_hold_diff_i.clear();

			mTools::addToSetIDs(-1, possible_hold_diff_i);
			mTools::addToSetIDs(from_hold_ids[i], possible_hold_diff_i);

			for (unsigned int j = 0; j < indices_lower_than[to_hold_id].size(); j++)
			{
				mTools::addToSetIDs(indices_lower_than[to_hold_id][j], possible_hold_diff_i);
			}

			for (unsigned int j = 0; j < from_hold_ids.size(); j++)
			{
				if (mTools::isInSetIDs(from_hold_ids[j], indices_lower_than[to_hold_id]))
					mTools::addToSetIDs(from_hold_ids[j], possible_hold_diff_i);
			}
		}
		bool flag_continue = true;
		while (flag_continue)
		{
			// create sample n
			std::vector<int> sample_n(4, -1);

			for (unsigned int i = 0; i < itr_index_diff_size; i++)
			{
				int itr_index_diff_i = itr_index_diff[i];
				std::vector<int> &possible_hold_diff_i = possible_hold_index_diff[i];
				sample_n[i] = possible_hold_diff_i[itr_index_diff_i];
			}

			// increase itr num
			for (unsigned int i = 0; i < itr_index_diff_size; i++)
			{
				itr_index_diff[i] = itr_index_diff[i] + 1;
				if (itr_index_diff[i] >= (int)possible_hold_index_diff[i].size())
				{
					if (i == itr_index_diff_size - 1)
					{
						flag_continue = false;
					}
					itr_index_diff[i] = 0;
				}
				else
				{
					break;
				}
			}

			///////////////////////////////////////////////////////////////
			// prior for adding sample_n to the list of possible samples //
			///////////////////////////////////////////////////////////////

			if (sample_n[2] != to_hold_id && sample_n[3] != to_hold_id)
			{
				continue;
			}

			if (!isFromStanceToStanceValid(from_hold_ids, sample_n, false))
			{
				continue;
			}

			if (!mTools::isInSampledStanceSet(sample_n, out_list_samples))
				out_list_samples.push_back(sample_n);
		}
	}

	// given a hold id: we assume that hold should be used for hand (right hand) and generate all possible stance samples
	// if right hand hold is -1 then left hand should get connected to the given hold id
	/////////////////////////////////////////////////////////////////////////////////////////////////// *********************** TODO: check this **************************
	void getListOfStanceAroundHandHold(int given_hold_id, std::vector<std::vector<int>> &out_list_samples)
	{
		out_list_samples.clear();

		static std::vector<std::vector<int>> possible_hold_index_diff;
		if (possible_hold_index_diff.size() != 4)
			possible_hold_index_diff.resize(4);
		const int itr_index_diff_size = 4;
		int itr_index_diff[itr_index_diff_size] = { 0,0,0,0 };

		for (int i = 0; i < itr_index_diff_size; i++)
		{
			possible_hold_index_diff[i].clear();
		}
		//set right hand possible hold ids
		mTools::addToSetIDs(given_hold_id, possible_hold_index_diff[3]);

		Vector3 center_hold = mContext->getHoldPos(given_hold_id);
		float theta_normal_dir = mContext->getHoldThetaNormal(given_hold_id);

		Vector3 y_local_axis(cosf(theta_normal_dir), sinf(theta_normal_dir), 0.0f);
		Vector3 x_local_axis(cosf(theta_normal_dir + PI / 2.0f), sinf(theta_normal_dir + PI / 2.0f), 0.0f);

		for (unsigned int j = 0; j < indices_around_hand[given_hold_id].size(); j++)
		{
			int hold_id_j = indices_around_hand[given_hold_id][j];
			Vector3 hold_j = mContext->getHoldPos(hold_id_j);

			Vector3 _displacement = (hold_j - center_hold);
			float _dis = _displacement.norm();

			if (_dis < climberHandHandDis)
			{
				mTools::addToSetIDs(hold_id_j, possible_hold_index_diff[2]);
			}

			mTools::addToSetIDs(hold_id_j, possible_hold_index_diff[0]);
			mTools::addToSetIDs(hold_id_j, possible_hold_index_diff[1]);
		}

		for (int i = 0; i < itr_index_diff_size; i++)
		{
			mTools::addToSetIDs(-1, possible_hold_index_diff[i]);
		}

		bool flag_continue = true;
		while (flag_continue)
		{
			// create sample n
			std::vector<int> sample_n(4, -1);

			for (unsigned int i = 0; i < itr_index_diff_size; i++)
			{
				int itr_index_diff_i = itr_index_diff[i];
				std::vector<int> &possible_hold_diff_i = possible_hold_index_diff[i];
				sample_n[i] = possible_hold_diff_i[itr_index_diff_i];
			}

			if (sample_n[3] == -1)
				sample_n[2] = given_hold_id;

			// increase itr num
			for (unsigned int i = 0; i < itr_index_diff_size; i++)
			{
				itr_index_diff[i] = itr_index_diff[i] + 1;
				if (itr_index_diff[i] >= (int)possible_hold_index_diff[i].size())
				{
					if (sample_n[3] == -1 && i == itr_index_diff_size - 3)
					{
						flag_continue = false;
					}
					if (i == itr_index_diff_size - 1)
					{
						flag_continue = false;
					}
					itr_index_diff[i] = 0;
				}
				else
				{
					break;
				}
			}

			///////////////////////////////////////////////////////////////
			// prior for adding sample_n to the list of possible samples //
			///////////////////////////////////////////////////////////////
			if (isValidStanceSample(sample_n))
				out_list_samples.push_back(sample_n);
		}
		return;
	}

	// the mid point is actual hold point on the wall and generated stances assumed to be around that point 
	// hands and feet holds in the stance should be reachable from each other
	// create target stance
	void getStanceSampleAroundPoint(Vector3& midPos, float theta_dir, std::vector<int>& out_hold_ids, bool just_hands)
	{
		std::vector<int> holds_around_mid;

		theta_dir = theta_dir * (PI / 180.0f);

		Vector3 y_local_axis(cosf(theta_dir), sinf(theta_dir), 0.0f);
		Vector3 x_local_axis(cosf(theta_dir + PI / 2.0f), sinf(theta_dir + PI / 2.0f), 0.0f);

		for (unsigned int h = 0; h < mContext->holds_body.size(); h++)
		{
			Vector3 hold_h = mContext->getHoldPos(h);
			float cDis = (midPos - hold_h).norm();
			Vector3 _dir = (hold_h - midPos).normalized();
			float angle = mTools::getAbsAngleBtwVectors(y_local_axis, _dir);
			if ((cDis < climberRadius / 2.0f) && (angle < angle_limit_wall))
			{
				holds_around_mid.push_back(h);
			}
		}

		int mIDs[] = { 0, 1, 2, 3 };
		std::vector<int> _handsFeetIds(mIDs, mIDs + 4);
		std::random_shuffle(_handsFeetIds.begin(), _handsFeetIds.end());

		std::vector<int> not_connected_hands;// we want at least one hand get connected
		bool is_one_hand_higher_feet = false;
		std::vector<int> _connected_bodies;// restricted to bodies

		for (unsigned int b = 0; b < _handsFeetIds.size(); b++)
		{
			int body_i = _handsFeetIds[b];
			if (just_hands)
			{
				if (body_i == 0 || body_i == 1)// if selected body belongs to feet
				{
					// continue if it is not hand
					continue;
				}
			}

			std::vector<int> reachable_holds_body_i;

			if (_connected_bodies.size() == 0)
			{
				reachable_holds_body_i = holds_around_mid;
			}
			else
			{
				for (unsigned int h = 0; h < holds_around_mid.size(); h++)
				{
					int hold_id = holds_around_mid[h];

					Vector3 hold_pos = mContext->getHoldPos(hold_id);
					Vector3 _dir = (hold_pos - midPos).normalized();
					float angle = mTools::getAbsAngleBtwVectors(x_local_axis, _dir);
					if (_dir[2] < 0.0f)
					{
						angle = -angle;
					}

					bool flag_add = true;
					switch (body_i)// current body that is adding
					{
					case 0://left leg
						if (angle < 0.75f * PI && angle >= 0)
							flag_add = false;
						if (angle > -0.25f * PI && angle <= 0)
							flag_add = false;
						break;
					case 1://right leg
						if (angle > 0.25f * PI && angle >= 0)
							flag_add = false;
						if (angle < -0.75f * PI && angle <= 0)
							flag_add = false;
						break;
					case 2://left hand
						if (angle < 0.25f * PI && angle >= 0)
							flag_add = false;
						if (angle > -0.75f * PI && angle <= 0)
							flag_add = false;
						break;
					case 3://right hand
						if (angle > 0.75f * PI && angle >= 0)
							flag_add = false;
						if (angle < -0.25f * PI && angle <= 0)
							flag_add = false;
						break;
					}

					for (unsigned int cb = 0; cb < _connected_bodies.size() && flag_add; cb++)
					{
						int connected_hold = out_hold_ids[_connected_bodies[cb]];
						Vector3 connected_hold_pos = mContext->getHoldPos(connected_hold);
						Vector3 _dir = (hold_pos - connected_hold_pos).normalized();
						float _dis = (hold_pos - connected_hold_pos).norm();

						float max_dis = climberRadius;
						switch (body_i)// current body that is adding
						{
						case 0:
						case 1:
							if (_connected_bodies[cb] < 2)// feet that is already added
							{
								max_dis = climberLegLegDis;
							}
							if (!is_one_hand_higher_feet)
							{
								// deciding on feet
								if (_dir[2] >= 0.0f && _connected_bodies[cb] >= 2)
								{
									flag_add = false;
								}
							}
							break;
						case 2:
						case 3:
							if (_connected_bodies[cb] >= 2)// hands that is already added
							{
								max_dis = climberHandHandDis;
							}
							if (!is_one_hand_higher_feet)
							{
								// deciding on hands
								if (_dir[2] < 0.0f && _connected_bodies[cb] < 2)
								{
									flag_add = false;
								}
							}
							break;
						default:
							break;
						}

						if (_dis > max_dis)
						{
							flag_add = false;
						}
					}

					if (flag_add)
					{
						reachable_holds_body_i.push_back(hold_id);
					}
				}
			}
			reachable_holds_body_i.push_back(-1);

			int rnd_hold_id = reachable_holds_body_i[mTools::getRandomIndex(reachable_holds_body_i.size())];

			if (rnd_hold_id == -1)
			{
				if (body_i >= 2)
				{
					if (not_connected_hands.size() == 0)
					{
						not_connected_hands.push_back(body_i);
					}
					else
					{
						// removing last element which is (-1)
						rnd_hold_id = reachable_holds_body_i[mTools::getRandomIndex(max(1, int(reachable_holds_body_i.size() - 1)))];
					}
				}
			}

			if (rnd_hold_id != -1)
			{
				if (!is_one_hand_higher_feet)
				{
					if (body_i >= 2)
					{
						for (unsigned int cb = 0; cb < _connected_bodies.size() && !is_one_hand_higher_feet; cb++)
						{
							int connected_hold = out_hold_ids[_connected_bodies[cb]];
							Vector3 _dir = (mContext->getHoldPos(rnd_hold_id) - mContext->getHoldPos(connected_hold)).normalized();

							if (_connected_bodies[cb] < 2 && _dir[2] > 0)
							{
								is_one_hand_higher_feet = true;
							}
						}
					}
				}

				_connected_bodies.push_back(body_i);
			}

			out_hold_ids[body_i] = rnd_hold_id;
		}

		return;
	}

	bool isValidStanceSample(std::vector<Vector3>& sample_hold_points)
	{
		Vector3 midStancePoint(0.0f, 0.0f, 0.0f);
		for (unsigned int i = 0; i < 4; i++)
			midStancePoint += sample_hold_points[i] / 4.0f;

		float theta_normal_dir = -PI/2;

		Vector3 x_local_axis(cosf(theta_normal_dir + PI / 2.0f), sinf(theta_normal_dir + PI / 2.0f), 0.0f);

		float dis_feet = (sample_hold_points[0] - sample_hold_points[1]).norm();
		
		if (dis_feet > climberLegLegDis)
			return false;

		float dis_hands = (sample_hold_points[2] - sample_hold_points[3]).norm();

		if (dis_hands > climberHandHandDis)
			return false;

		float dis_hand_feet = 0.0f;
		for (int f = 0; f < 2; f++)
		{

			for (int h = 2; h < 4; h++)
			{
				dis_hand_feet = (sample_hold_points[f] - sample_hold_points[h]).norm();
				if (dis_hand_feet > climberRadius)
					return false;
			}
		}

		// if both feet are higher than max hight hand
		float max_hight_zHand = -FLT_MAX;
		max_hight_zHand = max(max_hight_zHand, sample_hold_points[2].z());
		max_hight_zHand = max(max_hight_zHand, sample_hold_points[3].z());

		
		float d_feet_threshold = 0.5f;

		if (sample_hold_points[0].z() >= max_hight_zHand + d_feet_threshold)
			return false;
		
		
		if (sample_hold_points[1].z() >= max_hight_zHand + d_feet_threshold)
			return false;
		
		
		if (sample_hold_points[0].z() >= max_hight_zHand && sample_hold_points[1].z() >= max_hight_zHand)
			return false;
		
		// hand position related to the other hand
		//Vector3 _dir_rh = (sample_hold_points[3] - midStancePoint);
		//if (_dir_rh.norm() > 0)
		//{
		//	_dir_rh.normalize();
		//}
		//Vector3 _dir_lh = (sample_hold_points[2] - midStancePoint);
		//if (_dir_lh.norm() > 0)
		//{
		//	_dir_lh.normalize();
		//}
		//dis_hands = (sample_hold_points[3] - sample_hold_points[2]).norm();
		//float angle_rh = mTools::getAbsAngleBtwVectors(x_local_axis, _dir_rh);
		//float angle_lh = mTools::getAbsAngleBtwVectors(x_local_axis, _dir_lh);
		//if (_dir_rh[2] < 0)
		//{
		//	angle_rh = -angle_rh;
		//	angle_rh += 2 * PI;
		//}
		//if (_dir_lh[2] < 0)
		//{
		//	angle_lh = -angle_lh;
		//	angle_lh += 2 * PI;
		//}
		//float diff = angle_lh - angle_rh;
		//if (angle_rh >= 0 && angle_rh < PI) // diff is best when is positive (right hand should be before left hand for facing the wall)
		//{
		//	if (diff < 0 && dis_hands > climberHandHandDis / 2.0f)
		//		return false;
		//}
		//else // diff is best when is negative (left hand should be before right hand for facing the wall)
		//{
		//	if (diff > 0 && dis_hands > climberHandHandDis / 2.0f)
		//		return false;
		//}
		//// feet position related to the other foot and hands
		//Vector3 _dir_rf = (sample_hold_points[1] - midStancePoint);
		//if (_dir_rf.norm() > 0)
		//{
		//	_dir_rf.normalize();
		//}
		//Vector3 _dir_lf = (sample_hold_points[0] - midStancePoint);
		//if (_dir_lf.norm() > 0)
		//{
		//	_dir_lf.normalize();
		//}
		//dis_feet = (sample_hold_points[1] - sample_hold_points[0]).norm();
		//float angle_rf = mTools::getAbsAngleBtwVectors(x_local_axis, _dir_rf);
		//float angle_lf = mTools::getAbsAngleBtwVectors(x_local_axis, _dir_lf);
		//if (_dir_rf[2] < 0)
		//{
		//	angle_rf = -angle_rf;
		//	angle_rf += 2 * PI;
		//}
		//if (_dir_lf[2] < 0)
		//{
		//	angle_lf = -angle_lf;
		//	angle_lf += 2 * PI;
		//}
		//diff = angle_lf - angle_rf;
		//if (angle_rf >= 0 && angle_rf < PI) // diff is best when is positive (right hand should be before left hand for facing the wall)
		//{
		//	if (diff < 0 && dis_feet > climberLegLegDis / 2.0f)
		//		return false;
		//}
		//else // diff is best when is negative (left hand should be before right hand for facing the wall)
		//{
		//	if (diff > 0 && dis_feet > climberLegLegDis / 2.0f)
		//		return false;
		//}
		return true;
	}
private:

	void fillAroundHandHoldIndices()
	{
		float max_body_radius = climberRadius;

		for (unsigned int k = 0; k < mContext->holds_body.size(); k++)
		{
			Vector3 dHoldPos = mContext->getHoldPos(k);
			std::vector<int> ret_holds_ids;
			getPointsInRadius(dHoldPos, max_body_radius, ret_holds_ids);

			std::vector<int> hand_holds_ids;
			for (unsigned int i = 0; i < ret_holds_ids.size(); i++)
			{
				bool flag_add = true;
				Vector3 dir_hold_pos_i = mContext->getHoldPos(ret_holds_ids[i]) - dHoldPos;

				hand_holds_ids.push_back(ret_holds_ids[i]);
			}
			indices_around_hand.push_back(hand_holds_ids);
		}

		return;
	}

	void fillInLowerHigherHoldIndices()
	{
		float max_radius_around_hand = 1.0f * climberRadius;

		for (unsigned int k = 0; k < mContext->holds_body.size(); k++)
		{
			Vector3 dHoldPos = mContext->getHoldPos(k);
			std::vector<int> ret_holds_ids;
			getPointsInRadius(dHoldPos, max_radius_around_hand, ret_holds_ids);
			std::vector<int> lower_holds_ids;
			std::vector<int> higher_holds_ids;
			for (unsigned int i = 0; i < ret_holds_ids.size(); i++)
			{
				bool flag_add = true;
				Vector3 hold_pos_i = mContext->getHoldPos(ret_holds_ids[i]);

				float cDis = (hold_pos_i - dHoldPos).norm();

				if (cDis < 0.01f)
				{
					lower_holds_ids.push_back(ret_holds_ids[i]);
					higher_holds_ids.push_back(ret_holds_ids[i]);
					continue;
				}

				float angle_btw_l = mTools::getAngleBtwVectorsXZ(Vector3(1.0f, 0.0f, 0.0f), hold_pos_i - dHoldPos);
				if (((angle_btw_l <= 0) && (angle_btw_l >= -PI)) || (angle_btw_l >= PI))
				{
					lower_holds_ids.push_back(ret_holds_ids[i]);
				}
				if (((angle_btw_l >= 0) && (angle_btw_l <= PI)) || (angle_btw_l <= -PI))
				{
					higher_holds_ids.push_back(ret_holds_ids[i]);
				}
			}
			indices_lower_than.push_back(lower_holds_ids);
			indices_higher_than.push_back(higher_holds_ids);
		}
		return;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	bool isValidStanceSample(std::vector<int>& sample_stance)
	{
		if (sample_stance[2] == -1 && sample_stance[3] == -1)
			return false;

		float mSize = 0;
		std::vector<Vector3> sample_hold_points;
		Vector3 midStancePoint = mContext->getHoldStancePosFrom(sample_stance, sample_hold_points, mSize);
		float theta_normal_dir = mContext->getThetaDirHoldStance(sample_stance) * (PI / 180.0f);

		Vector3 x_local_axis(cosf(theta_normal_dir + PI / 2.0f), sinf(theta_normal_dir + PI / 2.0f), 0.0f);

		float dis_feet = 0.0f;
		if (sample_stance[0] != -1 && sample_stance[1] != -1)
		{
			dis_feet = (sample_hold_points[0] - sample_hold_points[1]).norm();
		}
		if (dis_feet > climberLegLegDis)
			return false;

		float dis_hands = 0.0f;
		if (sample_stance[2] != -1 && sample_stance[3] != -1)
		{
			dis_hands = (sample_hold_points[2] - sample_hold_points[3]).norm();
		}
		if (dis_hands > climberHandHandDis)
			return false;

		float dis_hand_feet = 0.0f;
		for (int f = 0; f < 2; f++)
		{
			if (sample_stance[f] != -1)
			{
				for (int h = 2; h < 4; h++)
				{
					if (sample_stance[h] != -1)
					{
						dis_hand_feet = (sample_hold_points[f] - sample_hold_points[h]).norm();
						if (dis_hand_feet > climberRadius)
							return false;
					}
				}
			}
		}

		// if both feet are higher than max hight hand
		float max_hight_zHand = -FLT_MAX;
		if (sample_stance[2] != -1)
			max_hight_zHand = max(max_hight_zHand, sample_hold_points[2].z());
		if (sample_stance[3] != -1)
			max_hight_zHand = max(max_hight_zHand, sample_hold_points[3].z());

		if (sample_stance[0] != -1 || sample_stance[1] != -1)
		{
			float d_feet_threshold = 0.5f;
			if (sample_stance[0] != -1)
			{
				if (sample_hold_points[0].z() >= max_hight_zHand + d_feet_threshold)
					return false;
			}
			if (sample_stance[1] != -1)
			{
				if (sample_hold_points[1].z() >= max_hight_zHand + d_feet_threshold)
					return false;
			}
			if (sample_stance[0] != -1 && sample_stance[1] != -1)
			{
				if (sample_hold_points[0].z() >= max_hight_zHand && sample_hold_points[1].z() >= max_hight_zHand)
					return false;
			}
		}
		//// hand position related to the other hand
		//if (sample_stance[2] != -1 && sample_stance[3] != -1)
		//{
		//	Vector3 _dir_rh = (sample_hold_points[3] - midStancePoint);
		//	if (_dir_rh.norm() > 0)
		//	{
		//		_dir_rh.normalize();
		//	}
		//	Vector3 _dir_lh = (sample_hold_points[2] - midStancePoint);
		//	if (_dir_lh.norm() > 0)
		//	{
		//		_dir_lh.normalize();
		//	}
		//	dis_hands = (sample_hold_points[3] - sample_hold_points[2]).norm();
		//	float angle_rh = mTools::getAbsAngleBtwVectors(x_local_axis, _dir_rh);
		//	float angle_lh = mTools::getAbsAngleBtwVectors(x_local_axis, _dir_lh);
		//	if (_dir_rh[2] < 0)
		//	{
		//		angle_rh = -angle_rh;
		//		angle_rh += 2 * PI;
		//	}
		//	if (_dir_lh[2] < 0)
		//	{
		//		angle_lh = -angle_lh;
		//		angle_lh += 2 * PI;
		//	}
		//	float diff = angle_lh - angle_rh;
		//	if (angle_rh >= 0 && angle_rh < PI) // diff is best when is positive (right hand should be before left hand for facing the wall)
		//	{
		//		if (diff < 0 && dis_hands > climberHandHandDis / 2.0f)
		//			return false;
		//	}
		//	else // diff is best when is negative (left hand should be before right hand for facing the wall)
		//	{
		//		if (diff > 0 && dis_hands > climberHandHandDis / 2.0f)
		//			return false;
		//	}
		//}
		//// feet position related to the other foot and hands
		//if (sample_stance[0] != -1 && sample_stance[1] != -1)
		//{
		//	Vector3 _dir_rf = (sample_hold_points[1] - midStancePoint);
		//	if (_dir_rf.norm() > 0)
		//	{
		//		_dir_rf.normalize();
		//	}
		//	Vector3 _dir_lf = (sample_hold_points[0] - midStancePoint);
		//	if (_dir_lf.norm() > 0)
		//	{
		//		_dir_lf.normalize();
		//	}
		//	float dis_feet = (sample_hold_points[1] - sample_hold_points[0]).norm();
		//	float angle_rf = mTools::getAbsAngleBtwVectors(x_local_axis, _dir_rf);
		//	float angle_lf = mTools::getAbsAngleBtwVectors(x_local_axis, _dir_lf);
		//	if (_dir_rf[2] < 0)
		//	{
		//		angle_rf = -angle_rf;
		//		angle_rf += 2 * PI;
		//	}
		//	if (_dir_lf[2] < 0)
		//	{
		//		angle_lf = -angle_lf;
		//		angle_lf += 2 * PI;
		//	}
		//	float diff = angle_lf - angle_rf;
		//	if (angle_rf >= 0 && angle_rf < PI) // diff is best when is positive (right hand should be before left hand for facing the wall)
		//	{
		//		if (diff < 0 && dis_feet > climberLegLegDis / 2.0f)
		//			return false;
		//	}
		//	else // diff is best when is negative (left hand should be before right hand for facing the wall)
		//	{
		//		if (diff > 0 && dis_feet > climberLegLegDis / 2.0f)
		//			return false;
		//	}
		//	
		//}
		return true;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	bool isFromStanceToStanceValid(std::vector<int>& _formStanceIds, std::vector<int>& _toStanceIds, bool isInitialStance)
	{
		if (!isAllowedHandsLegsInDSample(_formStanceIds, _toStanceIds, isInitialStance)) // are letting go of hands and legs allowed
		{
			return false;
		}

		std::vector<Vector3> sample_n_hold_points; float size_n = 0;
		Vector3 midPointN = mContext->getHoldStancePosFrom(_toStanceIds, sample_n_hold_points, size_n);
		if (size_n == 0)
		{
			return false;
		}

		if (!acceptDirectionLegsAndHands(midPointN, _toStanceIds, sample_n_hold_points))
		{
			return false;
		}

		if (!isFromStanceCloseEnough(_formStanceIds, _toStanceIds))
		{
			return false;
		}

		if (!earlyAcceptOfSample(_toStanceIds, isInitialStance)) // is it kinematically reachable
		{
			return false;
		}
		return true;
	}

	bool earlyAcceptOfSample(std::vector<int>& sample_desired_hold_ids, bool isStanceGiven)
	{
		if (!isStanceGiven)
		{
			if (sample_desired_hold_ids[0] == sample_desired_hold_ids[1] &&
				(sample_desired_hold_ids[2] == sample_desired_hold_ids[0] || sample_desired_hold_ids[3] == sample_desired_hold_ids[0])) // if hands and legs are on the same hold
				return false;
			if (sample_desired_hold_ids[2] == sample_desired_hold_ids[3] &&
				(sample_desired_hold_ids[2] == sample_desired_hold_ids[0] || sample_desired_hold_ids[2] == sample_desired_hold_ids[1])) // if hands and legs are on the same hold
				return false;

			if (mTools::isSetAGreaterThan(sample_desired_hold_ids, -1))
			{
				std::vector<int> diff_hold_ids;
				for (unsigned int i = 0; i < sample_desired_hold_ids.size(); i++)
				{
					if (sample_desired_hold_ids[i] != -1)
						mTools::addToSetIDs(sample_desired_hold_ids[i], diff_hold_ids);
				}
				if (diff_hold_ids.size() <= 2)
				{
					return false;
				}
			}
		}
		std::vector<Vector3> sample_desired_hold_points;
		float mSize = 0;
		Vector3 midPoint = mContext->getHoldStancePosFrom(sample_desired_hold_ids, sample_desired_hold_points, mSize);

		if (mSize == 0)
		{
			return false;
		}

		// early reject of the sample (do not try the sample, because it is not reasonable)

		float coefficient_hand = 1.5f;
		float coefficient_leg = 1.2f;
		float coefficient_all = 1.1f;
		// if hands or legs distance are violating
		if (sample_desired_hold_ids[0] != -1 && sample_desired_hold_ids[1] != -1)
		{
			float dis_ll = (sample_desired_hold_points[0] - sample_desired_hold_points[1]).norm();
			if (dis_ll > coefficient_leg * climberLegLegDis)
			{
				return false;
			}
		}

		if (sample_desired_hold_ids[2] != -1 && sample_desired_hold_ids[3] != -1)
		{
			float dis_hh = (sample_desired_hold_points[2] - sample_desired_hold_points[3]).norm();
			if (dis_hh > coefficient_hand * climberHandHandDis)
			{
				return false;
			}
		}

		if (sample_desired_hold_ids[0] != -1 && sample_desired_hold_ids[2] != -1)
		{
			float dis_h1l1 = (sample_desired_hold_points[2] - sample_desired_hold_points[0]).norm();

			if (dis_h1l1 > coefficient_all * climberRadius)
				return false;
		}

		if (sample_desired_hold_ids[0] != -1 && sample_desired_hold_ids[3] != -1)
		{
			float dis_h2l1 = (sample_desired_hold_points[3] - sample_desired_hold_points[0]).norm();
			if (dis_h2l1 > coefficient_all * climberRadius)
				return false;
		}

		if (sample_desired_hold_ids[1] != -1 && sample_desired_hold_ids[2] != -1)
		{
			float dis_h1l2 = (sample_desired_hold_points[2] - sample_desired_hold_points[1]).norm();
			if (dis_h1l2 > coefficient_all * climberRadius)
				return false;
		}

		if (sample_desired_hold_ids[1] != -1 && sample_desired_hold_ids[3] != -1)
		{
			float dis_h2l2 = (sample_desired_hold_points[3] - sample_desired_hold_points[1]).norm();
			if (dis_h2l2 > coefficient_all * climberRadius)
				return false;
		}

		return true;
	}

	bool isFromStanceCloseEnough(std::vector<int>& initial_holds_ids, std::vector<int>& iSample_desired_hold_ids)
	{
		int m_count = 0;
		for (unsigned int i = 0; i < iSample_desired_hold_ids.size(); i++)
		{
			if (iSample_desired_hold_ids[i] == initial_holds_ids[i])
			{
				m_count++;
			}
		}

		if (m_count >= 2)
		{
			return true;
		}
		return false;
	}

	bool acceptDirectionLegsAndHands(Vector3 mPoint, std::vector<int>& sample_n_ids, std::vector<Vector3>& sample_n_points)
	{
		if (sample_n_ids[0] != -1 || sample_n_ids[1] != -1)
		{
			float z = 0;
			if (sample_n_ids[2] != -1)
			{
				z = std::max<float>(z, sample_n_points[2].z());
			}
			if (sample_n_ids[3] != -1)
			{
				z = std::max<float>(z, sample_n_points[3].z());
			}

			if (sample_n_ids[0] != -1)
			{
				if (z < sample_n_points[0].z())
				{
					return false;
				}
			}
			if (sample_n_ids[1] != -1)
			{
				if (z < sample_n_points[1].z())
				{
					return false;
				}
			}
		}

		return true;
	}

	//////////////////////////////////////////////////// general use for sampler ////////////////////////////////////////////////////

	bool isAllowedHandsLegsInDSample(std::vector<int>& initial_hold_ids, std::vector<int>& sampled_desired_hold_ids, bool isStanceGiven)
	{
		static std::vector<int> sample_initial_id;  //static to prevent heap alloc (a quick hack)
		sample_initial_id.clear();
		if (isStanceGiven)
		{
			if (initial_hold_ids[2] == -1 && initial_hold_ids[3] == -1)
			{
				if (sampled_desired_hold_ids[2] != -1 || sampled_desired_hold_ids[3] != -1)
				{
					return true;
				}
			}
			else if ((initial_hold_ids[2] != -1 || initial_hold_ids[3] != -1) && initial_hold_ids[0] == -1 && initial_hold_ids[1] == -1)
			{
				if (initial_hold_ids[2] == -1 && sampled_desired_hold_ids[2] != -1)
				{
					return true;
				}
				if (initial_hold_ids[3] == -1 && sampled_desired_hold_ids[3] != -1)
				{
					return true;
				}
				if (sampled_desired_hold_ids[0] != -1 || sampled_desired_hold_ids[1] != -1)
				{
					return true;
				}
			}
		}

		for (unsigned int i = 0; i < initial_hold_ids.size(); i++)
		{
			if (sampled_desired_hold_ids[i] == initial_hold_ids[i])
			{
				sample_initial_id.push_back(initial_hold_ids[i]);
			}
			else
			{
				sample_initial_id.push_back(-1);
			}
		}

		for (unsigned int i = 0; i < sample_initial_id.size(); i++)
		{
			if (sample_initial_id[i] == -1)
			{
				if (!isAllowedToReleaseHand_i(sample_initial_id, i) || !isAllowedToReleaseLeg_i(sample_initial_id, i))
					return false;
			}
		}
		return true;
	}

	bool isAllowedToReleaseHand_i(std::vector<int>& sampled_rInitial_hold_ids, int i)
	{
		if (i == sampled_rInitial_hold_ids.size() - 1 && sampled_rInitial_hold_ids[i - 1] == -1)
		{
			return false;
		}

		if (i == sampled_rInitial_hold_ids.size() - 2 && sampled_rInitial_hold_ids[i + 1] == -1)
		{
			return false;
		}

		if (sampled_rInitial_hold_ids[0] == -1 && sampled_rInitial_hold_ids[1] == -1 && i >= 2) // check feet
		{
			return false;
		}
		return true;
	}

	bool isAllowedToReleaseLeg_i(std::vector<int>& sampled_rInitial_hold_ids, int i)
	{
		if (sampled_rInitial_hold_ids[sampled_rInitial_hold_ids.size() - 1] != -1 && sampled_rInitial_hold_ids[sampled_rInitial_hold_ids.size() - 2] != -1)
		{
			return true;
		}

		if (i == 1 && sampled_rInitial_hold_ids[i - 1] == -1)
		{
			return false;
		}

		if (i == 0 && sampled_rInitial_hold_ids[i + 1] == -1)
		{
			return false;
		}

		return true;
	}

	/////////////////////////////////////////// handling holds using kd-tree ///////////////////////////////////////////////////////

	std::vector<double> getHoldKey(Vector3& c)
	{
		std::vector<double> rKey;

		rKey.push_back(c.x());
		rKey.push_back(c.y());
		rKey.push_back(c.z());

		return rKey;
	}
public:

	/////////////////////////////////////////// handling holds using kd-tree ///////////////////////////////////////////////////////

	int getClosestPoint(Vector3& qP)
	{
		int min_index = -1;
		float min_dis = FLT_MAX;
		for (unsigned int i = 0; i < mContext->holds_body.size(); i++)
		{
			Vector3 hold_i = mContext->getHoldPos(i);
			float cDis = (hold_i - qP).norm();
			if (cDis < min_dis)
			{
				min_dis = cDis;
				min_index = i;
			}
		}

		return min_index;
	}

	void getPointsInRadius(Vector3& qP, float r, std::vector<int>& out_indices)
	{
		for (unsigned int i = 0; i < mContext->holds_body.size(); i++)
		{
			float theta_normal_dir = mContext->getHoldThetaNormal(i);

			Vector3 y_local_axis(cosf(theta_normal_dir), sinf(theta_normal_dir), 0.0f);
			Vector3 x_local_axis(cosf(theta_normal_dir + PI / 2.0f), sinf(theta_normal_dir + PI / 2.0f), 0.0f);

			Vector3 hold_i = mContext->getHoldPos(i);
			Vector3 _dir = (hold_i - qP).normalized();

			float cDis = (hold_i - qP).norm();
			float angle = mTools::getAbsAngleBtwVectors(y_local_axis, _dir);
			if (cDis < r && (angle < angle_limit_wall))
			{
				out_indices.push_back(i);
			}
		}
		return;
	}

}*mHoldSampler;

class mStanceNode
{
public:
	Vector3 midPoint;
	float theta_dir;

	bool isItExpanded; // for building the graph
	std::vector<int> hold_ids; // ll, rl, lh, rh

	int stanceIndex;
	std::vector<int> childStanceIds; // childNodes
	//std::vector<int> parentStanceIds; // parent Nodes

	// cost if a failure happened
	std::vector<float> cost_transition_to_child;

	// neural network realized cost after training
	std::vector<float> cost_neuralnetwork_to_child;
	std::vector<float> cost_success_rate_to_child;

	//	std::vector<bool> updated_neuralnetwork_to_child;
	// our hueristics
	float nodeCost; // cost of standing at node of graph
	std::vector<float> cost_moveLimb_to_child;

	float g_AStar;
	float h_AStar;

	bool dijkstraVisited;

	int number_updates_neural_networks;
	//	int count_u;
	int childIndexInbFather;
	int bFatherStanceIndex;
	int bChildStanceIndex;

	mStanceNode(std::vector<int>& iInitStance)
	{
		g_AStar = FLT_MAX;
		h_AStar = FLT_MAX;
		bFatherStanceIndex = -1;
		bChildStanceIndex = -1;
		childIndexInbFather = -1;
		number_updates_neural_networks = -1;

		hold_ids = iInitStance;
		isItExpanded = false;
		stanceIndex = -1;
	}
};

class mStancePathNode // for A* prune
{
public:
	mStancePathNode(std::vector<int>& iPath, float iG, float iH)
	{
		_path.reserve(100);

		mG = iG;
		mH = iH;
		for (unsigned int i = 0; i < iPath.size(); i++)
		{
			_path.push_back(iPath[i]);
		}
	}

	std::vector<int> addToEndPath(int stanceID)
	{
		std::vector<int> nPath;
		for (unsigned int i = 0; i < _path.size(); i++)
		{
			nPath.push_back(_path[i]);
		}
		nPath.push_back(stanceID);
		return nPath;
	}

	std::vector<int> _path;
	float mG;
	float mH;
};

static const float punish_success_threshold = 0.5f;// this should change to 0.5f for path planning and after training done
static const float _MaxFailedTransition_PredictedCost = 1e6;
static const float const_risk_cost = 5000.0f; //2000 for 4 limb
static const float const_var_cost = 2000.0f;// 800 for 4 limb
class mStanceGraph
{
public:
	static inline uint32_t stanceToKey(const std::vector<int>& stance)
	{
		uint32_t result = 0;
		for (int i = 0; i < 4; i++)
		{
			uint32_t uNode = (uint32_t)(stance[i] + 1);  //+1 to handle the -1
			if (!(uNode >= 0 && uNode < 256))
				Debug::throwError("Invalid hold index %d!\n", uNode);
			result = (result + uNode);
			if (i < 3)
				result = result << 8; //shift, assuming max 256 holds
		}
		return result;
	}

	mStanceGraph(SimulationContext* iContext, mSampler* iClimberSampler)
	{
		mClimberSampler = iClimberSampler;
		mContext = iContext;

		maxGraphDegree = 0;
		stanceToNode.rehash(1000003);  //a prime number of buckets to minimize different keys mapping to same bucket

		_MaxTriedTransitionCost = 1e10;
		//_MaxFailedTransition_PredictedCost = 1e6;
		stance_nodes.reserve(20000);
		mRootGraph = addGraphNode(-1, std::vector<int>(4, -1));

		initializeOpenListAStarPrune();

		initialStance.clear(); // if size is zero (do random search around trunk when h=-1 is selected)
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////// TODO: complete sampling stances first!
	void buildGraphGradually(int num_samples = 32)
	{
		float max_dis_up = mClimberSampler->climberRadius / 2.0f;
		float max_dis_down = mClimberSampler->climberRadius;

		int rnd_hold_id = -1;
		bool rnd_hold_selected = false;
		float rnd_num = mTools::getRandomBetween_01();
		if (rnd_num < 0.1f)
		{
			rnd_num = mTools::getRandomBetween_01();
			// sample around hands when climber is on the ground
			if (initialStance.size() == 0 && rnd_num < 0.5f)
			{
				rnd_hold_id = -1;
				rnd_hold_selected = true;
			}
			// go toward goal hold
			if (!rnd_hold_selected)
			{
				rnd_hold_id = mContext->goal_hold_id;
			}
		}
		else
		{
			rnd_hold_id = mTools::getRandomIndex(mContext->holds_body.size());
		}

		std::vector<int> neighbor_stance_id;
		float theta_dir = -90;

		if (stance_nodes.size() == 0)
		{
			rnd_hold_id = -1;
		}

		if (rnd_hold_id != -1)
		{
			Vector3 sampleMidPoint = mContext->getHoldPos(rnd_hold_id);
			// choose a mid point in the maximum range

			int close_stance_id = -1;
			float minDis = FLT_MAX;
			for (unsigned int s = 0; s < stance_nodes.size(); s++)
			{
				float cDis = (sampleMidPoint - stance_nodes[s].midPoint).norm();

				if (cDis < minDis)
				{
					minDis = cDis;
					close_stance_id = s;
				}
			}
			// update minPoint
			float max_dis = max_dis_down;
			if (stance_nodes[close_stance_id].midPoint.z() < sampleMidPoint.z()) // go up
			{
				max_dis = max_dis_up;
			}
			theta_dir = stance_nodes[close_stance_id].theta_dir;
			theta_dir = theta_dir * (PI / 180.0f);

			Vector3 y_local_axis(cosf(theta_dir), sinf(theta_dir), 0.0f);
			Vector3 x_local_axis(cosf(theta_dir + PI / 2.0f), sinf(theta_dir + PI / 2.0f), 0.0f);

			Vector3 _dir = (sampleMidPoint - stance_nodes[close_stance_id].midPoint).normalized();
			float angle = mTools::getAbsAngleBtwVectors(y_local_axis, _dir);

			if (minDis > max_dis || angle > mClimberSampler->angle_limit_wall)
			{
				int closest_hold_to_sample = -1;
				float min_sample_dis = FLT_MAX;
				for (unsigned int h = 0; h < mContext->holds_body.size(); h++)
				{
					Vector3 hold_pos = mContext->getHoldPos(h);
					float cDis = (stance_nodes[close_stance_id].midPoint - hold_pos).norm();
					_dir = (hold_pos - stance_nodes[close_stance_id].midPoint).normalized();
					angle = mTools::getAbsAngleBtwVectors(y_local_axis, _dir);
					// get holds around stance
					if ((cDis < max_dis) && (angle < mClimberSampler->angle_limit_wall))
					{
						cDis = (sampleMidPoint - hold_pos).norm();
						if (cDis < min_sample_dis)
						{
							// find closest one to sample hold
							min_sample_dis = cDis;
							closest_hold_to_sample = h;
						}
					}
				}

				sampleMidPoint = mContext->getHoldPos(closest_hold_to_sample);
			}

			// create sample stance
			std::vector<int> sample_stance(4, -1);
			mClimberSampler->getStanceSampleAroundPoint(sampleMidPoint, theta_dir, sample_stance, false);

		}

		if (rnd_hold_id == -1 || neighbor_stance_id.size() == 0)
		{
			// sample stance around climber on the ground
			std::vector<int> sample_stance(4, -1);
			theta_dir = -90;
			mClimberSampler->getStanceSampleAroundPoint(Vector3(0, -0.5f, 1.0f), theta_dir, sample_stance, true);
			addGraphNode(mRootGraph, sample_stance);
		}

		// update goal hold
	}

	std::string retStanceToStanceString(const mStanceNode& f_node, const mStanceNode& t_node)
	{
		mTransitionData mS;
		mS.fillInputStanceFrom(mContext, f_node.hold_ids, t_node.hold_ids);

		return mS.transitionStanceToString();
	}

	void loadNeuralNetworkValues(std::string _fileName)
	{
		mFileHandler cFile(_fileName, "r");

		std::vector<std::vector<float>> mvalues;
		cFile.readFile(mvalues);

		unsigned int cIndex = 0;
		for (unsigned int s = 0; s < stance_nodes.size(); s++)
		{
			mStanceNode& node_s = stance_nodes[s];

			for (unsigned int c = 0; c < node_s.childStanceIds.size(); c++)
			{
				const mStanceNode& node_c = stance_nodes[node_s.childStanceIds[c]];
				if (cIndex < mvalues.size())
				{
					node_s.cost_neuralnetwork_to_child[c] 
						= getEdgeCost(mvalues[cIndex][0], mvalues[cIndex][1], node_c.hold_ids, mvalues[cIndex][3]);
					node_s.cost_success_rate_to_child[c] = mvalues[cIndex][0];
					cIndex++;
				}
				else
				{
					return;
				}
			}
		}

		cFile.mCloseFile();
		return;
	}

	void writeToFile()
	{
		mFileHandler mFileHandler("ClimberInfo\\BayesianLearning\\StanceGraph.txt", "w");//

		for (unsigned int s = 0; s < stance_nodes.size(); s++)
		{
			const mStanceNode& node_s = stance_nodes[s];

			for (unsigned int c = 0; c < node_s.childStanceIds.size(); c++)
			{
				const mStanceNode& node_c = stance_nodes[node_s.childStanceIds[c]];
				mFileHandler.writeLine(retStanceToStanceString(node_s, node_c));
			}
		}

		mFileHandler.mCloseFile();
		return;
	}

	// creating the whole graph at once is not traceable for problems with close holds together
	void buildGraph(std::vector<int>& iInitStance, mANNBase* _successNet, mANNBase* _costNet, int _diff_transition_count = -1)
	{
		initialStance = iInitStance;

		if (mContext->holds_body.size() == 0)
		{
			return;
		}

		std::vector<int> expandNodes;
		expandNodes.push_back(mRootGraph);
		stance_nodes[mRootGraph].isItExpanded = true;
		//add all nodes connected to initial stance.
		while (expandNodes.size() > 0)
		{
			int exanpd_stance_id = expandNodes[0];
			expandNodes.erase(expandNodes.begin());

			std::vector<std::vector<int>> mInitialList = mClimberSampler->getListOfStanceSamples(stance_nodes[exanpd_stance_id].hold_ids, iInitStance, true);

			for (unsigned int i = 0; i < mInitialList.size(); i++)
			{
				int sId = insertNewGraphNode(mInitialList[i]);//addGraphNode(exanpd_stance_id, mInitialList[i]);

				updateNeighborStances(exanpd_stance_id, sId, _successNet, _costNet);

				if (!stance_nodes[sId].isItExpanded)//&& !mTools::isInSetIDs(sId, expandNodes)
				{
					expandNodes.push_back(sId);
					stance_nodes[sId].isItExpanded = true;
				}
			}
		}

		int fStance = findStanceFrom(iInitStance);

		/*std::vector<int> tStance(4, -1);
		tStance[2] = tStance[3] = 6;

		std::vector<int> _fStance(4, -1);
		_fStance[0] = 2;
		_fStance[1] = 3;
		_fStance[2] = 4;
		_fStance[3] = 5;*/
		float threshold_dis = 2.5f;
		for (unsigned int h = 0; h < mContext->holds_body.size(); h++)
		{
			std::vector<std::vector<int>> list_stances;
			mClimberSampler->getListOfStanceAroundHandHold(h, list_stances);

			unsigned int update_neighbor_from = stance_nodes.size();
			for (unsigned i = 0; i < list_stances.size(); i++)
			{
				std::vector<int>& sample_i = list_stances[i];

				insertNewGraphNode(sample_i);
			}

			for (unsigned i = update_neighbor_from; i < stance_nodes.size(); i++)
			{
				mStanceNode &node_i = stance_nodes[i];

				Vector3& midNewSample = node_i.midPoint;
				for (unsigned s = fStance; s < stance_nodes.size(); s++)
				{
					mStanceNode &node_s = stance_nodes[s];

					int cDiff = mTools::getDiffBtwSetASetB(node_i.hold_ids, node_s.hold_ids);
					float _cDis = (midNewSample - node_s.midPoint).norm();

					bool is_condition_met = (_cDis < threshold_dis);// (2.0f * mClimberSampler->climberRadius)); // changed from r/2 to r
					
					/*if (cDiff <= 2)
					{
						is_condition_met = (_cDis < mClimberSampler->climberRadius / 2.0f);
					}*/

					if (is_condition_met)
					{
						bool add_transition = false;
						for (unsigned int sb = 0; sb < 4 && !add_transition; sb++)
						{
							if (node_s.hold_ids[sb] != -1 && node_i.hold_ids[sb] != -1)
							{
								float sbDis = (mContext->getHoldPos(node_s.hold_ids[sb]) - mContext->getHoldPos(node_i.hold_ids[sb])).norm();
								
								// if you find one limb move less than climberRadius
								add_transition = (sbDis < threshold_dis); // 1.5f * mClimberSampler->climberRadius);

								/*if (cDiff <= 2)
								{
									add_transition = (sbDis < mClimberSampler->climberRadius);
								}*/
							}
						}

						is_condition_met = add_transition;
					}

					if (_diff_transition_count != -1 && is_condition_met)
					{
						is_condition_met = cDiff == _diff_transition_count;
					}

					if (is_condition_met)
					{
						updateNeighborStances(s, i, _successNet, _costNet);
					}
				}

			}
		}

		minmax.init(stance_nodes.size());
		printf("%d, %d", stance_nodes.size(), maxGraphDegree);

		for (unsigned int i = 0; i < stance_nodes.size(); i++)
		{
			updating_graph_node_costs.push_back(i);
		}

		return;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////// updating graph costs using neural networks
	bool useNeuralNetworkCost()
	{
		return updating_graph_node_costs.size() == 0;
	}

	float getPredictedCostVal(mANNBase* _costNet, mTransitionData& mSample, const std::vector<int>& _in_from, const std::vector<int>& _in_to)
	{
		float total_cost = 0.0f;

		VectorXf cCostState = mSample.getCostState();//h
		VectorXf cCostVal = VectorXf::Zero(1);
		_costNet->getMachineLearningOutput(cCostState.data(), cCostVal.data());
		total_cost += cCostVal[0];

		return total_cost;
	}

	static float getEdgeCost(float success_rate, float control_cost, const std::vector<int>& _tIds, float var_cost)
	{
		float free_body = float(int(_tIds[0] < 0) + int(_tIds[1] < 0) + int(_tIds[2] < 0) + int(_tIds[3] < 0));
		
		float coeff = 1.0f - success_rate;
		float edge_cost = control_cost;

		edge_cost += const_risk_cost * (exp(5.0f * coeff) - 1.0f);// coeff;

		edge_cost += free_body * const_var_cost;

		/*if (success_rate < 0.5f)
			edge_cost = 100;
		else
			edge_cost = 1;*/
		
		return edge_cost;
	}

	unsigned int updateGraphNodeNueralNetworkCost(mANNBase* _successNet, mANNBase* _costNet, const int& numUpdate)
	{
		if (updating_graph_node_costs.size() == 0)
		{
			if (stance_nodes[mRootGraph].number_updates_neural_networks < numUpdate)
			{
				for (unsigned int i = 0; i < stance_nodes.size(); i++)
				{
					updating_graph_node_costs.push_back(i);
					stance_nodes[i].number_updates_neural_networks = numUpdate;
				}
			}
		}
		else
		{
			// do breath first seach for updating graph nodes
			bool flag_continue = true;
			int counter_update = 0;
			int max_update_num = 1000;
			while (flag_continue)
			{
				int fIndex = *updating_graph_node_costs.begin();
				updating_graph_node_costs.erase(updating_graph_node_costs.begin());
				mStanceNode& fNode = stance_nodes[fIndex];

				stance_nodes[fIndex].number_updates_neural_networks = numUpdate;

				for (unsigned int c = 0; c < fNode.childStanceIds.size(); c++)
				{
					mStanceNode& childNode = stance_nodes[fNode.childStanceIds[c]];

					updateNNCostInfo(fNode, childNode.hold_ids, c, _successNet, _costNet);

					counter_update++;
				}

				if (counter_update > max_update_num || updating_graph_node_costs.size() == 0)
				{
					flag_continue = false;
				}
			}
		}

		return updating_graph_node_costs.size();
	}

	void initializeOpenListAStarPrune()
	{
		openListPath.clear();
		std::vector<int> rootPath; rootPath.push_back(0);
		openListPath.push_back(mStancePathNode(rootPath, 0, 0));
	}

	bool updateGraphNN(int _fNode, int _tNode)
	{
		for (unsigned int i = 0; i < stance_nodes[_fNode].childStanceIds.size(); i++)
		{
			if (stance_nodes[_fNode].childStanceIds[i] == _tNode)
			{
				stance_nodes[_fNode].cost_transition_to_child[i] += _MaxTriedTransitionCost;
				return true;
			}
		}
		return false;
	}

	float getNeuralNetworkCost(int _fNode, int _tNode)
	{
		if (_fNode >= 0 && _fNode < (int)(stance_nodes.size()))
		{
			for (unsigned int i = 0; i < stance_nodes[_fNode].childStanceIds.size(); i++)
			{
				if (stance_nodes[_fNode].childStanceIds[i] == _tNode)
				{
					return stance_nodes[_fNode].cost_neuralnetwork_to_child[i];
				}
			}
		}
		return -100.0f;
	}

	float getNeuralNetworkSuccessRate(int _fNode, int _tNode)
	{
		if (_fNode >= 0 && _fNode < (int)(stance_nodes.size()))
		{
			for (unsigned int i = 0; i < stance_nodes[_fNode].childStanceIds.size(); i++)
			{
				if (stance_nodes[_fNode].childStanceIds[i] == _tNode)
				{
					return stance_nodes[_fNode].cost_success_rate_to_child[i];
				}
			}
		}
		return -100.0f;
	}

	float solveGraph()
	{
		float ret_min_val = FLT_MAX;
		if (goal_stances.size() > 0)
		{
			solveAStarPrune(ret_min_val);
		}
		else
		{
			/////////////////////////////////////////////////////// TODO: if graph is not complete do sth
			retPath.clear();
		}
		return ret_min_val;
	}

	std::vector<std::vector<int>> returnPath()
	{
		return returnPathFrom(retPath);
	}

	int findStanceFrom(const std::vector<int>& _fStance)
	{
		uint32_t key = stanceToKey(_fStance);
		std::unordered_map<uint32_t, int>::iterator it = stanceToNode.find(key);
		if (it == stanceToNode.end())
			return -1;
		return it->second;
	}

	std::vector<int> getStanceGraph(int _graphNodeId)
	{
		return stance_nodes[_graphNodeId].hold_ids;
	}

	unsigned int getNumOfChildrenOfStance(int _graphNodeId)
	{
		return stance_nodes[_graphNodeId].childStanceIds.size();
	}

	int getChildStanceID(int _graphNodeId, int child_num)
	{
		return stance_nodes[_graphNodeId].childStanceIds[child_num];
	}

	int getIndexStanceNode(int indexPath)
	{
		int j = retPath.size() - 1 - indexPath;
		if (j < (int)retPath.size())
		{
			return retPath[j];
		}
		return mRootGraph;
	}

	void emptyGraph()
	{
		updating_graph_node_costs.clear();
		retPath.clear();
		openListPath.clear();
		goal_stances.clear();
		stance_nodes.clear();
		stanceToNode.clear();
		m_found_paths.clear();
		minmax.~DynamicMinMax();

		maxGraphDegree = 0;
		stanceToNode.rehash(1000003);  //a prime number of buckets to minimize different keys mapping to same bucket
		stance_nodes.reserve(20000);// = std::vector<mStanceNode>(20000, mStanceNode(std::vector<int>(4,-1)));

		mRootGraph = addGraphNode(-1, std::vector<int>(4, -1));

		initializeOpenListAStarPrune();

		initialStance.clear(); // if size is zero (do random search around trunk when h=-1 is selected)
	}

	float getMaxFailedPredictionCost()
	{
		return _MaxFailedTransition_PredictedCost;
	}

	std::vector<int> retPath;
	std::list<mStancePathNode> openListPath;
	std::list<int> updating_graph_node_costs;
	std::vector<std::vector<int>> m_found_paths;
	std::unordered_map<uint32_t, int> stanceToNode; //key is uint64_t computed from a stance (i.e., a vector of 4 ints), data is index to the stance nodes vector
	DynamicMinMax minmax;

	mSampler* mClimberSampler;
	SimulationContext* mContext;
private:
	int mRootGraph;
	std::vector<int> goal_stances;
	std::vector<mStanceNode> stance_nodes;
	unsigned int maxGraphDegree;
	std::vector<int> initialStance;
	float _MaxTriedTransitionCost;

	std::vector<int> reversePath(std::vector<int>& _iPath)
	{
		std::vector<int> nPath;
		for (int i = _iPath.size() - 1; i >= 0; i--)
		{
			nPath.push_back(_iPath[i]);
		}
		return nPath;
	}

	int getChildIndex(int _fNode, int _tNode)
	{
		for (unsigned int i = 0; i < stance_nodes[_fNode].childStanceIds.size(); i++)
		{
			if (stance_nodes[_fNode].childStanceIds[i] == _tNode)
			{
				return i;
			}
		}
		return -1;
	}

	float getHeuristicValue(mStanceNode* fNode, mStanceNode* cNode, int c)
	{
		float val = fNode->cost_neuralnetwork_to_child[c]; //
		if (val < 0)
		{
			int notifyme = 1;
		}
		if (val < 0)
			val = fNode->cost_moveLimb_to_child[c] + cNode->nodeCost;

		//if (val < 0)
		//	exit(0);
		return val;
	}

	void applyDijkstraAll()
	{
		//Run Dijkstra backwards, initializing all goal nodes to zero cost and others to infinity.
		//This results in each node having the cost as the minimum cost to go towards any of the goals.
		//This will then be used as the heuristic in A* prune. Note that if the climber is able to make
		//all the moves corresponding to all edges, the heuristic equals the actual cost, and A* will be optimal.
		//In case the climber fails, this function will be called again, i.e., the heuristic will be updated
		//after each failure.
		//Note: minmax is a DynamicMinMax instance - almost like std::priority_queue, but supports updating 
		//the priorities of elements at a logN cost without removal & insertion. 
		for (unsigned int i = 0; i < stance_nodes.size(); i++)
		{
			stance_nodes[i].g_AStar = FLT_MAX;
			stance_nodes[i].h_AStar = FLT_MAX;  //this is the output value, i.e., the "heuristic" used by A* prune
			stance_nodes[i].dijkstraVisited = false;
			minmax.setValue(i, FLT_MAX);
		}

		for (unsigned int i = 0; i < goal_stances.size(); i++)
		{
			stance_nodes[goal_stances[i]].h_AStar = 0;
			minmax.setValue(goal_stances[i], 0);
		}

		//	int sIndex = findStanceFrom(initialStance);
		int nVisited = 0;
		while (nVisited < (int)stance_nodes.size())
		{
			//get the node with least cost (initially, all goal nodes have zero cost)
			mStanceNode* cNode = &stance_nodes[minmax.getMinIdx()];

			//loop over neighbors (childStanceIds really contains all the neighbors)
			//and update their costs

			for (unsigned int f = 0; f < cNode->childStanceIds.size(); f++)
//			for (unsigned int f = 0; f < cNode->parentStanceIds.size(); f++)
			{
//				mStanceNode* fNode = &stance_nodes[cNode->parentStanceIds[f]];
//				int c = getChildIndex(cNode->parentStanceIds[f], cNode->stanceIndex);
				mStanceNode* fNode = &stance_nodes[cNode->childStanceIds[f]];
				int c = getChildIndex(cNode->childStanceIds[f], cNode->stanceIndex);
				if (c == -1)
					continue;
				float nH = cNode->h_AStar + fNode->cost_transition_to_child[c] + getHeuristicValue(fNode, cNode, c);

	//			if (nH < 0)
	//				exit(0);

				if (!fNode->dijkstraVisited)
				{
					if (nH < fNode->h_AStar)
					{
						fNode->h_AStar = nH;
						fNode->bChildStanceIndex = cNode->stanceIndex;
						minmax.setValue(fNode->stanceIndex, nH);
					}
				}
			} //all neighbors
			//Mark the node as visited. Also set it's priority to FLT_MAX so that it will not be returned by minmax.getMinIdx().
			//Note that since dijkstraVisited is now true, the priority will not be updated again and will stay at FLT_MAX for the remainder of the algorithm.
			cNode->dijkstraVisited = true;
			minmax.setValue(cNode->stanceIndex, FLT_MAX);
			nVisited++;
		}

		return;
	}

	std::vector<int> solveAStarPrune(float& ret_min_val)
	{
		int k = 100; // k shortest path in the paper
		int max_number_paths = max<int>(maxGraphDegree + 20, k);

		int sIndex = findStanceFrom(initialStance);

		applyDijkstraAll();

		bool flag_continue = true;

		if (openListPath.size() == 0)
		{
			initializeOpenListAStarPrune();
		}

		int counter_not_found = 0;

		while (flag_continue)
		{
			counter_not_found++;

			mStancePathNode fNode = openListPath.front();
			openListPath.pop_front();// erase(openListPath.begin());

			int eStanceIndex = fNode._path[fNode._path.size() - 1];

			if (isGoalFound(eStanceIndex))
			{
				retPath = fNode._path;
				ret_min_val = fNode.mG + fNode.mH;
				if (mTools::isInSetIDs(sIndex, retPath))
				{
					if (!mTools::isInSampledStanceSet(retPath, m_found_paths))
					{
						flag_continue = false;
					}
					else
					{
						continue;
					}
				}
				else
				{
					retPath.clear();
				}
			}
			else
			{
				if (counter_not_found > 1e6)
				{
					retPath = fNode._path;
					return retPath;
				}
			}

			if (flag_continue)
			{
				mStanceNode& stanceNode = stance_nodes[eStanceIndex];
				for (unsigned int c = 0; c < stanceNode.childStanceIds.size(); c++)
				{
					mStanceNode& stanceChild = stance_nodes[stanceNode.childStanceIds[c]];

					if (mTools::isInSetIDs(stanceChild.stanceIndex, fNode._path)) // no loop
						continue;

					float nG = fNode.mG + stanceNode.cost_transition_to_child[c] + getHeuristicValue(&stanceNode, &stanceChild, c);
					float nH = stanceChild.h_AStar;

					//AALTO_ASSERT1(nH >= 0);
					std::vector<int> nPath = fNode.addToEndPath(stanceChild.stanceIndex);

					mStancePathNode nStancePath(nPath, nG, nH);

					insertNodePath(openListPath, nStancePath, max_number_paths);
				}
			}

			if (openListPath.size() == 0)
			{
				flag_continue = false;
			}
		}

		return retPath;
	}

	void insertNodePath(std::list<mStancePathNode>& openList, mStancePathNode& nNode, int max_num)
	{
		int k = 0;
		std::list<mStancePathNode>::iterator iter_pathNode = openList.begin();
		while (iter_pathNode != openList.end())
		{
			mStancePathNode& n_i = *iter_pathNode;//openList[i];
			float cF = n_i.mG + n_i.mH;
			float nF = nNode.mG + nNode.mH;
			if (nF < cF)
			{
				break;
			}
			iter_pathNode++;
			k++;
		}
		if (k < max_num)
			openList.insert(iter_pathNode, nNode);

		if ((int)openList.size() > max_num)
		{
			openList.pop_back();
		}
		return;
	}

	//////////////////////////////////////////////////////// the huristic values to goal is not admisible

	float getCostAtNode(std::vector<int>& iCStance, bool printDebug = false)
	{
		float k_crossing = 100;
		float k_hanging_hand = 200 + 50; // 200
		float k_hanging_leg = 10 + 10; // 10
									   //		float k_hanging_more_than2 = 0;//100;
		float k_matching = 100;
		float k_dis = 1000;

		float _cost = 0.0f;

		// punish for hanging more than one limb
		int counter_hanging = 0;
		for (unsigned int i = 0; i < iCStance.size(); i++)
		{
			if (iCStance[i] == -1)
			{
				counter_hanging++;

				if (i >= 2)
				{
					// punish for having hanging hand
					_cost += k_hanging_hand;
					if (printDebug) rcPrintString("Hanging hand!");
				}
				else
				{
					// punish for having hanging hand
					_cost += k_hanging_leg;
				}
			}
		}

		// crossing hands
		if (iCStance[2] != -1 && iCStance[3] != -1)
		{
			Vector3 rHand = mContext->getHoldPos(iCStance[3]);
			Vector3 lHand = mContext->getHoldPos(iCStance[2]);

			if (rHand.x() < lHand.x())
			{
				_cost += k_crossing;
				if (printDebug) rcPrintString("Hands crossed!");
			}
		}

		// crossing feet
		if (iCStance[0] != -1 && iCStance[1] != -1)
		{
			Vector3 lLeg = mContext->getHoldPos(iCStance[0]);
			Vector3 rLeg = mContext->getHoldPos(iCStance[1]);

			if (rLeg.x() < lLeg.x())
			{
				_cost += k_crossing;
				if (printDebug) rcPrintString("Legs crossed!");
			}
		}

		// crossing hand and foot
		for (unsigned int i = 0; i <= 1; i++)
		{
			if (iCStance[i] != -1)
			{
				Vector3 leg = mContext->getHoldPos(iCStance[i]);
				for (unsigned int j = 2; j <= 3; j++)
				{
					if (iCStance[j] != -1)
					{
						Vector3 hand = mContext->getHoldPos(iCStance[j]);

						if (hand.z() <= leg.z())
						{
							_cost += k_crossing;
						}
					}
				}
			}
		}

		//feet matching
		if (iCStance[0] == iCStance[1])
		{
			_cost += k_matching;
			if (printDebug) rcPrintString("Feet matched!");
		}

		//punishment for hand and leg being close
		for (unsigned int i = 0; i <= 1; i++)
		{
			if (iCStance[i] != -1)
			{
				Vector3 leg = mContext->getHoldPos(iCStance[i]);
				for (unsigned int j = 2; j <= 3; j++)
				{
					if (iCStance[j] != -1)
					{
						Vector3 hand = mContext->getHoldPos(iCStance[j]);

						float cDis = (hand - leg).norm();

						const float handAndLegDistanceThreshold = 0.5f;//mClimberSampler->climberRadius / 2.0f;
						if (cDis < handAndLegDistanceThreshold)
						{
							cDis /= handAndLegDistanceThreshold;
							_cost += k_dis*max(0.0f, 1.0f - cDis);
							if (printDebug) rcPrintString("Hand and leg too close! v = %f", k_dis*max(0.0f, 1.0f - cDis));
						}
					}
				}
			}
		}

		return _cost;
	}

	std::vector<Vector3> getExpectedPositionSigma(Vector3 midPoint)
	{
		float r = mClimberSampler->climberRadius;

		std::vector<float> theta_s;
		theta_s.push_back(PI + PI / 4.0f);
		theta_s.push_back(1.5 * PI + PI / 4.0f);
		theta_s.push_back(0.5 * PI + PI / 4.0f);
		theta_s.push_back(PI / 4.0f);

		std::vector<Vector3> expectedPoses;
		for (unsigned int i = 0; i < theta_s.size(); i++)
		{
			Vector3 iDir(cosf(theta_s[i]), 0.0f, sinf(theta_s[i]));
			expectedPoses.push_back(midPoint + (r / 2.0f) * iDir);
		}

		return expectedPoses;
	}

	float getDisFromStanceToStance(std::vector<int>& si, std::vector<int>& sj)
	{
		float cCount = 0.0f;
		std::vector<Vector3> hold_points_i;
		Vector3 midPoint1 = mContext->getHoldStancePosFrom(si, hold_points_i, cCount);
		//		std::vector<Vector3> e_hold_points_i = getExpectedPositionSigma(midPoint1);

		std::vector<Vector3> hold_points_j;
		Vector3 midPoint2 = mContext->getHoldStancePosFrom(sj, hold_points_j, cCount);
		//		std::vector<Vector3> e_hold_points_j = getExpectedPositionSigma(midPoint2);

		float cCost = 0.0f;
		float hangingLimbExpectedMovement = 2.0f;
		for (unsigned int i = 0; i < si.size(); i++)
		{
			float coeff_cost = 1.0f;
			if (si[i] != sj[i])
			{
				Vector3 pos_i;
				if (si[i] != -1)
				{
					pos_i = hold_points_i[i];
				}
				else
				{
					//pos_i = e_hold_points_i[i];
					cCost += 0.5f;
					continue;
				}
				Vector3 pos_j;
				if (sj[i] != -1)
				{
					pos_j = hold_points_j[i];
				}
				else
				{
					//pos_j = e_hold_points_j[i];
					cCost += hangingLimbExpectedMovement;
					continue;
				}

				//favor moving hands
				if (i >= 2)
					coeff_cost = 0.9f;

				cCost += coeff_cost * (pos_i - hold_points_j[i]).squaredNorm();
			}
			else
			{
				if (sj[i] == -1)
				{
					cCost += hangingLimbExpectedMovement;
				}
			}
		}

		return sqrtf(cCost);
	}

	bool firstHoldIsLower(int hold1, int hold2)
	{
		if (hold1 == -1 && hold2 == -1)
			return false;
		if (hold1 != -1 && mContext->getHoldPos(hold1).z() < mContext->getHoldPos(hold2).z())
		{
			return true;
		}
		//first hold is "free" => we can't really know 
		return false;
	}

	float getCostMovementLimbs(std::vector<int>& si, std::vector<int>& sj)
	{
		float k_dis = 1.0f;
		float k_2limbs = 120.0f;//20.0f;
		float k_pivoting_close_dis = 500.0f;

		//First get the actual distance between holds. We scale it up 
		//as other penalties are not expressed in meters
		float _cost = k_dis * getDisFromStanceToStance(si, sj);

		//penalize moving 2 limbs, except in "ladder climbing", i.e., moving opposite hand and leg
		bool flag_punish_2Limbs = true;
		bool is2LimbsPunished = false;

		if (mTools::getDiffBtwSetASetB(si, sj) > 1.0f)
		{

			if (si[0] != sj[0] && si[3] != sj[3] && firstHoldIsLower(si[0], sj[0]))
			{
				flag_punish_2Limbs = false;
				if (sj[0] != -1 && sj[3] != -1 && mContext->getHoldPos(sj[3]).x() - mContext->getHoldPos(sj[0]).x() < 0.5f)
					flag_punish_2Limbs = true;
				if (sj[0] != -1 && sj[3] != -1 && mContext->getHoldPos(sj[3]).z() - mContext->getHoldPos(sj[0]).z() < 0.5f)
					flag_punish_2Limbs = true;
			}

			if (si[1] != sj[1] && si[2] != sj[2] && firstHoldIsLower(si[1], sj[1]))
			{
				flag_punish_2Limbs = false;
				if (sj[1] != -1 && sj[2] != -1 && mContext->getHoldPos(sj[1]).x() - mContext->getHoldPos(sj[2]).x() < 0.5f)
					flag_punish_2Limbs = true;
				if (sj[1] != -1 && sj[2] != -1 && mContext->getHoldPos(sj[2]).z() - mContext->getHoldPos(sj[1]).z() < 0.5f)
					flag_punish_2Limbs = true;
			}

			if (flag_punish_2Limbs)
				_cost += k_2limbs;
		}

		// calculating the stance during the transition
		std::vector<int> sn(4);
		int count_free_limbs = 0;
		for (unsigned int i = 0; i < si.size(); i++)
		{
			if (si[i] != sj[i])
			{
				sn[i] = -1;
				count_free_limbs++;
			}
			else
			{
				sn[i] = si[i];
			}
		}
		// free another
		if (count_free_limbs >= 2 && mTools::getDiffBtwSetASetB(si, sj) == 1.0f)
			_cost += k_2limbs;

		// punish for pivoting!!!
		float v = 0.0f;
		float max_dis = -FLT_MAX;
		for (unsigned int i = 0; i <= 1; i++)
		{
			if (sn[i] != -1)
			{
				Vector3 leg = mContext->getHoldPos(sn[i]);
				for (unsigned int j = 2; j <= 3; j++)
				{
					if (sn[j] != -1)
					{
						Vector3 hand = mContext->getHoldPos(sn[j]);

						float cDis = (hand - leg).norm();

						if (max_dis < cDis)
							max_dis = cDis;
					}
				}
			}
		}
		if (max_dis >= 0 && max_dis < mClimberSampler->climberRadius / 2.0f && count_free_limbs > 1.0f)
		{
			v += k_pivoting_close_dis;
		}
		_cost += v;

		return _cost;
	}

	float getCostTransition(std::vector<int>& si, std::vector<int>& sj)
	{
		return 1.0f;
	}

	////////////////////////////////////////////////////////
	std::vector<int> returnPathFrom(int iStanceIndex)
	{
		std::vector<int> rPath;

		int cIndex = iStanceIndex;

		int counter = 0;
		while (cIndex >= 0) // cIndex == 0 is (-1,-1,-1,-1)
		{
			mStanceNode nSi = stance_nodes[cIndex];
			rPath.push_back(nSi.stanceIndex);

			cIndex = nSi.bFatherStanceIndex;

			counter++;

			if (counter > (int)stance_nodes.size())
				break;
		}

		return rPath;
	}

	std::vector<std::vector<int>> returnPathFrom(std::vector<int>& pathIndex)
	{
		std::vector<std::vector<int>> lPath;
		for (int i = pathIndex.size() - 1; i >= 0; i--)
		{
			mStanceNode nSi = stance_nodes[pathIndex[i]];
			lPath.push_back(nSi.hold_ids);
		}
		return lPath;
	}

	bool isGoalFound(int iStanceIndex)
	{
		return mTools::isInSetIDs(iStanceIndex, goal_stances);
	}

	bool isLoopCreated(int iStanceIndex)
	{
		int cIndex = iStanceIndex;

		int counter = 0;
		while (cIndex != 0)
		{
			mStanceNode nSi = stance_nodes[cIndex];

			cIndex = nSi.bFatherStanceIndex;

			counter++;

			if (counter > (int)stance_nodes.size())
			{
				return true;
			}
		}
		return false;
	}

	int insertNewGraphNode(std::vector<int>& _sStance)
	{
		AALTO_ASSERT1(_sStance.size() == 4);

		int stance_id = findStanceFrom(_sStance);

		Vector3 goalPos = mContext->getGoalPos();

		if (stance_id == -1)
		{
			mStanceNode nStance(_sStance);
			nStance.stanceIndex = stance_nodes.size();
			nStance.midPoint = mContext->getMidPointHoldStance(_sStance);
			nStance.theta_dir = mContext->getThetaDirHoldStance(_sStance);

			nStance.nodeCost = getCostAtNode(_sStance);

			if (nStance.hold_ids[3] != -1) // search for right hand
			{
				Vector3 holdPos = mContext->getHoldPos(nStance.hold_ids[3]);
				if ((holdPos - goalPos).norm() < 0.1f)
				{
					mTools::addToSetIDs(nStance.stanceIndex, goal_stances);
				}
			}
			if (nStance.hold_ids[2] != -1) // search for left hand
			{
				Vector3 holdPos = mContext->getHoldPos(nStance.hold_ids[2]);
				if ((holdPos - goalPos).norm() < 0.1f)
				{
					mTools::addToSetIDs(nStance.stanceIndex, goal_stances);
				}
			}

			//if (nStance.hold_ids[2] != -1 && nStance.hold_ids[3] != -1) // search for left hand
			//{
			//	Vector3 holdPos1 = mContext->getHoldPos(nStance.hold_ids[2]);
			//	Vector3 holdPos2 = mContext->getHoldPos(nStance.hold_ids[3]);
			//	if ((holdPos1 - goalPos).norm() < 0.1f && (holdPos2 - goalPos).norm() < 0.1f)
			//	{
			//		mTools::addToSetIDs(nStance.stanceIndex, goal_stances);
			//	}
			//}


			stance_nodes.push_back(nStance);
			if (stance_nodes.size() % 100 == 0)  //don't printf every node, will slow things down
				printf("Number of nodes: %d\n", stance_nodes.size());
			int index = stance_nodes.size() - 1;
			stance_id = index;
			stanceToNode[stanceToKey(nStance.hold_ids)] = index;
		}

		return stance_id;
	}

	void updateNNCostInfo(mStanceNode& fNode, std::vector<int>& _tChildHoldIds, int childIndex, mANNBase* _successNet, mANNBase* _costNet)
	{
		if (!_successNet || !_costNet)
		{
			return;
		}

		mTransitionData mSample;
		mSample.fillInputStanceFrom(mContext, fNode.hold_ids, _tChildHoldIds);

		VectorXf cSuccessState = mSample.getSuccessState();
		VectorXf cSuccessRate = VectorXf::Zero(1);
		_successNet->getMachineLearningOutput(cSuccessState.data(), cSuccessRate.data());


		float total_cost = getPredictedCostVal(_costNet, mSample, fNode.hold_ids, _tChildHoldIds);

		fNode.cost_success_rate_to_child[childIndex] = cSuccessRate[0];
		fNode.cost_neuralnetwork_to_child[childIndex] 
			= getEdgeCost(cSuccessRate[0], total_cost, _tChildHoldIds, const_var_cost);
		
		return;
	}

	float getSuccessRate(mANNBase* _successNet, int _fromIndex, int _toIndex)
	{
		mTransitionData mSample;
		mSample.fillInputStanceFrom(mContext, stance_nodes[_fromIndex].hold_ids, stance_nodes[_toIndex].hold_ids);

		VectorXf cSuccessState = mSample.getSuccessState();
		VectorXf cSuccessRate = VectorXf::Zero(1);
		_successNet->getMachineLearningOutput(cSuccessState.data(), cSuccessRate.data());

		return cSuccessRate[0];
	}

	void updateNeighborStances(int _fromIndex, int stance_id, mANNBase* _successNet, mANNBase* _costNet)
	{
		unsigned int max_degree_graph = 2500;

		if (_fromIndex == stance_id)
			return;
		float s = getSuccessRate(_successNet, _fromIndex, stance_id);
		if (s >= 0.5f)
		{
			if (mTools::addToSetIDs(stance_nodes[stance_id].stanceIndex, stance_nodes[_fromIndex].childStanceIds))
			{
				stance_nodes[_fromIndex].cost_moveLimb_to_child.push_back(getCostMovementLimbs(stance_nodes[_fromIndex].hold_ids, stance_nodes[stance_id].hold_ids));
				stance_nodes[_fromIndex].cost_transition_to_child.push_back(getCostTransition(stance_nodes[_fromIndex].hold_ids, stance_nodes[stance_id].hold_ids));
				stance_nodes[_fromIndex].cost_neuralnetwork_to_child.push_back(-1.0f);
				stance_nodes[_fromIndex].cost_success_rate_to_child.push_back(-1.0f);

				int index_child = (int)(stance_nodes[_fromIndex].childStanceIds.size() - 1);

				updateNNCostInfo(stance_nodes[_fromIndex], stance_nodes[stance_id].hold_ids, index_child, _successNet, _costNet);

				//			mTools::addToSetIDs(stance_nodes[_fromIndex].stanceIndex, stance_nodes[stance_id].parentStanceIds);

				/*if (stance_nodes[_fromIndex].childStanceIds.size() > max_degree_graph)
				{
					if (stance_nodes[_fromIndex].cost_success_rate_to_child[index_child] >= 0 && stance_nodes[_fromIndex].cost_success_rate_to_child[index_child] < 0.5f)
					{
						stance_nodes[_fromIndex].childStanceIds.erase(stance_nodes[_fromIndex].childStanceIds.begin() + index_child);
						stance_nodes[_fromIndex].cost_moveLimb_to_child.erase(stance_nodes[_fromIndex].cost_moveLimb_to_child.begin() + index_child);
						stance_nodes[_fromIndex].cost_transition_to_child.erase(stance_nodes[_fromIndex].cost_transition_to_child.begin() + index_child);
						stance_nodes[_fromIndex].cost_neuralnetwork_to_child.erase(stance_nodes[_fromIndex].cost_neuralnetwork_to_child.begin() + index_child);
						stance_nodes[_fromIndex].cost_success_rate_to_child.erase(stance_nodes[_fromIndex].cost_success_rate_to_child.begin() + index_child);
					}
				}*/
				if (stance_nodes[_fromIndex].childStanceIds.size() > maxGraphDegree)
					maxGraphDegree = stance_nodes[_fromIndex].childStanceIds.size();
			}
		}
		s = getSuccessRate(_successNet, stance_id, _fromIndex);
		if (s >= 0.5f)
		{
			if (mTools::addToSetIDs(stance_nodes[_fromIndex].stanceIndex, stance_nodes[stance_id].childStanceIds))
			{
				stance_nodes[stance_id].cost_moveLimb_to_child.push_back(getCostMovementLimbs(stance_nodes[stance_id].hold_ids, stance_nodes[_fromIndex].hold_ids));
				stance_nodes[stance_id].cost_transition_to_child.push_back(getCostTransition(stance_nodes[stance_id].hold_ids, stance_nodes[_fromIndex].hold_ids));
				stance_nodes[stance_id].cost_neuralnetwork_to_child.push_back(-1.0f);
				stance_nodes[stance_id].cost_success_rate_to_child.push_back(-1.0f);

				int index_child = (int)(stance_nodes[stance_id].childStanceIds.size() - 1);

				updateNNCostInfo(stance_nodes[stance_id], stance_nodes[_fromIndex].hold_ids, index_child, _successNet, _costNet);

				//			mTools::addToSetIDs(stance_nodes[stance_id].stanceIndex, stance_nodes[_fromIndex].parentStanceIds);
				/*if (stance_nodes[stance_id].childStanceIds.size() > max_degree_graph)
				{
					if (stance_nodes[stance_id].cost_success_rate_to_child[index_child] >= 0 && stance_nodes[stance_id].cost_success_rate_to_child[index_child] < 0.5f)
					{
						stance_nodes[stance_id].childStanceIds.erase(stance_nodes[stance_id].childStanceIds.begin() + index_child);
						stance_nodes[stance_id].cost_moveLimb_to_child.erase(stance_nodes[stance_id].cost_moveLimb_to_child.begin() + index_child);
						stance_nodes[stance_id].cost_transition_to_child.erase(stance_nodes[stance_id].cost_transition_to_child.begin() + index_child);
						stance_nodes[stance_id].cost_neuralnetwork_to_child.erase(stance_nodes[stance_id].cost_neuralnetwork_to_child.begin() + index_child);
						stance_nodes[stance_id].cost_success_rate_to_child.erase(stance_nodes[stance_id].cost_success_rate_to_child.begin() + index_child);
					}
				}*/
				if (stance_nodes[stance_id].childStanceIds.size() > maxGraphDegree)
					maxGraphDegree = stance_nodes[stance_id].childStanceIds.size();
			}
		}
		return;
	}

	int addGraphNode(int _fromIndex, std::vector<int>& _sStance)
	{
		AALTO_ASSERT1(_sStance.size() == 4);
		mStanceNode nStance(_sStance);
		nStance.stanceIndex = stance_nodes.size();
		nStance.midPoint = mContext->getMidPointHoldStance(_sStance);
		nStance.theta_dir = mContext->getThetaDirHoldStance(_sStance);

		Vector3 goalPos = mContext->getGoalPos();

		if (_fromIndex == -1)
		{
			nStance.nodeCost = 0.0f;

			stance_nodes.push_back(nStance);
			int index = stance_nodes.size() - 1;
			stanceToNode[stanceToKey(nStance.hold_ids)] = index;
			return index;
		}

		int stance_id = findStanceFrom(nStance.hold_ids);

		if (stance_id == -1)
		{
			nStance.nodeCost = getCostAtNode(_sStance);

			if (nStance.hold_ids[3] != -1) // search for right hand
			{
				Vector3 holdPos = mContext->getHoldPos(nStance.hold_ids[3]);
				if ((holdPos - goalPos).norm() < 0.1f)
				{
					mTools::addToSetIDs(nStance.stanceIndex, goal_stances);
				}
			}
			if (nStance.hold_ids[2] != -1) // search for left hand
			{
				Vector3 holdPos = mContext->getHoldPos(nStance.hold_ids[2]);
				if ((holdPos - goalPos).norm() < 0.1f)
				{
					mTools::addToSetIDs(nStance.stanceIndex, goal_stances);
				}
			}

			stance_nodes.push_back(nStance);
			if (stance_nodes.size() % 100 == 0)  //don't printf every node, will slow things down
				printf("Number of nodes: %d\n", stance_nodes.size());
			int index = stance_nodes.size() - 1;
			stance_id = index;
			stanceToNode[stanceToKey(nStance.hold_ids)] = index;
		}
		else
		{
			if (stance_nodes[stance_id].stanceIndex == _fromIndex)
			{
				return stance_nodes[stance_id].stanceIndex;
			}
		}

		//if (mTools::addToSetIDs(stance_nodes[_fromIndex].stanceIndex, stance_nodes[stance_id].parentStanceIds))
		//{
		//	if (stance_nodes[stance_id].parentStanceIds.size() > maxGraphDegree)
		//		maxGraphDegree = stance_nodes[stance_id].parentStanceIds.size();
		//}

		if (mTools::addToSetIDs(stance_nodes[stance_id].stanceIndex, stance_nodes[_fromIndex].childStanceIds))
		{
			stance_nodes[_fromIndex].cost_moveLimb_to_child.push_back(getCostMovementLimbs(stance_nodes[_fromIndex].hold_ids, stance_nodes[stance_id].hold_ids));
			stance_nodes[_fromIndex].cost_transition_to_child.push_back(getCostTransition(stance_nodes[_fromIndex].hold_ids, stance_nodes[stance_id].hold_ids));
			stance_nodes[_fromIndex].cost_neuralnetwork_to_child.push_back(-1.0f);
			stance_nodes[_fromIndex].cost_success_rate_to_child.push_back(-1.0f);

			if (stance_nodes[_fromIndex].childStanceIds.size() > maxGraphDegree)
				maxGraphDegree = stance_nodes[_fromIndex].childStanceIds.size();

//			mTools::addToSetIDs(stance_nodes[_fromIndex].stanceIndex, stance_nodes[stance_id].parentStanceIds);
		}

		//if (mTools::addToSetIDs(stance_nodes[stance_id].stanceIndex, stance_nodes[_fromIndex].parentStanceIds))
		//{
		//	if (stance_nodes[_fromIndex].parentStanceIds.size() > maxGraphDegree)
		//		maxGraphDegree = stance_nodes[_fromIndex].parentStanceIds.size();
		//}

		if (mTools::addToSetIDs(stance_nodes[_fromIndex].stanceIndex, stance_nodes[stance_id].childStanceIds))
		{
			stance_nodes[stance_id].cost_moveLimb_to_child.push_back(getCostMovementLimbs(stance_nodes[stance_id].hold_ids, stance_nodes[_fromIndex].hold_ids));
			stance_nodes[stance_id].cost_transition_to_child.push_back(getCostTransition(stance_nodes[stance_id].hold_ids, stance_nodes[_fromIndex].hold_ids));
			stance_nodes[stance_id].cost_neuralnetwork_to_child.push_back(-1.0f);
			stance_nodes[stance_id].cost_success_rate_to_child.push_back(-1.0f);

			if (stance_nodes[stance_id].childStanceIds.size() > maxGraphDegree)
				maxGraphDegree = stance_nodes[stance_id].childStanceIds.size();

//			mTools::addToSetIDs(stance_nodes[stance_id].stanceIndex, stance_nodes[_fromIndex].parentStanceIds);
		}

		return stance_id;
	}

}*mGraph;

class mTreeNode
{
public:
	mTreeNode(const BipedState& cState, int graph_index)
	{
		climber_state = cState;
		graph_index_node = graph_index;
		mFatherIndex = -1;
		cost_to_father = 0;
		nodeIndex = -1;
	}

	float getSumDisEndPosTo(std::vector<int>& iIDs, std::vector<Vector3>& iP)
	{
		float dis = 0.0f;

		for (unsigned int i = 0; i < iIDs.size(); i++)
		{
			if (iIDs[i] != -1)
			{
				dis += (climber_state.getEndPointPosBones(SimulationContext::ContactPoints::LeftLeg + i) - iP[i]).squaredNorm();
			}
		}

		return dis;
	}

	float getSumDisEndPosTo(Vector3& iP)
	{
		float dis = 0.0f;
		dis += (climber_state.getEndPointPosBones(SimulationContext::ContactPoints::LeftLeg) - iP).squaredNorm();
		dis += (climber_state.getEndPointPosBones(SimulationContext::ContactPoints::RightLeg) - iP).squaredNorm();
		dis += (climber_state.getEndPointPosBones(SimulationContext::ContactPoints::LeftArm) - iP).squaredNorm();
		dis += (climber_state.getEndPointPosBones(SimulationContext::ContactPoints::RightArm) - iP).squaredNorm();
		dis += (climber_state.getEndPointPosBones(SimulationContext::BodyName::BodyTrunk) - iP).squaredNorm();
		return sqrtf(dis);
	}

	BipedState climber_state;
	
	int graph_index_node;

	int mFatherIndex;
	int nodeIndex;
	std::vector<BipedState> statesFromFatherToThis;
	float cost_to_father;
	std::vector<int> mChildrenTreeNodes;
};