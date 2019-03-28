/*************************************************************************
 *                                                                       *
 * Climber's Interface done by Kourosh Naderi and Perttu Hamalainen      *
 * All rights reserved.  Email: firstname.lastname@aalto.fi              *
 *                                                                       *    
 *                                                                       *
 *************************************************************************/

#include <stdio.h>
#include <conio.h>
#include <chrono>
#include <list>
#include <unordered_map>
#include <queue>
#include <stdint.h>
#include "ode/ode.h"
#include "MathUtils.h"
#include <Math.h>

#include "mUnityODE\UnityOde.h"
#include "RenderClient.h"
#include "RenderCommands.h"
#include <Eigen/Eigen>
#include "FileUtils.h"
#include "Debug.h"

using namespace std::chrono;
using namespace std;
using namespace AaltoGames;
using namespace Eigen;

#ifdef _MSC_VER
#pragma warning(disable:4244 4305)  /* for VC++, no precision loss complaints */
#endif

/// select correct drawing functions 
#ifdef dDOUBLE
#define dsDrawBox dsDrawBoxD
#define dsDrawSphere dsDrawSphereD
#define dsDrawCylinder dsDrawCylinderD
#define dsDrawCapsule dsDrawCapsuleD
#endif

enum OptimizerTypes
{
	otCPBP = 0, otCMAES = 1, otLearnerCPBP = 2, otANNCPBP = 3
};
static const OptimizerTypes optimizerType = otCMAES;

float xyz[3] = {4.0,-5.0f,0.5f}; 
float lightXYZ[3] = {-10,-10,10};
float lookAt[3] = {0.75f,0,2.0};

static const bool useOfflinePlanning = optimizerType == otCMAES ? true : /*for C-PBP user can decide wheather do it online (false) or offline (true)*/ true;
static bool pause = false;
static bool flag_capture_video = false;

static const bool testClimber = true;

enum mEnumTestCaseClimber
{
	TestAngle = 0, TestCntroller = 1, RunLearnerRandomly = 2
};
enum mDemoTestClimber { DemoRouteFromFile = 1, DemoLongWall = 2, Demo45Wall = 3, 
	DemoRoute1 = 4, DemoRoute2 = 5, DemoRoute3 = 6, 
	DemoPillar = 7, DemoHorRotatedWall = 8, 
	DemoJump1 = 9, DemoJump2 = 10, DemoJump3 = 11, DemoJump4 = 12, DemoJump5 = 13, DemoJump6 = 14};

mEnumTestCaseClimber TestID = mEnumTestCaseClimber::RunLearnerRandomly;
mDemoTestClimber DemoID = mDemoTestClimber::DemoLongWall;


#define Vector2 Eigen::Vector2f
#define Vector3 Eigen::Vector3f
#define Vector4 Eigen::Vector4f

#define DEG_TO_RAD (M_PI/180.0)

#include "mTools.h"

//timing
high_resolution_clock::time_point t1;
void startPerfCount()
{
	t1 = high_resolution_clock::now();
}
int getDurationMs()
{
	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	return (int)(time_span.count()*1000.0);
}

class mFileHandler
{
private:
	std::string mfilename;
	bool mExists;
	FILE * rwFileStream;

	bool mOpenCreateFile(const std::string& name) 
	{
		fopen_s(&rwFileStream, name.c_str(), "r");
		if (rwFileStream == nullptr) 
		{
			fopen_s(&rwFileStream, name.c_str(), "w+");
		}
		return true;
	}

	bool mOpenCreateFile(const std::string& name, char* mode)
	{
		fopen_s(&rwFileStream, name.c_str(), mode);
		if (rwFileStream == nullptr)
		{
			return false;
		}
		return true;
	}

	template<typename Out>
	void split(const std::string &s, char delim, Out result)
	{
		std::stringstream ss;
		ss.str(s);
		std::string item;
		while (std::getline(ss, item, delim)) 
		{
			*(result++) = item;
		}
	}

public:

	std::vector<std::string> split(const std::string &s, char delim)
	{
		std::vector<std::string> elems;
		split(s, delim, std::back_inserter(elems));
		return elems;
	}

	void mCloseFile()
	{
		if (mExists)
		{
			fclose(rwFileStream);
			mExists = false;
		}
	}

	bool reWriteFile(std::vector<float>& iValues)
	{
		mCloseFile();
		
		fopen_s(&rwFileStream , mfilename.c_str(), "w");
		fprintf(rwFileStream, "#val \n");
		for (unsigned int i = 0; i < iValues.size(); i++)
		{
			fprintf(rwFileStream, "%f \n", iValues[i]);
		}
		mExists = true;
		return true;
	}

	bool openFileForWritingOn()
	{
		mCloseFile();
		
		fopen_s(&rwFileStream , mfilename.c_str(), "w");

		return true;
	}

	void writeLine(std::string _str)
	{
		fputs(_str.c_str(), rwFileStream);
		return;
	}

	void readFile(std::vector<std::vector<float>>& values)
	{
		const int max_count_read = 5000;
		if (!mExists)
		{
			return;
		}

		int found_comma = -1;
		while (!feof(rwFileStream))
		{
			char buff[max_count_read];
			char* mLine = fgets(buff, max_count_read, rwFileStream);
			if (mLine)
			{
				if (mLine[0] == '#')
				{
					found_comma = 0;
					for (unsigned int c = 0; c < strlen(mLine); c++)
					{
						if (mLine[c] == ',') found_comma++;
					}
				}
				else
				{
					std::vector<float> val;
					
					std::vector<std::string> _STR = split(buff, ',');
					val = std::vector<float>(_STR.size(), -1);

					for (unsigned int i = 0; i < val.size(); i++)
					{
						val[i] = (float)strtod(_STR[i].c_str(), NULL);
					}
					values.push_back(val);
				}
			}
			else
			{
				return;
			}
		}
		return;
	}

	bool exists()
	{
		return mExists;
	}

	mFileHandler(const std::string& iFilename)
	{
		mfilename = iFilename;
		mExists = mOpenCreateFile(iFilename);
	}

	mFileHandler(const std::string& iFilename, char* mode)
	{
		mfilename = iFilename;
		mExists = mOpenCreateFile(iFilename, mode);
	}

	~mFileHandler()
	{
		mCloseFile();
	}
};

#include "mSimulationContext.h"

// keeps track of high-level data
class mTransitionData
{
	std::vector<Vector3> _from; // 1st
	std::vector<int> _fromIDs; // 2nd
	Vector3 _meanFrom;
	std::vector<Vector3> _to; // third
	std::vector<int> _toIDs; // forth
	Vector3 _meanTo;

	int _diff_index;

	unsigned int controlSize; // seventh
	Eigen::VectorXf _controlPoints; // eighth

	BipedState _initialBodyState; // ninth

	static void pushStateFeature(int &featureIdx, float *stateFeatures, const Vector3& v)
	{
		stateFeatures[featureIdx++] = v.x();
		stateFeatures[featureIdx++] = v.y();
		stateFeatures[featureIdx++] = v.z();
	}

	Vector3 computeFullBodyState(Eigen::VectorXf& outState, Eigen::Quaternionf& out_root_inverse)
	{
		outState = Eigen::VectorXf(3 + (3 + 3) * 5);

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

		//dReal codeq[4];
		Vector4 root_angle = _initialBodyState.bodyStates[SimulationContext::BodySpine].getAngle();
		//codeq[0] = root_angle[0]; codeq[1] = root_angle[1]; codeq[2] = root_angle[2]; codeq[3] = root_angle[3];
		const Quaternionf root_rotation = Eigen::Quaternionf(root_angle[0], root_angle[1], root_angle[2], root_angle[3]);// ode2eigenq(codeq);
		const Quaternionf root_inverse = root_rotation.inverse();

		Vector3f root_pos(_initialBodyState.bodyStates[SimulationContext::BodySpine].getPos());//mContext->getBonePosition(SimulationContext::BodySpine));
		Vector3f root_vel(_initialBodyState.bodyStates[SimulationContext::BodySpine].getVel());// mContext->getBoneLinearVelocity(SimulationContext::BodySpine));
//		Vector3f root_angle_vel(_initialBodyState.bodyStates[SimulationContext::BodySpine].getAVel());// mContext->getBoneAngularVelocity(SimulationContext::BodySpine));

		pushStateFeature(featureIdx, outState.data(), root_inverse * root_vel);
//		pushStateFeature(featureIdx, outState.data(), root_inverse * root_angle_vel);

		Vector3f bone_pos;
		Vector3f bone_vel;

		Vector3f pos_relative_root;
		Vector3f vel_relative_root;
//		Vector3f bone_angle_angle_vel;

		for (int i = 0; i < nStateBones; i++)
		{
			pos_relative_root.noalias() = mContext->getBonePosition(stateBones[i]) - root_pos;
			vel_relative_root.noalias() = mContext->getBoneLinearVelocity(stateBones[i]) - root_vel;

			bone_pos.noalias() = root_inverse*pos_relative_root;
			bone_vel.noalias() = root_inverse*vel_relative_root;

//			bone_angle_angle_vel = mContext->getBoneAngularVelocity(stateBones[i]) - root_angle_vel;

			pushStateFeature(featureIdx, outState.data(), bone_pos);
			pushStateFeature(featureIdx, outState.data(), bone_vel);
//			pushStateFeature(featureIdx, outState.data(), bone_angle_angle_vel);
		}


		out_root_inverse = root_inverse;
		return root_pos;
	}

public:
//	const float _costThreshold[4] = {3000.0f, 8000.0f, 50000.0f, 50000.0f};//3000.0f for 1 limb, 5000 for 2 limbs

	bool _succeed; // fifth
	float _cost; // sixth

	int sample_num;

	mTransitionData()
	{
		sample_num = -10;
		////////////////////// input feature (stance to stance)
		_from = std::vector<Vector3>(4, Vector3());
		_meanFrom = Vector3();
		_fromIDs = std::vector<int>(4, -1);

		_to = std::vector<Vector3>(4, Vector3());
		_meanTo = Vector3();
		_toIDs = std::vector<int>(4, -1);

		_succeed = false;
		_cost = FLT_MAX;
		controlSize = 0;
	}

	VectorXf norm_success_state;
	VectorXf norm_cost_state;
	VectorXf norm_policy_state;
	VectorXf norm_debug_state;

	VectorXf norm_success_output;
	VectorXf norm_cost_output;
	VectorXf norm_policy_output;
	VectorXf norm_debug_output;

	VectorXf derivative_success_output;
	VectorXf derivative_cost_output;
	VectorXf derivative_policy_output;
	VectorXf derivative_debug_output;

	VectorXf variance_success_output;
	VectorXf variance_cost_output;
	VectorXf variance_policy_output;
	VectorXf variance_debug_output;

	float input_test;
	float output_test;

	int calculate_diff_limbs()
	{
		int _diff = 0.0f;

		for (unsigned int i = 0; i < 4; i++)
		{
			float _dis = (_from[i] - _to[i]).norm();

			if (_dis > 0)
				_diff++;
		}

		_diff_index = _diff;

		return _diff;
	}

	//////////////////////////////////////////////////////////////// this is used in test scenario only!!!!!!!!
	void fillInputStanceFrom(const std::vector<Vector3>& iFromPos, const std::vector<Vector3>& iToPos)
	{
		_from = iFromPos;
		_to = iToPos;
		_meanFrom = Vector3(0.0f, 0.0f, 0.0f);
		_meanTo = Vector3(0.0f, 0.0f, 0.0f);
		for (unsigned int i = 0; i < 4; i++)
		{
			_meanFrom += iFromPos[i] / 4.0f;
			_meanTo += iToPos[i] / 4.0f;
		}

		for (unsigned int i = 0; i < 4; i++)
		{
			_fromIDs[i] = 1;
			_toIDs[i] = 1;
		}

		calculate_diff_limbs();
		return;
	}

	//// this is used for predicting cost, success and control points in graph and low-level controller
	void fillInputStanceFrom(SimulationContext* mContext, const std::vector<int>& iFromIDs, const std::vector<int>& iToIDs)
	{
		float _fromSize = 0;
		float _toSize = 0;

		_from.clear();
		_to.clear();

		_meanFrom = mContext->getHoldStancePosFrom(iFromIDs, _from, _fromSize);
		_meanTo = mContext->getHoldStancePosFrom(iToIDs, _to, _toSize);

		for (unsigned int i = 0; i < 4; i++)
		{
			if (iFromIDs[i] != -1)
			{
				_fromIDs[i] = mContext->holds_body[iFromIDs[i]].public_prototype_hold_id;
			}
			else
			{
				_fromIDs[i] = -1;
			}
			if (iToIDs[i] != -1)
			{
				_toIDs[i] = mContext->holds_body[iToIDs[i]].public_prototype_hold_id;
			}
			else
			{
				_toIDs[i] = -1;
			}
		}

		calculate_diff_limbs();
		return;
	}

	void fillInputStateFrom(const BipedState& init_biped_state)//const Eigen::VectorXf& iState, const Vector3& iSpinePos)
	{
		_initialBodyState = init_biped_state;
	}

	void setControlPoints(Eigen::VectorXf& _c)
	{
		_controlPoints = _c;
		controlSize = _c.size();
	}

	// used only for tensor flow
	std::string transitionStanceToString()
	{
		std::string write_buff;
		char _buff[2000];

		// from
		for (unsigned int i = 0; i < _from.size(); i++)
		{
			sprintf_s(_buff, "%.3f,", _from[i][0]); write_buff += _buff;
			sprintf_s(_buff, "%.3f,", _from[i][1]); write_buff += _buff;
			sprintf_s(_buff, "%.3f,", _from[i][2]); write_buff += _buff;
		}

		// from mean point
		sprintf_s(_buff, "%.3f,", _meanFrom[0]); write_buff += _buff;
		sprintf_s(_buff, "%.3f,", _meanFrom[1]); write_buff += _buff;
		sprintf_s(_buff, "%.3f,", _meanFrom[2]); write_buff += _buff;

		// from ids
		for (unsigned int i = 0; i < _fromIDs.size(); i++)
		{
			sprintf_s(_buff, "%d,", _fromIDs[i]); write_buff += _buff;
		}

		// to
		for (unsigned int i = 0; i < _to.size(); i++)
		{
			sprintf_s(_buff, "%.3f,", _to[i][0]); write_buff += _buff;
			sprintf_s(_buff, "%.3f,", _to[i][1]); write_buff += _buff;
			sprintf_s(_buff, "%.3f,", _to[i][2]); write_buff += _buff;
		}

		// to mean point
		sprintf_s(_buff, "%.3f,", _meanTo[0]); write_buff += _buff;
		sprintf_s(_buff, "%.3f,", _meanTo[1]); write_buff += _buff;
		sprintf_s(_buff, "%.3f,", _meanTo[2]); write_buff += _buff;

		// to ids
		for (unsigned int i = 0; i < _toIDs.size(); i++)
		{
			if (i == _toIDs.size() - 1)
			{
				sprintf_s(_buff, "%d\n", _toIDs[i]); write_buff += _buff;
			}
			else
			{
				sprintf_s(_buff, "%d,", _toIDs[i]); write_buff += _buff;
			}
		}

		return write_buff;
	}

	std::string toString()
	{
		//#x,y,z,f,dx,dy,dz,k,s
		std::string write_buff;
		char _buff[5000];

		// from
		for (unsigned int i = 0; i < _from.size(); i++)
		{
			sprintf_s(_buff, "%.3f,", _from[i][0]); write_buff += _buff;
			sprintf_s(_buff, "%.3f,", _from[i][1]); write_buff += _buff;
			sprintf_s(_buff, "%.3f,", _from[i][2]); write_buff += _buff;
		}

		// from ids
		for (unsigned int i = 0; i < _fromIDs.size(); i++)
		{
			sprintf_s(_buff, "%d,", _fromIDs[i]); write_buff += _buff;
		}

		// to
		for (unsigned int i = 0; i < _to.size(); i++)
		{
			sprintf_s(_buff, "%.3f,", _to[i][0]); write_buff += _buff;
			sprintf_s(_buff, "%.3f,", _to[i][1]); write_buff += _buff;
			sprintf_s(_buff, "%.3f,", _to[i][2]); write_buff += _buff;
		}

		// to ids
		for (unsigned int i = 0; i < _toIDs.size(); i++)
		{
			sprintf_s(_buff, "%d,", _toIDs[i]); write_buff += _buff;
		}

		// succeed
		sprintf_s(_buff, "%d,", _succeed ? 1 : 0); write_buff += _buff;

		// cost
		sprintf_s(_buff, "%.3f,", _cost); write_buff += _buff;

		// control size and values
		sprintf_s(_buff, "%d,", controlSize); write_buff += _buff;
		for (unsigned int c = 0; c < controlSize; c++)
		{
			sprintf_s(_buff, "%.3f,", _controlPoints[c]); write_buff += _buff;
		}

		for (int i = 0; i < BodyNUM; i++)
		{
			Vector4 angle = _initialBodyState.bodyStates[i].getAngle();

			for (int j = 0; j < 4; j++)
			{
				sprintf_s(_buff, "%.3f,", angle[j]); write_buff += _buff;
			}
			Vector3 pos = _initialBodyState.bodyStates[i].getPos();
			Vector3 vel = _initialBodyState.bodyStates[i].getVel();
			Vector3 avel = _initialBodyState.bodyStates[i].getAVel();
			for (int j = 0; j < 3; j++)
			{
				sprintf_s(_buff, "%.3f,", pos[j]); write_buff += _buff;
				sprintf_s(_buff, "%.3f,", vel[j]); write_buff += _buff;
				if (i == BodyNUM - 1 && j == 2)
				{
					sprintf_s(_buff, "%.3f\n", avel[j]); write_buff += _buff;
				}
				else
				{
					sprintf_s(_buff, "%.3f,", avel[j]); write_buff += _buff;
				}
			}
		}

		return write_buff;
	}

	bool loadFrom(std::vector<float>& cVal)
	{
		int cIndex = 0;
		
		// from
		for (unsigned int i = 0; i < _from.size(); i++)
		{
			_from[i][0] = cVal[cIndex++];
			_from[i][1] = cVal[cIndex++];
			_from[i][2] = cVal[cIndex++];
		}

		_meanFrom = Vector3(0.0f, 0.0f, 0.0f);
		Vector3 default_mean(0.0f, -0.5f, 1.0f);
		int s = 0;
		for (unsigned int i = 0; i < _fromIDs.size(); i++)
		{
			_fromIDs[i] = cVal[cIndex++];
			if (_fromIDs[i] > -1)
			{
				s++;
				_meanFrom += _from[i];
			}
		}
		if (s > 0)
		{
			_meanFrom /= (float)s;
		}
		else
		{
			_meanFrom = default_mean;
		}

		// to
		for (unsigned int i = 0; i < _to.size(); i++)
		{
			_to[i][0] = cVal[cIndex++];
			_to[i][1] = cVal[cIndex++];
			_to[i][2] = cVal[cIndex++];
		}

		_meanTo = Vector3(0.0f, 0.0f, 0.0f);
		s = 0;
		for (unsigned int i = 0; i < _toIDs.size(); i++)
		{
			_toIDs[i] = cVal[cIndex++];
			if (_toIDs[i] > -1)
			{
				s++;
				_meanTo += _to[i];
			}
		}
		if (s > 0)
		{
			_meanTo /= (float)s;
		}
		else
		{
			_meanTo = default_mean;
		}

		// succeed
		_succeed = cVal[cIndex++] == 1.0f;

		// cost
		_cost = cVal[cIndex++];

		controlSize = (unsigned int)(cVal[cIndex++]);
		_controlPoints = VectorXf::Zero(controlSize);
		for (unsigned int c = 0; c < controlSize; c++)
		{
			_controlPoints[c] = cVal[cIndex++];
		}

		for (int i = 0; i < BodyNUM; i++)
		{
			Vector4 angle(0, 0, 0, 0);

			for (int j = 0; j < 4; j++)
			{
				angle[j] = cVal[cIndex++];
			}
			Vector3 pos(0, 0, 0);
			Vector3 vel(0, 0, 0);
			Vector3 avel(0, 0, 0);
			for (int j = 0; j < 3; j++)
			{
				pos[j] = cVal[cIndex++];
				vel[j] = cVal[cIndex++];
				avel[j] = cVal[cIndex++];
			}

			_initialBodyState.bodyStates[i].setAngle(angle);
			_initialBodyState.bodyStates[i].setPos(pos);
			_initialBodyState.bodyStates[i].setVel(vel);
			_initialBodyState.bodyStates[i].setAVel(avel);
		}

		return !(calculate_diff_limbs() == 0);
	}

	void loadToFloatTestSample(float input, float output)
	{
		input_test = input;
		output_test = output;
	}

	Eigen::VectorXf getSuccessState()
	{
		int cIndex = 0;
		VectorXf v(4 * 3 + 2 + 4 * 3 );//+ 1 /*max z lower than climber radius*/

		float max_z = -FLT_MAX;
		for (unsigned int i = 0; i < _from.size(); i++)
		{
			Vector3 _c_i = _from[i];
			if (_fromIDs[i] != -1)
			{
				max_z = max(max_z, _c_i[2]);

				_c_i -= _meanFrom;
				v[cIndex++] = _c_i[0];
				v[cIndex++] = _c_i[2];
			}
			else
			{
				max_z = max(max_z, 1.0f);

				if (i == 0 || i == 2)
					v[cIndex++] = -1.0f;
				else
					v[cIndex++] = 1.0f;
				if (i <= 1)
					v[cIndex++] = -1.0f;
				else
					v[cIndex++] = 1.0f;
			}

			v[cIndex++] = _fromIDs[i] >= 0 ? 0 : -1;
		}

		Vector3 relative_means = _meanTo - _meanFrom;

		v[cIndex++] = relative_means[0];
		v[cIndex++] = relative_means[2];

		for (unsigned int i = 0; i < _to.size(); i++)
		{
			Vector3 _c_i = _to[i];
			if (_toIDs[i] != -1)
			{
				_c_i -= _meanTo;
				v[cIndex++] = _c_i[0];
				v[cIndex++] = _c_i[2];
			}
			else
			{
				if (i == 0 || i == 2)
					v[cIndex++] = -1.0f;
				else
					v[cIndex++] = 1.0f;
				if (i <= 1)
					v[cIndex++] = -1.0f;
				else
					v[cIndex++] = 1.0f;
			}

			v[cIndex++] = _toIDs[i] >= 0 ? 0 : -1;
		}

		return v;
	}

	Eigen::VectorXf getCostState()
	{
		int cIndex = 0;
		VectorXf v(4 * 3 + 2 + 4 * 3);// + 1 /*max z lower than climber radius*/

		float max_z = -FLT_MAX;
		for (unsigned int i = 0; i < _from.size(); i++)
		{
			Vector3 _c_i = _from[i];
			if (_fromIDs[i] != -1)
			{
				max_z = max(max_z, _c_i[2]);

				_c_i -= _meanFrom;
				v[cIndex++] = _c_i[0];
				v[cIndex++] = _c_i[2];
			}
			else
			{
				max_z = max(max_z, 1.0f);

				if (i == 0 || i == 2)
					v[cIndex++] = -1.0f;
				else
					v[cIndex++] = 1.0f;
				if (i <= 1)
					v[cIndex++] = -1.0f;
				else
					v[cIndex++] = 1.0f;
			}

			v[cIndex++] = _fromIDs[i] >= 0 ? 0 : -1;
		}

		//v[cIndex++] = max_z < 2.0f ? 0 : 1;

		Vector3 relative_means = _meanTo - _meanFrom;

		v[cIndex++] = relative_means[0];
		v[cIndex++] = relative_means[2];

		for (unsigned int i = 0; i < _to.size(); i++)
		{
			Vector3 _c_i = _to[i];
			if (_toIDs[i] != -1)
			{
				_c_i -= _meanTo;
				v[cIndex++] = _c_i[0];
				v[cIndex++] = _c_i[2];
			}
			else
			{
				if (i == 0 || i == 2)
					v[cIndex++] = -1.0f;
				else
					v[cIndex++] = 1.0f;
				if (i <= 1)
					v[cIndex++] = -1.0f;
				else
					v[cIndex++] = 1.0f;
			}

			v[cIndex++] = _toIDs[i] >= 0 ? 0 : -1;
		}

		return v;
	}

	// not considering y-axis at the moment
	Eigen::VectorXf getPolicyState()
	{
		VectorXf v;
		int cIndex = 0;
		
		VectorXf _fullBodyStateFeature;
		Eigen::Quaternionf _inverse_root;
		Vector3 _spinePos = computeFullBodyState(_fullBodyStateFeature, _inverse_root);
		unsigned int featureSize = _fullBodyStateFeature.size();

		v.resize(featureSize + 4 * 2 + 4 * 3 + 3);

		//featuresize
		for (unsigned int i = 0; i < featureSize; i++)
		{
			v[cIndex++] = _fullBodyStateFeature[i];
		}
		
		// 4 * 2
		for (unsigned int i = 0; i < 4; i++)
		{
			v[cIndex++] = _fromIDs[i] >= 0 ? 0 : -1;
			v[cIndex++] = _toIDs[i] >= 0 ? 0 : -1;
		}

		// 4 * 2
		Vector3 bone_pos;
		for (unsigned int i = 0; i < 4; i++)
		{
			Vector3 _c_i = _to[i];
			if (_toIDs[i] != -1)
			{
				_c_i -= _spinePos;
			}
			else
			{
				_c_i = Vector3(0.0f,0.0f,0.0f);
				if (i == 0 || i == 2)
					_c_i[0] = -1.0f;
				else
					_c_i[0] = 1.0f;
				if (i <= 1)
					_c_i[2] = -1.0f;
				else
					_c_i[2] = 1.0f;
			}

			bone_pos.noalias() = _inverse_root*_c_i;

			v[cIndex++] = bone_pos[0];
			v[cIndex++] = bone_pos[1];
			v[cIndex++] = bone_pos[2];
		}

		// 2
		Vector3 _disTrunk = _meanTo - _spinePos;
		bone_pos.noalias() = _inverse_root*_disTrunk;
		v[cIndex++] = bone_pos[0];
		v[cIndex++] = bone_pos[1];
		v[cIndex++] = bone_pos[2];

		return v;
	}

	Eigen::VectorXf getDebugState()
	{
		VectorXf v(1);
		v[0] = input_test;
		return v;
	}

	Eigen::VectorXf getSucceedVal()
	{
		VectorXf v(1);

		/*int _diff = _diff_index - 1;

		if (_diff < 0)
			_diff = 0;
		if (_diff > 3)
			_diff = 3;*/

		v[0] = _succeed ? 1.0f : 0.0f;//_cost < _costThreshold[_diff] 
		return v;
	}

	Eigen::VectorXf getCostVal()
	{
		VectorXf v(1);
		v[0] = _cost == FLT_MAX ? 1e6 : _cost;
		return v;
	}
	
	Eigen::VectorXf getPolicyVal()
	{
		return _controlPoints;
	}

	Eigen::VectorXf getDebugVal()
	{
		VectorXf v(1);
		v[0] = output_test;
		return v;
	}
};

#include "mController.h"

#include "mGraph.h"


class mMouseClass
{
	SimulationContext* mContextEnv;
public:
	mMouseClass(SimulationContext* iContextEnv)
	{
		mContextEnv = iContextEnv;

		debugShow = false;

		update_hold = false;
		disFromCOM = 6.5f;

		rayDis = 0.0f;
		bMouse = -1;
		selected_body = -1;
		selected_hold = -1;

		state_selection = 0;
		last_hit_geom_id = -1;

		currentIndexBody = 0;

		cX = 0; cY = 0; lX = 0; lY = 0;

		lookDir = Vector3(0.0f, 0.0f, 0.0f);

		desired_holds_ids = std::vector<int>(4,-1);
		desired_COM = mContextEnv->computeCOM();
		desired_COM[1] = 0;
		resetTorsoDir();
	}

	void resetTorsoDir()
	{
		theta_torso = 90;
		phi_torso = 0;
	}

	void trackMouse(bool showDebugMouseInfo = false)
	{
		if (showDebugMouseInfo)
		{
			// mouse info
			rcPrintString("bX:%f, bY:%f, bZ:%f \n", rayBegin.x(), rayBegin.y(), rayBegin.z());
			rcPrintString("eX:%f, eY:%f, eZ:%f, B:%d \n", rayEnd.x(), rayEnd.y(), rayEnd.z(), bMouse);

			rcSetColor(1, 0, 0);
			Vector3 rayStart = rayBegin + lookDir;
			SimulationContext::drawLine(rayStart, rayEnd);
		}

		Vector3 rayDir = (rayEnd - rayBegin).normalized();
		rayDis = (rayEnd - rayBegin).norm();

		dVector3 out_pos;
		float out_depth;
		int hitGeomID = -1;
		int cIndexBody = -1;

		if (!this->update_hold)
		{
			if (bMouse == 0) // any click other than left click
			{
				hitGeomID = odeRaycastGeom(rayBegin.x(), rayBegin.y(), rayBegin.z(),
					rayDir.x(), rayDir.y(), rayDir.z(),
					rayDis, out_pos, out_depth,
					unsigned long(0x7FFF), unsigned long(0x7FFF));

				cIndexBody = (int)mContextEnv->getIndexHandsAndLegsFromGeom(selected_body);
				if (state_selection == 1 && selected_body >= 0)
				{
					selected_hold = last_hit_geom_id;
					state_selection = 0;

					if (cIndexBody >= (int)SimulationContext::MouseTrackingBody::MouseLeftLeg && cIndexBody <= (int)SimulationContext::MouseTrackingBody::MouseRightHand)
					{
						currentIndexBody = cIndexBody;
						desired_holds_ids[currentIndexBody] = mContextEnv->getIndexHoldFromGeom(selected_hold);
						if (desired_holds_ids[currentIndexBody] < 0)
							desired_holds_ids[currentIndexBody] = -1;
					}
				}
				else
				{
					selected_hold = -1;
					state_selection = 0;
				}
				if (mContextEnv->getIndexHandsAndLegsFromGeom(hitGeomID) >= 0)
					selected_body = hitGeomID;
				else
					selected_body = -1;
			}
			else if (bMouse == 1) // left click
			{
				hitGeomID = odeRaycastGeom(rayBegin.x(), rayBegin.y(), rayBegin.z(),
					rayDir.x(), rayDir.y(), rayDir.z(),
					rayDis, out_pos, out_depth,
					unsigned long(0x8000), unsigned long(0x8000));

				if (state_selection == 0)
				{
					state_selection = 1;
				}
				cIndexBody = (int)mContextEnv->getIndexHandsAndLegsFromGeom(selected_body);
				if (cIndexBody >= (int)SimulationContext::MouseTrackingBody::MouseLeftLeg && cIndexBody <= (int)SimulationContext::MouseTrackingBody::MouseRightHand)
				{
					// choose hold body
					if (selected_body >= 0)
						selected_hold = hitGeomID;
					else
						selected_hold = -1;
				}
				else if (cIndexBody == (int)SimulationContext::MouseTrackingBody::MouseTorsoDir)
				{
					selected_hold = -1;
					int deltax = cX - lX;
					int deltay = cY - lY;

					theta_torso -= float(deltax) * 0.5f;
					//phi_torso -= float(deltay) * 0.5f;

					lX = cX;
					lY = cY;

					/*if (debugShow)
						rcPrintString("dx:%d, dy:%d, theta:%f \n", deltax, deltay, theta_torso);*/
				}
				else
				{
					selected_hold = -1;
				}
			}
			else if (bMouse == 2)
			{
				cIndexBody = (int)mContextEnv->getIndexHandsAndLegsFromGeom(selected_body);

				if (cIndexBody == (int)SimulationContext::MouseTrackingBody::MouseTorsoDir)
				{
					selected_hold = -1;
					int deltax = cX - lX;
					int deltay = cY - lY;

					desired_COM[0] -= float(deltax) * 0.05f;
					desired_COM[2] -= float(deltay) * 0.05f;

					lX = cX;
					lY = cY;
				}

			}
			last_hit_geom_id = hitGeomID;
		}
		else
		{
			hitGeomID = odeRaycastGeom(rayBegin.x(), rayBegin.y(), rayBegin.z(),
				rayDir.x(), rayDir.y(), rayDir.z(),
				rayDis, out_pos, out_depth,
				unsigned long(0x8000), unsigned long(0x8000));
			if (bMouse == 1) // left click
			{
				int deltax = cX - lX;
				int deltay = cY - lY;

				int cIndex = mContextEnv->getIndexHold(last_hit_geom_id);
				if (cIndex >= 0)
				{
					mContextEnv->holds_body[cIndex].theta -= float(deltax) * 0.5f;
					mContextEnv->holds_body[cIndex].phi -= float(deltay) * 0.5f;

					if (mContextEnv->holds_body[cIndex].phi > 90.0f) mContextEnv->holds_body[cIndex].phi = 90.0f;
					if (mContextEnv->holds_body[cIndex].phi < -90.0f) mContextEnv->holds_body[cIndex].phi = -90.0f;

					if (mContextEnv->holds_body[cIndex].theta > 180.0f) mContextEnv->holds_body[cIndex].theta = 180.0f;
					if (mContextEnv->holds_body[cIndex].theta < -180.0f) mContextEnv->holds_body[cIndex].theta = -180.0f;

					Vector3 dir;
					dir[0] = (float)(cosf(mContextEnv->holds_body[cIndex].theta*DEG_TO_RAD) * cosf(mContextEnv->holds_body[cIndex].phi*DEG_TO_RAD));
					dir[1] = (float)(sinf(mContextEnv->holds_body[cIndex].theta*DEG_TO_RAD) * cosf(mContextEnv->holds_body[cIndex].phi*DEG_TO_RAD));
					dir[2] = (float)(sinf(mContextEnv->holds_body[cIndex].phi*DEG_TO_RAD));
					dir.normalize();

					mContextEnv->holds_body[cIndex].setIdealForceDirection(0, dir);

					/*if (debugShow)
						rcPrintString("th:%f, ph:%f \n", mContextEnv->holds_body[cIndex].theta, mContextEnv->holds_body[cIndex].phi);*/
				}
				lX = cX;
				lY = cY;
			}
			else
			{
				last_hit_geom_id = hitGeomID;
				selected_hold = hitGeomID;
			}
		}

		/*if (debugShow)
			rcPrintString("hit body Id:%d \n", hitGeomID);*/
		return;
	}

	void zoomInOut()
	{
		if (bMouse >= 8 && bMouse <= 10)
		{
			if (bMouse == 8)
			{
				disFromCOM -= 0.25f;
			}
			else if (bMouse == 10)
			{
				disFromCOM += 0.25f;
			}
			bMouse = 0;
		}
	}

	void updateCameraPositionByMouse(float& theta, float& phi, bool showDebug = false)
	{
		int deltax = cX - lX;
		int deltay = cY - lY;
		if (showDebug)
			rcPrintString("button mouse: %d, deltax: %d, deltay: %d", bMouse, deltax, deltay);
		if (bMouse == 4) //middle click (in unity)
		{
			theta += float(deltax) * 0.5f;
			phi -= float(deltay) * 0.5f;

			deltax = 0;
			deltay = 0;

			lX = cX;
			lY = cY;
		}
		return;
	}

	void cameraAdjustment(Vector3 _climberCOM)
	{
		zoomInOut();

		Vector3 camera_pos = _climberCOM;

		static float theta = -110;
		static float phi = -5;

		updateCameraPositionByMouse(theta, phi);
		
		Vector3 cameraDir = mTools::getDirectionFromAngles(theta, phi);

		Vector3 cameraLocation = camera_pos + disFromCOM * cameraDir;

		float old_xyz[3] = { xyz[0], xyz[1], xyz[2] };

		xyz[0] = cameraLocation[0];
		xyz[1] = cameraLocation[1];
		xyz[2] = cameraLocation[2];

		bool flag_update_lookAtPos = false;

		if (xyz[1] > -0.25f && !(DemoID == mDemoTestClimber::DemoPillar)) // not going through wall
		{
			xyz[1] = -0.25f;
			old_xyz[1] = xyz[1];

			flag_update_lookAtPos = true;
		}

		if (xyz[2] < 0.25f) // not going under ground
		{
			xyz[2] = 0.25f;
			old_xyz[2] = xyz[2];

			flag_update_lookAtPos = true;
		}

		if (!flag_update_lookAtPos)
		{
			lookAt[0] = _climberCOM[0];
			lookAt[1] = _climberCOM[1];
			lookAt[2] = _climberCOM[2];
		}
		else
		{
			cameraLocation[0] = xyz[0];
			cameraLocation[1] = xyz[1];
			cameraLocation[2] = xyz[2];

			Vector3 lookAtPos = cameraLocation - disFromCOM * cameraDir;

			lookAt[0] = lookAtPos[0];
			lookAt[1] = lookAtPos[1];
			lookAt[2] = lookAtPos[2];
		}
		rcSetViewPoint(xyz[0], xyz[1], xyz[2], lookAt[0], lookAt[1], lookAt[2]);

		lookDir = cameraDir;
	}

	bool debugShow;

	bool update_hold;
	float disFromCOM;

	// comes from mouse function
	Vector3 rayBegin, rayEnd;
	float rayDis;
	int bMouse;
	int cX, cY;//current x, y
	int lX, lY;//last x, y

	Vector3 lookDir; // we are looking at climber's center of mass 

	int selected_body; // contains hit geom of humanoid climber
	int selected_hold; // contains hit geom of hold on the wall

	int state_selection; // {0: for selecting humanoid body,1: for selecting hold}
	int last_hit_geom_id;

	int currentIndexBody;

	// output
	std::vector<int> desired_holds_ids; // for hold ids for the hands and feet of the agent {0: ll, 1: rl, 2: lh, 3: rh}
	Vector3 desired_COM;
	float theta_torso;
	float phi_torso;
}* mMouseReader;

class mTestControllerClass
{
public:
	bool useMachineLearning;

	class mSavedDataState
	{
	public:
		BipedState _bodyState;
		std::vector<int> _desired_holds_ids;
		std::vector<outSavedData> _fromToMotions;
	};

	class mSavedUserStudyData
	{
	public:
		std::vector<int> _from_holds_ids;
		std::vector<int> _to_holds_ids;
		float _starting_time;
		float _end_time;
		bool _succedd;
		bool _isBackSpaceHit;

		std::string toString()
		{
			//#x,y,z,f,dx,dy,dz,k,s
			std::string write_buff;
			char _buff[200];
			sprintf_s(_buff, "%d,", _from_holds_ids[0]); write_buff += _buff;
			sprintf_s(_buff, "%d,", _from_holds_ids[1]); write_buff += _buff;
			sprintf_s(_buff, "%d,", _from_holds_ids[2]); write_buff += _buff;
			sprintf_s(_buff, "%d,", _from_holds_ids[3]); write_buff += _buff;

			sprintf_s(_buff, "%d,", _to_holds_ids[0]); write_buff += _buff;
			sprintf_s(_buff, "%d,", _to_holds_ids[1]); write_buff += _buff;
			sprintf_s(_buff, "%d,", _to_holds_ids[2]); write_buff += _buff;
			sprintf_s(_buff, "%d,", _to_holds_ids[3]); write_buff += _buff;

			sprintf_s(_buff, "%f,", _starting_time); write_buff += _buff;
			sprintf_s(_buff, "%f,", _end_time); write_buff += _buff;

			sprintf_s(_buff, "%d,", _succedd ? 1 : 0); write_buff += _buff;

			sprintf_s(_buff, "%d\n", _isBackSpaceHit ? 1 : 0); write_buff += _buff;

			return write_buff;
		}
	};
	float timeSpendOnAnimation;
	float timeSpendInPlanning;
	float preVisitingTime;
	bool isBackSpaceHit;

	std::vector<mSavedUserStudyData> _savedUserData;

	void mSaveUserDataFunc()
	{
		mSavedUserStudyData cData;
		cData._from_holds_ids = mSample.initial_hold_ids;
		cData._to_holds_ids = mSample.desired_hold_ids;

		cData._starting_time = mSample.starting_time;
		cData._end_time = timeSpendInPlanning; // one event happened (backspace, reached, failed, change)

		cData._succedd = mTools::isSetAEqualsSetB(mOptimizer->startState.hold_bodies_ids, mSample.desired_hold_ids);

		cData._isBackSpaceHit = isBackSpaceHit;

		_savedUserData.push_back(cData);
	}

	mController* mOptimizer;
	SimulationContext* mContextRRT;
	mSampleStructure mSample;
	mMouseClass* mMouseTracker;

	//handling offline - online method
	enum targetMethod { Offiline = 0, Online = 1 };

	mANNBase mNNStanceSuccessNet;
	mANNBase mNNStanceCostNet;
	mANNBase mNNStancePolicyNet;

	// handling falling problem of the climber
	std::list<mSavedDataState> lOptimizationBipedStates;
	std::list<int> deletedSavingSlots;
	std::vector<outSavedData> _fromToMotions;

	int itr_optimization_for_each_sample;
	int max_itr_optimization_for_each_sample;
	int max_no_improvement_iterations;
	float current_cost_togoal;

	bool pauseOptimization;

	int currentIndexOfflineState;

	bool isOptDone;
	bool isRestartedForOfflineOpt;

	bool debugShow;

	// for playing animation
	unsigned int indexOfSavedStates;
	unsigned int index_in_currentState;
	int showing_state;
	BipedState lastStateBeforeAnimation;

	// some analysis
	std::vector<int> itrCMAESLearningMovingLimbs;
	std::vector<int> succCMAESLearningMovingLimbs;
	std::vector<int> countCMAESLearningMovingLimbs;
	std::vector<int> itrCMAESMovingLimbs;
	std::vector<int> succCMAESMovingLimbs;
	std::vector<int> countCMAESMovingLimbs;

	void play_animation()
	{
		if (showing_state == 1)
		{
			if (index_in_currentState < _fromToMotions.size())
			{
				mOptimizer->startState = _fromToMotions[index_in_currentState]._s;
				index_in_currentState++;
			}
			else
			{
				showing_state = 2;
			}
		}

		if (showing_state == 0 || showing_state == 2)
		{
			std::list<mSavedDataState>::iterator pointerToSavedStates = lOptimizationBipedStates.begin();
			for (unsigned int counter = 0; counter < indexOfSavedStates && showing_state == 0; counter++)
			{
				if (pointerToSavedStates != lOptimizationBipedStates.end())
					pointerToSavedStates++;
				else
				{
					indexOfSavedStates = lOptimizationBipedStates.size() - 1;
					break;
				}
			}

			if (index_in_currentState < pointerToSavedStates->_fromToMotions.size() && showing_state == 0)
			{
				mOptimizer->startState = pointerToSavedStates->_fromToMotions[index_in_currentState]._s;
				index_in_currentState++;
			}
			else
			{
				index_in_currentState = 0;
				if (showing_state == 0)
				{
					if (indexOfSavedStates < lOptimizationBipedStates.size() - 1)
					{
						indexOfSavedStates++;
					}
					else if (_fromToMotions.size() > 0)
					{
						showing_state = 1;
					}
					else
					{
						indexOfSavedStates = 0;
						showing_state = 0;
					}
				}
				else
				{
					indexOfSavedStates = 0;
					showing_state = 0;
				}
			}
		}
		//Sleep(60);
		mOptimizer->syncMasterContextWithStartState(true);
		return;
	}

	void loadNeuralNetwork()
	{
		mNNStanceSuccessNet.loadFromFile("ClimberInfo\\InterfaceNNs\\SuccessNet");
		mNNStanceCostNet.loadFromFile("ClimberInfo\\InterfaceNNs\\CostNet");
		mNNStancePolicyNet.loadFromFile("ClimberInfo\\InterfaceNNs\\PolicyNet");
	}

	mTestControllerClass(SimulationContext* iContextRRT, mController* iController, mMouseClass* iMouseTracker)
		:mNNStanceSuccessNet(NULL, mMachineLearningType::Success),
		mNNStanceCostNet(NULL, mMachineLearningType::Cost),
		mNNStancePolicyNet(NULL, mMachineLearningType::Policy)
	{
		timeSpendInPlanning = 0;
		timeSpendOnAnimation = 0;
		preVisitingTime = 0;
		isBackSpaceHit = false;

		mOptimizer = iController;
		mContextRRT = iContextRRT;
		mMouseTracker = iMouseTracker;

		loadNeuralNetwork();

		for (unsigned int i = 0; i < mOptimizer->startState.hold_bodies_ids.size(); i++)
		{
			if (mSample.initial_hold_ids.size() < mOptimizer->startState.hold_bodies_ids.size())
				mSample.initial_hold_ids.push_back(mOptimizer->startState.hold_bodies_ids[i]);
			else
				mSample.initial_hold_ids[i] = mOptimizer->startState.hold_bodies_ids[i];
			if (mSample.desired_hold_ids.size() < mOptimizer->startState.hold_bodies_ids.size())
				mSample.desired_hold_ids.push_back(mOptimizer->startState.hold_bodies_ids[i]);
			else
				mSample.desired_hold_ids[i] = mOptimizer->startState.hold_bodies_ids[i];
		}

		mSavedDataState ndata;
		ndata._bodyState = mOptimizer->startState.getNewCopy(mContextRRT->getNextFreeSavingSlot(), mContextRRT->getMasterContextID());
		ndata._desired_holds_ids = mMouseTracker->desired_holds_ids;

		lOptimizationBipedStates.push_back(ndata);

		mOptimizer->using_sampling = true;
		max_itr_optimization_for_each_sample = useOfflinePlanning ? (int)(4.0f * cTime) : (int)(3.0f * cTime);
		max_no_improvement_iterations = useOfflinePlanning ? (int)(cTime) : (int)(cTime / 2.0f);
		itr_optimization_for_each_sample = 0;
		current_cost_togoal = FLT_MAX;
		pauseOptimization = true;
		currentIndexOfflineState = 0;
		isOptDone = false;
		isRestartedForOfflineOpt = false;

		mOptimizer->syncMasterContextWithStartState(true);

		debugShow = false;

		// for playing animation
		index_in_currentState = 0;
		indexOfSavedStates = 0;
		lastStateBeforeAnimation = mOptimizer->startState.getNewCopy(mContextRRT->getNextFreeSavingSlot(), mContextRRT->getMasterContextID());
		showing_state = 0;

		// some analysis
		itrCMAESLearningMovingLimbs = std::vector<int>(4, 0);
		succCMAESLearningMovingLimbs = std::vector<int>(4, 0);
		countCMAESLearningMovingLimbs = std::vector<int>(4, 0);
		itrCMAESMovingLimbs = std::vector<int>(4, 0);
		succCMAESMovingLimbs = std::vector<int>(4, 0);
		countCMAESMovingLimbs = std::vector<int>(4, 0);

		useMachineLearning = true;
	}

	~mTestControllerClass()
	{
		if (CreateDirectory("ClimberResults", NULL) || ERROR_ALREADY_EXISTS == GetLastError())
		{
			/*mFileHandler writeFile(mContextRRT->getAppendAddress("ClimberResults\\OutRoute", mContextRRT->_RouteNum + 1, ".txt"));

			writeFile.openFileForWritingOn();

			std::string write_buff;
			char _buff[100];
			sprintf_s(_buff, "%f,%f\n", timeSpendInPlanning, timeSpendOnAnimation); write_buff += _buff;
			writeFile.writeLine(write_buff);
			for (unsigned int i = 0; i < _savedUserData.size(); i++)
			{
				std::string cData = _savedUserData[i].toString();
				writeFile.writeLine(cData);
			}

			writeFile.mCloseFile();*/
		}
	}

	void runLoopTest(bool advance_time, bool playAnimation, float current_time)
	{
		targetMethod itargetMethd = useOfflinePlanning ? targetMethod::Offiline : targetMethod::Online;

		float delta_time = current_time - preVisitingTime;
		preVisitingTime = current_time;

		if (!playAnimation)
		{
			timeSpendInPlanning += delta_time;
			mOptimizer->startState = lastStateBeforeAnimation;

			//update sample info if neccessary
			mMouseTracker->trackMouse();
			updateSampleInfo();

			// do the optimization
			if (!pauseOptimization)
			{
				if (itargetMethd == targetMethod::Online)
				{
					runTestOnlineOpt(advance_time);
				}
				else
				{
					runTestOfflineOpt(advance_time);
				}
			}
			// when optimization is done just sync the current state with optimized one
			else
			{
				mOptimizer->syncMasterContextWithStartState(true);
			}

			// play best simulated trajectory after the optimization is done
			if (itargetMethd == targetMethod::Offiline)
			{
				if (isOptDone)
				{
					std::vector<outSavedData> nStates;
					isOptDone = !mOptimizer->simulateBestTrajectory(true, mSample.desired_hold_ids, nStates);

					for (unsigned int i = 0; i < nStates.size(); i++)
					{
						_fromToMotions.push_back(nStates[i]);
					}

					if (!isOptDone)
					{
						// update the analysis
						int diff_limbs = mTools::getDiffBtwSetASetB(mSample.initial_hold_ids, mSample.desired_hold_ids) - 1;
						if (mSample.isUsingMachineLearning)
						{
							if (diff_limbs >= 0 && diff_limbs <= 3)
							{
								countCMAESLearningMovingLimbs[diff_limbs]++;
								itrCMAESLearningMovingLimbs[diff_limbs] += itr_optimization_for_each_sample;
								if (mTools::isSetAEqualsSetB(mOptimizer->startState.hold_bodies_ids, mSample.desired_hold_ids))
									succCMAESLearningMovingLimbs[diff_limbs]++;
							}
						}
						else
						{
							if (diff_limbs >= 0 && diff_limbs <= 3)
							{
								countCMAESMovingLimbs[diff_limbs]++;
								itrCMAESMovingLimbs[diff_limbs] += itr_optimization_for_each_sample;
								if (mTools::isSetAEqualsSetB(mOptimizer->startState.hold_bodies_ids, mSample.desired_hold_ids))
									succCMAESMovingLimbs[diff_limbs]++;
							}
						}

						mSample.numItrFixedCost = 0;
						itr_optimization_for_each_sample = 0;

						float cCost = mOptimizer->getCost(mSample.sourceP, mSample.destinationP, mSample.desired_hold_ids, mSample.isAllowedOnGround);
						if (cCost < current_cost_togoal && !mTools::isSetAEqualsSetB(mOptimizer->startState.hold_bodies_ids, mSample.desired_hold_ids))
						{
							runOptimization(false);
						}
					}
				}
			}
			lastStateBeforeAnimation = mOptimizer->startState.getNewCopy(lastStateBeforeAnimation.saving_slot_state, mContextRRT->getMasterContextID());
		}
		else
		{
			play_animation();
			timeSpendOnAnimation += delta_time;
		}

		drawDesiredLines(playAnimation);
		debugPrint();
	}

	void removeLastSavingState(bool playAnimation)
	{
		if (isOptDone || playAnimation)
		{
			// we are playing the animation
			return;
		}

		isBackSpaceHit = true;
		if (mSample.isReached || mSample.isRejected || !pauseOptimization) // if new state is about to get added
		{
			// get back to the last saved state
			mOptimizer->startState = lOptimizationBipedStates.back()._bodyState;
			mMouseTracker->desired_holds_ids = lOptimizationBipedStates.back()._desired_holds_ids;

			_fromToMotions.clear();
		}
		else
		{
			if (lOptimizationBipedStates.size() > 1)
			{
				int deletedSlot = lOptimizationBipedStates.back()._bodyState.saving_slot_state;

				deletedSavingSlots.push_back(deletedSlot);
				lOptimizationBipedStates.pop_back();
			}

			mOptimizer->startState = lOptimizationBipedStates.back()._bodyState;
			mMouseTracker->desired_holds_ids = lOptimizationBipedStates.back()._desired_holds_ids;
		}

		if (indexOfSavedStates > lOptimizationBipedStates.size() - 1)
		{
			indexOfSavedStates = lOptimizationBipedStates.size() - 1;
		}

		pauseOptimization = true;
		mSample.isReached = false;
		mSample.isRejected = false;
		itr_optimization_for_each_sample = 0;
		mSample.numItrFixedCost = 0;
		mSample.isOdeConstraintsViolated = false;
		mSample.statesFromTo.clear();
		mSample.cOptimizationCost = FLT_MAX;

		mOptimizer->syncMasterContextWithStartState(true);

		lastStateBeforeAnimation = mOptimizer->startState.getNewCopy(lastStateBeforeAnimation.saving_slot_state, mContextRRT->getMasterContextID());
		return;
	}

	void runOptimization(bool save_state = true)
	{
		if (!pauseOptimization)
		{
			return;
		}

		if (save_state)
			saveLastReachedState();

		if (pauseOptimization)
		{
			mSample.starting_time = timeSpendInPlanning;
		}

		pauseOptimization = false;
		mSample.isReached = false;
		mSample.isRejected = false;
	}

private:
	void debugPrint()
	{
		rcPrintString("Route Num is: %d", mContextRRT->_RouteNum + 1);
		rcPrintString("Climber's height is: %f", mContextRRT->climberHeight);

		if (useMachineLearning)
		{
			rcPrintString("Using Machine Learning");
		}
		else
		{
			rcPrintString("NOT Using Machine Learning");
		}

//		rcPrintString("%0.3f, %0.3f, %0.3f, %0.3f", mContextRRT->mColorBodies[0], mContextRRT->mColorBodies[1], mContextRRT->mColorBodies[2], mContextRRT->mColorBodies[3]);

		float edgeCost = mStanceGraph::getEdgeCost(mSample.success_rate, mOptimizer->current_cost_control, mSample.desired_hold_ids, mSample.cost_var);
		rcPrintString("TCost:%0.3f, CCost:%0.3f, Success:%0.3f", mSample.target_cost, edgeCost, mSample.success_rate);
		//rcPrintString("Target and Current Control Cost: (%0.3f, %0.3f)", mSample.target_cost, edgeCost);

		rcPrintString("Current and Best Traj. state cost: %0.3f, %0.3f", mOptimizer->current_cost_state, mOptimizer->best_trajectory_cost);

		// print efficiecy data

		rcPrintString("ML Count, 1L:%d, 2L:%d, 3L:%d, 4L:%d"
			, countCMAESLearningMovingLimbs[0], countCMAESLearningMovingLimbs[1]
			, countCMAESLearningMovingLimbs[2], countCMAESLearningMovingLimbs[3]);
		rcPrintString("ML Itr, 1L:%0.2f, 2L:%0.2f, 3L:%0.3f, 4L:%0.2f"
			, itrCMAESLearningMovingLimbs[0] / float(countCMAESLearningMovingLimbs[0] + 0.001f), itrCMAESLearningMovingLimbs[1] / float(countCMAESLearningMovingLimbs[1] + 0.001f)
			, itrCMAESLearningMovingLimbs[2] / float(countCMAESLearningMovingLimbs[2] + 0.001f), itrCMAESLearningMovingLimbs[3] / float(countCMAESLearningMovingLimbs[3] + 0.001f));
		rcPrintString("ML Success, 1L:%0.2f, 2L:%0.2f, 3L:%0.3f, 4L:%0.2f"
			, succCMAESLearningMovingLimbs[0] / float(countCMAESLearningMovingLimbs[0] + 0.001f), succCMAESLearningMovingLimbs[1] / float(countCMAESLearningMovingLimbs[1] + 0.001f)
			, succCMAESLearningMovingLimbs[2] / float(countCMAESLearningMovingLimbs[2] + 0.001f), succCMAESLearningMovingLimbs[3] / float(countCMAESLearningMovingLimbs[3] + 0.001f));


		rcPrintString("CMA-ES Count, 1L:%d, 2L:%d, 3L:%d, 4L:%d"
			, countCMAESMovingLimbs[0], countCMAESMovingLimbs[1]
			, countCMAESMovingLimbs[2], countCMAESMovingLimbs[3]);
		rcPrintString("CMA-ES Itr, 1L:%0.2f, 2L:%0.2f, 3L:%0.3f, 4L:%0.2f"
			, itrCMAESMovingLimbs[0] / float(countCMAESMovingLimbs[0] + 0.001f), itrCMAESMovingLimbs[1] / float(countCMAESMovingLimbs[1] + 0.001f)
			, itrCMAESMovingLimbs[2] / float(countCMAESMovingLimbs[2] + 0.001f), itrCMAESMovingLimbs[3] / float(countCMAESMovingLimbs[3] + 0.001f));
		rcPrintString("CMA-ES Success, 1L:%0.2f, 2L:%0.2f, 3L:%0.3f, 4L:%0.2f"
			, succCMAESMovingLimbs[0] / float(countCMAESMovingLimbs[0] + 0.001f), succCMAESMovingLimbs[1] / float(countCMAESMovingLimbs[1] + 0.001f)
			, succCMAESMovingLimbs[2] / float(countCMAESMovingLimbs[2] + 0.001f), succCMAESMovingLimbs[3] / float(countCMAESMovingLimbs[3] + 0.001f));


		return;
	}

	bool saveLastReachedState()
	{
		//save last state if we moved to any desired position
		if (mSample.isReached || mSample.isRejected)
		{
			// save last optimization state
			int nSavingSlot = -1;
			if (deletedSavingSlots.size() > 0)
			{
				nSavingSlot = deletedSavingSlots.front();
				deletedSavingSlots.pop_front();
			}
			else
			{
				nSavingSlot = mContextRRT->getNextFreeSavingSlot();
			}

			mSavedDataState ndata;
			ndata._bodyState = mOptimizer->startState.getNewCopy(nSavingSlot, mContextRRT->getMasterContextID());

			ndata._desired_holds_ids = mMouseTracker->desired_holds_ids;
			ndata._fromToMotions = _fromToMotions;

			lOptimizationBipedStates.push_back(ndata);

			if (showing_state == 1)
			{
				showing_state = 0;
				indexOfSavedStates = lOptimizationBipedStates.size() - 1;
			}

			_fromToMotions.clear();

			return true;
		}

		//_fromToMotions.clear();
		lOptimizationBipedStates.back()._desired_holds_ids = mMouseTracker->desired_holds_ids;
		return false;
	}

	/////////////////////////////////////////////// choose offline or online mode
	void runTestOfflineOpt(bool advance_time)
	{
		// sync all contexts with stating state and current stance
		mOptimizer->syncMasterContextWithStartState(true);

		mOptimizer->using_sampling = !useMachineLearning;

		if (useMachineLearning)
		{
			if (optimizerType == otCMAES)
			{
				mTransitionData cVal;
				cVal.fillInputStanceFrom(mContext, mSample.initial_hold_ids, mSample.desired_hold_ids);
				cVal.fillInputStateFrom(mOptimizer->startState);

				mSample.initBodyState = mOptimizer->startState;

				VectorXf cPolicyState = cVal.getPolicyState();
				VectorXf cPolicy = VectorXf::Zero(92);
				mNNStancePolicyNet.getMachineLearningOutput(cPolicyState.data(), cPolicy.data());
				((mOptCMAES*)mOptimizer)->machineLearningInput = cPolicy;
			}
		}

		if (itr_optimization_for_each_sample == 0)
		{
			if (optimizerType == otCMAES)
			{
				mSample.isUsingMachineLearning = useMachineLearning;

				mTransitionData mS;
				mS.fillInputStanceFrom(mContext, mSample.initial_hold_ids, mSample.desired_hold_ids);
				VectorXf cSuccessState = mS.getSuccessState();
				VectorXf cSuccessRate = VectorXf::Zero(1);
				mNNStanceSuccessNet.getMachineLearningOutput(cSuccessState.data(), cSuccessRate.data());
				mSample.success_rate = cSuccessRate[0];

				VectorXf cCostState = mS.getCostState();
				VectorXf cCostVal = VectorXf::Zero(1);
				mNNStanceCostNet.getMachineLearningOutput(cCostState.data(), cCostVal.data());

				mSample.cost_var = 2000;

				mSample.target_cost = mStanceGraph::getEdgeCost(cSuccessRate[0], cCostVal[0], mSample.desired_hold_ids, mSample.cost_var);
			}

			// need to store current cost before optimization
			current_cost_togoal = mOptimizer->getCost(mSample.sourceP, mSample.destinationP, mSample.desired_hold_ids, mSample.isAllowedOnGround);
			mOptimizer->reset();
		}
		if (optimizerType == otCMAES)
			mOptimizer->optimize_the_cost(itr_optimization_for_each_sample == 0, mSample.sourceP, mSample.destinationP, mSample.desired_hold_ids, debugShow, mSample.isAllowedOnGround);
		else
			mOptimizer->optimize_the_cost(false, mSample.sourceP, mSample.destinationP, mSample.desired_hold_ids, debugShow, mSample.isAllowedOnGround);

		float edgeCost = mStanceGraph::getEdgeCost(mSample.success_rate, mOptimizer->current_cost_control, mSample.desired_hold_ids, mSample.cost_var);
		int current_max_no_improvement_iterations = max_no_improvement_iterations;
		//rcPrintString("TCost:%0.3f, CCost:%0.3f, Success:%0.3f", mSample.target_cost, edgeCost, mSample.success_rate);
		if (useMachineLearning)
		{
			if (edgeCost <= mSample.target_cost && mOptimizer->isReachedToTargetStance)
			{
				current_max_no_improvement_iterations = max_no_improvement_iterations / 2;
			}
		}

		if (advance_time)
		{
			float costImprovement = max(0.0f, mSample.cOptimizationCost - mOptimizer->current_cost_state);
			//rcPrintString("Traj. cost improvement: %f", costImprovement);
			if (costImprovement < noCostImprovementThreshold)
			{
				mSample.numItrFixedCost++;
			}
			else
			{
				mSample.numItrFixedCost = 0;
			}
			mSample.cOptimizationCost = mOptimizer->best_trajectory_cost;

			itr_optimization_for_each_sample++;
		}
		// offline optimization is done, simulate forward on the best trajectory
		if ((advance_time && mSample.numItrFixedCost > current_max_no_improvement_iterations)
			|| itr_optimization_for_each_sample > max_itr_optimization_for_each_sample)
		{
			pauseOptimization = true;
			isOptDone = true;
			mSample.isReached = true;
			mSaveUserDataFunc();
		}
		else
		{
			isOptDone = false;
			mSample.isReached = false;
		}
		return;
	}

	void runTestOnlineOpt(bool advance_time)
	{
		mSteerFunc(advance_time);

		if (advance_time)
		{
			itr_optimization_for_each_sample++;
		}

		if (itr_optimization_for_each_sample > max_itr_optimization_for_each_sample ||
			mSample.numItrFixedCost > max_no_improvement_iterations)
		{
			itr_optimization_for_each_sample = 0;
			mSample.numItrFixedCost = 0;
			mSample.isOdeConstraintsViolated = false;
			mSample.isReached = true;
		}

		if (mSample.isReached || mSample.isRejected)
		{
			pauseOptimization = true;
			mSaveUserDataFunc();
		}

		return;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void drawDesiredLines(bool playAnimation)
	{
		mSample.drawDesiredTorsoDir(mContextRRT->getBonePosition(SimulationContext::BodyName::BodyTrunk));
		rcSetColor(1, 0, 0, 1);
		Vector3 drawPoint = mMouseTracker->desired_COM;
		drawPoint[1] = -0.1f;
		SimulationContext::drawCross(drawPoint);

		if (!playAnimation)
		{
			rcSetColor(0, 1, 0, 1);
			for (unsigned int i = 0; i < 4; i++)
			{
				if (mSample.desired_hold_ids[i] != -1)
				{
					Vector3 toPoint = mContextRRT->getHoldPos(mSample.desired_hold_ids[i]);
					float _s = mContextRRT->getHoldSize(mSample.desired_hold_ids[i]);
					toPoint[1] -= (_s / 2);
					SimulationContext::drawLine(mContextRRT->getEndPointPosBones((int)(SimulationContext::BodyName::BodyLeftLeg + i)), toPoint);
				}
				else
				{
					SimulationContext::drawCross(mContextRRT->getEndPointPosBones((int)(SimulationContext::BodyName::BodyLeftLeg + i)));
				}
			}
		}
		if (pauseOptimization && debugShow)
			mOptimizer->visualizeForceDirections();
		return;
	}

	void updateSampleInfo()
	{
		bool updateSample = false;
		if (mSample.destinationP.size() == 0)
			updateSample = true;
		else
		{
			if ((mSample.destinationP[mSample.destinationP.size() - 2] - mMouseTracker->desired_COM).norm() > 0.01f)
			{
				updateSample = true;
			}

			if ((mTools::getDirectionFromAngles(mMouseTracker->theta_torso, mMouseTracker->phi_torso) - mSample.destinationP[mSample.destinationP.size() - 1]).norm() > 0.01f)
			{
				updateSample = true;
			}
		}
		if (!mTools::isSetAEqualsSetB(mMouseTracker->desired_holds_ids, mSample.desired_hold_ids) || updateSample || isBackSpaceHit)
		{
			if (!pauseOptimization)
			{
				//				mSaveUserDataFunc();
				//				pauseOptimization = true;
				//				mSample.isReached = true; // sample is marked by the user as reached
				//				mSample.isRejected = true; // or sample is marked rejected by the user
			}

			// needs to optimize again
			itr_optimization_for_each_sample = 0;
			mSample.cOptimizationCost = FLT_MAX;
			mSample.numItrFixedCost = 0;

			mSample.destinationP.clear();
			mSample.sourceP.clear();
			//			mSample.dPoint.clear();

			mSample.initial_hold_ids = mOptimizer->startState.hold_bodies_ids;
			mSample.isAllowedOnGround = mTools::isSetAEqualsSetB(mSample.initial_hold_ids, std::vector<int>(4, -1));
			for (unsigned int i = 0; i < mMouseTracker->desired_holds_ids.size(); i++)
			{
				if (mMouseTracker->desired_holds_ids[i] != mSample.desired_hold_ids[i])
				{
					mSample.desired_hold_ids[i] = mMouseTracker->desired_holds_ids[i];
				}

				if (mMouseTracker->desired_holds_ids[i] != -1) // && mSample.initial_hold_ids[i] == -1
				{
					Vector3 dPos = mContextRRT->getHoldPos(mMouseTracker->desired_holds_ids[i]);

					mSample.sourceP.push_back((mOptCPBP::ControlledPoses)(mOptCPBP::ControlledPoses::LeftLeg + i));
					mSample.destinationP.push_back(dPos);
				}
			}

			//Vector3 midPoint(0, 0, 0);
			//int _count = 0;
			//for (unsigned int i = 0; i < mSample.initial_hold_ids.size(); i++)
			//{
			//	if (mSample.initial_hold_ids[i] != -1)
			//	{
			//		Vector3 hold_pos = mContextRRT->getHoldPos(mSample.initial_hold_ids[i]);
			//		midPoint += hold_pos;
			//		_count++;
			//	}
			//}
			//if (_count != 0)
			//{
			//	midPoint /= (float)_count;
			//	Vector3 disVec = (midPoint - mContextRRT->getBonePosition(SimulationContext::BodyName::BodyTrunk));
			//	mSample.sourceP.push_back(mOptCPBP::ControlledPoses::LeanBack);
			//	mSample.destinationP.push_back(midPoint);
			//}


			mSample.sourceP.push_back(mOptCPBP::ControlledPoses::MiddleTrunk);
			mSample.destinationP.push_back(mMouseTracker->desired_COM);
			
			// always the last element is the torso direction, we use it later for online changing of the torso direction
			mSample.sourceP.push_back(mOptCPBP::ControlledPoses::TorsoDir);
			mSample.destinationP.push_back(mTools::getDirectionFromAngles(mMouseTracker->theta_torso, mMouseTracker->phi_torso));


			//			if (isBackSpaceHit)
			//			{
			//				mSaveUserDataFunc();
			//			}
			//			mSample.starting_time = timeSpendInPlanning;
			isBackSpaceHit = false;
		}

		return;
	}

	//////////////////////////////////////////// used for online optimization //////////////////////////////////////

	//used for online steering in the test mode
	void mSteerFunc(bool isOnlineOptimization)
	{
		mOptimizer->syncMasterContextWithStartState(!isOnlineOptimization);

		mOptimizer->optimize_the_cost(isOnlineOptimization, mSample.sourceP, mSample.destinationP, mSample.desired_hold_ids, debugShow, mSample.isAllowedOnGround);

		// check changing of the cost
		float costImprovement = max(0.0f, mSample.cOptimizationCost - mOptimizer->current_cost_state);
		rcPrintString("Traj. cost improvement: %f", costImprovement);
		if (costImprovement < noCostImprovementThreshold)
		{
			mSample.numItrFixedCost++;
		}
		else
		{
			mSample.numItrFixedCost = 0;

		}
		//		if (mOptimizerOnline->current_cost_state < mSample.cOptimizationCost)
		mSample.cOptimizationCost = mOptimizer->current_cost_state;

		//apply the best control to get the start state of next frame
		if (isOnlineOptimization)
		{
			mStepOptimization(0, true, isOnlineOptimization);
		}

		return;
	}

	void mStepOptimization(int cTimeStep, bool saveIntermediateStates, bool debugPrint = false)
	{
		std::vector<outSavedData> nStates;

		mOptimizer->simulateBestTrajectory(saveIntermediateStates, mSample.desired_hold_ids, nStates);// advance_simulation(cTimeStep, debugPrint);

		mSample.control_cost += mOptimizer->current_cost_control;

		for (unsigned int i = 0; i < nStates.size(); i++)
		{
			_fromToMotions.push_back(nStates[i]);
		}
	}

}*mTestClimber;


static void write_vector_to_file(std::string filename, std::vector<std::vector<float>>& vector_to_write) {
	std::ofstream myfile;
	myfile.open(filename);

	for (std::vector<float>& vect : vector_to_write) {

		for (int i = 0; i < (int)vect.size(); i++) {

			myfile << vect[i];

			if (i < (int)vect.size() - 1) {
				myfile << ",";
			}

		}
		myfile << std::endl;

	}

	myfile.close();
}

class mLearnerSamplingClass
{
	const bool useMachineLearningGraph = true; // set this to true for using machine learning in Graph search
	const bool useMachineLearning = true; // set this to true for using machine learning in CMA-ES Opt
	const bool useTensorFlow = false;
	const bool debugNeuralNetwork = false;
	const bool testEfficiency = false;
	const bool onlyTrainNN = !testEfficiency && false;
	const bool allowTrainNN = onlyTrainNN || false;
	const float minExplorationRate = 0.0f;//testEfficiency ? 0.0f : 0.25f;
	const float maxExplorationRate = 0.0f;//testEfficiency ? 0.0f : 0.25f;
	//handling offline - online method
	const int rnd_diff_transition = 2;//{1,2,3,4} movements// !useMachineLearningGraph ? 2 : 4;
	
	enum targetMethod { Offiline = 0, Online = 1 };

	mController* mOptimizer;
	SimulationContext* mContext;

	bool debugShow;

	mSampleStructure mSample;

	bool pauseOptimization;
	bool isOptDone;
	int itr_optimization_for_each_sample;
	int max_itr_optimization_for_each_sample;
	int max_no_improvement_iterations;
	float current_cost_togoal;
	bool flag_use_reducedCost_forOpt;
	float minCostThreshold;

	mDataManager mDataCollector;
	mANNBase mNNStanceSuccessNet;
	mANNBase mNNStanceCostNet;
	mANNBase mNNStancePolicyCostNet;
	mANNBase mNNStancePolicyNet;
	bool isPolicyNetUsable;
	int numEpochTotrain;
	float currentExplorationRate;
	int fist_init_data_size;

	std::vector<mTransitionData> mStoredData;
	int max_stored_data;
	int current_datum_num;
	int file_num;
	int loaded_file_num;

	mTreeNode startingStateLearning;

	// planning using graph search
	mStanceGraph* mySamplingGraph;
	std::vector<int> init_stance;
	float _cPathCost;
	std::vector<int> stance_graph_path;
	int current_index_on_graph_path;
	int current_index_on_tree_nodes;
	std::vector<mTreeNode> treeNodes;
	float _preCostTransition;

	// some analysis
	std::vector<int> itrCMAESLearningMovingLimbs;
	std::vector<int> succCMAESLearningMovingLimbs;
	std::vector<int> countCMAESLearningMovingLimbs;
	std::vector<int> itrCMAESMovingLimbs;
	std::vector<int> succCMAESMovingLimbs;
	std::vector<int> countCMAESMovingLimbs;

	int num_path_tried;
	std::vector<float> statistics_res;
public:
	mLearnerSamplingClass(SimulationContext* iContext, mController* iController, mStanceGraph* iStanceGraph, const mEnumTestCaseClimber& cTestID)
		: startingStateLearning(iController->startState.getNewCopy(iContext->getNextFreeSavingSlot(), iContext->getMasterContextID()), 0),
		mDataCollector(iContext), 
		mNNStanceSuccessNet(&mDataCollector, mMachineLearningType::Success),
		mNNStanceCostNet(&mDataCollector, mMachineLearningType::Cost),
		mNNStancePolicyNet(&mDataCollector, mMachineLearningType::Policy),
		mNNStancePolicyCostNet(&mDataCollector, mMachineLearningType::PolicyCost)
	{
		statistics_res = std::vector<float>(4, 0);
		num_path_tried = 0;
		isPolicyNetUsable = false;

		mContext = iContext;
		mOptimizer = iController;
		mySamplingGraph = iStanceGraph;

		sinCosData();

		debugShow = false;

		int batch_size = 500;//500;//200;

		mNNStanceSuccessNet.bach_size = batch_size;
		mNNStanceSuccessNet.use_variance_loss_function = false;
		mNNStanceSuccessNet.flag_normalize_output = false;

		mNNStanceCostNet.bach_size = batch_size;
		mNNStanceCostNet.use_variance_loss_function = false;
		mNNStanceCostNet.flag_normalize_output = true;

		mNNStancePolicyNet.bach_size = batch_size;
		mNNStancePolicyNet.use_variance_loss_function = false;
		mNNStancePolicyNet.flag_normalize_output = false;

		mNNStancePolicyCostNet.bach_size = batch_size;
		mNNStancePolicyCostNet.use_variance_loss_function = false;
		mNNStancePolicyCostNet.flag_normalize_output = true;

		loadNeuralNetwork();

//		writeFileSuccessRateFromGivenStance(&mNNStanceSuccessNet); // mNNStanceCostNet

		mNNStanceSuccessNet.setLearningRate(0.001f);
		mNNStanceCostNet.setLearningRate(0.001f);
		mNNStancePolicyNet.setLearningRate(0.001f);
		mNNStancePolicyCostNet.setLearningRate(0.001f);

		if (mNNStancePolicyNet.isMinMaxValuesSet && !onlyTrainNN)
		{
			numEpochTotrain = 5;
		}
		else
		{
			numEpochTotrain = 5;
		}

		pauseOptimization = false;
		isOptDone = false;
		itr_optimization_for_each_sample = 0;
		max_itr_optimization_for_each_sample = useOfflinePlanning ? (int)(4.0f * cTime) : (int)(3.0f * cTime);
		max_no_improvement_iterations = useOfflinePlanning ? (int)(cTime) : (int)(cTime / 2.0f);
		current_cost_togoal = FLT_MAX;
		flag_use_reducedCost_forOpt = false;
		minCostThreshold = 100.0f;
		mOptimizer->using_sampling = true;

		if (!allowTrainNN)
			max_stored_data = 5000;
		else
			max_stored_data = 50;
		mStoredData = std::vector<mTransitionData>(max_stored_data, mTransitionData());
		current_datum_num = 0;

		init_stance = std::vector<int>(4,-1);
		startingStateLearning.nodeIndex = 0;
		startingStateLearning.cost_to_father = 0.0f;
		treeNodes.push_back(startingStateLearning);
		current_index_on_graph_path = -1;
		current_index_on_tree_nodes = -1;

		_cPathCost = 0.0f;
		_preCostTransition = -1.0f;

		// some analysis
		itrCMAESLearningMovingLimbs = std::vector<int>(4,0);
		succCMAESLearningMovingLimbs = std::vector<int>(4, 0);
		countCMAESLearningMovingLimbs = std::vector<int>(4, 0);
		itrCMAESMovingLimbs = std::vector<int>(4, 0);
		succCMAESMovingLimbs = std::vector<int>(4, 0);
		countCMAESMovingLimbs = std::vector<int>(4, 0);

		// play animation
		isPathFound = false;
		lastPlayingNode = 0;
		lastPlayingStateInNode = 0;
		isNodeShown = false;
//		cTimeElapsed = 0.0f;
		isAnimationFinished = false;
		cGoalPathIndex = 0;

		file_num = 0; loaded_file_num = 0;
		fist_init_data_size = mDataCollector.getDataSize(mDataCollector.transition_data_);
		if (allowTrainNN)
		{
			int size_loaded = 0;
			size_loaded = loadTrainingData(1, mDataCollector.getDataSize(mDataCollector.transition_data_));

			printf("%d data added for 1 limb \n", size_loaded);

			mDataCollector.learning_budget_ += 10000;
			file_num = 0; loaded_file_num = 0;
			size_loaded = loadTrainingData(2, mDataCollector.getDataSize(mDataCollector.transition_data_));

			printf("%d data added for 2 limbs \n", size_loaded);

			mDataCollector.learning_budget_ += 10000;
			file_num = 0; loaded_file_num = 0;
			size_loaded = loadTrainingData(3, mDataCollector.getDataSize(mDataCollector.transition_data_));

			printf("%d data added for 3 limbs \n", size_loaded);

			mDataCollector.learning_budget_ += 10000;
			file_num = 0; loaded_file_num = 0;
			size_loaded = loadTrainingData(4, mDataCollector.getDataSize(mDataCollector.transition_data_));

			printf("%d data added for 4 limbs \n", size_loaded);
			
			fist_init_data_size = mDataCollector.getDataSize(mDataCollector.transition_data_);

			mDataCollector.learning_budget_ += 20000;
			file_num = 0; loaded_file_num = 0;
			size_loaded = loadTrainingData(-1, mDataCollector.getDataSize(mDataCollector.transition_data_));

			printf("%d data added for all other \n", size_loaded);
//			file_num = 279; loaded_file_num = 279;
		}
		currentExplorationRate = maxExplorationRate;

		mNNStancePolicyNet.calculateMSEOnTrainAndTestSet();
		mNNStancePolicyCostNet.calculateMSEOnTrainAndTestSet();
		mNNStanceCostNet.calculateMSEOnTrainAndTestSet();
		mNNStanceSuccessNet.calculateMSEOnTrainAndTestSet();

		resetLearningScene(true);
	}

	~mLearnerSamplingClass()
	{
		saveNeuralNetwork();

		if (current_datum_num > 0)
		{
			writeDataToFile();
		}

		/*mFileHandler writeTrainNumFile("ClimberInfo\\mTrainingNum.txt", "w");
		std::string write_buff;
		char _buff[200];
		sprintf_s(_buff, "%d\n", currentNumTrainItr); write_buff += _buff;
		writeTrainNumFile.writeLine(write_buff);
		writeTrainNumFile.mCloseFile();*/
	}

	void writeDataToFile()
	{
		if (testEfficiency || !allowTrainNN)
			return;
		if (CreateDirectory("ClimberResults", NULL) || ERROR_ALREADY_EXISTS == GetLastError())
		{
			char writeBuff[200];

			bool flag_continue = true;
			while (flag_continue)
			{
				mContext->getAppendAddress("ClimberResults\\Data", file_num, ".txt", writeBuff);
				if (!fileExists(writeBuff))
				{
					flag_continue = false;
				}
				else
				{
					file_num++;
				}
			}

			mFileHandler mFileHandler(writeBuff, "a");

			for (int i = 0; i < current_datum_num; i++)
			{
				mFileHandler.writeLine(mStoredData[i].toString());
			}

			mFileHandler.mCloseFile();
		}
		return;
	}

	void runLearnerLoop(bool advance_time, const bool& _playAnimation, const mEnumTestCaseClimber& cTestID)
	{
		targetMethod itargetMethd = useOfflinePlanning ? targetMethod::Offiline : targetMethod::Online;

//		rcPrintString("%f success rate in paths tried", goal_nodes.size() / (num_path_tried + 0.001f));

		if (_playAnimation)
		{
			playAnimation();
			return;
		}
		
		if (mSample.restartSampleStartState)
		{
			mOptimizer->startState = treeNodes[current_index_on_tree_nodes].climber_state;
			mOptimizer->syncMasterContextWithStartState(true);
			mSample.restartSampleStartState = false;
		}
		// do the learning for prediction
		if (useMachineLearning && allowTrainNN)
			doLearning();

		// do the optimization for reaching
		if (!pauseOptimization && mSample.isSet)
		{
			if (itargetMethd == targetMethod::Online)
			{
				runOnlineOpt(advance_time);
			}
			else
			{
				runOfflineOpt(advance_time);
			}
		}
		// play animation in offline mode
		else if (pauseOptimization && mSample.isSet && isOptDone)
		{
			// play best simulated trajectory after the optimization is done
			if (itargetMethd == targetMethod::Offiline)
			{
				std::vector<outSavedData> nStates;
				isOptDone = !mOptimizer->simulateBestTrajectory(true, mSample.desired_hold_ids, nStates);

				for (unsigned int i = 0; i < nStates.size(); i++)
				{
					mSample.statesFromTo.push_back(nStates[i]._s);
				}

				if (!isOptDone)// when playing of last animation is done, check if cost reduced and climber reached the target or not
				{
					if (flag_use_reducedCost_forOpt)
					{
						float cCost = mOptimizer->getCost(mSample.sourceP, mSample.destinationP, mSample.desired_hold_ids, mSample.isAllowedOnGround);
						if (cCost < current_cost_togoal && !mTools::isSetAEqualsSetB(mOptimizer->startState.hold_bodies_ids, mSample.desired_hold_ids))
						{
							// if climber has not reached, do the same optimization again until cost increases
							resetOptimization();
						}
					}
				}
			}
		}
		// plan a path and see the predicted cost is working or not
		else
		{
			getNextTransitionOnPlannedPath();
		}

		isPathFound = false;

		debugPrint();
		drawDesiredLines(false);

		return;
	}

	void sinCosData()
	{
		if (!debugNeuralNetwork)
			return;

		mANNBase mNNDebuggNet(&mDataCollector, mMachineLearningType::_Debugg);

		mNNDebuggNet.bach_size = 100;
		mNNDebuggNet.use_variance_loss_function = true;
		mNNDebuggNet.flag_normalize_output = false;
		float lr = 0.001f;
		mNNDebuggNet.setLearningRate(lr);
		mNNDebuggNet.setMaxGradientNorm(1000000000000.0f);

		int amount_data = 5000;

		std::vector<std::vector<float>> true_data;
		float noise = 0.05f;
		for (int i = 0; i < amount_data; ++i) 
		{

			std::vector<float> tmp;

			float input = (i % 200) / 200.0f * (2 * PI);
			float output = 0;
			int rnd = (int)(mTools::getRandomBetween_01() + 0.5f);
			if (rnd)
				output = sinf(input) - noise/2 + mTools::getRandomBetween_01() * (noise);
			else
				output = cosf(input) - noise / 2 + mTools::getRandomBetween_01() * (noise);

			output = (output + 1.0f) / 2.0f;

			mTransitionData* cVal = new mTransitionData();
			cVal->loadToFloatTestSample(input, output);
			mDataCollector.addSample(cVal, 0);

			tmp.push_back(input);
			tmp.push_back(output);

			true_data.push_back(tmp);
		}

		mNNDebuggNet.updateMinMaxMeanVals();
		mNNDebuggNet.curEpochNum = 1000;
		write_vector_to_file("data.csv",true_data);

		int counter = 0;
		int last_shown = mNNDebuggNet.curEpochNum;
		while (true)
		{
			mNNDebuggNet.train();
			
			
			if (last_shown - mNNDebuggNet.curEpochNum > 5)
			{
				//lr *= 0.75;
				//mNNDebuggNet.setLearningRate(lr);

				last_shown = mNNDebuggNet.curEpochNum;
				std::vector<std::vector<float>> data;

				for (int i = 0; i < 200; i++)
				{
					std::vector<float> tmp;

					VectorXf cDebugState = VectorXf(1);
					cDebugState[0] = true_data[i][0];
					VectorXf cDebugVal = VectorXf::Zero(1);
					VectorXf cDebugVar = VectorXf::Zero(1);
					mNNDebuggNet.getMachineLearningOutput(cDebugState.data(), cDebugVal.data(), cDebugVar.data());

					tmp.push_back(cDebugState[0]);
					tmp.push_back(true_data[i][1]);
					tmp.push_back(cDebugVal[0]);
					tmp.push_back(cDebugVar[0]);

					data.push_back(tmp);
				}

				write_vector_to_file("mean_var_data.csv", data);
				std::system("python mean_std_plot.py");
				
			}
			if (counter % 100 == 0)
			{
				printf("Data - Size, NN: %d, Epoch: %d, lr: %.5f \n", mDataCollector.getDataSize(mDataCollector.transition_data_), mNNDebuggNet.curEpochNum, lr);
				printf("MSE DebugNN - (Train: %0.3f, Test: %0.3f) \n", mNNDebuggNet.getMSETrain(), mNNDebuggNet.getMSETest());
				
			}

			counter++;
		}

		exit(0);
	}
	
	void writeFileSuccessRateFromGivenStance(mANNBase* _successNet)
	{
		bool add_dis = true;
		mTransitionData mSample;
		std::vector<Vector3> _fromPos;
		_fromPos.push_back(Vector3(0.0f, 0.0f, 0.5f));//ll
		_fromPos.push_back(Vector3(0.8f, 0.0f, 0.5f));//rl
		_fromPos.push_back(Vector3(0.0f, 0.0f, 1.67f));//lh
		_fromPos.push_back(Vector3(0.8f, 0.0f, 1.67f));//rh

		std::vector<Vector3> _toPos = _fromPos;

		std::vector<std::vector<float>> true_data;

		std::vector<int> target_id;

		target_id.push_back(3);
//		target_id.push_back(3);
		if (target_id.size() > 1)
			add_dis = false;
		for (unsigned int i = 0; i < 10000; i++)
		{
			std::vector<Vector3> _dis;
			for (unsigned int j = 0; j < target_id.size(); j++)
			{
				float theta_r = mTools::getRandomBetween(0.0f, 2.0f * PI);
				float r_r = mTools::getRandomBetween(0.01f, 5.0f);

				Vector3 cdis = r_r * Vector3(cosf(theta_r), 0.0f, sinf(theta_r));
				_dis.push_back(cdis);

				_toPos[target_id[j]] = _fromPos[target_id[j]] + _dis[j];
			}
			
			mSample.fillInputStanceFrom(_fromPos, _toPos);

			VectorXf cSuccessState = mSample.getSuccessState();
			VectorXf cSuccessRate = VectorXf::Zero(1);
			_successNet->getMachineLearningOutput(cSuccessState.data(), cSuccessRate.data());

			std::vector<float> tmp;
			
			
			if (add_dis)
			{
				tmp.push_back(4);
				tmp.push_back(_dis[0][0]);
				tmp.push_back(_dis[0][2]);
			}
			else
			{
				Vector3 _dis_from = _toPos[target_id[1]] - _toPos[target_id[0]];

				tmp.push_back(4);
				tmp.push_back(_dis_from[0]);
				tmp.push_back(_dis_from[2]);
			}
			
			if (mGraph->mClimberSampler->isValidStanceSample(_toPos))
				tmp.push_back(cSuccessRate[0]);
			else
				tmp.push_back(0.0f);

			true_data.push_back(tmp);
		}

		write_vector_to_file("data_.csv", true_data);

		exit(0);
		return;
	}

	int cGoalPathIndex;
	std::vector<int> goal_nodes;
	void findBestPathToGoal()
	{
		Vector3 desiredPos = mContext->getGoalPos();

		int min_index = -1;
		if (goal_nodes.size() == 0)
		{
			float minDis = FLT_MAX;
			for (unsigned int i = 0; i < treeNodes.size(); i++)
			{
				mTreeNode& nodei = treeNodes[i];

				float cDis = nodei.getSumDisEndPosTo(desiredPos);

				if (minDis > cDis)
				{
					minDis = cDis;
					min_index = i;
				}
			}
		}
		else
		{
			/*std::vector<int> _sameCostIndices;
			float minCost = FLT_MAX;
			float maxCost = -FLT_MAX;

			for (unsigned int i = 0; i < goal_nodes.size(); i++)
			{
			float cCost = getNodeCost(&mRRTNodes[goal_nodes[i]]);
			if (cCost < minCost)
			{
			_sameCostIndices.clear();
			_sameCostIndices.push_back(i);
			minCost = cCost;
			}
			else if (cCost == minCost)
			{
			_sameCostIndices.push_back(i);
			}
			}*/

			min_index = goal_nodes[cGoalPathIndex];
		}

		std::vector<int> cPath;
		mTreeNode cNode = treeNodes[min_index];
		while (cNode.mFatherIndex != -1)
		{
			cPath.insert(cPath.begin(), cNode.nodeIndex);
			if (cNode.mFatherIndex != -1)
				cNode = treeNodes[cNode.mFatherIndex];
		}
		cPath.insert(cPath.begin(), cNode.nodeIndex);
		path_nodes_indices = cPath;

		return;
	}

private:
	std::vector<int> path_nodes_indices;
	bool isPathFound;
	bool isAnimationFinished;
	bool isNodeShown;
	int lastPlayingNode;
	int lastPlayingStateInNode;

	void playAnimation()
	{
		clock_t begin = clock();
		if (!mSample.restartSampleStartState)
			mSample.restartSampleStartState = true;

		if (!isPathFound)
		{
			findBestPathToGoal();

			isPathFound = true;

			isAnimationFinished = false;
		}

		if (!isNodeShown)
		{
			if (lastPlayingNode < (int)path_nodes_indices.size() && path_nodes_indices[lastPlayingNode] < (int)treeNodes.size())
			{
				mOptimizer->startState = treeNodes[path_nodes_indices[lastPlayingNode]].climber_state;
				isNodeShown = true;
				lastPlayingStateInNode = 0;
				lastPlayingNode++;
			}
			else
			{
				// restart showing animation
				lastPlayingNode = 0;
//				cTimeElapsed = 0;
				isAnimationFinished = true;
			}
		}
		else
		{
			if (lastPlayingNode < (int)path_nodes_indices.size() && path_nodes_indices[lastPlayingNode] < (int)treeNodes.size())
			{
				if (lastPlayingStateInNode < (int)treeNodes[path_nodes_indices[lastPlayingNode]].statesFromFatherToThis.size())
				{
					mOptimizer->startState = treeNodes[path_nodes_indices[lastPlayingNode]].statesFromFatherToThis[lastPlayingStateInNode];
					lastPlayingStateInNode++;
				}
				else
				{
					isNodeShown = false;
				}
			}
			else
			{
				lastPlayingNode = 0;
				isNodeShown = false;
//				cTimeElapsed = 0;
				isAnimationFinished = true;
			}
		}
		Sleep(10); // 1 frame every 30 ms

		//rcPrintString("%f, %f, %f, %f", mOptimizer->startState._body_control_cost[0]
		//	, mOptimizer->startState._body_control_cost[1]
		//	, mOptimizer->startState._body_control_cost[2]
		//	, mOptimizer->startState._body_control_cost[3]);

		mOptimizer->syncMasterContextWithStartState(true);
//		clock_t end = clock();
//		cTimeElapsed += double(end - begin) / CLOCKS_PER_SEC;
		return;
	}

	void resetLearningScene(bool _randomize)
	{
		if (onlyTrainNN)
			return;

		treeNodes.clear();
		startingStateLearning.mChildrenTreeNodes.clear();
		treeNodes.push_back(startingStateLearning);
		
		mOptimizer->startState = startingStateLearning.climber_state;
		mOptimizer->syncMasterContextWithStartState(true);

		stance_graph_path.clear();
		mySamplingGraph->emptyGraph();

		if (_randomize)
		{
			std::vector<int> ret_index;
			float _dir = mContext->randomizeHoldPositions(DemoID, (rnd_diff_transition <= 0) ? -1 : mTools::getRandomBetween(0.25f * PI, 0.75f * PI), ret_index);

			if (DemoID == mDemoTestClimber::DemoLongWall || DemoID == mDemoTestClimber::DemoHorRotatedWall)
			{
				init_stance[2] = ret_index[0]; 
				init_stance[3] = ret_index[1];
				/*if (_dir > 0)
				{
					init_stance[2] = 2; init_stance[3] = 3;
				}
				else
				{
					init_stance[2] = 3; init_stance[3] = 2;
				}*/
			}
			else
			{
				if (DemoID == mDemoTestClimber::DemoJump4 || DemoID == mDemoTestClimber::DemoJump5)
				{
					init_stance[2] = 1; init_stance[3] = 2;
				}
				else
				{
					init_stance[2] = 2; init_stance[3] = 3;
				}
			}
			
			if (DemoID == mDemoTestClimber::DemoJump6)
			{
				init_stance[2] = 1; init_stance[3] = 1;
			}
			if (DemoID == mDemoTestClimber::DemoRouteFromFile)
			{
				init_stance[2] = 0; init_stance[3] = 1;
			}
			mGraph->mClimberSampler->init();
		}

		if (!useTensorFlow)
		{
			int nDataSize = mDataCollector.getDataSize(mDataCollector.transition_data_);
			float p_rand = 0.0f;//mTools::getRandomBetween_01();

//			if ((nDataSize > int(10 * mNNStancePolicyNet.bach_size) &&
//				(!testEfficiency || (testEfficiency && current_datum_num >= max_stored_data / 2))) && p_rand < 0.5f)
			if (useMachineLearningGraph)
			{
				mySamplingGraph->buildGraph(init_stance, &mNNStanceSuccessNet, &mNNStanceCostNet, rnd_diff_transition);
			}
			else
			{
				mySamplingGraph->buildGraph(init_stance, NULL, NULL, rnd_diff_transition);
			}
		}
		else
		{
			mySamplingGraph->buildGraph(init_stance, NULL, NULL, rnd_diff_transition);

			mySamplingGraph->writeToFile();

			Sleep(1000);

			// do some python reading
			std::system("python ClimberInfo\\BayesianLearning\\main.py");

			mySamplingGraph->loadNeuralNetworkValues("ClimberInfo\\BayesianLearning\\StanceGraph-Output.txt");
		}
	}

	void getNextTransitionOnPlannedPath()
	{
		if (stance_graph_path.size() == 0)
		{
			_cPathCost = mySamplingGraph->solveGraph();
			stance_graph_path = mySamplingGraph->retPath;
			current_index_on_graph_path = 1;
			current_index_on_tree_nodes = 0;

			mSample.isSet = false;
			num_path_tried++;
		}
		else
		{
			// add a node if the sample was set
			if (mSample.isSet)
			{
				mTransitionData& d_i = mStoredData[current_datum_num];
				float _formSize = 0, _toSize = 0;
				d_i.fillInputStanceFrom(mContext, mSample.initial_hold_ids, mSample.desired_hold_ids);
				d_i.setControlPoints(mSample.controlPoints);
				d_i.fillInputStateFrom(mSample.initBodyState);
				d_i._succeed = true;
				d_i._cost = 0.0f;
				for (unsigned int i = 0; i < mSample.statesFromTo.size(); i++)
				{
					d_i._cost += mSample.statesFromTo[i].control_cost;
				}

				_preCostTransition = d_i._cost;

				mTreeNode& fNode = treeNodes[current_index_on_tree_nodes];
				fNode.mChildrenTreeNodes.push_back(treeNodes.size());
				BipedState nCopy = mOptimizer->startState.getNewCopy(mContext->getNextFreeSavingSlot(), mContext->getMasterContextID());
				// set child info
				mTreeNode childNode(nCopy, mySamplingGraph->findStanceFrom(mOptimizer->startState.hold_bodies_ids));
				childNode.nodeIndex = treeNodes.size();
				childNode.mFatherIndex = current_index_on_tree_nodes;
				childNode.cost_to_father = _preCostTransition;

				treeNodes.push_back(childNode);
				mTreeNode& refChildNode = treeNodes[treeNodes.size() - 1];
				refChildNode.statesFromFatherToThis = mSample.statesFromTo;
				for (unsigned int g = 0; g < mSample.statesFromTo.size(); g++)
				{
					refChildNode.statesFromFatherToThis[g] = mSample.statesFromTo[g];
				}

				statistics_res[0] += 1;
				if (mTools::isSetAEqualsSetB(mOptimizer->startState.hold_bodies_ids, mSample.desired_hold_ids))
				{
					statistics_res[1] += 1;
					statistics_res[2] += d_i._cost;
				}
				statistics_res[3] += itr_optimization_for_each_sample;
				// do some analysis
				if (testEfficiency)
				{
					int diff_limbs = mTools::getDiffBtwSetASetB(mSample.initial_hold_ids, mSample.desired_hold_ids) - 1;
					if (mSample.isUsingMachineLearning)
					{
						if (diff_limbs >= 0 && diff_limbs <= 3)
						{
							countCMAESLearningMovingLimbs[diff_limbs]++;
							itrCMAESLearningMovingLimbs[diff_limbs] += itr_optimization_for_each_sample;
							if (mTools::isSetAEqualsSetB(mOptimizer->startState.hold_bodies_ids, mSample.desired_hold_ids))
								succCMAESLearningMovingLimbs[diff_limbs]++;
						}
					}
					else
					{
						if (diff_limbs >= 0 && diff_limbs <= 3)
						{
							countCMAESMovingLimbs[diff_limbs]++;
							itrCMAESMovingLimbs[diff_limbs] += itr_optimization_for_each_sample;
							if (mTools::isSetAEqualsSetB(mOptimizer->startState.hold_bodies_ids, mSample.desired_hold_ids))
								succCMAESMovingLimbs[diff_limbs]++;
						}
					}
				}

				if (!mTools::isSetAEqualsSetB(mOptimizer->startState.hold_bodies_ids, mSample.desired_hold_ids))
				{
					// failed case
					d_i._succeed = false;
					d_i._cost = mySamplingGraph->getMaxFailedPredictionCost();

					mySamplingGraph->updateGraphNN(mySamplingGraph->findStanceFrom(mSample.initial_hold_ids), mySamplingGraph->findStanceFrom(mSample.desired_hold_ids));
					mySamplingGraph->initializeOpenListAStarPrune();

					if (current_index_on_graph_path >= 0)
					{
						if (mTools::isSetAEqualsSetB(mySamplingGraph->getStanceGraph(stance_graph_path[current_index_on_graph_path]), mSample.desired_hold_ids))
						{
							stance_graph_path.clear();
							current_index_on_graph_path = -1;
						}
					}

					mSample.isSet = false;
				}
				else if (current_index_on_graph_path == stance_graph_path.size() - 1)
				{
					goal_nodes.push_back(childNode.nodeIndex);

					if (current_index_on_graph_path >= 0)
					{
						if (mTools::isSetAEqualsSetB(mySamplingGraph->getStanceGraph(stance_graph_path[current_index_on_graph_path]), mSample.desired_hold_ids))
						{
							// success case
							if (mTools::isSetAEqualsSetB(mOptimizer->startState.hold_bodies_ids, mSample.desired_hold_ids))
							{
								if (!mTools::isInSampledStanceSet(mySamplingGraph->retPath, mySamplingGraph->m_found_paths))
								{
									mySamplingGraph->m_found_paths.push_back(mySamplingGraph->retPath);
								}
							}

							stance_graph_path.clear();
							current_index_on_graph_path = -1;
						}
					}

					mSample.isSet = false;
				}

				current_datum_num++;
				if (current_datum_num >= (int)mStoredData.size())
				{
					writeDataToFile();
					resetLearningScene(true);
					file_num++;
					current_datum_num = 0;
				}

				if (testEfficiency && current_datum_num == max_stored_data / 2)
				{
					resetLearningScene(false);
				}

				if (statistics_res[0] >= 10)
				{
					mFileHandler m_file("statistics_our_method.txt","a+");
					std::string write_buff;
					char _buff[5000];
					sprintf_s(_buff, "%.3f,%.3f,%.3f,%.3f \n", statistics_res[0], statistics_res[1], statistics_res[2], statistics_res[3]); write_buff += _buff;
					
					m_file.writeLine(write_buff);
					m_file.mCloseFile();

					exit(0);
				}
			}

			bool flag_continue = true;
			if (current_index_on_graph_path < 0 || current_index_on_graph_path >= (int)stance_graph_path.size())
			{
				flag_continue = false;
			}
			while (flag_continue)
			{
				mTreeNode& cNode = treeNodes[current_index_on_tree_nodes];
				bool flag_found = false;
				int nTreeIndex = -1;
				for (unsigned int m = 0; m < cNode.mChildrenTreeNodes.size() && !flag_found; m++)
				{
					mTreeNode& ccNode = treeNodes[cNode.mChildrenTreeNodes[m]];
					if (ccNode.graph_index_node == stance_graph_path[current_index_on_graph_path])
					{
						flag_found = true;
						nTreeIndex = cNode.mChildrenTreeNodes[m];
					}
				}
				if (!flag_found)
				{
					flag_continue = false;
				}
				else
				{
					current_index_on_tree_nodes = nTreeIndex;
					if (current_index_on_graph_path >= (int)(stance_graph_path.size() - 1))
					{
						flag_continue = false;
					}
					else
					{
						current_index_on_graph_path++;
					}
				}
			}
			if (current_index_on_tree_nodes >= 0 && current_index_on_tree_nodes < (int)treeNodes.size())
			{
				mOptimizer->startState = treeNodes[current_index_on_tree_nodes].climber_state;
				mOptimizer->syncMasterContextWithStartState(true);
			}

			float r = mTools::getRandomBetween_01();
			currentExplorationRate = max(currentExplorationRate, minExplorationRate);
			if (r < currentExplorationRate)
			{
				if (current_index_on_graph_path >= 1 && current_index_on_graph_path < (int)stance_graph_path.size())
				{
					int stance_index = stance_graph_path[current_index_on_graph_path - 1];
					unsigned int numChildren = mySamplingGraph->getNumOfChildrenOfStance(stance_index);

					std::vector<int> mRandIndex;
					for (unsigned int i = 0; i < numChildren; ++i)
					{
						int cChildStanceID = mySamplingGraph->getChildStanceID(stance_index, i);
						if (stance_graph_path[current_index_on_graph_path] != cChildStanceID || currentExplorationRate > 0.8f)
							mRandIndex.push_back(cChildStanceID);
					}
					
					if (mRandIndex.size() > 0)
					{
						std::random_shuffle(mRandIndex.begin(), mRandIndex.end());
						mSample.desired_hold_ids = mySamplingGraph->getStanceGraph(mRandIndex[0]);
						updateSampleInfo();
					}
				}
				else
				{
					stance_graph_path.clear();
				}
			}
			else
			{
				if (current_index_on_graph_path >= 0 && current_index_on_graph_path < (int)stance_graph_path.size())
				{
					mSample.desired_hold_ids = mySamplingGraph->getStanceGraph(stance_graph_path[current_index_on_graph_path]);
					updateSampleInfo();
				}
				else
				{
					stance_graph_path.clear();
				}
			}
		}

		mSample.statesFromTo.clear();
	}

	void saveNeuralNetwork()
	{
		mNNStanceSuccessNet.waitForLearning();
		mNNStanceSuccessNet.saveToFile("ClimberInfo\\SuccessNet");
		mNNStanceCostNet.waitForLearning();
		mNNStanceCostNet.saveToFile("ClimberInfo\\CostNet");
		mNNStancePolicyNet.waitForLearning();
		mNNStancePolicyNet.saveToFile("ClimberInfo\\PolicyNet");
		mNNStancePolicyCostNet.waitForLearning();
		mNNStancePolicyCostNet.saveToFile("ClimberInfo\\PolicyCostNet");
	}

	void loadNeuralNetwork()
	{
		mNNStanceSuccessNet.loadFromFile("ClimberInfo\\SuccessNet");
		mNNStanceCostNet.loadFromFile("ClimberInfo\\CostNet");
		isPolicyNetUsable = mNNStancePolicyNet.loadFromFile("ClimberInfo\\PolicyNet");
		mNNStancePolicyCostNet.loadFromFile("ClimberInfo\\PolicyCostNet");
	}

	void debugPrint()
	{
		float edgeCost = mySamplingGraph->getEdgeCost(mSample.success_rate, mOptimizer->current_cost_control, mSample.desired_hold_ids, mSample.cost_var);
		float edgeSuccess = mySamplingGraph->getNeuralNetworkSuccessRate(mySamplingGraph->findStanceFrom(mSample.initial_hold_ids), mySamplingGraph->findStanceFrom(mSample.desired_hold_ids));
		float edgeCostGraph = mySamplingGraph->getNeuralNetworkCost(mySamplingGraph->findStanceFrom(mSample.initial_hold_ids), mySamplingGraph->findStanceFrom(mSample.desired_hold_ids));

		if (useMachineLearning || useMachineLearningGraph)
		{
			if (useMachineLearning && useMachineLearningGraph)
				rcPrintString("Using Machine Learning");
			else if (isPolicyNetUsable && useMachineLearning)
				rcPrintString("Using Machine Learning in Low-Level Controller");
			else if (useMachineLearningGraph)
				rcPrintString("Using Machine Learning in Graph");
			else
				rcPrintString("NOT Using Machine Learning");
		}
		else
		{
			rcPrintString("NOT Using Machine Learning");
		}

		rcPrintString("Predicted and Current Control Cost: %0.3f, %0.3f", mSample.target_cost, edgeCost);
		//rcPrintString("CSuccess:%0.3f, EdgeSuccess:%0.3f, EdgeCost:%0.3f", mSample.success_rate, edgeSuccess, edgeCostGraph);
		rcPrintString("Edge Success and Cost: %0.3f, %0.3f", edgeSuccess, edgeCostGraph);
		//rcPrintString("Previous Transition Cost: %f", _preCostTransition);

		

		rcPrintString("Best and Current Traj. cost for controller: %0.3f, %0.3f", mOptimizer->best_trajectory_cost, mOptimizer->current_cost_state);

		/*if (false)
		{
			VectorXf cSuccessState = mSample.cStanceData.getSuccessState();
			VectorXf cSuccess = VectorXf::Zero(1);
			VectorXf cSuccessVar = VectorXf::Zero(1);
			mNNStanceSuccessNet.getMachineLearningOutput(cSuccessState.data(), cSuccess.data());
			mNNStanceSuccessNet.getMachineLearningVarOutput(cSuccessState.data(), cSuccessVar.data());

			float total_cost = mSample.cStanceData.getPredictedCostVal(&mNNStanceCostNet, mSample.initial_hold_ids, mSample.desired_hold_ids);

			rcPrintString("Success: %f", cSuccess[0]);
			rcPrintString("Success Var: %f", cSuccessVar[0]);
			rcPrintString("Predicted Cost: %f", total_cost);
			

			rcPrintString("Graph Total Cost: %f", mySamplingGraph->getNeuralNetworkCost(mySamplingGraph->findStanceFrom(mSample.initial_hold_ids), mySamplingGraph->findStanceFrom(mSample.desired_hold_ids)));
			rcPrintString("Graph Success Rate: %f", mySamplingGraph->getNeuralNetworkSuccessRate(mySamplingGraph->findStanceFrom(mSample.initial_hold_ids), mySamplingGraph->findStanceFrom(mSample.desired_hold_ids)));
		}*/
		if (useMachineLearning && allowTrainNN)
		{
			rcPrintString("MSE PolicyNN - (Train: %0.3f, Test: %0.3f)", mNNStancePolicyNet.getMSETrain(), mNNStancePolicyNet.getMSETest());
			rcPrintString("MSE PolicyCostNN - (Train: %0.3f, Test: %0.3f)", mNNStancePolicyCostNet.getMSETrain(), mNNStancePolicyCostNet.getMSETest());
			rcPrintString("MSE SuccessNN - (Train: %0.3f, Test: %0.3f)", mNNStanceSuccessNet.getMSETrain(), mNNStanceSuccessNet.getMSETest());
			rcPrintString("MSE CostNN - (Train: %0.3f, Test: %0.3f)", mNNStanceCostNet.getMSETrain(), mNNStanceCostNet.getMSETest());
			rcPrintString("Data - Size, NN: %d", mDataCollector.getDataSize(mDataCollector.transition_data_));

			rcPrintString("Data Storage Num: %d", current_datum_num);
			rcPrintString("(Cost, Success, Policy, CostPolicy) Train Iteration: %d, %d, %d, %d", mNNStanceCostNet.curEpochNum, mNNStanceSuccessNet.curEpochNum, mNNStancePolicyNet.curEpochNum, mNNStancePolicyCostNet.curEpochNum);
		}
		rcPrintString("Graph Path Cost: %0.3f", _cPathCost);
		// print efficiecy data
		if (testEfficiency)
		{
			rcPrintString("ML Count, 1L:%d, 2L:%d, 3L:%d, 4L:%d"
				, countCMAESLearningMovingLimbs[0], countCMAESLearningMovingLimbs[1]
				, countCMAESLearningMovingLimbs[2], countCMAESLearningMovingLimbs[3]);
			rcPrintString("ML Itr, 1L:%0.2f, 2L:%0.2f, 3L:%0.3f, 4L:%0.2f"
				, itrCMAESLearningMovingLimbs[0] / float(countCMAESLearningMovingLimbs[0] + 0.001f), itrCMAESLearningMovingLimbs[1] / float(countCMAESLearningMovingLimbs[1] + 0.001f)
				, itrCMAESLearningMovingLimbs[2] / float(countCMAESLearningMovingLimbs[2] + 0.001f), itrCMAESLearningMovingLimbs[3] / float(countCMAESLearningMovingLimbs[3] + 0.001f));
			rcPrintString("ML Success, 1L:%0.2f, 2L:%0.2f, 3L:%0.3f, 4L:%0.2f"
				, succCMAESLearningMovingLimbs[0] / float(countCMAESLearningMovingLimbs[0] + 0.001f), succCMAESLearningMovingLimbs[1] / float(countCMAESLearningMovingLimbs[1] + 0.001f)
				, succCMAESLearningMovingLimbs[2] / float(countCMAESLearningMovingLimbs[2] + 0.001f), succCMAESLearningMovingLimbs[3] / float(countCMAESLearningMovingLimbs[3] + 0.001f));


			rcPrintString("CMA-ES Count, 1L:%d, 2L:%d, 3L:%d, 4L:%d"
				, countCMAESMovingLimbs[0], countCMAESMovingLimbs[1]
				, countCMAESMovingLimbs[2], countCMAESMovingLimbs[3]);
			rcPrintString("CMA-ES Itr, 1L:%0.2f, 2L:%0.2f, 3L:%0.3f, 4L:%0.2f"
				, itrCMAESMovingLimbs[0] / float(countCMAESMovingLimbs[0] + 0.001f), itrCMAESMovingLimbs[1] / float(countCMAESMovingLimbs[1] + 0.001f)
				, itrCMAESMovingLimbs[2] / float(countCMAESMovingLimbs[2] + 0.001f), itrCMAESMovingLimbs[3] / float(countCMAESMovingLimbs[3] + 0.001f));
			rcPrintString("CMA-ES Success, 1L:%0.2f, 2L:%0.2f, 3L:%0.3f, 4L:%0.2f"
				, succCMAESMovingLimbs[0] / float(countCMAESMovingLimbs[0] + 0.001f), succCMAESMovingLimbs[1] / float(countCMAESMovingLimbs[1] + 0.001f)
				, succCMAESMovingLimbs[2] / float(countCMAESMovingLimbs[2] + 0.001f), succCMAESMovingLimbs[3] / float(countCMAESMovingLimbs[3] + 0.001f));
		}
		return;
	}

	int loadTrainingData(int diff_limb, int starting_index)
	{
		int cDataSize = mDataCollector.getDataSize(mDataCollector.transition_data_);
		int numLoadedData = 0;
		bool flag_continue = true;

		int counter_failed = 0;

		rcPrintString("fileNum: %d", loaded_file_num);
		while (flag_continue)
		{
			mFileHandler* cFile;
			switch (diff_limb)
			{
			case -1:
				cFile = new mFileHandler(mContext->getAppendAddress("ClimberResults\\Data", loaded_file_num, ".txt"), "r");
				break;
			case 1:
				cFile = new mFileHandler(mContext->getAppendAddress("ClimberResults\\Data_1Limb\\selected\\Data", loaded_file_num, ".txt"), "r");
				break;
			case 2:
				cFile = new mFileHandler(mContext->getAppendAddress("ClimberResults\\Data_2Limb\\selected\\Data", loaded_file_num, ".txt"), "r");
				break;
			case 3:
				cFile = new mFileHandler(mContext->getAppendAddress("ClimberResults\\Data_3Limb\\selected\\Data", loaded_file_num, ".txt"), "r");
				break;
			case 4:
				cFile = new mFileHandler(mContext->getAppendAddress("ClimberResults\\Data_4Limb\\selected\\Data", loaded_file_num, ".txt"), "r");
				break;
			default:
				cFile = new mFileHandler(mContext->getAppendAddress("ClimberResults\\Data", loaded_file_num, ".txt"), "r");
			}
			
			if (cFile->exists())
			{
				std::vector<std::vector<float>> mvalues;
				cFile->readFile(mvalues);

				//numLoadedData += (int)mvalues.size();

				for (unsigned int i = 0; i < mvalues.size(); i++)
				{
					mTransitionData* cVal = new mTransitionData();
					bool isAddable = cVal->loadFrom(mvalues[i]);

					cVal->sample_num = diff_limb;

					if (isAddable)
					{
						mDataCollector.addSample(cVal, starting_index);
						numLoadedData++;
						if (!cVal->_succeed)
							counter_failed++;
					}
					else
						delete cVal;
				}

				loaded_file_num++;
				currentExplorationRate -= 0.1f;
				cFile->mCloseFile();
				delete cFile;
			}
			else
			{
				flag_continue = false;
			}
		}

//		printf("failed data: %d \n", counter_failed);

		if (cDataSize == mDataCollector.learning_budget_)
		{
			cDataSize -= numLoadedData;
		}

		int nDataSize = mDataCollector.getDataSize(mDataCollector.transition_data_);
		if (cDataSize != nDataSize)
		{
			mNNStanceCostNet.curEpochNum = numEpochTotrain;
			mNNStanceSuccessNet.curEpochNum = numEpochTotrain;
			mNNStancePolicyNet.curEpochNum = numEpochTotrain;
			mNNStancePolicyCostNet.curEpochNum = numEpochTotrain;
		}
		// need some data for statistics
		else if (nDataSize > (int)(mNNStancePolicyNet.bach_size * 10) || (numLoadedData == 0 && nDataSize == mDataCollector.learning_budget_))
		{
			mNNStancePolicyNet.updateMinMaxMeanVals();
			mNNStanceCostNet.updateMinMaxMeanVals();
			mNNStanceSuccessNet.updateMinMaxMeanVals();
			mNNStancePolicyCostNet.updateMinMaxMeanVals();
		}

		return numLoadedData;
	}

	void doLearning()
	{
		if (testEfficiency)
		{
			if (current_datum_num < max_stored_data / 2)
				isPolicyNetUsable = false;
			else
				isPolicyNetUsable = true;

			return;
		}

		int loadedNumData = loadTrainingData(-1, fist_init_data_size);
		
		int nDataSize = mDataCollector.getDataSize(mDataCollector.transition_data_);
		if (nDataSize > (int)(mNNStancePolicyNet.bach_size * 10) || (loadedNumData >= 0 && nDataSize == mDataCollector.learning_budget_))
		{
			if (!useTensorFlow)
			{
				if (debugNeuralNetwork)
					mNNStanceCostNet.train();
				else
				{
					mNNStanceCostNet.train();
					mNNStanceSuccessNet.train();
				}
			}
			mNNStancePolicyNet.train();
			mNNStancePolicyCostNet.train();
		}

		return;
	}

	void drawDesiredLines(bool playAnimation)
	{
		mSample.drawDesiredTorsoDir(mContext->getBonePosition(SimulationContext::BodyName::BodyTrunk));
		//rcSetColor(1,0,0,1);
		//Vector3 drawPoint = desired_COM;
		//drawPoint[1] = -0.1f;
		//SimulationContext::drawCross(drawPoint);

		if (!playAnimation)
		{
			rcSetColor(0,1,0,1);
			for (unsigned int i = 0; i < 4; i++)
			{
				if (mSample.desired_hold_ids[i] != -1)
				{
					Vector3 toPoint = mContext->getHoldPos(mSample.desired_hold_ids[i]);
					float _s = mContext->getHoldSize(mSample.desired_hold_ids[i]);
					SimulationContext::drawLine(mContext->getEndPointPosBones((int)(SimulationContext::BodyName::BodyLeftLeg + i)), toPoint);
				}
				else
				{
					SimulationContext::drawCross(mContext->getEndPointPosBones((int)(SimulationContext::BodyName::BodyLeftLeg + i)));
				}
			}

			/*rcSetColor(0, 0, 1, 1);
			for (unsigned int i = 0; i < 4; i++)
			{
				if (mSample.initial_hold_ids[i] != -1)
				{
					Vector3 toPoint = mContext->getHoldPos(mSample.initial_hold_ids[i]);
					float _s = mContext->getHoldSize(mSample.initial_hold_ids[i]);
					toPoint[1] -= (_s / 2);
					SimulationContext::drawLine(mContext->getEndPointPosBones((int)(SimulationContext::BodyName::BodyLeftLeg + i)), toPoint);
				}
				else
				{
					SimulationContext::drawCross(mContext->getEndPointPosBones((int)(SimulationContext::BodyName::BodyLeftLeg + i)));
				}
			}*/
		}
		if (pauseOptimization && debugShow)
			mOptimizer->visualizeForceDirections();
		return;
	}

	/////////////////////////////////////////////// choose offline or online mode
	void runOfflineOpt(bool advance_time)
	{
		// sync all contexts with stating state and current stance
		mOptimizer->syncMasterContextWithStartState(true);

		mOptimizer->using_sampling = !useMachineLearning;

		if (!isPolicyNetUsable || !useMachineLearning)
		{
			mOptimizer->using_sampling = true;
		}

		if (itr_optimization_for_each_sample == 0)
		{
			if (optimizerType == otCMAES)
			{
				mSample.isUsingMachineLearning = useMachineLearning && isPolicyNetUsable;
				
				mTransitionData mS;
				mS.fillInputStanceFrom(mContext, mSample.initial_hold_ids, mSample.desired_hold_ids);
				mS.fillInputStateFrom(mOptimizer->startState);

				VectorXf cPolicyState = mS.getPolicyState();
				if (useMachineLearning && isPolicyNetUsable)
				{
					VectorXf cPolicy = VectorXf::Zero(92);
					mNNStancePolicyNet.getMachineLearningOutput(cPolicyState.data(), cPolicy.data());
					((mOptCMAES*)mOptimizer)->machineLearningInput = cPolicy;
				}

				mSample.initBodyState = mOptimizer->startState; //mS._fullBodyStateFeature;
				//mSample.spinePos = mS._spinePos;
				
				VectorXf cPolicyCost = VectorXf::Zero(1);
				mNNStancePolicyCostNet.getMachineLearningOutput(cPolicyState.data(), cPolicyCost.data());

				VectorXf cSuccessState = mS.getSuccessState();
				VectorXf cSuccessRate = VectorXf::Zero(1);
				mNNStanceSuccessNet.getMachineLearningOutput(cSuccessState.data(), cSuccessRate.data());
				mSample.success_rate = cSuccessRate[0];

				if (!useTensorFlow)
				{
	//				VectorXf cCostState = mS.getCostState();
	//				VectorXf cCostVar = VectorXf::Zero(1);
	//				mNNStanceCostNet.getMachineLearningVarOutput(cCostState.data(), cCostVar.data());
					mSample.cost_var = 2000;// mySamplingGraph->const_var_cost;//cCostVar[0];
				}

				mSample.target_cost = mySamplingGraph->getEdgeCost(mSample.success_rate, cPolicyCost[0], mSample.desired_hold_ids, mSample.cost_var);
			}
			// need to store current cost before optimization
			current_cost_togoal = mOptimizer->getCost(mSample.sourceP, mSample.destinationP, mSample.desired_hold_ids, mSample.isAllowedOnGround);
			mOptimizer->reset();
		}

		if (optimizerType == otCMAES)
			mOptimizer->optimize_the_cost(itr_optimization_for_each_sample == 0, mSample.sourceP, mSample.destinationP, mSample.desired_hold_ids, debugShow, mSample.isAllowedOnGround);
		else
			mOptimizer->optimize_the_cost(false, mSample.sourceP, mSample.destinationP, mSample.desired_hold_ids, debugShow, mSample.isAllowedOnGround);

		float edgeCost = mySamplingGraph->getEdgeCost(mSample.success_rate, mOptimizer->current_cost_control, mSample.desired_hold_ids, mSample.cost_var);
		int current_max_no_improvement_iterations = max_no_improvement_iterations;
		//rcPrintString("TCost:%0.3f, CCost:%0.3f, Success:%0.3f", mSample.target_cost, edgeCost, mSample.success_rate);
		if (useMachineLearning && isPolicyNetUsable)
		{
			if (edgeCost <= mSample.target_cost && mOptimizer->isReachedToTargetStance)
			{
				current_max_no_improvement_iterations = max_no_improvement_iterations / 2;
			}
		}
		if (advance_time)
		{
			float costImprovement = max(0.0f, mSample.cOptimizationCost - mOptimizer->current_cost_state);
			//rcPrintString("Traj. cost improvement: %f", costImprovement);
			if (costImprovement < noCostImprovementThreshold)
			{
				mSample.numItrFixedCost++;
			}
			else
			{
				mSample.numItrFixedCost = 0;
			}
			mSample.cOptimizationCost = mOptimizer->best_trajectory_cost;

			itr_optimization_for_each_sample++;
		}
		// offline optimization is done, simulate forward on the best trajectory
		if ((advance_time && mSample.numItrFixedCost > current_max_no_improvement_iterations)
			|| itr_optimization_for_each_sample > max_itr_optimization_for_each_sample)
		{
			pauseOptimization = true;
			isOptDone = true;
			mSample.isReached = true;

			mSample.controlPoints = mOptimizer->getBestControl();
			mSample.initBodyState = mOptimizer->startState;
			mSample.control_cost = mOptimizer->current_cost_control;
		}
		else
		{
			isOptDone = false;
			mSample.isReached = false;
		}
		return;
	}

	void runOnlineOpt(bool advance_time)
	{
		mSteerFunc(advance_time);

		if (advance_time)
		{
			itr_optimization_for_each_sample++;
		}

		if (itr_optimization_for_each_sample > max_itr_optimization_for_each_sample ||
			mSample.numItrFixedCost > max_no_improvement_iterations)
		{
			itr_optimization_for_each_sample = 0;
			mSample.numItrFixedCost = 0;
			mSample.isOdeConstraintsViolated = false;
			mSample.isReached = true;
		}

		if (mSample.isReached || mSample.isRejected)
		{
			pauseOptimization = true;
		}

		return;
	}

	void resetOptimization()
	{
		itr_optimization_for_each_sample = 0;
		mSample.cOptimizationCost = FLT_MAX;
		mSample.numItrFixedCost = 0;
		mSample.isReached = false;
		mSample.isRejected = false;
		mSample.isSet = true;

		pauseOptimization = false;
	}

	void updateSampleInfo()
	{
		float desired_theta_torso = 90.0f;
		float desired_phi_torso = 0.0f;

		/*for (unsigned int i = 0; i < mSample.desired_hold_ids.size(); i++)
		{
			if (mSample.desired_hold_ids[i] >= 0)
			{
				SimulationContext::holdContent& hold_body = mContext->holds_body[mSample.desired_hold_ids[i]];

				int rnd_hold_prototype = mTools::getRandomIndex(mContext->holds_prototypes.size());
				hold_body.setIdealForceDirection(0, mContext->holds_prototypes[rnd_hold_prototype].getIdealForceDirection(0));
				hold_body.public_prototype_hold_id = mContext->holds_prototypes[rnd_hold_prototype].public_prototype_hold_id;
			}

			if (mOptimizer->startState.hold_bodies_ids[i] >= 0)
			{
				SimulationContext::holdContent& hold_body = mContext->holds_body[mOptimizer->startState.hold_bodies_ids[i]];

				int rnd_hold_prototype = mTools::getRandomIndex(mContext->holds_prototypes.size());
				hold_body.setIdealForceDirection(0, mContext->holds_prototypes[rnd_hold_prototype].getIdealForceDirection(0));
				hold_body.public_prototype_hold_id = mContext->holds_prototypes[rnd_hold_prototype].public_prototype_hold_id;
			}
		}*/

		// needs to optimize again
		resetOptimization();

		mSample.destinationP.clear();
		mSample.sourceP.clear();
		mSample.initial_hold_ids = mOptimizer->startState.hold_bodies_ids;
		mSample.isAllowedOnGround = mTools::isSetAEqualsSetB(mSample.initial_hold_ids, std::vector<int>(4, -1));

		/*if (!mSample.isAllowedOnGround)
		{
			int idxContext = getCurrentOdeContext();
			setCurrentOdeContext(ALLTHREADS);
			odeGeomPlaneSetParams(mContext->spaceID, 0, 0, 1, -1.0f);
			setCurrentOdeContext(idxContext);
		}
		else
		{
			int idxContext = getCurrentOdeContext();
			setCurrentOdeContext(ALLTHREADS);
			odeGeomPlaneSetParams(mContext->spaceID, 0, 0, 1, 0.0f);
			setCurrentOdeContext(idxContext);
		}*/

		int counter_mean_com = 0;
		for (unsigned int i = 0; i < mSample.desired_hold_ids.size(); i++)
		{
			if (mSample.desired_hold_ids[i] != -1)
			{
				Vector3 dPos = mContext->getHoldPos(mSample.desired_hold_ids[i]);

				mSample.sourceP.push_back((mOptCPBP::ControlledPoses)(mOptCPBP::ControlledPoses::LeftLeg + i));
				mSample.destinationP.push_back(dPos);
			}

			if (mSample.initial_hold_ids[i] != -1)
			{
				mSample.initial_hold_prototypeIDs[i] = mContext->holds_body[mSample.initial_hold_ids[i]].public_prototype_hold_id;
			}
			else
			{
				mSample.initial_hold_prototypeIDs[i] = -1;
			}

			if (mSample.desired_hold_ids[i] != -1)
			{
				mSample.desired_hold_prototypeIDs[i] = mContext->holds_body[mSample.desired_hold_ids[i]].public_prototype_hold_id;
			}
			else
			{
				mSample.desired_hold_prototypeIDs[i] = -1;
			}
		}

		if (mSample.sourceP.size() == 0)
		{
			return;
		}

		mSample.sourceP.push_back(mOptCPBP::ControlledPoses::MiddleTrunk);
		mSample.destinationP.push_back(Vector3(0.0f,0.0f,0.0f));

		// always the last element is the torso direction, we use it later for online changing of the torso direction
		mSample.sourceP.push_back(mOptCPBP::ControlledPoses::TorsoDir);
		mSample.destinationP.push_back(mTools::getDirectionFromAngles(desired_theta_torso, desired_phi_torso));
		return;
	}

	////////////////////////////////////////////////////////////////// not used functions /////////////////////////////////////////////////
	Vector3 setSampleHandFeetPosition(std::vector<int>& desired_holds_ids, int max_body_move)
	{
		// setting hold places of current start state
		for (unsigned int i = 0; i < mOptimizer->startState.hold_bodies_ids.size(); i++)
		{
			if (mOptimizer->startState.hold_bodies_ids[i] >= 0)
			{
				SimulationContext::holdContent& hold_body = mContext->holds_body[mOptimizer->startState.hold_bodies_ids[i]];
				hold_body = mOptimizer->startState.hold_bodies_info[i];
			}
		}

		int mIDs[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
		std::list<int> _freeHoldsIDs(mIDs, mIDs + 8);

		// use info about being connected or not
		// to draw new sample transition
		bool rHandDisconnected = mContext->getHoldBodyIDs(mController::sbRightHand, mContext->getMasterContextID()) == -1;
		bool lHandDisconnected = mContext->getHoldBodyIDs(mController::sbLeftHand, mContext->getMasterContextID()) == -1;
		bool isEitherOfTheHandsConnected = !rHandDisconnected || !lHandDisconnected;

		Vector3 rfootPos = mContext->getEndPointPosBones(SimulationContext::BodyName::BodyRightFoot);
		Vector3 lfootPos = mContext->getEndPointPosBones(SimulationContext::BodyName::BodyLeftFoot);
		Vector3 trunkPos = mContext->getEndPointPosBones(SimulationContext::BodyName::BodyTrunk);
		Vector3 rhandPos = mContext->getEndPointPosBones(SimulationContext::BodyName::BodyRightHand);
		Vector3 lhandPos = mContext->getEndPointPosBones(SimulationContext::BodyName::BodyLeftHand);

		// find free hold iDs and Limiting_Bodies
		std::vector<int> limiting_bodies;
		std::vector<int> assigned_holds(4, -1);
		for (unsigned int i = 0; i < mOptimizer->startState.hold_bodies_ids.size(); i++)
		{
			if (mOptimizer->startState.hold_bodies_ids[i] != -1)
			{
				_freeHoldsIDs.remove(mOptimizer->startState.hold_bodies_ids[i]);
				limiting_bodies.push_back(i);
				assigned_holds[i] = mOptimizer->startState.hold_bodies_ids[i];
			}
		}

		std::vector<int> _handsFeetIds(mIDs, mIDs + 4);
		if (isEitherOfTheHandsConnected)
		{
			// if at least one of the hands connected and both feet are above the ground => then all rand
			std::random_shuffle(_handsFeetIds.begin(), _handsFeetIds.end());
		}
		else
		{
			// if neither of the hands are connected => then just choose from hands
			std::reverse(_handsFeetIds.begin(), _handsFeetIds.end());
			std::random_shuffle(_handsFeetIds.begin(), _handsFeetIds.begin() + 2);
		}

		// make random number for moving bodies
		// we alwayse want to move at least one limb
		int rnd_n_bodies = mTools::getRandomBetween(1, min(4, max_body_move));
		if (!isEitherOfTheHandsConnected)
		{
			rnd_n_bodies = mTools::getRandomBetween(1, min(2, max_body_move));
		}

		Vector3 minArea(-2.0f, 0.0f, -4.5f);
		Vector3 maxArea(2.0f, 0.0f, 4.5f);

		// make random position and max direction of force
		Vector3 center_pos(0.0f, 0.0f, 0.0f);
		unsigned int counter = 0;

		std::random_shuffle(limiting_bodies.begin(), limiting_bodies.end());

		for (std::list<int>::iterator it = _freeHoldsIDs.begin(); it != _freeHoldsIDs.end() && (int)(counter) < rnd_n_bodies; it++)
		{
			int hold_index_i = *it;

			float rnd_free = mTools::getRandomBetween_01();

			if (rnd_free < 0.1f)
			{
				desired_holds_ids[_handsFeetIds[counter]] = -1;
				assigned_holds[_handsFeetIds[counter]] = -1;
			}
			else
			{
				desired_holds_ids[_handsFeetIds[counter]] = hold_index_i;
				assigned_holds[_handsFeetIds[counter]] = hold_index_i;

				SimulationContext::holdContent& hold_body = mContext->holds_body[hold_index_i];

				Vector3 nHoldPos;

				if (!isEitherOfTheHandsConnected) // assuming starting from T-pos
				{
					float max_radius = mContext->climberRadius; //mContext->climberHandHandDis / 2.0f;
																//				float min_radius = 0.1f;

					float radius_sample = mTools::getRandomBetween_01() * max_radius;// -min_radius);
																					 //				radius_sample += min_radius;
					float theta = (2 * PI) * mTools::getRandomBetween_01();
					//				theta += (_handsFeetIds[counter] == mController::StanceBodies::sbRightHand ? (-PI / 2.0f) : (PI /2.0f));
					Vector3 _center = mContext->getEndPointPosBones(SimulationContext::BodyName::BodyTrunkUpper);
					_center[1] = 0.0f;
					nHoldPos = _center + radius_sample * Vector3(cosf(theta), 0, sinf(theta));

					center_pos = _center;
				}
				else
				{
					Vector3 _center = mContext->getHoldPos(assigned_holds[limiting_bodies[0]]);

					float max_radius = mContext->climberRadius;
					float theta = (2 * PI) * mTools::getRandomBetween_01();
					float radius_sample = mTools::getRandomBetween_01() * max_radius;

					nHoldPos = _center + radius_sample * Vector3(cosf(theta), 0, sinf(theta));

					center_pos = _center;
				}
				//	bool flag_not_chosen = true;
				//	Vector3 _center;
				//	for (unsigned int l = 0; l < limiting_bodies.size(); l++)
				//	{
				//		float max_radius = mContext->climberRadius;
				//		float min_radius = 0.5f;
				//		if (limiting_bodies[l] != _handsFeetIds[counter])
				//		{
				//			if (abs(limiting_bodies[l] - _handsFeetIds[counter]) <= 1)
				//			{
				//				if (limiting_bodies[l] <= int(mController::sbRightLeg))
				//				{
				//					max_radius = mContext->climberLegLegDis;
				//				}
				//				else
				//				{
				//					max_radius = mContext->climberHandHandDis;
				//				}
				//			}
				//			_center = mContext->getHoldPos(assigned_holds[limiting_bodies[l]]);
				//			float sTheta = 0;
				//			float _duration = 2 * PI;
				//			decideLimitingRegion(limiting_bodies[l], _handsFeetIds[counter], _center, sTheta, _duration);
				//			limitPosInArea(minArea, maxArea, _center);
				//			if (flag_not_chosen)
				//			{
				//				float nTheta = sTheta;
				//				nTheta += _duration * mTools::getRandomBetween_01();
				//				float radius_sample = min_radius;
				//				radius_sample += (max_radius - min_radius) * mTools::getRandomBetween_01();
				//				flag_not_chosen = false;
				//				nHoldPos = _center + radius_sample * Vector3(cosf(nTheta), 0.0f, sinf(nTheta));
				//				center_pos = _center;
				//			}
				//			else
				//			{
				//				Vector3 cDir = (nHoldPos - _center);
				//				float cDis = cDir.norm();
				//				cDir = cDir.normalized();
				//				if (cDis > max_radius)
				//				{
				//					cDis = max_radius;
				//				}
				//				nHoldPos = _center + cDis * cDir;
				//			}
				//		}
				//	}
				//}
				//limitPosInArea(minArea, maxArea, nHoldPos);

				hold_body.holdPos = nHoldPos;
				int rnd_hold_prototype = mTools::getRandomIndex(mContext->holds_prototypes.size());
				hold_body.setIdealForceDirection(0, mContext->holds_prototypes[rnd_hold_prototype].getIdealForceDirection(0));
				hold_body.public_prototype_hold_id = mContext->holds_prototypes[rnd_hold_prototype].public_prototype_hold_id;

				if (desired_holds_ids[_handsFeetIds[counter]] != -1)
					mTools::addToSetIDs(_handsFeetIds[counter], limiting_bodies);
			}

			counter++;
		}
		// remain still
		for (; counter < mOptimizer->startState.hold_bodies_ids.size(); counter++)
		{
			desired_holds_ids[_handsFeetIds[counter]] = mOptimizer->startState.hold_bodies_ids[_handsFeetIds[counter]];
		}

		return center_pos;
	}

	void decideLimitingRegion(const int& limiting_body, const int& movingBody, Vector3& _center, float& sTheta, float& duration)
	{
		switch (limiting_body)
		{
		case 0:
			_center -= Vector3(0.5f, 0.0f, 0.0f);
			if (movingBody > int(mController::sbRightLeg))
			{
				// for hands
				_center -= Vector3(0.0f, 0.0f, 0.5f);
				sTheta = 0.0f;
				duration = PI / 2.0f;
			}
			else
			{
				// for feet
				sTheta = -PI / 2.0f;
				duration = PI;
			}
			break;
		case 1:
			_center += Vector3(0.5f, 0.0f, 0.0f);
			if (movingBody > int(mController::sbRightLeg))
			{
				_center -= Vector3(0.0f, 0.0f, 0.5f);
				sTheta = PI / 2.0f;
				duration = PI / 2.0f;
			}
			else
			{
				sTheta = PI / 2;
				duration = PI;
			}
			break;
		case 2:
			_center -= Vector3(0.5f, 0.0f, 0.0f);
			if (movingBody > int(mController::sbRightLeg))
			{
				sTheta = -PI / 2.0f;
				duration = PI;
			}
			else
			{
				_center += Vector3(0.0f, 0.0f, 0.5f);
				sTheta = -PI / 2.0f;
				duration = PI / 2.0f;
			}
			break;
		case 3:
			_center += Vector3(0.5f, 0.0f, 0.0f);
			if (movingBody > int(mController::sbRightLeg))
			{
				sTheta = PI / 2.0f;
				duration = PI;
			}
			else
			{
				_center += Vector3(0.0f, 0.0f, 0.5f);
				sTheta = PI;
				duration = PI / 2.0f;
			}
			break;
		}
	}

	void limitPosInArea(const Vector3& minArea, const Vector3& maxArea, Vector3& nHoldPos)
	{
		if (nHoldPos[0] < minArea[0]) nHoldPos[0] = minArea[0];
		if (nHoldPos[2] < minArea[2]) nHoldPos[2] = minArea[2];
		if (nHoldPos[0] > maxArea[0]) nHoldPos[0] = maxArea[0];
		if (nHoldPos[2] > maxArea[2]) nHoldPos[2] = maxArea[2];
		return;
	}

	//used for online steering in the test mode
	void mSteerFunc(bool isOnlineOptimization)
	{
		mOptimizer->syncMasterContextWithStartState(!isOnlineOptimization);

		mOptimizer->optimize_the_cost(isOnlineOptimization, mSample.sourceP, mSample.destinationP, mSample.desired_hold_ids, debugShow, mSample.isAllowedOnGround);

		// check changing of the cost
		float costImprovement = max(0.0f,mSample.cOptimizationCost-mOptimizer->current_cost_state);
		rcPrintString("Traj. cost improvement: %f",costImprovement);
		if (costImprovement < noCostImprovementThreshold)
		{
			mSample.numItrFixedCost++;
		}
		else
		{
			mSample.numItrFixedCost = 0;
			
		}

		mSample.cOptimizationCost = mOptimizer->current_cost_state;

		//apply the best control to get the start state of next frame
		if (isOnlineOptimization) 
		{
			mStepOptimization(true, isOnlineOptimization);
		}

		return;
	}

	void mStepOptimization(bool saveIntermediateStates, bool debugPrint = false)
	{
		std::vector<outSavedData> nStates;

		mOptimizer->simulateBestTrajectory(saveIntermediateStates, mSample.desired_hold_ids, nStates);

		mSample.control_cost += mOptimizer->current_cost_control;

		// for saving animation
		for (unsigned int i = 0; i < nStates.size(); i++)
		{
			mSample.statesFromTo.push_back(nStates[i]._s);
		}
	}

}* mLearnerSampler;

bool advance_time;
bool play_animation;

double total_time_elasped = 0.0f;

int cBodyNum;
int cAxisNum;
float dAngle;
bool revertToLastState;

void forwardSimulation()
{
	if (!testClimber)
	{
//		myRRTPlanner->mRunPathPlanner(
//			useOfflinePlanning ? mRRT::targetMethod::Offiline : mRRT::targetMethod::Online, 
//			advance_time, 
//			play_animation, 
//			(mCaseStudy)(max<int>((int)current_case_study, 0)));
	}
	else
	{
		switch (TestID)
		{
		case TestAngle:
			if (revertToLastState)
			{
				revertToLastState = !revertToLastState;
				mOptimizer->startState = mOptimizer->resetState;
				mOptimizer->syncMasterContextWithStartState(true);
			}
			mContext->setMotorAngle(cBodyNum, cAxisNum, dAngle);
			rcPrintString("axis: %d, angle: %f \n",cAxisNum, dAngle);
			stepOde(timeStep,false);
			break;
		case TestCntroller:
			mTestClimber->runLoopTest(
				advance_time,
				play_animation,
				0); // total_time_elasped
			break;
		case RunLearnerRandomly:
			mLearnerSampler->runLearnerLoop(advance_time, play_animation, TestID);
			break;
		default:
			break;
		}
		
	}
}

////////////////////////////////////////////////////////////// interface functions with drawstuff and unity /////////////////////////////////
void EXPORT_API rcInit()
{
	srand(time(NULL));

	advance_time = true;

	revertToLastState = false;

	mContext = new SimulationContext(testClimber, TestID, DemoID);

	mMouseReader = new mMouseClass(mContext);

	mHoldSampler = new mSampler(mContext);

	BipedState startState;

	startState.saving_slot_state = mContext->getMasterContextID();

	switch (optimizerType)
	{
	case otCPBP:
		mOptimizer = (mController*)(new mOptCPBP(mContext, startState));
		break;
	case otCMAES:
		mOptimizer = (mController*)(new mOptCMAES(mContext, startState));
		break;
	default:
		mOptimizer = (mController*)(new mOptCMAES(mContext, startState));
		break;
	}
	
	mGraph = new mStanceGraph(mContext, mHoldSampler);

	if (TestID == mEnumTestCaseClimber::TestCntroller)
		mTestClimber = new mTestControllerClass(mContext, mOptimizer, mMouseReader);
	else
		mTestClimber = NULL;

	if (TestID == mEnumTestCaseClimber::RunLearnerRandomly)
	{
		mLearnerSampler = new mLearnerSamplingClass(mContext, mOptimizer, mGraph, TestID);
	}
	else
	{
		mLearnerSampler = NULL;
	}

	if (testClimber)
	{
		switch (TestID)
		{
		case mEnumTestCaseClimber::TestAngle:
			cBodyNum = 0;
			cAxisNum = 0;
			dAngle = mContext->getDesMotorAngle(cBodyNum, cAxisNum);
			break;
		default:
			cBodyNum = 0;
			break;
		} 
	}
	else
	{
		cBodyNum = SimulationContext::BodyName::BodyRightArm;
	}

	play_animation = false;

	dAllocateODEDataForThread(dAllocateMaskAll);

}

void EXPORT_API rcGetClientData(RenderClientData &data)
{
	data.physicsTimeStep = timeStep;
	data.defaultMouseControlsEnabled = true;
	data.maxAllowedTimeStep = timeStep;
}

void EXPORT_API rcUninit()
{
	if (mTestClimber)
		delete mTestClimber;
	if (mLearnerSampler)
		delete mLearnerSampler;

	mOptimizer->unInit();

	delete mOptimizer;
	delete mHoldSampler;
	delete mMouseReader;
	delete mContext;

	if (flag_capture_video && testClimber)
	{
		if (fileExists("out.mp4"))
			remove("out.mp4");
		system("screencaps2mp4.bat");
	}
}

void EXPORT_API rcUpdate()
{
	static bool firstRun = true;
	if (firstRun)
	{
		mMouseReader->cameraAdjustment(mContext->computeCOM());
		rcSetLightPosition(lightXYZ[0],lightXYZ[1],lightXYZ[2]);
		firstRun = false;
	}

	if (!play_animation)
	{
		// reporting total time elapsed
		rcPrintString("Total CPU time used: %f \n", total_time_elasped);
	}
	else
	{
		rcPrintString("Path Number: %d", mLearnerSampler->cGoalPathIndex + 1);
	}
	startPerfCount();

	if (!pause)
	{
		// forward simulation for reaching desired stances
		forwardSimulation();
	}

	/////////////////////////////////////////////////////////// draw stuff ///////////////////////////
	if (testClimber)
	{
		switch (TestID)
		{
		case TestAngle:
			mContext->mDrawStuff(cBodyNum, -1, mContext->masterContext,false,false);
			break;
		case TestCntroller:
			mContext->mDrawStuff(mMouseReader->selected_body, mMouseReader->selected_hold, mContext->masterContext, false, mMouseReader->debugShow);
			break;
		case RunLearnerRandomly:
		default:
			mContext->mDrawStuff(-1, -1, mContext->masterContext, false, mMouseReader->debugShow);
			break;
		}
		
	}
	else
	{
		mContext->mDrawStuff(-1, -1, mContext->masterContext,false,false);
	}

	/////////////////////////////////// Adjust Camera Position ////////////////////////////
//	mMouseReader->cameraAdjustment(mContext->computeCOM());

	if (advance_time && !pause)
	{
		total_time_elasped += getDurationMs() / 1000.0f;

		if (flag_capture_video)
			rcTakeScreenShot();
	}


	return;
}

void EXPORT_API rcOnKeyUp(int key)
{

}

void EXPORT_API rcOnKeyDown(int cmd)
{
	switch (cmd) 
	{
	case 'f':
		flag_capture_video = !flag_capture_video;
		break;
	case '+':
		if (mLearnerSampler)
		{
			mLearnerSampler->cGoalPathIndex++;
			if (mLearnerSampler->cGoalPathIndex >= (int)(mLearnerSampler->goal_nodes.size()))
				mLearnerSampler->cGoalPathIndex = 0;
			mLearnerSampler->findBestPathToGoal();
		}
		break;
	case '-':
		if (mLearnerSampler)
		{
			mLearnerSampler->cGoalPathIndex--;
			if (mLearnerSampler->cGoalPathIndex < 0)
				mLearnerSampler->cGoalPathIndex = mLearnerSampler->goal_nodes.size() - 1;
			mLearnerSampler->findBestPathToGoal();
		}
		break;
	case 13: // enter
		if (testClimber)
		{
			if (!play_animation && mTestClimber != NULL)
				mTestClimber->runOptimization();
		}
		break;
	case 9: // tab
		if (mTestClimber)
			mTestClimber->debugShow = !mTestClimber->debugShow;
		if (mMouseReader)
			mMouseReader->debugShow = !mMouseReader->debugShow;
		break;
	case 'm':
		if (mTestClimber)
			mTestClimber->useMachineLearning = !mTestClimber->useMachineLearning;
		break;
	case 8: //backspace
		if (testClimber)
		{
			switch (TestID)
			{
			case TestAngle:
				revertToLastState = !revertToLastState;
				break;
			case TestCntroller:
				mTestClimber->removeLastSavingState(play_animation);
				break;
			default:
				break;
			}
		}
		break;
	case 'q':	
		rcUninit();
		exit(0);
		break;
	case ' ':
		play_animation = !play_animation;
		break;
	case 'o':
		advance_time = !advance_time;
		break;
	case 'p':
		pause = !pause;
		break;
	case 'z':
		if (testClimber)
		{
			cBodyNum += 1;
			dAngle = mContext->getDesMotorAngle(cBodyNum, cAxisNum);
		}
		break;
	case 'x':
		if (testClimber)
		{
			cBodyNum -= 1;
			dAngle = mContext->getDesMotorAngle(cBodyNum, cAxisNum);
		}
		break;
	case 'a':
		if (testClimber) 
		{
			cAxisNum += 1;
			dAngle = mContext->getDesMotorAngle(cBodyNum, cAxisNum);
		}
		break;
	case 's':
		if (testClimber)
		{
			cAxisNum -= 1;
			dAngle = mContext->getDesMotorAngle(cBodyNum, cAxisNum);
		}
		break;
	case 'c':
		if (testClimber)
		{
			dAngle += 0.1f;
		}
		break;
	case 'v':
		if (testClimber)
		{
			dAngle -= 0.1f;
		}
		break;
	default:
		break;
	}
	
	return;
}

void EXPORT_API rcOnMouse(float rayStartX, float rayStartY, float rayStartZ, float rayDirX, float rayDirY, float rayDirZ, int button, int x, int y)
{
	Vector3 rBegin(rayStartX,rayStartY,rayStartZ);
	Vector3 rEnd(rayStartX+rayDirX*100.0f,rayStartY+rayDirY*100.0f,rayStartZ+rayDirZ*100.0f);
	if (mMouseReader)
	{
		mMouseReader->rayBegin = rBegin;
		mMouseReader->rayEnd = rEnd;
		mMouseReader->bMouse = button;
		mMouseReader->lX = mMouseReader->cX;
		mMouseReader->lY = mMouseReader->cY;
		mMouseReader->cX = x;
		mMouseReader->cY = y;
	}
}