
static float mBiasWallY = -0.2f;
static float maxSpeed = optimizerType == otCMAES ? 2.0f*3.1415f : 2.0f*3.1415f;

static const float cTime = 30.0f;
static float contactCFM = 0; 

//Simulation globals
static float worldCfM = 1e-3;
static float worldERP = 1e-5;
static const float maximumForce = 300.0f;
static const float minimumForce = 200.0f; //non-zero FMax better for joint stability
static const float forceCostSd = maximumForce;  //for computing control cost

// some constants 
static const float timeStep = 1.0f / cTime;   //physics simulation time step

#define BodyNUM 11+4		       // number of boxes

static const int contextNUM = optimizerType==otCMAES ? 256 : 265; //maximum number of contexts, i.e., forward-simulated trajectories per animation frame.
#define boneRadius (0.2f)	   // sphere radius 
#define boneLength (0.5f)	   // sphere radius
#define DENSITY 1.0f		   // density of all objects
#define holdSize 0.5f

//static const float _connectionThreshold = 0.5 * holdSize;

// initial rotation (for cost computing in t-pose)
static	__declspec(align(16)) Eigen::Quaternionf initialRotations[100];  //static because the class of which this should be a member should be modified to support alignment...

static inline Eigen::Quaternionf ode2eigenq(ConstOdeQuaternion odeq)
{
	return Eigen::Quaternionf(odeq[0],odeq[1],odeq[2],odeq[3]); 
}
static inline void eigen2odeq(const Eigen::Quaternionf &q, OdeQuaternion out_odeq)
{
	out_odeq[0]=q.w();
	out_odeq[1]=q.x();
	out_odeq[2]=q.y();
	out_odeq[3]=q.z();
}

enum FMaxGroups
{
	fmTorso=0,fmLeftArm ,fmLeftLeg,fmRightArm,fmRightLeg,
	fmEnd
};

//the full state of a body
class BodyState
{
public:
	BodyState()
	{
		setAngle(Vector4(0.0f,0.0f,0.0f,0.0f));
		setPos(Vector3(0.0f,0.0f,0.0f));
		setVel(Vector3(0.0f,0.0f,0.0f));
		setAVel(Vector3(0.0f,0.0f,0.0f));
		boneSize = boneLength;
		bodyType = 0;
	}

	Vector4 getAngle()
	{
		return Vector4(angle[0], angle[1], angle[2], angle[3]);
	}

	Vector3 getPos()
	{
		return Vector3(pos[0], pos[1], pos[2]);
	}

	Vector3 getVel()
	{
		return Vector3(vel[0], vel[1], vel[2]);
	}

	Vector3 getAVel()
	{
		return Vector3(aVel[0], aVel[1], aVel[2]);
	}

	float getBoneSize()
	{
		return boneSize;
	}

	int getBodyType()
	{
		return bodyType;
	}

	void setAngle(Vector4& iAngle)
	{
		angle[0] = iAngle[0];
		angle[1] = iAngle[1];
		angle[2] = iAngle[2];
		angle[3] = iAngle[3];
		return;
	}

	void setPos(Vector3& iPos)
	{
		pos[0] = iPos[0];
		pos[1] = iPos[1];
		pos[2] = iPos[2];
		return;
	}

	void setVel(Vector3& iVel)
	{
		vel[0] = iVel[0];
		vel[1] = iVel[1];
		vel[2] = iVel[2];
		return;
	}

	void setAVel(Vector3& iAVel)
	{
		aVel[0] = iAVel[0];
		aVel[1] = iAVel[1];
		aVel[2] = iAVel[2];
		return;
	}

	void setBoneSize(float iBonSize)
	{
		boneSize = iBonSize;
	}

	void setBodyType(float iBodyType)
	{
		bodyType = iBodyType;
	}
	
private:
	float angle[4];
	float pos[3];
	float vel[3];
	float aVel[3];

	float boneSize;
	int bodyType;
};

class SimulationContext
{
public:
// this model is coming from application (linear or non-linear)
class holdContent
{
private:
	std::vector<int> preBestForceDirIndex;
	std::vector<Vector3> preBodyDir;

	///////////////////////////////////////////////////////////////////////// TODO: right now we have only one f_ideal and CMA-ES is not working with it
	std::vector<float> f_ideal;
	std::vector<float> k;
	std::vector<Vector3> d_ideal;
	std::vector<int> HoldPushPullMode;// 0 = pulling, 1 = pushing

	float calculateValueForceDirection(int i, Vector3& dirBody)
	{
		float _angle = mTools::getAbsAngleBtwVectors(d_ideal[i], dirBody);

		bool _pushBody = HoldPushPullMode[i] == 1;

		float val = f_ideal[i];
		if (_angle > 1.0f && _pushBody)
			val *= (1 / _angle);
		if (_angle < 1.0f && !_pushBody)
			val *= _angle;
		return val; // calculated f_max gotten from different direction
	}

	void getBestLinearForceDirectionTo(Vector3& cBodyDirection, int bIndex)
	{
		if (bIndex <= 1 && preBestForceDirIndex[bIndex] >= 0)
			return;

		if (bIndex >= 2 && (cBodyDirection - preBodyDir[bIndex]).norm() < 0.1f)
			return;

		int index = -1;
		float bFmax = -FLT_MAX;
		for (unsigned int i = 0; i < d_ideal.size(); i++)
		{
			float cFmax = f_ideal[i];
			if (bIndex >= 2)
				cFmax = calculateValueForceDirection(i, cBodyDirection);

			if (cFmax > bFmax)
			{
				index = i;
				bFmax = cFmax;
			}
		}

		preBestForceDirIndex[bIndex] = index;
		preBodyDir[bIndex] = cBodyDirection;
	}

	Vector3 getBodyDirection(SimulationContext* iContextCPBP, int bIndex)
	{
		Vector3 dirBody = iContextCPBP->getBodyDirectionZ(SimulationContext::BodyName::BodyLeftFoot + bIndex); // for hands
		if (bIndex <= 1)
			dirBody = -iContextCPBP->getBodyDirectionY(SimulationContext::BodyName::BodyLeftFoot + bIndex); // for feet
		return dirBody;
	}

	//assumes we are in the correct context!
	float getRatioForceToBodyPart(SimulationContext* iContextCPBP, int bIndex, Vector3& d_ideal, int closestLinearIndex)
	{
		Vector3 dirBody = getBodyDirection(iContextCPBP, bIndex);
		float _angle = mTools::getAbsAngleBtwVectors(d_ideal, dirBody);

		bool _pushBody = true; // alwayse true for feet
		if (bIndex >= 2)
			_pushBody = HoldPushPullMode[closestLinearIndex] == 1;

		if (_angle > 1.0f && _pushBody)
			return squared(1 / _angle);
		if (_angle < 1.0f && !_pushBody) // for pulling (pulling is modeled for hands only!!!)
			return squared(_angle);
		return 1.0f;
	}

	//assumes we are in the correct context!
	Vector3 getDIdealBodyPart(SimulationContext* iContextCont, int bIndex, Vector3& cBodyDirection)//, int targetContext
	{
		getBestLinearForceDirectionTo(cBodyDirection, bIndex);
		int closestLinearIndex = preBestForceDirIndex[bIndex];

		Vector3 d_ideal = this->d_ideal[closestLinearIndex];
		if (bIndex <= 1)
		{
			Vector3 bPos = iContextCont->getBonePosition(SimulationContext::BodyName::BodyLeftFoot + bIndex);
			Vector3 _comPos = iContextCont->computeCOM();
			d_ideal = (bPos - _comPos);
			if (d_ideal.norm() > 0)
				d_ideal = d_ideal.normalized();
		}
		return d_ideal;
	}

	void initialize()
	{
		preBestForceDirIndex = std::vector<int>(4, -1);
		preBodyDir = std::vector<Vector3>(4, Vector3(0, 0, 0));

		theta_normal_direction = -90.0f;
	}

public:
	// this constructor is used to create prototype holds
	holdContent(Vector3 _d_ideal, int _hold_id)
	{
		initialize();

		holdPos = Vector3(0.0f, 0.0f, 0.0f);

		size = 0.5f;

		if (_d_ideal.norm() > 0)
			_d_ideal.normalize();
		addHoldLinearForce(1000.0f, _d_ideal, 1.0f / PI, 0);

		theta = atan2f(_d_ideal[1], _d_ideal[0]) * (180 / PI);
		phi = asinf(_d_ideal[2]) * (180 / PI);

		public_prototype_hold_id = _hold_id;

		geomID = -1;
	}

	holdContent(Vector3 _holdPos, float _f_ideal, Vector3 _d_ideal, float _k, float _size, int _geomID, int _HoldPushPullMode, float theta_axis) 
	{
		initialize();

		holdPos = _holdPos;
		size = _size;

		if (_d_ideal.norm() > 0)
			_d_ideal.normalize();
		addHoldLinearForce(_f_ideal, _d_ideal, _k, _HoldPushPullMode);

		theta_normal_direction = theta_axis;

		theta = atan2f(_d_ideal[1], _d_ideal[0]) * (180 / PI);
		phi = asinf(_d_ideal[2]) * (180 / PI);

		public_prototype_hold_id = -1;

		geomID = _geomID;
	}

	float disFromIdeal(const Vector3& cF)
	{
		return (cF.normalized() - d_ideal[0]).norm();
	}

	void addHoldLinearForce(float _f_ideal, Vector3 _d_ideal, float _k, int _HoldPushPullMode)
	{
		f_ideal.push_back(_f_ideal);

		if (_d_ideal.norm() > 0)
			_d_ideal.normalize();
		d_ideal.push_back(_d_ideal);

		k.push_back(_k);
		HoldPushPullMode.push_back(_HoldPushPullMode);

		return;
	}

	float calculateLinearForceModel(SimulationContext* iContextCPBP, int bIndex, Vector3& cFi)//, int targetContext)
	{
		Vector3 cBodyDirection = getBodyDirection(iContextCPBP, bIndex);
		getBestLinearForceDirectionTo(cBodyDirection, bIndex);
		int closestLinearIndex = preBestForceDirIndex[bIndex];

		float f_ideal = this->f_ideal[closestLinearIndex];
		float k = this->k[closestLinearIndex];

		Vector3 d_ideal = getDIdealBodyPart(iContextCPBP, bIndex, cBodyDirection);//, targetContext
		float ratioBody = getRatioForceToBodyPart(iContextCPBP, bIndex, d_ideal, closestLinearIndex);

		return ratioBody * f_ideal * max<float>(0, 1 - k * mTools::getAbsAngleBtwVectors(cFi, d_ideal));
	}

	// for adjusting direction in the interface
	void setIdealForceDirection(int index, Vector3& cDir)
	{
		d_ideal[index] = cDir;
	}

	// for debug show
	Vector3 getIdealForceDirection(int index)
	{
		return d_ideal[index];
	}

	std::string toString(float _width)
	{
		//#x,y,z,f,dx,dy,dz,k,s
		std::string write_buff;
		char _buff[100];

		sprintf_s(_buff, "%f,", holdPos[0] + _width / 2); write_buff += _buff; // notTransformedHoldPos[0]
		sprintf_s(_buff, "%f,", holdPos[1] + size / 4); write_buff += _buff; // notTransformedHoldPos[1]
		sprintf_s(_buff, "%f,", holdPos[2]); write_buff += _buff; // notTransformedHoldPos[2]

		sprintf_s(_buff, "%f,", f_ideal[0]); write_buff += _buff;

		sprintf_s(_buff, "%f,", d_ideal[0][0]); write_buff += _buff;
		sprintf_s(_buff, "%f,", d_ideal[0][1]); write_buff += _buff;
		sprintf_s(_buff, "%f,", d_ideal[0][2]); write_buff += _buff;

		sprintf_s(_buff, "%f,", k[0]); write_buff += _buff;

		sprintf_s(_buff, "%f,", size); write_buff += _buff;

		sprintf_s(_buff, "%d\n", HoldPushPullMode[0]); write_buff += _buff;
		return write_buff;
	}

	Vector3 holdPos; // each hold has one position
	float size; // each hold has one size

	// this is used to set axis of the surface that hold is located on it
	float theta_normal_direction;

	// for updating the hold directions
	float theta;
	float phi;

	int public_prototype_hold_id;

	int geomID;
};

enum MouseTrackingBody { MouseLeftLeg = 0, MouseRightLeg = 1, MouseLeftHand = 2, MouseRightHand = 3, MouseTorsoDir = 4, MouseNone = -1 };

enum JointType { FixedJoint = 0, OneDegreeJoint = 1, TwoDegreeJoint = 2, ThreeDegreeJoint = 3, BallJoint = 4 };
enum BodyType { BodyCapsule = 0, BodyBox = 1, BodySphere = 2 };

enum ContactPoints { LeftLeg = 5, RightLeg = 6, LeftArm = 7, RightArm = 8 };
enum BodyName {
	BodyTrunk = 0, BodyLeftThigh = 1, BodyRightThigh = 2, BodyLeftShoulder = 3, BodyRightShoulder = 4
	, BodyLeftLeg = 5, BodyRightLeg = 6, BodyLeftArm = 7, BodyRightArm = 8, BodyHead = 9, BodySpine = 10
	, BodyLeftFoot = 11, BodyRightFoot = 12, BodyLeftHand = 13, BodyRightHand = 14
	, BodyTrunkUpper = 15, BodyTrunkLower = 16, BodyHold = 17
};

class BipedState
{
	void init()
	{
		bodyStates = std::vector<BodyState>(BodyNUM, BodyState());
		forces = std::vector<Vector3>(4, Vector3(0, 0, 0));
		_body_control_cost = std::vector<float>(4, 10);
		counter_let_go = std::vector<int>(4, 0);
		hold_bodies_ids = std::vector<int>(4, -1);
		hold_bodies_info = std::vector<holdContent>(4, holdContent(Vector3(0.0, 0.0, -1.0f), -1));

		control_cost = FLT_MAX;
	}
public:
	BipedState()
	{
		init();
	}

	BipedState(const BipedState& b_i)
	{
		init();
		for (int i = 0; i < BodyNUM; i++)
		{
			bodyStates[i] = b_i.bodyStates[i];
		}
		control_cost = b_i.control_cost;
		saving_slot_state = b_i.saving_slot_state;

		for (int i = 0; i < 4; i++)
		{
			hold_bodies_ids[i] = b_i.hold_bodies_ids[i];
			hold_bodies_info[i] = b_i.hold_bodies_info[i];
			forces[i] = b_i.forces[i];
			counter_let_go[i] = b_i.counter_let_go[i];
			_body_control_cost[i] = b_i._body_control_cost[i];
		}
	}

	BipedState& operator=(const BipedState& b_i)
	{
		init();
		for (int i = 0; i < BodyNUM; i++)
		{
			bodyStates[i] = b_i.bodyStates[i];
		}
		control_cost = b_i.control_cost;
		saving_slot_state = b_i.saving_slot_state;

		for (int i = 0; i < 4; i++)
		{
			hold_bodies_ids[i] = b_i.hold_bodies_ids[i];
			hold_bodies_info[i] = b_i.hold_bodies_info[i];
			forces[i] = b_i.forces[i];
			counter_let_go[i] = b_i.counter_let_go[i];
			_body_control_cost[i] = b_i._body_control_cost[i];
		}

		return *this;
	}

	BipedState getNewCopy(int freeSlot, int fromContextSlot)
	{
		BipedState c;

		c.bodyStates = bodyStates;
		c.control_cost = control_cost;
		c.saving_slot_state = freeSlot;

		c.hold_bodies_ids = hold_bodies_ids;
		c.hold_bodies_info = hold_bodies_info;

		c.forces = forces;
		c.counter_let_go = counter_let_go;
		c._body_control_cost = _body_control_cost;

		saveOdeState(freeSlot, fromContextSlot);

		return c;
	}

	std::vector<BodyState> bodyStates;
	float control_cost;
	std::vector<float> _body_control_cost;
	int saving_slot_state;

	std::vector<int> hold_bodies_ids; // left-leg, right-get, left-hand, right-hand
	std::vector<holdContent> hold_bodies_info;

	std::vector<Vector3> forces;
	std::vector<int> counter_let_go;

	Vector3 getBodyDirectionZ(int i)
	{
		dMatrix3 R;
		Vector4 mAngle;
		if (i == BodyName::BodyTrunkLower)
			mAngle = bodyStates[BodyName::BodySpine].getAngle();
		else
			mAngle = bodyStates[i].getAngle();
		dReal mQ[] = { mAngle[0], mAngle[1], mAngle[2], mAngle[3] };
		dRfromQ(R, mQ);

		int targetDirection = 2; // alwayse in z direction
		int t_i = i;
		if (i == BodyName::BodyTrunkLower)
			t_i = BodyName::BodySpine;
		if (bodyStates[t_i].getBodyType() == BodyType::BodyBox)
		{
			if (i == BodyName::BodyLeftArm || i == BodyName::BodyLeftShoulder || i == BodyName::BodyRightArm || i == BodyName::BodyRightShoulder)
				targetDirection = 0; // unless one of these bones is wanted
		}
		Vector3 targetDirVector(R[4 * 0 + targetDirection], R[4 * 1 + targetDirection], R[4 * 2 + targetDirection]);
		if (i == BodyName::BodyLeftArm || i == BodyName::BodyLeftShoulder || i == BodyName::BodyLeftThigh || i == BodyName::BodyRightThigh
			|| i == BodyName::BodyLeftLeg || i == BodyName::BodyRightLeg || i == BodyName::BodyTrunkLower || i == BodyName::BodyLeftHand
			|| i == BodyName::BodyLeftFoot || i == BodyName::BodyRightFoot)
		{
			targetDirVector = -targetDirVector;
		}

		return targetDirVector;
	}

	Vector3 getEndPointPosBones(int i)
	{
		switch (i)
		{
		case BodyName::BodyLeftArm:
			i = BodyName::BodyLeftHand;
			break;
		case BodyName::BodyLeftLeg:
			i = BodyName::BodyLeftFoot;
			break;
		case BodyName::BodyRightArm:
			i = BodyName::BodyRightHand;
			break;
		case BodyName::BodyRightLeg:
			i = BodyName::BodyRightFoot;
			break;
		default:
			break;
		}

		if (i == BodyName::BodyTrunk)
			i = BodyName::BodySpine;
		if (i == BodyName::BodyTrunkUpper)
			i = BodyName::BodyTrunk;

		Vector3 targetDirVector = getBodyDirectionZ(i);

		if (i == BodyName::BodyTrunkLower)
			i = BodyName::BodySpine;

		Vector3 mPos = bodyStates[i].getPos();

		float bone_size = bodyStates[i].getBoneSize() / 2;
		Vector3 ePos_i(mPos[0] + targetDirVector.x() * bone_size, mPos[1] + targetDirVector.y() * bone_size, mPos[2] + targetDirVector.z() * bone_size);

		return ePos_i;
	}
};

	Vector3 initialPosition[contextNUM - 1];
	Vector3 resultPosition[contextNUM - 1];
	//current holds in the environment
	std::vector<holdContent> holds_body;
	//a set of hold samples given by the user. each holds_body element can be chosen from all_holds_types but with different position!
	std::vector<holdContent> holds_prototypes;

	mFileHandler mDesiredAngleFile;
	int _RouteNum;

	bool isTestClimber;

	mEnumTestCaseClimber _TestID;

	~SimulationContext()
	{
		if (_TestID == mEnumTestCaseClimber::TestAngle)
		{
			mDesiredAngleFile.reWriteFile(desiredAnglesBones);
			mDesiredAngleFile.mCloseFile();
		}
		////#x,y,z,f,dx,dy,dz,k,s"
		//mFileHandler _holdFile(getAppendAddress("ClimberInfo\\mHoldsRoute", _RouteNum, ".txt"));
		//_holdFile.openFileForWritingOn();
		//_holdFile.writeLine("#x,y,z,f,dx,dy,dz,k,s\n");
		//for (unsigned int i = 0; i < holds_body.size(); i++)
		//{
		//	_holdFile.writeLine(holds_body[i].toString());
		//}
		//_holdFile.mCloseFile();
	}

	char mReadBuff[100];
	char* getAppendAddress(char* firstText, int num, char* secondText, char* buff = NULL)
	{
		if (buff == NULL)
		{
			sprintf_s(&mReadBuff[0], 100, "%s%d%s", firstText, num, secondText);
			return &mReadBuff[0];
		}
		else
		{
			sprintf_s(buff, 100, "%s%d%s", firstText, num, secondText);
			return buff;
		}
		return NULL;
	}

	int readRouteNumFromFile()
	{
		mFileHandler mRouteNumFile("ClimberInfo\\mRouteNumFile.txt");
		std::vector<std::vector<float>> readFileRouteInfo;
		mRouteNumFile.readFile(readFileRouteInfo);
		if (readFileRouteInfo.size() > 0)
			_RouteNum = (int)readFileRouteInfo[0][0];
		else
			_RouteNum = 1;
		mRouteNumFile.mCloseFile();
		return _RouteNum;
	}

	int spaceID;

	SimulationContext(bool testClimber, mEnumTestCaseClimber TestID, const mDemoTestClimber& DemoID)
		 :mDesiredAngleFile("ClimberInfo\\mDesiredAngleFile.txt")
	{
		_TestID = TestID;
		isTestClimber = testClimber;

		bodyIDs = std::vector<int>(BodyNUM);
		mGeomID = std::vector<int>(BodyNUM); // for drawing stuff
		
		mColorBodies = std::vector<float>(4, 0.0f);

		bodyTypes = std::vector<int>(BodyNUM);
		boneSize = std::vector<float>(BodyNUM);
		fatherBodyIDs = std::vector<int>(BodyNUM);
		jointIDs = std::vector<int>(BodyNUM - 1, -1);
		jointTypes = std::vector<int>(BodyNUM - 1);

		jointHoldBallIDs = std::vector<std::vector<int>>(4, std::vector<int>(3,-1));
		holdPosIndex = std::vector<std::vector<int>>(contextNUM, std::vector<int>(4,-1));

		for (int i = 0; i <= 5; i++)
		{
			for (int j = 0; j <= 5; j++)
			{
				int cID = (int)holds_prototypes.size();
				holds_prototypes.push_back(holdContent(mTools::getDirectionFromAngles(i * (180.0f / 5.0f) + 180, j * (90.f / 5.0f) - 90.0f), cID));
			}
		}

		maxNumContexts = contextNUM; 

		currentFreeSavingStateSlot = 0;

		//Allocate one simulation context for each sample, plus one additional "master" context
		initOde(maxNumContexts);  // contactgroup = dJointGroupCreate (1000000); //might be needed, right now it is 0
		setCurrentOdeContext(ALLTHREADS);
		odeRandSetSeed(0);
		odeSetContactSoftCFM(contactCFM);
		
		odeWorldSetCFM(worldCfM);
		odeWorldSetERP(worldERP);

		odeWorldSetGravity(0, 0, -9.81f);

		spaceID = odeCreatePlane(0,0,0,1,0);

		float wWidth = 10.0f, wHeight = 15.0f;
		float climberHeight = 1.84f, climberRelX = wWidth / 2.0f, climberRelY = 0.5f;
		std::vector<std::vector<float>> readFileHoldsInfo;
		std::vector<std::vector<float>> readFileClimberKinectInfo;
		
		if (DemoID == mDemoTestClimber::DemoRouteFromFile)
		{
			_RouteNum = readRouteNumFromFile();
			mFileHandler mClimberInfo(getAppendAddress("ClimberInfo\\mClimberInfoFile", _RouteNum, ".txt"));
			//mFileHandler mClimberKinectInfo("ClimberInfo\\mClimberReadKinect.txt");
			mFileHandler mWallInfo("ClimberInfo\\mWallInfoFile.txt");
			mFileHandler mHoldsInfo(getAppendAddress("ClimberInfo\\mHoldsRoute", _RouteNum, ".txt"));

			std::vector<std::vector<float>> readFileWallInfo;
			mWallInfo.readFile(readFileWallInfo);
			if (readFileWallInfo.size() > 0)
			{
				wWidth = readFileWallInfo[0][0];
				wHeight = readFileWallInfo[0][1];
			}
			mWallInfo.mCloseFile();

			std::vector<std::vector<float>> readFileClimberInfo;
			mClimberInfo.readFile(readFileClimberInfo);
			mClimberInfo.mCloseFile();

			climberRelX = readFileClimberInfo[0][0];
			climberRelY = readFileClimberInfo[0][1];
			climberHeight = readFileClimberInfo[0][2];

			//mClimberKinectInfo.readFile(readFileClimberKinectInfo);
			//mClimberKinectInfo.mCloseFile();

			mHoldsInfo.readFile(readFileHoldsInfo);
			mHoldsInfo.mCloseFile();

		}
		else if (DemoID >= mDemoTestClimber::DemoRoute1 && DemoID <= mDemoTestClimber::DemoRoute3)
		{
			_RouteNum = int(DemoID - mDemoTestClimber::DemoRoute1) + 1;
		}
		else
		{
			_RouteNum = 0; // learning
		}

//		_feetHandsWidth = std::vector<float>(4, 0.5f);

		createHumanoidBody(-wWidth / 2 + climberRelX /*relative dis x with wall (m)*/,
						   -climberRelY /*relative dis y with wall (m)*/, 
						   climberHeight /*height of climber (m)*/, 
						   70.0f /*climber's mass (kg)*/,
						   readFileClimberKinectInfo);

		// calculate joint size
		mJointSize = 0;
		desiredAnglesBones.clear();
		for (int i = 0; i < BodyNUM - 1; i++)
		{
			mJointSize += jointTypes[i];
			for (int j = 0; j < jointTypes[i]; j++)
			{
				desiredAnglesBones.push_back(0.0f);
			}
		}

		if (!testClimber)
		{
			createEnvironment(DemoID, wWidth, wHeight, readFileHoldsInfo);

			attachContactPointToHold(ContactPoints::LeftArm, 2, ALLTHREADS);
			attachContactPointToHold(ContactPoints::RightArm, 3, ALLTHREADS);
			attachContactPointToHold(ContactPoints::LeftLeg, 0, ALLTHREADS);
			attachContactPointToHold(ContactPoints::RightLeg, 1, ALLTHREADS);
		}
		else
		{
			Vector3 hPos;
			switch (TestID)
			{
			case mEnumTestCaseClimber::TestAngle:
				hPos = getEndPointPosBones(SimulationContext::BodyName::BodyTrunk);
				createJointType(hPos.x(), hPos.y(), hPos.z(), -1, SimulationContext::BodyName::BodyTrunk, JointType::FixedJoint);
				break;
			case mEnumTestCaseClimber::TestCntroller:
			case  mEnumTestCaseClimber::RunLearnerRandomly:
			default:
				createEnvironment(DemoID, wWidth, wHeight, readFileHoldsInfo);

	//			hPos = getEndPointPosBones(SimulationContext::BodyName::BodyTrunk);
	//			createJointType(hPos.x(), hPos.y(), hPos.z(), -1, SimulationContext::BodyName::BodyTrunk, JointType::FixedJoint);

				attachContactPointToHold(ContactPoints::LeftArm, 2, ALLTHREADS);
				attachContactPointToHold(ContactPoints::RightArm, 3, ALLTHREADS);
				attachContactPointToHold(ContactPoints::LeftLeg, 0, ALLTHREADS);
				attachContactPointToHold(ContactPoints::RightLeg, 1, ALLTHREADS);
				break;
			}
		}

		// read desired angles from file
		std::vector<std::vector<float>> readFiledAngles;
		mDesiredAngleFile.readFile(readFiledAngles);
		if (readFiledAngles.size() > 0)
		{
			for (unsigned int i = 0; i < readFiledAngles.size(); i++)
			{
				desiredAnglesBones[i] = readFiledAngles[i][0];
			}
		}
		if (TestID != mEnumTestCaseClimber::TestAngle)
		{
			mDesiredAngleFile.mCloseFile();
		}

		for (int i = 0; i < maxNumContexts; i++)
		{
			int cContextSavingSlotNum = getNextFreeSavingSlot();
			saveOdeState(cContextSavingSlotNum);
		}

		//We're done, now we should have nSamples+1 copies of a model
		masterContext = contextNUM - 1;
		setCurrentOdeContext(masterContext);
	}

	void detachContactPoint(ContactPoints iEndPos, int targetContext)
	{
		int cContextNum = getCurrentOdeContext();

		setCurrentOdeContext(targetContext);

		int jointIndex = iEndPos - ContactPoints::LeftLeg;

		for (unsigned int m = 0; m < jointHoldBallIDs[jointIndex].size(); m++)
		{
			if (jointHoldBallIDs[jointIndex][m] != -1)
				odeJointAttach(jointHoldBallIDs[jointIndex][m], 0, 0);
		}

		//if (mENVGeoms.size() > 0)
		//{
		//	unsigned long cCollideBits = odeGeomGetCollideBits(mENVGeoms[0]); // collision bits of wall
		//	switch (jointIndex)
		//	{
		//	case 0: // left leg should collide with 0x0040
		//		odeGeomSetCollideBits(mENVGeoms[0], cCollideBits | unsigned long(0x0040));
		//		break;
		//	case 1: // right leg should collide with 0x0008
		//		odeGeomSetCollideBits(mENVGeoms[0], cCollideBits | unsigned long(0x0008));
		//		break;
		//	case 2: // left hand should collide with 0x2000
		//		odeGeomSetCollideBits(mENVGeoms[0], cCollideBits | unsigned long(0x2000));
		//		break;
		//	case 3: // right hand should collide with 0x0400
		//		odeGeomSetCollideBits(mENVGeoms[0], cCollideBits | unsigned long(0x0400));
		//		break;
		//	default:
		//		break;
		//	}
		//}

		if (targetContext == ALLTHREADS)
		{
			for (int i = 0; i < maxNumContexts; i++)
			{
				holdPosIndex[i][jointIndex] = -1;
			}
		}
		else
			holdPosIndex[targetContext][jointIndex] = -1;

		setCurrentOdeContext(cContextNum);
	}

	void attachContactPointToHold(ContactPoints iEndPos, int iHoldID, int targetContext)
	{
		int cContextNum = getCurrentOdeContext();

		setCurrentOdeContext(targetContext);

		int jointIndex = iEndPos - ContactPoints::LeftLeg;
		int boneIndex = BodyName::BodyLeftLeg + jointIndex;
		Vector3 hPos = getEndPointPosBones(boneIndex);

		switch (boneIndex)
		{
			case BodyName::BodyLeftArm:
				boneIndex = BodyName::BodyLeftHand;
				break;
			case BodyName::BodyLeftLeg:
				boneIndex = BodyName::BodyLeftFoot;
				break;
			case BodyName::BodyRightArm:
				boneIndex = BodyName::BodyRightHand;
				break;
			case BodyName::BodyRightLeg:
				boneIndex = BodyName::BodyRightFoot;
				break;
		default:
			break;
		}

		bool flag_set_holdIndex = false;

		Vector3 zDir = getBodyDirectionZ(boneIndex);
		Vector3 yDir = getBodyDirectionY(boneIndex);
		Vector3 p0 = hPos;
		/*Vector3 p1 = hPos - (boneSize[boneIndex] / 2) * zDir + (this->_feetHandsWidth[jointIndex] / 2) * yDir;
		Vector3 p2 = hPos - (boneSize[boneIndex] / 2) * zDir - (this->_feetHandsWidth[jointIndex] / 2) * yDir;*/

		if (jointHoldBallIDs[jointIndex][0] == -1) // create the hold joint only once
		{
			int cJointBallID = createJointType(p0.x(), p0.y(), p0.z(), -1, boneIndex);
			jointHoldBallIDs[jointIndex][0] = cJointBallID;

			//cJointBallID = createJointType(p1.x(), p1.y(), p1.z(), -1, boneIndex);
			//jointHoldBallIDs[jointIndex][1] = cJointBallID;

			//cJointBallID = createJointType(p2.x(), p2.y(), p2.z(), -1, boneIndex);
			//jointHoldBallIDs[jointIndex][2] = cJointBallID;

			flag_set_holdIndex = true;
		}

		Vector3 holdPos = getHoldPos(iHoldID);
		float _dis = (holdPos - hPos).norm();

		float _connectionThreshold = 0.5f * getHoldSize(iHoldID);

		if (_dis <= _connectionThreshold + 0.1f)
		{
			if (jointHoldBallIDs[jointIndex][0] != -1)
			{
				odeJointAttach(jointHoldBallIDs[jointIndex][0], 0, bodyIDs[boneIndex]);
				odeJointSetBallAnchor(jointHoldBallIDs[jointIndex][0], p0.x(), p0.y(), p0.z());

				flag_set_holdIndex = true;
			}
			/*if (jointHoldBallIDs[jointIndex][1] != -1)
			{
				odeJointAttach(jointHoldBallIDs[jointIndex][1], 0, bodyIDs[boneIndex]);
				odeJointSetBallAnchor(jointHoldBallIDs[jointIndex][1], p1.x(), p1.y(), p1.z());

				flag_set_holdIndex = true;
			}
			if (jointHoldBallIDs[jointIndex][2] != -1)
			{
				odeJointAttach(jointHoldBallIDs[jointIndex][2], 0, bodyIDs[boneIndex]);
				odeJointSetBallAnchor(jointHoldBallIDs[jointIndex][2], p2.x(), p2.y(), p2.z());

				flag_set_holdIndex = true;
			}*/
		}
		
		if (flag_set_holdIndex)
		{
			//if (mENVGeoms.size() > 0)
			//{
			//	unsigned long cCollideBits = odeGeomGetCollideBits(mENVGeoms[0]); // collision bits of wall
			//	switch (jointIndex)
			//	{
			//	case 0: // left leg should not collide with 0x0040
			//		odeGeomSetCollideBits(mENVGeoms[0], cCollideBits & unsigned long(0xFFBF));
			//		break;
			//	case 1: // right leg should not collide with 0x0008
			//		odeGeomSetCollideBits(mENVGeoms[0], cCollideBits & unsigned long(0xFFF7));
			//		break;
			//	case 2: // left hand should not collide with 0x2000
			//		odeGeomSetCollideBits(mENVGeoms[0], cCollideBits & unsigned long(0xDFFF));
			//		break;
			//	case 3: // right hand should not collide with 0x0400
			//		odeGeomSetCollideBits(mENVGeoms[0], cCollideBits & unsigned long(0xFBFF));
			//		break;
			//	default:
			//		break;
			//	}
			//}

			if (targetContext == ALLTHREADS)
			{
				for (int i = 0; i < maxNumContexts; i++)
				{
					holdPosIndex[i][jointIndex] = iHoldID;
				}
			}
			else
				holdPosIndex[targetContext][jointIndex] = iHoldID;
		}

		setCurrentOdeContext(cContextNum);

		return;
	}

	int getNextFreeSavingSlot() 
	{
		return currentFreeSavingStateSlot++;
	}

	int getMasterContextID()
	{
		return masterContext;
	}

	////////////////////////////////////////// get humanoid and hold bodies info /////////////////////////
	Vector3 getHoldPos(int i)
	{
		if (i >= 0 && i < (int)holds_body.size())
			return holds_body[i].holdPos;
		return Vector3(0.0f, 0.0f, 0.0f);
	}

	float getHoldThetaNormal(int i)
	{
		if (i >= 0 && i < (int)holds_body.size())
			return (holds_body[i].theta_normal_direction * (PI / 180.0f));
		return -PI / 2.0f;// default wall dir
	}

	//////////////////////////////////////////////////////////////////////////////// TODO: Gorund location

	Vector3 getMidPointHoldStance(std::vector<int>& _hold_ids)
	{
		int mSize = 0;
		Vector3 midPoint(0.0f, 0.0f, 0.0f);
		for (unsigned int i = 0; i < _hold_ids.size(); i++)
		{
			if (_hold_ids[i] != -1)
			{
				midPoint += getHoldPos(_hold_ids[i]);
				mSize++;
			}
		}

		if (mSize > 0)
		{
			midPoint = midPoint / mSize;
		}
		else//we are on the ground
		{
			midPoint = Vector3(0.0f, -0.5f, 1.0f);
		}
		return midPoint;
	}

	float getThetaDirHoldStance(std::vector<int>& _hold_ids)
	{
		int mSize = 0;
		float theta_dir = 0.0f;
		for (unsigned int i = 0; i < _hold_ids.size(); i++)
		{
			if (_hold_ids[i] != -1)
			{
				theta_dir += holds_body[_hold_ids[i]].theta_normal_direction;
				mSize++;
			}
		}

		if (mSize > 0)
		{
			theta_dir = theta_dir / mSize;
		}
		else
		{
			theta_dir = -90.0f;
		}
		return theta_dir;
	}

	Vector3 getHoldStancePosFrom(const std::vector<int>& _hold_ids, std::vector<Vector3>& _hold_points, float& mSize)
	{
		Vector3 midPoint(0.0f, 0.0f, 0.0f);
		mSize = 0.0f;
		_hold_points.clear();
		for (unsigned int i = 0; i < _hold_ids.size(); i++)
		{
			if (_hold_ids[i] != -1)
			{
				_hold_points.push_back(getHoldPos(_hold_ids[i]));
				midPoint += _hold_points[i];
				mSize++;
			}
			else
				_hold_points.push_back(Vector3(0, 0, 0));
		}

		if (mSize > 0)
		{
			midPoint = midPoint / mSize;
		}
		else//we are on the ground
		{
			midPoint = Vector3(0.0f, -0.5f, 1.0f);
		}
		return midPoint;
	}

	float getHoldSize(int i)
	{
		if (i >= 0 && i < (int)holds_body.size())
			return holds_body[i].size;
		return 0.0f;
	}

	int getIndexHoldFromGeom(int cGeomID)
	{
		for (unsigned int i = 0; i < mENVGeoms.size(); i++)
		{
			if (mENVGeoms[i] == cGeomID)
			{
				return i - startingHoldGeomsIndex;
			}
		}
		return -1;
	}

	int getIndexHold(int cGeomID)
	{
		for (unsigned int i = 0; i < holds_body.size(); i++)
		{
			if (holds_body[i].geomID == cGeomID)
			{
				return i;
			}
		}
		return -1;
	}

	MouseTrackingBody getIndexHandsAndLegsFromGeom(int cGeomID)
	{
		if (cGeomID == -1)
			return MouseNone;
		for (int i = 0; i < 5; i++)
		{
			switch (i)
			{
			case 0:
				if (mGeomID[BodyLeftFoot] == cGeomID)
					return MouseLeftLeg;
				break;
			case 1:
				if (mGeomID[BodyRightFoot] == cGeomID)
					return MouseRightLeg;
				break;
			case 2:
				if (mGeomID[BodyLeftHand] == cGeomID)
					return MouseLeftHand;
				break;
			case 3:
				if (mGeomID[BodyRightHand] == cGeomID)
					return MouseRightHand;
				break;
			case 4:
				if (mGeomID[BodyTrunk] == cGeomID)
					return MouseTorsoDir;
				break;
			default:
				break;
			}
		}
		return MouseNone;
	}

	bool checkViolatingRelativeDis()
	{
		for (unsigned int i = 0; i < fatherBodyIDs.size(); i++)
		{
			if (fatherBodyIDs[i] != -1)
			{
				Vector3 bone_i = getEndPointPosBones(i, true);
				Vector3 f_bone_i = getEndPointPosBones(fatherBodyIDs[i], true);

				float coeff = 1.5f;

				if (i == BodyName::BodyLeftShoulder || i == BodyName::BodyRightShoulder)
					coeff = 2.0f;

				if ((bone_i - f_bone_i).norm() > (coeff * boneSize[i]))
				{
					return true;
				}
			}
		}

		return false;
	}

	Vector3 getBodyDirectionZ(int i)
	{
		dMatrix3 R;
		ConstOdeQuaternion mQ;
		if (i == BodyName::BodyTrunkLower)
			mQ = odeBodyGetQuaternion(bodyIDs[BodyName::BodySpine]);
		else
			mQ = odeBodyGetQuaternion(bodyIDs[i]);
		dRfromQ(R, mQ);

		int targetDirection = 2; // alwayse in z direction
		int t_i = i;
		if (i == BodyName::BodyTrunkLower)
			t_i = BodyName::BodySpine;
		if (bodyTypes[t_i] == BodyType::BodyBox)
		{
			if (i == BodyName::BodyLeftArm || i == BodyName::BodyLeftShoulder || i == BodyName::BodyRightArm || i == BodyName::BodyRightShoulder)
				targetDirection = 0; // unless one of these bones is wanted
		}
		Vector3 targetDirVector(R[4 * 0 + targetDirection], R[4 * 1 + targetDirection], R[4 * 2 + targetDirection]);
		if (i == BodyName::BodyLeftArm || i == BodyName::BodyLeftShoulder || i == BodyName::BodyLeftThigh || i == BodyName::BodyRightThigh 
			|| i == BodyName::BodyLeftLeg || i == BodyName::BodyRightLeg || i == BodyName::BodyTrunkLower || i == BodyName::BodyLeftHand
			|| i == BodyName::BodyLeftFoot || i == BodyName::BodyRightFoot)
		{
			targetDirVector = -targetDirVector;
		}

		return targetDirVector;
	}

	int getHoldBodyIDs(int i, int targetContext)
	{
		if (targetContext == ALLTHREADS)
			return holdPosIndex[maxNumContexts-1][i];
		else
			return holdPosIndex[targetContext][i];
	}

	int getHoldBodyIDsSize()
	{
		return holdPosIndex[0].size(); // the value is 4
	}

	int getJointSize()
	{
		return mJointSize;
	}

	Vector3 getGeomPosition(int i)
	{
		ConstOdeVector rPos;

		if (i < (int)mENVGeoms.size())
			rPos = odeGeomGetPosition(mENVGeoms[i]);
		else
		{
			return Vector3(0.0f, 0.0f, 0.0f);
		}

		return Vector3(rPos[0], rPos[1], rPos[2]);
	}

	Vector3 getBonePosition(int i)
	{
		ConstOdeVector rPos;
		
		if (i < BodyNUM)
			rPos = odeBodyGetPosition(bodyIDs[i]);
		else
		{
			//rPos = odeBodyGetPosition(bodyHoldIDs[i - BodyNUM]);
			return Vector3(0.0f,0.0f,0.0f);
		}
		
		return Vector3(rPos[0], rPos[1], rPos[2]);
	}
	
	Vector3 getBoneLinearVelocity(int i)
	{
		ConstOdeVector rVel = odeBodyGetLinearVel(bodyIDs[i]);
		
		return Vector3(rVel[0], rVel[1], rVel[2]);
	}

	Vector4 getBoneAngle(int i)
	{
		ConstOdeQuaternion rAngle = odeBodyGetQuaternion(bodyIDs[i]);

		return Vector4(rAngle[0], rAngle[1], rAngle[2], rAngle[3]);
	}

	Vector3 getBoneAngularVelocity(int i)
	{
		ConstOdeVector rAVel = odeBodyGetAngularVel(bodyIDs[i]);
		
		return Vector3(rAVel[0], rAVel[1], rAVel[2]);
	}

	float getJointAngle(int i)
	{
		if (jointTypes[jointIDIndex[i]] == JointType::ThreeDegreeJoint)
		{
			float angle = odeJointGetAMotorAngle(jointIDs[jointIDIndex[i]], jointAxisIndex[i]);
			return angle;
		}
		return odeJointGetHingeAngle(jointIDs[jointIDIndex[i]]);
	}
	
	float getJointAngleRate(int i)
	{
		if (jointTypes[jointIDIndex[i]] == JointType::ThreeDegreeJoint)
		{
			float angle = odeJointGetAMotorAngleRate(jointIDs[jointIDIndex[i]], jointAxisIndex[i]);
			return angle;
		}
		return odeJointGetHingeAngleRate(jointIDs[jointIDIndex[i]]);
	}
	
	float getJointFMax(int i)
	{
		int bodynames[5] = { BodySpine , BodyLeftArm , BodyLeftLeg ,BodyRightArm ,BodyRightLeg };

		int joint_id_index = bodynames[i] - 1;

		if (jointTypes[joint_id_index] == JointType::ThreeDegreeJoint)
		{
			int axis_index = 0; // jointAxisIndex[i]
			return odeJointGetAMotorParam(jointIDs[joint_id_index], dParamFMax1 + dParamGroup*axis_index);
		}
		return odeJointGetHingeParam(jointIDs[joint_id_index], dParamFMax1 );
	}

	float getJointAngleMin(int i)
	{
		//Vector2 mLimits = jointLimits[i];
		//const float angleLimitBuffer = deg2rad*2.5f;
		//return  mLimits[0];// +angleLimitBuffer;

		if (jointTypes[jointIDIndex[i]] == JointType::ThreeDegreeJoint)
		{
			return odeJointGetAMotorParam(jointIDs[jointIDIndex[i]], dParamLoStop1 + dParamGroup*jointAxisIndex[i]);
		}
		return odeJointGetHingeParam(jointIDs[jointIDIndex[i]], dParamLoStop1 );
	}
	
	float getJointAngleMax(int i)
	{
		//Vector2 mLimits = jointLimits[i];
		//const float angleLimitBuffer = deg2rad*2.5f;
		//return  mLimits[1];// -angleLimitBuffer;

		if (jointTypes[jointIDIndex[i]] == JointType::ThreeDegreeJoint)
		{
			return odeJointGetAMotorParam(jointIDs[jointIDIndex[i]], dParamHiStop1 + dParamGroup*jointAxisIndex[i]);
		}
		return odeJointGetHingeParam(jointIDs[jointIDIndex[i]], dParamHiStop1 );
	}

	Vector3 getEndPointPosBones(int i, bool flag_calculate_exact_val = false)
	{
		ConstOdeVector mPos;
		/*switch (i)
		{
			case BodyName::BodyLeftArm:
				mPos = odeBodyGetPosition(bodyIDs[BodyName::BodyLeftHand]);
				return Vector3(mPos[0], mPos[1], mPos[2]);
				break;
			case BodyName::BodyLeftLeg:
				mPos = odeBodyGetPosition(bodyIDs[BodyName::BodyLeftFoot]);
				return Vector3(mPos[0], mPos[1], mPos[2]);
				break;
			case BodyName::BodyRightArm:
				mPos = odeBodyGetPosition(bodyIDs[BodyName::BodyRightHand]);
				return Vector3(mPos[0], mPos[1], mPos[2]);
				break;
			case BodyName::BodyRightLeg:
				mPos = odeBodyGetPosition(bodyIDs[BodyName::BodyRightFoot]);
				return Vector3(mPos[0], mPos[1], mPos[2]);
				break;
		default:
			break;
		}*/
		if (!flag_calculate_exact_val)
		{
			switch (i)
			{
				case BodyName::BodyLeftArm:
					i = BodyName::BodyLeftHand;
					break;
				case BodyName::BodyLeftLeg:
					i = BodyName::BodyLeftFoot;
					break;
				case BodyName::BodyRightArm:
					i = BodyName::BodyRightHand;
					break;
				case BodyName::BodyRightLeg:
					i = BodyName::BodyRightFoot;
					break;
			default:
				break;
			}
		}
		if (i == BodyName::BodyTrunk)
			i = BodyName::BodySpine;
		if (i == BodyName::BodyTrunkUpper)
			i = BodyName::BodyTrunk;
		
		Vector3 targetDirVector = getBodyDirectionZ(i);

		if (i == BodyName::BodyTrunkLower)
			i = BodyName::BodySpine;

		mPos = odeBodyGetPosition(bodyIDs[i]);

		float bone_size = boneSize[i] / 2;
		Vector3 ePos_i(mPos[0] + targetDirVector.x() * bone_size, mPos[1] + targetDirVector.y() * bone_size, mPos[2] + targetDirVector.z() * bone_size);

		return ePos_i;
	}

	Vector3 getBodyDirectionY(int i)
	{
		dMatrix3 R;
		ConstOdeQuaternion mQ;
		if (i == BodyName::BodyTrunkLower)
			mQ = odeBodyGetQuaternion(bodyIDs[BodyName::BodySpine]);
		else
			mQ = odeBodyGetQuaternion(bodyIDs[i]);
		dRfromQ(R, mQ);

		int targetDirection = 1; // alwayse in y direction
		int t_i = i;
		if (i == BodyName::BodyTrunkLower)
			t_i = BodyName::BodySpine;
		if (bodyTypes[t_i] == BodyType::BodyBox)
		{
			if (i == BodyName::BodyLeftArm || i == BodyName::BodyLeftShoulder || i == BodyName::BodyRightArm || i == BodyName::BodyRightShoulder)
				targetDirection = 0; // unless one of these bones is wanted
		}
		Vector3 targetDirVector(R[4 * 0 + targetDirection], R[4 * 1 + targetDirection], R[4 * 2 + targetDirection]);
		if (i == BodyName::BodyTrunkLower)
		{
			targetDirVector = -targetDirVector;
		}

		return targetDirVector;
	}

	float getClimberRadius()
	{
		return climberRadius;
	}

	float getClimberLegLegDis()
	{
		return climberLegLegDis;
	}

	float getClimberHandHandDis()
	{
		return climberHandHandDis;
	}

	int getJointBody(BodyName iBodyName)
	{
		return jointIDs[iBodyName - 1];
	}

	Vector3 computeCOM()
	{
		float totalMass=0;
		Vector3 result=Vector3::Zero();
		for (int i = 0; i < BodyNUM; i++)
		{
			float mass = odeBodyGetMass(bodyIDs[i]);
			Vector3 pos(odeBodyGetPosition(bodyIDs[i]));
			result+=mass*pos;
			totalMass+=mass;
		}
		return result/totalMass;
	}

	/////////////////////////////////////////// setting motor speed to control humanoid body /////////////

	BodyName getBodyFromJointIndex(int joint_index)
	{
		if (joint_index < 0 && joint_index >= (int)(jointIDIndex.size()))
			return BodyName::BodyHead;
		return (BodyName)(jointIDIndex[joint_index] + 1);
	}

	void setMotorSpeed(int i, float iSpeed, int nPhysicsPerStep)
	{
		if (jointIDs[jointIDIndex[i]] == -1)
			return;
		if ((jointIDIndex[i] + 1) == BodyName::BodyHead || (jointIDIndex[i] + 1) == BodyName::BodyLeftHand || (jointIDIndex[i] + 1) == BodyName::BodyRightHand )
		{
			float angle=odeJointGetAMotorAngle(jointIDs[jointIDIndex[i]],jointAxisIndex[i]);
			iSpeed=-angle; //p control, keep head and wriat zero rotation
			//Vector3 dir_trunk = getBodyDirectionZ(BodyName::BodyTrunk);
			//Vector3 dir_head = getBodyDirectionZ(BodyName::BodyTrunk);
			//Vector3 diff_head_trunk = dir_trunk - dir_head;
			//diff_head_trunk[2] += FLT_EPSILON;
			//switch (jointAxisIndex[i])
			//{
			//case 0:// rotation about local x
			//	diff_head_trunk[0] = 0;
			//	diff_head_trunk.normalize();
			//	iSpeed = atan2(diff_head_trunk[1], diff_head_trunk[2]);
			//	break;
			//case 1:
			//	diff_head_trunk[1] = 0;
			//	diff_head_trunk.normalize();
			//	iSpeed = atan2(diff_head_trunk[0], diff_head_trunk[2]);
			//	break;
			//case 2:
			//	iSpeed = 0;
			//	break;
			//}
		}
		Vector2 mLimits = jointLimits[i];
		const float angleLimitBuffer=deg2rad*2.5f;
		if (jointTypes[jointIDIndex[i]] == JointType::ThreeDegreeJoint)
		{
			float cAngle = odeJointGetAMotorAngle(jointIDs[jointIDIndex[i]], jointAxisIndex[i]);
			float nAngle = cAngle + (float)(nPhysicsPerStep * iSpeed * timeStep);
			if (((nAngle < mLimits[0]+angleLimitBuffer) && iSpeed<0)
				|| ((nAngle > mLimits[1]+angleLimitBuffer) && iSpeed>0))
			{
				iSpeed = 0;
			}
			switch (jointAxisIndex[i])
			{
			case 0:
				odeJointSetAMotorParam(jointIDs[jointIDIndex[i]], dParamVel1, iSpeed);
				break;
			case 1:
				odeJointSetAMotorParam(jointIDs[jointIDIndex[i]], dParamVel2, iSpeed);
				break;
			case 2:
				odeJointSetAMotorParam(jointIDs[jointIDIndex[i]], dParamVel3, iSpeed);
				break;
			}
			return;
		}
		float cAngle = odeJointGetHingeAngle(jointIDs[jointIDIndex[i]]);
		float nAngle = cAngle + (float)(nPhysicsPerStep * iSpeed * timeStep);
		if (((nAngle < mLimits[0]+angleLimitBuffer) && iSpeed<0)
			|| ((nAngle > mLimits[1]+angleLimitBuffer) && iSpeed>0))
		{
			iSpeed = 0;
		}
		odeJointSetHingeParam(jointIDs[jointIDIndex[i]], dParamVel1, iSpeed);
		return;
	}
	
	void driveMotorToPose(int i, float targetAngle, float motorPoseInterpolationTime)
	{
		if (jointIDs[jointIDIndex[i]] == -1)
			return;

		float kp = 1.0f / motorPoseInterpolationTime;
		float max_speed = maxSpeed;//3.0f * PI;
		Vector2 mLimits = jointLimits[i];
		const float angleLimitBuffer = 0;// deg2rad*2.5f;
		float cAngle;
		if (jointTypes[jointIDIndex[i]] == JointType::ThreeDegreeJoint)
		{
			cAngle = odeJointGetAMotorAngle(jointIDs[jointIDIndex[i]], jointAxisIndex[i]);
		}
		else
		{
			cAngle = odeJointGetHingeAngle(jointIDs[jointIDIndex[i]]);
		}

		if (targetAngle < mLimits[0] + angleLimitBuffer)
			Debug::throwError("Invalid target angle!");
			//targetAngle = mLimits[0] + angleLimitBuffer;
		if (targetAngle > mLimits[1] - angleLimitBuffer)
			Debug::throwError("Invalid target angle!");

			//targetAngle = mLimits[1] - angleLimitBuffer;

		float iSpeed = (targetAngle - cAngle) * kp;
		//if ((jointIDIndex[i] + 1) == BodyName::BodyHead || (jointIDIndex[i] + 1) == BodyName::BodyLeftHand || (jointIDIndex[i] + 1) == BodyName::BodyRightHand)
		//{
		//	float angle=odeJointGetAMotorAngle(jointIDs[jointIDIndex[i]],jointAxisIndex[i]);
		//	iSpeed = -angle;
		//}
		
		if (fabs(iSpeed) > max_speed * (mLimits[1] - mLimits[0]))
		{
			iSpeed = sign(iSpeed) * max_speed * (mLimits[1] - mLimits[0]);
		}

		if (jointTypes[jointIDIndex[i]] == JointType::ThreeDegreeJoint)
		{
			//float nAngle = cAngle + (float)(iSpeed * timeStep);
			//if (nAngle < mLimits[0])
			//{
			//	targetAngle = mLimits[0];
			//	iSpeed = (targetAngle - cAngle) * kp;
			//}
			//if (nAngle > mLimits[1])
			//{
			//	targetAngle = mLimits[1];
			//	iSpeed = (targetAngle - cAngle) * kp;
			//}

			switch (jointAxisIndex[i])
			{
			case 0:
				odeJointSetAMotorParam(jointIDs[jointIDIndex[i]], dParamVel1, iSpeed);
				break;
			case 1:
				odeJointSetAMotorParam(jointIDs[jointIDIndex[i]], dParamVel2, iSpeed);
				break;
			case 2:
				odeJointSetAMotorParam(jointIDs[jointIDIndex[i]], dParamVel3, iSpeed);
				break;
			}
			return;
		}
		//float nAngle = cAngle + (float)(iSpeed * timeStep);
		//if (nAngle < mLimits[0])
		//{
		//	targetAngle = mLimits[0];
		//	iSpeed = (targetAngle - cAngle) * kp;
		//}
		//if (nAngle > mLimits[1])
		//{
		//	targetAngle = mLimits[1];
		//	iSpeed = (targetAngle - cAngle) * kp;
		//}

		odeJointSetHingeParam(jointIDs[jointIDIndex[i]], dParamVel1, iSpeed);
	}
	
	void setFmax(int joint, float fmax)
	{
		if (odeJointGetType(joint)==dJointTypeAMotor)
		{
			odeJointSetAMotorParam(joint,dParamFMax1,fmax);
			odeJointSetAMotorParam(joint,dParamFMax2,fmax);
			odeJointSetAMotorParam(joint,dParamFMax3,fmax);
		}
		else
		{
			odeJointSetHingeParam(joint,dParamFMax1,fmax);
		}
	}
	
	void setMotorGroupFmaxes(const float *fmax, float torsoMinFMax)
	{
		setFmax(getJointBody(BodySpine),max(torsoMinFMax,fmax[fmTorso]));
		//setFmax(getJointBody(BodyHead),fmax[fmTorso]);

		setFmax(getJointBody(BodyLeftShoulder),fmax[fmLeftArm]);
		setFmax(getJointBody(BodyLeftArm),fmax[fmLeftArm]);
		setFmax(getJointBody(BodyLeftHand),fmax[fmLeftArm]);

		setFmax(getJointBody(BodyLeftThigh),fmax[fmLeftLeg]);
		setFmax(getJointBody(BodyLeftLeg),fmax[fmLeftLeg]);
		setFmax(getJointBody(BodyLeftFoot),fmax[fmLeftLeg]);

		setFmax(getJointBody(BodyRightShoulder),fmax[fmRightArm]);
		setFmax(getJointBody(BodyRightArm),fmax[fmRightArm]);
		setFmax(getJointBody(BodyRightHand),fmax[fmRightArm]);

		setFmax(getJointBody(BodyRightThigh),fmax[fmRightLeg]);
		setFmax(getJointBody(BodyRightLeg),fmax[fmRightLeg]);
		setFmax(getJointBody(BodyRightFoot),fmax[fmRightLeg]);
	}

	inline static float odeVectorSquaredNorm(ConstOdeVector v)
	{
		return squared(v[0])+squared(v[1])+squared(v[2]);
	}
	
	float getMotorAppliedSqTorque(int i)
	{
		if (jointIDs[jointIDIndex[i]] == -1)
			return 0;
		return odeVectorSquaredNorm(odeJointGetAccumulatedTorque(jointIDs[jointIDIndex[i]],0))
			+ odeVectorSquaredNorm(odeJointGetAccumulatedTorque(jointIDs[jointIDIndex[i]],1));
	}
	
	Vector3 getForceVectorOnEndBody(ContactPoints iEndPos, int targetContext)
	{
		int jointIndex = iEndPos - ContactPoints::LeftLeg;
		Vector3 res(0.0f,0.0f,0.0f);
		int _s = 0;
		if (getHoldBodyIDs(jointIndex, targetContext) >= 0)
		{
			/*int bodyId;
			switch (iEndPos)
			{
			case ContactPoints::LeftLeg:
				bodyId = bodyIDs[BodyName::BodyLeftFoot];
				break;
			case ContactPoints::RightLeg:
				bodyId = bodyIDs[BodyName::BodyRightFoot];
				break;
			case ContactPoints::LeftArm:
				bodyId = bodyIDs[BodyName::BodyLeftHand];
				break;
			case ContactPoints::RightArm:
				bodyId = bodyIDs[BodyName::BodyRightHand];
				break;
			default:
				break;
			}
			int jointType = dJointTypeBall;*/
			for (unsigned int i = 0; i < jointHoldBallIDs[jointIndex].size(); i++)
			{
				if (jointHoldBallIDs[jointIndex][i] != -1)
				{
					ConstOdeVector res_i = odeJointGetAccumulatedForce(jointHoldBallIDs[jointIndex][i], 0);
					Vector3 mOut_i(res_i[0],res_i[1],res_i[2]);
					res += -mOut_i;
					_s++;
				}
			}

			if (_s > 0)
			{
				res /= (float)_s;
			}

			/*if (mOut.norm() == 0)
			{
				res = odeJointGetAccumulatedForce(jointHoldBallIDs[jointIndex], 1);
				mOut[0] = res[0];
				mOut[1] = res[1];
				mOut[2] = res[2];
			}*/
			//if (iEndPos == ContactPoints::LeftLeg || iEndPos == ContactPoints::RightLeg)
			
			//else
			//	return -mOut;
		}
		return res;
	}
	
	float getSqForceOnFingers(int targetContext)
	{
		/*float result=0;
		dVector3 f;
		odeBodyGetAccumulatedForce(bodyIDs[BodyName::BodyLeftHand],-1,f);
		result+=odeVectorSquaredNorm(f);
		odeBodyGetAccumulatedForce(bodyIDs[BodyName::BodyRightHand],-1,f);
		result+=odeVectorSquaredNorm(f);
		return result;*/
		
		float result=0;
		for (int j = 2; j < 4; j++)
		{
			if (getHoldBodyIDs(j, targetContext) >= 0)
			{
				for (unsigned int  m = 0; m < jointHoldBallIDs[j].size(); m++)
				{
					if (jointHoldBallIDs[j][m] != -1)
					{
						int joint = jointHoldBallIDs[j][m];
						if (odeJointGetBody(joint, 0) >= 0)
						{
							result += odeVectorSquaredNorm(odeJointGetAccumulatedForce(joint, 0));
							//result += odeVectorSquaredNorm(odeJointGetAccumulatedTorque(joint, 0));
						}
						if (odeJointGetBody(joint, 1) >= 0)
						{
							result += odeVectorSquaredNorm(odeJointGetAccumulatedForce(joint, 1));
							//result += odeVectorSquaredNorm(odeJointGetAccumulatedTorque(joint, 1));
						}
					}
				}
			}
		}
		return result;
	}
	
	float getDesMotorAngleFromID(int i)
	{
		int bodyID = jointIDIndex[i] + 1;
		int axisID = jointAxisIndex[i];
		return getDesMotorAngle(bodyID, axisID);
	}

	float getDesMotorAngle(int &iBodyName, int &iAxisNum)
	{
		int jointIndex = iBodyName - 1;
		if (jointIndex > (int)(jointIDs.size()-1))
		{
			jointIndex = jointIDs.size()-1;
		}
		if (jointIndex < 0)
		{
			jointIndex = 0;
		}
		iBodyName = jointIndex + 1;

		if (jointTypes[jointIndex] == JointType::ThreeDegreeJoint)
		{
			if (iAxisNum > 2)
			{
				iAxisNum = 0;
			}
			if (iAxisNum < 0)
			{
				iAxisNum = 2;
			}
		}
		else
		{
			if (iAxisNum != 0)
			{
				iAxisNum = 0;
			}
		}

		int m_angle_index = 0;
		for (int b = 0; b < BodyNUM; b++)
		{
			for (int j = 0; j < jointTypes[b]; j++)
			{
				if (b == iBodyName - 1 && j == iAxisNum)
				{
					return desiredAnglesBones[m_angle_index];
				}
				m_angle_index++;
			}
		}
		return 0.0f;
	}

	void setMotorAngle(int &iBodyName, int &iAxisNum, float& dAngle)
	{
		int jointIndex = iBodyName - 1;
		if (jointIndex > (int)(jointIDs.size()-1))
		{
			jointIndex = jointIDs.size()-1;
		}
		if (jointIndex < 0)
		{
			jointIndex = 0;
		}
		iBodyName = jointIndex + 1;

		if (jointTypes[jointIndex] == JointType::ThreeDegreeJoint)
		{
			if (iAxisNum > 2)
			{
				iAxisNum = 0;
			}
			if (iAxisNum < 0)
			{
				iAxisNum = 2;
			}
		}
		else
		{
			if (iAxisNum != 0)
			{
				iAxisNum = 0;
			}
		}

		int m_angle_index = 0;
		for (int b = 0; b < BodyNUM; b++)
		{
			int mJointIndex = b - 1;
			if (mJointIndex < 0)
			{
				continue;
			}
			if (jointIDs[mJointIndex] == -1)
			{
				m_angle_index += jointTypes[mJointIndex];
				continue;
			}
			float source_angle = 0.0f;
			for (int axis = 0; axis < jointTypes[mJointIndex]; axis++)
			{
				if (jointTypes[mJointIndex] == JointType::ThreeDegreeJoint)
				{
					source_angle = odeJointGetAMotorAngle(jointIDs[mJointIndex],axis);
				}
				else
				{
					source_angle = odeJointGetHingeAngle(jointIDs[mJointIndex]);
				}

				if (b == iBodyName && axis == iAxisNum)
				{
					desiredAnglesBones[m_angle_index] = dAngle;
				}

				float iSpeed = desiredAnglesBones[m_angle_index] - source_angle;
				m_angle_index++;

				if (fabs(iSpeed) > maxSpeed)
				{
					iSpeed = fsign(iSpeed) * maxSpeed;
				}

				if (jointTypes[mJointIndex] == JointType::ThreeDegreeJoint)
				{
					switch (axis)
					{
					case 0:
						odeJointSetAMotorParam(jointIDs[mJointIndex], dParamVel1, iSpeed);
						break;
					case 1:
						odeJointSetAMotorParam(jointIDs[mJointIndex], dParamVel2, iSpeed);
						break;
					case 2:
						odeJointSetAMotorParam(jointIDs[mJointIndex], dParamVel3, iSpeed);
						break;
					}
				}
				else
				{
					odeJointSetHingeParam(jointIDs[mJointIndex], dParamVel1, iSpeed);
				}
			}
		}

		return;
	}

	void saveContextIn(BipedState& c)
	{
		for (int i = 0; i < BodyNUM; i++)
		{
			if (c.bodyStates.size() < BodyNUM)
			{
				c.bodyStates.push_back(BodyState());
			}
		}

		for (int i = 0; i < BodyNUM; i++)
		{
			c.bodyStates[i].setPos(getBonePosition(i));
			c.bodyStates[i].setAngle(getBoneAngle(i));
			c.bodyStates[i].setVel(getBoneLinearVelocity(i));
			c.bodyStates[i].setAVel(getBoneAngularVelocity(i));
			c.bodyStates[i].setBoneSize(boneSize[i]);
			c.bodyStates[i].setBodyType(bodyTypes[i]);
		}

		return;
	}

	//////////////////////////////////////////// for drawing bodies ///////////////////////////////////////
	static void drawLine(const Vector3& mP1, const Vector3& mP2) // color should be set beforehand!
	{
		float p1[] = {mP1.x(),mP1.y(),mP1.z()};
		float p2[] = {mP2.x(),mP2.y(),mP2.z()};
		rcDrawLine(p1, p2);
		return;
	}

	static void drawCross(const Vector3& p)
	{
		float cross_size = 0.3f;
		float p1[] = {p.x() - cross_size / 2, p.y(), p.z()};
		float p2[] = {p.x() + cross_size / 2, p.y(), p.z()};
		rcDrawLine(p1, p2);

		p1[0] = p.x();
		p1[1] = p.y() - cross_size / 2;
		p2[0] = p.x();
		p2[1] = p.y() + cross_size / 2;
		rcDrawLine(p1, p2);

		p1[1] = p.y();
		p1[2] = p.z() - cross_size / 2;
		p2[1] = p.y();
		p2[2] = p.z() + cross_size / 2;
		rcDrawLine(p1, p2);

		return;
	}

	static void drawCube(Vector3& mCenter, float mCubeSize)
	{
		float p1[] = {mCenter.x() - mCubeSize, mCenter.y(), mCenter.z() - mCubeSize};
		float p2[] = {mCenter.x() - mCubeSize, mCenter.y(), mCenter.z() + mCubeSize};
		float p3[] = {mCenter.x() + mCubeSize, mCenter.y(), mCenter.z() + mCubeSize};
		float p4[] = {mCenter.x() + mCubeSize, mCenter.y(), mCenter.z() - mCubeSize};
		rcDrawLine(p1, p2);
		rcDrawLine(p2, p3);
		rcDrawLine(p3, p4);
		rcDrawLine(p4, p1);

		return;
	}

	void setColorConnectedBody(int bodyID, int targetContext)
	{
		Vector3 wColor(1, 1, 1);
		Vector3 rColor(1, 0, 0);

		Vector3 tColor(0,1,1);
		float transparency = 1;// 0.5f;
		switch (bodyID)
		{
		case BodyName::BodyLeftFoot:
//			tColor = (1 - mColorBodies[0]) * wColor + mColorBodies[0] * rColor;
			if (getHoldBodyIDs(0, targetContext) != -1)
				rcSetColor(tColor.x(),tColor.y(),tColor.z(),transparency);
			break;
		case BodyName::BodyRightFoot:
//			tColor = (1 - mColorBodies[1]) * wColor + mColorBodies[1] * rColor;
			if (getHoldBodyIDs(1, targetContext) != -1)
				rcSetColor(tColor.x(),tColor.y(),tColor.z(),transparency);
			break;
		case BodyName::BodyLeftHand:
//			tColor = (1 - mColorBodies[2]) * wColor + mColorBodies[2] * rColor;
			if (getHoldBodyIDs(2, targetContext) != -1)
				rcSetColor(tColor.x(),tColor.y(),tColor.z(),transparency);
			break;
		case BodyName::BodyRightHand:
//			tColor = (1 - mColorBodies[3]) * wColor + mColorBodies[3] * rColor;
			if (getHoldBodyIDs(3, targetContext) != -1)
				rcSetColor(tColor.x(),tColor.y(),tColor.z(),transparency);
			break;
		default:
			break;
		}
		return;
	}

	void mDrawStuff(int iControlledBody, int iControlledHold, int targetContext, bool whileOptimizing, bool debug_show)
	{
		if (!whileOptimizing)
		{
			setCurrentOdeContext(masterContext);
			float _d = 0;
			odeGeomPlaneGetParams(mContext->spaceID, _d);
			rcDrawGround(_d);
		}

		for (int i = 0; i < BodyNUM; i++)
		{
			rcSetColor(1,1,1,1.0f);

			setColorConnectedBody(i, targetContext);

			if (((mGeomID[i] == iControlledBody && TestID == mEnumTestCaseClimber::TestCntroller) 
				|| (i == iControlledBody && TestID == mEnumTestCaseClimber::TestAngle)) && iControlledBody >= 0 && isTestClimber)
			{
				rcSetColor(0,1,0,1.0f);
			}
			if (bodyTypes[i] == BodyType::BodyBox)
			{
				float lx, ly, lz;
				odeGeomBoxGetLengths(mGeomID[i], lx, ly, lz);
				float sides[] = {lx, ly, lz};
				rcDrawBox(odeBodyGetPosition(bodyIDs[i]), odeBodyGetRotation(bodyIDs[i]), sides);
			}
			else if (bodyTypes[i] == BodyType::BodyCapsule)
			{
				float radius,length;
				odeGeomCapsuleGetParams(mGeomID[i], radius, length);
				rcDrawCapsule(odeBodyGetPosition(bodyIDs[i]), odeBodyGetRotation(bodyIDs[i]), length, radius);
			}
			else
			{
				float radius = odeGeomSphereGetRadius(mGeomID[i]);
				//rcDrawMarkerCircle(odeBodyGetPosition(bodyIDs[i]), radius);
				rcDrawSphere(odeBodyGetPosition(bodyIDs[i]), odeBodyGetRotation(bodyIDs[i]), radius);
			}
		}
		
		// draw wall and holds
		if (!whileOptimizing)
		{
			//dsSetTexture(DS_TEXTURE_NUMBER::DS_NONE);
			for (unsigned int i = 0; i < mENVGeomTypes.size(); i++)
			{
				if (mENVGeomTypes[i] == BodyType::BodyBox)
				{
					rcSetColor(1, 1, 1, 0.5f);//rcSetColor(1, 1, 1, 1.0f);//
					float lx, ly, lz;
					odeGeomBoxGetLengths(mENVGeoms[i], lx, ly, lz);
					float sides[] = {lx, ly, lz};
					rcDrawBox(odeGeomGetPosition(mENVGeoms[i]), odeGeomGetRotation(mENVGeoms[i]), sides); 
				}
				else if (mENVGeomTypes[i] == BodyType::BodyCapsule)
				{
					rcSetColor(1, 1, 1, 1.0f);
					float radius, length;
					odeGeomCapsuleGetParams(mENVGeoms[i], radius, length);
					rcDrawCapsule(odeGeomGetPosition(mENVGeoms[i]), odeGeomGetRotation(mENVGeoms[i]), length, radius);
				}
				else
				{
					float radius = holds_body[i - startingHoldGeomsIndex].size / 2;
					float pos[3] = {holds_body[i - startingHoldGeomsIndex].holdPos.x(), holds_body[i - startingHoldGeomsIndex].holdPos.y(), holds_body[i - startingHoldGeomsIndex].holdPos.z()};
					rcSetColor(1,1,0,0.5f);
					if (mENVGeoms[i] == iControlledHold && iControlledHold >= 0 && isTestClimber && (TestID == TestAngle || TestID == TestCntroller))
					{
						rcSetColor(0,1,0,1.0f);
					}
					
					rcDrawSphere(pos, odeGeomGetRotation(mENVGeoms[i]), radius);//odeGeomGetPosition(mENVGeoms[i])

					if (debug_show)
					{
						Vector3 p1 = holds_body[i - startingHoldGeomsIndex].holdPos;
						Vector3 p2 = p1 + (holds_body[i - startingHoldGeomsIndex].getIdealForceDirection(0)) * 0.5f;
						rcSetColor(0,1,1);
						drawLine(p1, p2);
					}
				}
			}

			rcSetColor(0,1,0,1.0f);
//			for (unsigned int i = 0; i < BodyNUM; i++)
//			{
			int i = BodyName::BodyTrunk;
			Vector3 dir_i = getBodyDirectionY(i);

			/*if (i == BodyName::BodyRightHand || i == BodyName::BodyLeftHand)
			{
				dir_i = getBodyDirectionZ(i);
			}
			if (i == BodyName::BodyLeftFoot || i == BodyName::BodyRightFoot)
			{
				dir_i = -dir_i;
			}*/
			Vector3 pos_i = getBonePosition(i);

			Vector3 n_pos_i = pos_i + 0.5f * dir_i;
			if (i == BodyName::BodyTrunk)
			{
				drawLine(pos_i, n_pos_i);
			}

			//if (i == BodyName::BodyLeftFoot)
			//{
			//	drawLine(pos_i, n_pos_i);
			//}

			//if (i == BodyName::BodyRightFoot)
			//{
			//	drawLine(pos_i, n_pos_i);
			//}

			//if (i == BodyName::BodyRightHand)
			//{
			//	drawLine(pos_i, n_pos_i);
			//}

			//if (i == BodyName::BodyLeftHand)
			//{
			//	drawLine(pos_i, n_pos_i);
			//}

//			}
		}

		return;
	}
	//////////////////////////////////////////// goal info ///////////////////////////////////////////////

	Vector3 getGoalPos()
	{
		return holds_body[goal_hold_id].holdPos;
	}

	//////////////////////////////////////////// create environment //////////////////////////////////////
	
	float randomizeHoldPositions(const mDemoTestClimber& DemoID, float _angle, std::vector<int>& ret_near_index)
	{
		if (!(DemoID == mDemoTestClimber::DemoLongWall || DemoID == mDemoTestClimber::DemoHorRotatedWall || DemoID == mDemoTestClimber::DemoJump2))
			return 0;
		
		if (DemoID == mDemoTestClimber::DemoJump2)
			_angle = -1;
		// there is alwayse the same amount of holds but with randomized hold positions

		float radius = climberLegLegDis + mTools::getRandomBetween_01() * (climberRadius / 2.5f - climberLegLegDis);
		Vector3 startPos = getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg);
		float _dir_sign = sign(cosf(_angle));

		ret_near_index.clear();
		ret_near_index = std::vector<int>(2, -1);
		std::vector<float> near_dis(2, FLT_MAX);
		Vector3 trunkPos = getEndPointPosBones(SimulationContext::BodyName::BodyTrunk);
		//Vector3 rightH = getEndPointPosBones(SimulationContext::BodyName::BodyRightHand);
		trunkPos[1] = 0.0f;
		if (_angle < 0)
		{
			_dir_sign = 1.0f;
			float r_sign = (float)rand() / (float)RAND_MAX;
			if (r_sign < 0.5f)
			{
				_dir_sign = -1.0f;
			}

			if (DemoID == mDemoTestClimber::DemoJump2)
				_dir_sign = 1.0f;
		}
		
		if (_dir_sign < 0)
		{
			startPos = getEndPointPosBones(SimulationContext::BodyName::BodyRightLeg);
		}

		float seperate_dis = 100.0f;
		float seperate_angle = 0.0f;
		if (DemoID == mDemoTestClimber::DemoHorRotatedWall)
		{
			seperate_dis = (climberHandHandDis / 2.5f) + mTools::getRandomBetween_01() * (climberHandHandDis / 2.5f);
			seperate_angle = -PI / 2.0f + PI * mTools::getRandomBetween_01();

			Vector3 v1_w1 = startPos - _dir_sign * Vector3(1.0f, 0.0f, 0.0f);
			v1_w1[1] = 0;
			v1_w1[2] = 0;
			Vector3 v2_w1 = startPos + _dir_sign * Vector3(seperate_dis, 0.0f, 0.0f);
			v2_w1[1] = 0;
			v2_w1[2] = 0;
			Vector3 v3_w1 = (v1_w1 + v2_w1) / 2.0f + 10.0f * Vector3(0.0f, 0.0f, 1.0f);
			createRotateSurface(v1_w1, v2_w1, v3_w1, (v1_w1 + v2_w1) / 2.0f + Vector3(0.0f, -1.0f, 0.0f), mENVGeoms[0]);

			Vector3 v1_w2 = v2_w1;
			Vector3 v2_w2 = v1_w2 + _dir_sign * 5.0f * Vector3(cosf(seperate_angle), sinf(seperate_angle), 0.0f);
			Vector3 v3_w2 = (v1_w2 + v2_w2) / 2.0f + 10.0f * Vector3(0.0f, 0.0f, 1.0f);

			createRotateSurface(v1_w2, v2_w2, v3_w2, (v1_w2 + v2_w2) / 2.0f + Vector3(sinf(seperate_angle), -cosf(seperate_angle), 0.0f), mENVGeoms[1]);
		}
				
		float cHeightZ = 0.5f;
		int index = 0;

		int des_row = 5;
		int des_col = 2;
		int miss_row = 50;
		int des_num_holds = des_col * (des_row - 1);
		if (_angle < 0)
		{
			des_col = 4;
			des_row = 4;
			miss_row = 50;
			int row = 0;
			float _minZ = FLT_MAX;
			while (row < des_row)
			{
				float cDisX = 0.0f;
				
				_minZ = FLT_MAX;

				int col = 0;

				Vector3 rIndex(0.0f, 0.0f, 0.0f);
				while (col < des_col)
				{
					float r1 = (float)rand() / (float)RAND_MAX;
					float r2 = (float)rand() / (float)RAND_MAX;

					float nZ = cHeightZ + (climberRadius / 5.0f) * r2; // cHeightZ + 0.3f * r2 // 
					float nX = startPos.x() - (climberHandHandDis / 6.0f) + (r1 * (climberHandHandDis / 3.0f)) + (cDisX * _dir_sign); //startPos.x() - 0.4f + r1 * 0.2f + (cDisX * _dir_sign); // 
					Vector3 cPos = Vector3(nX, 0.0f, nZ);

					if (nX * _dir_sign > seperate_dis - 0.25f)
					{
						cPos[0] = cosf(seperate_angle) * (nX + _dir_sign * (boneRadius / 2.0f - seperate_dis)) + (seperate_dis - boneRadius / 2.0f) * _dir_sign;
						cPos[1] = sinf(seperate_angle) * (nX + _dir_sign * (boneRadius / 2.0f - seperate_dis));
					}

					//float dis1 = (trunkPos - cPos).norm();
					//if (dis1 < near_dis[0])
					//{
					//	near_dis[0] = dis1;
					//	ret_near_index[0] = index;

					//	// swap
					//	if (near_dis[0] < near_dis[1])
					//	{
					//		swap(near_dis[1], near_dis[0]);
					//		swap(ret_near_index[1], ret_near_index[0]);
					//	}
					//}

					if (row == 1)
					{
						if (_dir_sign > 0)
							ret_near_index[col] = index;
						else
							ret_near_index[1 - col] = index;
					}

					if (row != miss_row)
					{
						if (index < (int)holds_body.size())
						{
							holds_body[index].holdPos = cPos;
							int rnd_hold_prototype = mTools::getRandomIndex(holds_prototypes.size());
							holds_body[index].setIdealForceDirection(0, holds_prototypes[rnd_hold_prototype].getIdealForceDirection(0));
							holds_body[index].public_prototype_hold_id = holds_prototypes[rnd_hold_prototype].public_prototype_hold_id;
						}
						else
						{
							addHoldBodyToWorld(cPos);
						}
						index++;
					}
					
					cDisX += (climberHandHandDis / 2.5f); // 0.8f;
					
					_minZ = min<float>(_minZ, nZ);

					col++;

				}

				if (row != miss_row)
					cHeightZ = _minZ + climberRadius / 2.5f;
				else
					cHeightZ = _minZ + climberRadius / 4.0f;

				row++;
			}
		}
		else
		{
			// create holds in the direction of _dir from _angle starting from startPos
			Vector3 _dir(cosf(_angle), 0.0f, sinf(_angle));
			int index_on_line = 0;
			while (index < des_num_holds)
			{
				float row = _dir.z() * index_on_line;
				float col = _dir.x() * index_on_line;
				for (int i = 0; i < 2; i++)
				{
					float r1 = (float)rand() / (float)RAND_MAX;
					float r2 = (float)rand() / (float)RAND_MAX;

					float nZ = cHeightZ + row * (climberRadius / 2.5f) + (climberRadius / 5.0f) * r2;
					float nX = startPos.x() + col * (climberHandHandDis / 2.0f) - (climberHandHandDis / 2.1f) + (2.0f * r1 * (climberHandHandDis / 2.1f));
					Vector3 cPos = Vector3(nX, 0.0f, nZ);

					/*float dis1 = (trunkPos - cPos).norm();
					if (dis1 < near_dis[0])
					{
						near_dis[0] = dis1;
						ret_near_index[0] = index;

						if (near_dis[0] < near_dis[1])
						{
							swap(near_dis[1], near_dis[0]);
							swap(ret_near_index[1], ret_near_index[0]);
						}
					}*/

					if (index_on_line == 1)
					{
						if (col > 0)
							ret_near_index[i] = index;
						else
							ret_near_index[1 - i] = index;
					}

					if (index_on_line != miss_row)
					{
						if (index < (int)holds_body.size())
						{
							holds_body[index].holdPos = cPos;
							int rnd_hold_prototype = mTools::getRandomIndex(holds_prototypes.size());
							holds_body[index].setIdealForceDirection(0, holds_prototypes[rnd_hold_prototype].getIdealForceDirection(0));
							holds_body[index].public_prototype_hold_id = holds_prototypes[rnd_hold_prototype].public_prototype_hold_id;
						}
						else
						{
							addHoldBodyToWorld(cPos);
						}
						index++;
					}
				}
				index_on_line++;
			}
		}

		goal_hold_id = (int)(holds_body.size() - 1);
		return _dir_sign;
	}

	void createEnvironment(const mDemoTestClimber& DemoID, float wWidth, float wHeight, std::vector<std::vector<float>>& mHoldInfo)
	{
		Vector3 _wallInfo = createWall(DemoID, wWidth, wHeight);

		startingHoldGeomsIndex = mENVGeoms.size();

		Vector3 cPos;
		if (mHoldInfo.size() > 0)
		{
			for (unsigned int i = 0; i < mHoldInfo.size(); i++)
			{
				if (mHoldInfo[i].size() == 7 || mHoldInfo[i].size() == 2)
				{
					cPos[0] = -wWidth / 2 + mHoldInfo[i][0];
					cPos[1] = 0.0f;
					cPos[2] = mHoldInfo[i][1];
					if (mHoldInfo[i].size() > 2)
						addHoldBodyToWorld(cPos, 
										   mHoldInfo[i][2],
										   Vector3(mHoldInfo[i][3], mHoldInfo[i][4], mHoldInfo[i][5]), 
										   mHoldInfo[i][6], 
										   0.25f);
					else
						addHoldBodyToWorld(cPos);
				}
				else
				{
					cPos[0] = -wWidth / 2 + mHoldInfo[i][0];
					cPos[1] = -(mHoldInfo[i][8] / 4) + mHoldInfo[i][1];
					cPos[2] = mHoldInfo[i][2];
					
					addHoldBodyToWorld(cPos,
									   mHoldInfo[i][3],
									   Vector3(mHoldInfo[i][4], mHoldInfo[i][5], mHoldInfo[i][6]),
									   mHoldInfo[i][7],
									   mHoldInfo[i][8]);
				}
			}
			goal_hold_id = (int)(holds_body.size() - 1);
			return;
		}

		float betweenHolds = 0.2f;
				
		Vector3 rLegPos;
		Vector3 lLegPos;

		float cHeightZ = 0.35f;
		float cWidthX = -FLT_MAX;

		float wall_1_z = 0.5f;
		float wall_1_x = 0.245f;
		float btwHolds1 = 0.2f;

		float wall_2_z = 1.37f + 0.3f;
		float wall_2_x = 0.25f;
		float btwHolds2_middle = 0.2f;
		float btwHolds2_topRow = 0.22f;

		float wall_3_z = 2.495f + 0.3f;
		float wall_3_x = 0.3f;
		float btwHolds3 = 0.17f;

		float rPillar = _wallInfo.z();
		float disBetweenPillar = _wallInfo.x();
		Vector3 startPillarPos(0, _wallInfo.y(), 0.5f);
		float theta_pillar = 0.0f;

		std::vector<int> ret_index;
		int row = 0;
		switch (DemoID)
		{
		case mDemoTestClimber::DemoRoute1:
		{
			cPos = getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg);
			cPos[1] = 0.0f;
			cPos[2] = wall_1_z;
			addHoldBodyToWorld(cPos); // leftLeg
			lLegPos = cPos;

			rLegPos = cPos;
			rLegPos[0] += 4 * btwHolds1;
			addHoldBodyToWorld(rLegPos); // rightLeg

			cPos[2] = wall_2_z;
			addHoldBodyToWorld(cPos); // LeftHand

			cPos[0] += 3 * btwHolds2_middle;
			addHoldBodyToWorld(cPos); // RightHand

			cPos = lLegPos;
			cPos[2] += 3 * btwHolds1;
			cPos[0] += 2 * btwHolds1;
			addHoldBodyToWorld(cPos); // helper 1

			cPos = lLegPos;
			cPos[2] = wall_2_z + 3 * btwHolds2_middle + btwHolds2_topRow;
			cPos[0] += btwHolds2_topRow;
			addHoldBodyToWorld(cPos); // helper 2

			cPos = lLegPos;
			cPos[2] = wall_3_z + btwHolds3;
			cPos[0] += (4 * btwHolds3 - 2 * btwHolds1);
			addHoldBodyToWorld(cPos); // helper 3

			cPos[2] += 3 * btwHolds3;
			cPos[0] += (1 * btwHolds3);
			addHoldBodyToWorld(cPos); // goal

			break;
		}
		case mDemoTestClimber::DemoRoute2:
		{
			cPos = getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg);
			cPos[1] = 0.0f;
			cPos[2] = wall_1_z;
			addHoldBodyToWorld(cPos); // leftLeg
			lLegPos = cPos;

			rLegPos = cPos;
			rLegPos[0] += 2 * btwHolds1;
			addHoldBodyToWorld(rLegPos); // rightLeg

			cPos[2] = wall_2_z;
			addHoldBodyToWorld(cPos); // LeftHand

			cPos[0] += 2 * btwHolds2_middle;
			addHoldBodyToWorld(cPos); // RightHand

			cPos = lLegPos;
			cPos[2] += 3 * btwHolds1;
			cPos[0] += 3 * btwHolds1;
			addHoldBodyToWorld(cPos); // helper 1

			cPos = lLegPos;
			cPos[2] = wall_2_z;
			cPos[0] += 9 * btwHolds2_middle;
			addHoldBodyToWorld(cPos); // helper 2

			cPos = lLegPos;
			cPos[2] = wall_2_z + 3 * btwHolds2_middle;
			cPos[0] += (5 * btwHolds2_middle);
			addHoldBodyToWorld(cPos); // helper 3

			cPos = lLegPos;
			cPos[2] = wall_2_z + 3 * btwHolds2_middle + btwHolds2_topRow;
			cPos[0] += (2 * btwHolds2_middle + btwHolds2_topRow);
			addHoldBodyToWorld(cPos); // helper 4

			cPos = lLegPos;
			cPos[2] = wall_3_z + 2 * btwHolds3;
			cPos[0] += (9 * btwHolds3);
			addHoldBodyToWorld(cPos); // helper 5

			cPos[2] += 2 * btwHolds3;
			addHoldBodyToWorld(cPos); // goal

			cPos[2] += 4 * btwHolds3;
			addHoldBodyToWorld(cPos); // goal

			break;
		}
		case mDemoTestClimber::DemoRoute3:
		{
			cPos = getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg);
			cPos[1] = 0.0f;
			cPos[2] = wall_1_z;
			addHoldBodyToWorld(cPos); // leftLeg
			lLegPos = cPos;

			rLegPos = cPos;
			rLegPos[0] += 2 * btwHolds1;
			addHoldBodyToWorld(rLegPos); // rightLeg

			cPos[2] = wall_2_z;
			addHoldBodyToWorld(cPos); // LeftHand

			cPos[0] += 2 * btwHolds2_middle;
			addHoldBodyToWorld(cPos); // RightHand

			cPos = lLegPos;
			cPos[2] += 3 * btwHolds1;
			cPos[0] += 3 * btwHolds1;
			addHoldBodyToWorld(cPos); // helper 1

			cPos = lLegPos;
			cPos[2] = wall_2_z + 7 * btwHolds2_middle + btwHolds2_topRow;//+ 3 * btwHolds2_middle;
			cPos[0] += (5 * btwHolds2_middle);
			addHoldBodyToWorld(cPos); // helper 3

			cPos = lLegPos;
			cPos[2] = wall_2_z + 5 * btwHolds2_middle + btwHolds2_topRow;//3 * btwHolds2_middle + btwHolds2_topRow;
			cPos[0] += btwHolds2_middle;//(2 * btwHolds2_middle + btwHolds2_topRow);
			addHoldBodyToWorld(cPos); // helper 4

			cPos = lLegPos;
			cPos[2] = wall_3_z + 2 * btwHolds3;
			cPos[0] += (15 * btwHolds3);//(9 * btwHolds3);
			addHoldBodyToWorld(cPos); // helper 5

			cPos[2] += 2 * btwHolds3;
			addHoldBodyToWorld(cPos); // goal

			break;
		}
		case mDemoTestClimber::DemoJump1:
		{
			cPos = getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg);
			cPos[1] = 0.0f;
			cPos[2] = wall_1_z;
			addHoldBodyToWorld(cPos); // leftLeg
			lLegPos = cPos;

			rLegPos = cPos;
			rLegPos[0] += 2 * btwHolds1;
			addHoldBodyToWorld(rLegPos); // rightLeg

			cPos[2] = wall_2_z;
			cPos[0] -= btwHolds1;
			addHoldBodyToWorld(cPos); // LeftHand

			cPos[0] += 2 * btwHolds2_middle;
			addHoldBodyToWorld(cPos); // RightHand

			cPos = lLegPos;
			cPos[2] = wall_2_z - 2 * btwHolds1;
			cPos[0] += (6 * btwHolds2_middle); 
			addHoldBodyToWorld(cPos); // helper 3

			cPos = lLegPos;
			cPos[2] = wall_3_z;
			cPos[0] += (12 * btwHolds3);//(12 * btwHolds3);
			addHoldBodyToWorld(cPos); // helper 5

			cPos[2] += 4 * btwHolds3; // make this 6 to see the effect of nn in low-level controller
			addHoldBodyToWorld(cPos); // goal

			break;
		}
		case mDemoTestClimber::DemoJump2:
		{
			randomizeHoldPositions(DemoID, -1, ret_index);

			break;
		}
		case mDemoTestClimber::DemoJump3:
		{
			cPos = getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg);
			cPos[1] = 0.0f;
			cPos[2] = wall_1_z;
			addHoldBodyToWorld(cPos); // leftLeg
			lLegPos = cPos;

			rLegPos = cPos;
			rLegPos[0] += 2 * btwHolds1;
			addHoldBodyToWorld(rLegPos); // rightLeg

			cPos[2] = wall_2_z;
			cPos[0] -= btwHolds1;
			addHoldBodyToWorld(cPos); // LeftHand

			cPos[0] += 2 * btwHolds2_middle;
			addHoldBodyToWorld(cPos); // RightHand

			cPos[0] += 3 * btwHolds1;
			cPos[2] += 3 * btwHolds1;
			addHoldBodyToWorld(cPos); // help1

			cPos[0] += 3 * btwHolds1;
			cPos[2] += 3 * btwHolds1;
			addHoldBodyToWorld(cPos); // help2

			cPos[0] += 3 * btwHolds1;
			cPos[2] += 3 * btwHolds1;
			addHoldBodyToWorld(cPos); // help3 

			cPos[0] += 3 * btwHolds1;
			cPos[2] += 3 * btwHolds1;
//			addHoldBodyToWorld(cPos); // help4 // remove for jumping

			cPos[0] += 4 * btwHolds1;
			cPos[2] += 4 * btwHolds1;
			addHoldBodyToWorld(cPos); // goal

			break;
		}
		case mDemoTestClimber::DemoJump4:
		{
			cPos = getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg);
			cPos[1] = 0.0f;
			cPos[2] = wall_1_z;
			addHoldBodyToWorld(cPos); // leftLeg

			cPos[2] += 4 * btwHolds1;
			addHoldBodyToWorld(cPos);

			cPos[2] += 4 * btwHolds1;
			addHoldBodyToWorld(cPos);

			cPos[2] += 4 * btwHolds1;
			addHoldBodyToWorld(cPos);

			cPos[0] += 4 * btwHolds1;
			addHoldBodyToWorld(cPos);

			cPos[0] += 4 * btwHolds1;
			addHoldBodyToWorld(cPos);

			cPos[2] += 4 * btwHolds1;
			cPos[0] += 6 * btwHolds1;
			addHoldBodyToWorld(cPos);

			cPos[0] += 4 * btwHolds1;
			addHoldBodyToWorld(cPos);

			break;
		}
		case mDemoTestClimber::DemoJump5:
		{
			cPos = getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg);
			cPos[1] = 0.0f;
			cPos[2] = wall_1_z;
			addHoldBodyToWorld(cPos); // leftLeg

			cPos[2] += 4 * btwHolds1;
			addHoldBodyToWorld(cPos);

			cPos[2] += 4 * btwHolds1;
			addHoldBodyToWorld(cPos);

			cPos[2] += 4 * btwHolds1;
			addHoldBodyToWorld(cPos);

			cPos[2] += 4 * btwHolds1;
			addHoldBodyToWorld(cPos);

			cPos[2] += 6 * btwHolds1;
			cPos[0] += 6 * btwHolds1;
			addHoldBodyToWorld(cPos);

			cPos[2] += 4 * btwHolds1;
			addHoldBodyToWorld(cPos);

			cPos[2] += 4 * btwHolds1;
			addHoldBodyToWorld(cPos);

			break;
		}
		case mDemoTestClimber::DemoJump6:
		{
			cPos = getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg);
			cPos[1] = 0.0f;
			cPos[2] = wall_1_z;
//			addHoldBodyToWorld(cPos); // leftLeg

			cPos[2] += 2 * btwHolds1;
//			addHoldBodyToWorld(cPos);

			cPos[2] += 4 * btwHolds1;
			addHoldBodyToWorld(cPos);

			cPos[2] += 4 * btwHolds1;
			addHoldBodyToWorld(cPos);

			cPos[2] += 4 * btwHolds1;
			addHoldBodyToWorld(cPos);

			cPos[2] += 8 * btwHolds1;
//			cPos[0] += 6 * btwHolds1;
			addHoldBodyToWorld(cPos);

			cPos[2] += 4 * btwHolds1;
			addHoldBodyToWorld(cPos);

			cPos[2] += 4 * btwHolds1;
			addHoldBodyToWorld(cPos);

			break;
		}
		case mDemoTestClimber::DemoHorRotatedWall:
		case mDemoTestClimber::DemoLongWall:
		{
			randomizeHoldPositions(DemoID, mTools::getRandomBetween(0.25f * PI, 0.75f * PI), ret_index);
			break;
		}
		case mDemoTestClimber::Demo45Wall:
		{
			cPos = getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg);
			cPos[1] = 0.0f;
			cPos[2] = 0.35f;
			addHoldBodyToWorld(cPos); // leftLeg

			rLegPos = cPos;
			rLegPos[0] += 2 * betweenHolds;
			addHoldBodyToWorld(rLegPos); // rightLeg

			cPos[2] += 3 * betweenHolds + 0.36f;
			addHoldBodyToWorld(cPos); // LeftHand

			cPos[0] += 2 * betweenHolds;
			addHoldBodyToWorld(cPos); // RightHand

			cPos[2] = _wallInfo.y() + betweenHolds * sinf(_wallInfo.z());
			cPos[1] += -betweenHolds * cosf(_wallInfo.z());
			addHoldBodyToWorld(cPos); // RightHand 1

			cPos[0] -= 2 * betweenHolds;
			addHoldBodyToWorld(cPos); // LeftHand 1

			cPos[2] += 4 * betweenHolds * sinf(_wallInfo.z());
			cPos[1] += -4 * betweenHolds * cosf(_wallInfo.z());
			addHoldBodyToWorld(cPos); // LeftHand 2

			cPos[0] += 2 * betweenHolds;
			addHoldBodyToWorld(cPos); // RightHand 2

			cPos[2] += 4 * betweenHolds * sinf(_wallInfo.z());
			cPos[1] += -4 * betweenHolds * cosf(_wallInfo.z());
			addHoldBodyToWorld(cPos); // LeftHand 2

			cPos[0] -= 2 * betweenHolds;
			addHoldBodyToWorld(cPos); // RightHand 2

			cPos[2] += 4 * betweenHolds * sinf(_wallInfo.z());
			cPos[1] += -4 * betweenHolds * cosf(_wallInfo.z());
			addHoldBodyToWorld(cPos); // LeftHand 2

			cPos[0] += 2 * betweenHolds;
			addHoldBodyToWorld(cPos); // RightHand 2
			break;
		}
		case mDemoTestClimber::DemoPillar:
		{
			while (startPillarPos.z() < wHeight - 2.0f)
			{
				if (startPillarPos.z() < 1.0f || startPillarPos.z() > wHeight - 3.0f)
				{
					rPillar = _wallInfo.z() - 0.15f;
					cPos = startPillarPos 
						 + rPillar * mTools::getDirectionFromAngles(theta_pillar + 5.0f * (0.5f - mTools::getRandomBetween_01()), 0.0f);
				}
				else
				{
					rPillar = _wallInfo.z();
					cPos = startPillarPos + Vector3(0,0,0.25f * (0.5f - mTools::getRandomBetween_01())) 
						 + rPillar * mTools::getDirectionFromAngles(theta_pillar + 5.0f * (0.5f - mTools::getRandomBetween_01()), 0.0f);
				}

				theta_pillar += 20.0f;

				if (360 - theta_pillar < 20)
				{
					theta_pillar = 0.0f;
					startPillarPos[2] += 1.0f;
				}

				addHoldBodyToWorld(cPos, 1000, Vector3(0,0,-1), 1, 0.25, 0, theta_pillar);
			}

			startPillarPos[0] += disBetweenPillar;
			startPillarPos[2] = 0.5f;
			theta_pillar = 0.0f;

			while (startPillarPos.z() < wHeight - 2.0f)
			{
				if (startPillarPos.z() < 1.0f || startPillarPos.z() > wHeight - 3.0f)
				{
					rPillar = _wallInfo.z() - 0.15f;
					cPos = startPillarPos
						+ rPillar * mTools::getDirectionFromAngles(theta_pillar + 5.0f * (0.5f - mTools::getRandomBetween_01()), 0.0f);
				}
				else
				{
					rPillar = _wallInfo.z();
					cPos = startPillarPos + Vector3(0, 0, 0.25f * (0.5f - mTools::getRandomBetween_01()))
						+ rPillar * mTools::getDirectionFromAngles(theta_pillar + 5.0f * (0.5f - mTools::getRandomBetween_01()), 0.0f);
				}

				theta_pillar += 20.0f;

				if (360 - theta_pillar < 20)
				{
					theta_pillar = 0.0f;
					startPillarPos[2] += 1.0f;
				}
				
				addHoldBodyToWorld(cPos, 1000, Vector3(0, 0, -1), 1, 0.25, 0, theta_pillar);
			}

		}
		default:
			break;
		};

		goal_hold_id = (int)(holds_body.size() - 1);
	}

	Vector3 crossProduct(Vector3 a, Vector3 b)
	{
		a.normalize();
		b.normalize();
		Vector3 _n(a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]);
		_n.normalize();
		return _n;
	}

	Vector3 getGeomAxisDirection(int geomID, int axis)
	{
		dMatrix3 R;
		dReal mQ[4];
		odeGeomGetQuaternion(geomID, mQ);
		
		dRfromQ(R, mQ);

		int targetDirection = axis; // alwayse in y direction		
		return Vector3(R[4 * 0 + targetDirection], R[4 * 1 + targetDirection], R[4 * 2 + targetDirection]);
	}

	Vector3 getAxisDirectionFrom(const dMatrix3 R, int axis)
	{
		int targetDirection = axis; // alwayse in y direction		
		return Vector3(R[4 * 0 + targetDirection], R[4 * 1 + targetDirection], R[4 * 2 + targetDirection]);
	}

	void createRotateSurface(Vector3 v1, Vector3 v2, Vector3 v3, Vector3 lookAtPos, int cGeomID = -1)
	{
		float wallTickness = 2 * boneRadius;

		Vector3 _pos = (0.5f * (v1 + v2) + v3) / 2.0f;

		Vector3 oN(0,1,0);
		Vector3 nN = crossProduct(v2 - v1, v3 - v1);
		Vector3 dirSurface = (lookAtPos-_pos).normalized();

		float _width = (v2 - v1).norm();
		float _height = (0.5f * (v1+v2) - v3).norm();

		if (cGeomID == -1)
		{
			cGeomID = odeCreateBox(_width, wallTickness, _height);
			mENVGeoms.push_back(cGeomID);
			mENVGeomTypes.push_back(BodyType::BodyBox);

			odeGeomSetCategoryBits(cGeomID, unsigned long(0));
			odeGeomSetCollideBits(cGeomID, unsigned long(0x7FFF));
		}
		else
		{
			odeGeomBoxSetLengths(cGeomID, _width, wallTickness, _height);
		}


		if (mTools::getAbsAngleBtwVectors(dirSurface, nN) > PI / 2)
		{
			_pos = _pos + (0.5f * wallTickness) * nN;
		}
		else
		{
			_pos = _pos - (0.5f * wallTickness) * nN;
		}
		odeGeomSetPosition(cGeomID, _pos[0], _pos[1], _pos[2]);

		Vector3 rV = crossProduct(oN, nN);
		float angle = mTools::getAbsAngleBtwVectors(oN, nN);

		dMatrix3 R1;
		dRFromAxisAndAngle(R1, rV[0], rV[1], rV[2], angle);

		Vector3 xDir = getAxisDirectionFrom(R1, 0);
		Vector3 yDir = getAxisDirectionFrom(R1, 1);
		Vector3 zDir = getAxisDirectionFrom(R1, 2);

		dMatrix3 R2;
		Vector3 nXDir = (v2 - v1).normalized();
		angle = mTools::getAbsAngleBtwVectors(xDir, nXDir);
		
		//nN = yDir;
		dRFromAxisAndAngle(R2, nN[0], nN[1], nN[2], angle);

		dMatrix3 fR;
		dMultiply0(fR, R2, R1, 3, 3, 3);
		fR[3] = 0.0f;
		fR[7] = 0.0f;
		fR[11] = 0.0f;

		xDir = getAxisDirectionFrom(fR, 0);
		yDir = getAxisDirectionFrom(fR, 1);
		zDir = getAxisDirectionFrom(fR, 2);

		if ((xDir - nXDir).norm() > 0.001f)
		{
			dRFromAxisAndAngle(R2, nN[0], nN[1], nN[2], -angle);

			dMultiply0(fR, R2, R1, 3, 3, 3);
			fR[3] = 0.0f;
			fR[7] = 0.0f;
			fR[11] = 0.0f;

			xDir = getAxisDirectionFrom(fR, 0);
			yDir = getAxisDirectionFrom(fR, 1);
			zDir = getAxisDirectionFrom(fR, 2);
		}

		odeGeomSetRotation(cGeomID, fR);

		xDir = getGeomAxisDirection(cGeomID, 0);
		yDir = getGeomAxisDirection(cGeomID, 1);
		zDir = getGeomAxisDirection(cGeomID, 2);

		

		return;
	}

	void createWallAroundPyramid(Vector3 _from, Vector3 _to, Vector3 _boundary)
	{
		float dis = min((_boundary - _from).norm(), (_boundary - _to).norm());
		Vector3 _dir = (_to - _from).normalized();
		Vector3 dir(-_dir[2],_dir[1],_dir[0]);
		if (dis > 0 && dir.norm() > 0)
			createRotateSurface(_from, _to, 0.5f * (_from + _to) + dis * dir, 0.5f * (_from + _to) + Vector3(0,-1,0));
		return;
	}

	bool createPyramid(std::vector<Vector3>& _boundaries, std::vector<Vector3>& _points)
	{
		Vector3 pp1 = _points[0];
		Vector3 pp2 = _points[1];
		Vector3 pp3 = _points[2];
		Vector3 pp4 = _points[3];
		Vector3 _midPos = (pp1 + pp2 + pp3 + pp4) / 4.0f;
		Vector3 midPos = _points[4];
		
		createRotateSurface(pp1, pp2, midPos, _midPos);
		createRotateSurface(pp2, pp3, midPos, _midPos);
		createRotateSurface(pp3, pp4, midPos, _midPos);
		createRotateSurface(pp4, pp1, midPos, _midPos);

		Vector3 up = pp1; // search for max z
		Vector3 right = pp2; // search for max x
		Vector3 down = pp3; // search for min z
		Vector3 left = pp1; // search for min x

		for (int i = 0; i < 4; i++)
		{
			if (_points[i].z() > up.z())
			{
				up = _points[i];
			}
			if (_points[i].x() > right.x())
			{
				right = _points[i];
			}
			if (_points[i].z() < down.z())
			{
				down = _points[i];
			}
			if (_points[i].x() < left.x())
			{
				left = _points[i];
			}
		}

		Vector3 up_right(max(up[0],right[0]), 0.0f, max(up[2],right[2]));
		createWallAroundPyramid(up, right, up_right);

		Vector3 down_right(max(down[0],right[0]), 0.0f, min(down[2],right[2]));
		createWallAroundPyramid(right, down, down_right);
		
		Vector3 down_left(min(down[0],left[0]), 0.0f, min(down[2],left[2]));
		createWallAroundPyramid(down, left, down_left);

		Vector3 left_up(min(up[0],left[0]), 0.0f, max(up[2],left[2]));
		createWallAroundPyramid(left, up, left_up);

		_boundaries.push_back(up);
		_boundaries.push_back(right);
		_boundaries.push_back(down);
		_boundaries.push_back(left);

		return true;
	}

	void createWallFromTo(Vector3 _from, Vector3 _to)
	{
		float _width = _to[0] - _from[0];
		float _height = _to[2] - _from[2];
		int cGeomID = odeCreateBox(_width, 2 * boneRadius, _height);

		Vector3 _pos = (_to + _from) * 0.5f;

		odeGeomSetPosition(cGeomID, _pos[0], (1.5) * boneRadius + mBiasWallY + 0.5f * boneRadius, _pos[2]);
			
		mENVGeoms.push_back(cGeomID);
		mENVGeomTypes.push_back(BodyType::BodyBox);

		odeGeomSetCategoryBits(cGeomID, unsigned long(0));
		odeGeomSetCollideBits (cGeomID, unsigned long(0x7FFF)); //unsigned long(0x7FFF)
	}

	Vector3 createWall(const mDemoTestClimber& DemoID, float wWidth, float wHeight)
	{
		std::vector<std::vector<float>> xyz_points;
		if (DemoID == mDemoTestClimber::DemoRouteFromFile)
		{
			mFileHandler mWholeInfo("ClimberInfo\\wholeInfo.txt");
			mWholeInfo.readFile(xyz_points);
		}
		std::vector<Vector3> _boundaries1;
		std::vector<Vector3> _boundaries2;
		std::vector<Vector3> _iPoints;
		std::vector<Vector3> maxBoundary;
		std::vector<Vector3> minBoundary;

		float dis_pillar = 0.0f;
		float y_pilar = 2.0f;
		float r_pilar = 2.0f;

		int cGeomID;

		switch (DemoID)
		{
		case mDemoTestClimber::DemoRouteFromFile:
		{
			for (int i = 0; i < 5; i++)
			{
				_iPoints.push_back(Vector3(-wWidth / 2 + xyz_points[i][0], xyz_points[i][1], xyz_points[i][2]));
			}
			createPyramid(_boundaries1, _iPoints);
			_iPoints.clear();

			for (int i = 5; i < 10; i++)
			{
				_iPoints.push_back(Vector3(-wWidth / 2 + xyz_points[i][0], xyz_points[i][1], xyz_points[i][2]));
			}
			createPyramid(_boundaries2, _iPoints);
			_iPoints.clear();

			wWidth = 1.5 * wWidth;
			wHeight = 1.2 * wHeight;

			if (_boundaries1[2][2] > _boundaries2[0][2])
			{
				maxBoundary = _boundaries1;
				minBoundary = _boundaries2;
			}
			else
			{
				maxBoundary = _boundaries2;
				minBoundary = _boundaries1;
			}

			createWallFromTo(Vector3(-wWidth / 2, 0, 0), Vector3(wWidth / 2, 0, minBoundary[2][2]));
			createWallFromTo(Vector3(minBoundary[1][0], 0, minBoundary[2][2]), Vector3(wWidth / 2, 0, minBoundary[0][2]));
			createWallFromTo(Vector3(-wWidth / 2, 0, minBoundary[2][2]), Vector3(minBoundary[3][0], 0, minBoundary[0][2]));

			createWallFromTo(Vector3(-wWidth / 2, 0, minBoundary[0][2]), Vector3(wWidth / 2, 0, maxBoundary[2][2]));
			createWallFromTo(Vector3(maxBoundary[1][0], 0, maxBoundary[2][2]), Vector3(wWidth / 2, 0, maxBoundary[0][2]));
			createWallFromTo(Vector3(-wWidth / 2, 0, maxBoundary[2][2]), Vector3(maxBoundary[3][0], 0, maxBoundary[0][2]));

			createWallFromTo(Vector3(-wWidth / 2, 0, maxBoundary[0][2]), Vector3(wWidth / 2, 0, wHeight));

			return Vector3(0.0f, 0.0f, 0.0f);
			break;
		}
		case mDemoTestClimber::Demo45Wall:
		{
			// starting from vertical wall
			cGeomID = odeCreateBox(5, 2 * boneRadius, 2);
			odeGeomSetPosition(cGeomID, 0, (1.5) * boneRadius + mBiasWallY + 0.5f * boneRadius, 1.0f);
			mENVGeoms.push_back(cGeomID);
			mENVGeomTypes.push_back(BodyType::BodyBox);

			// going to 45 degree wall
			float l = 5.0f;
			float angle = PI / 4;
			float x_b = ((1.5) * boneRadius + mBiasWallY + 0.5f * boneRadius); // -boneRadius -0.5f * l * sinf(angle)
			cGeomID = odeCreateBox(5, 2 * boneRadius, l);
			odeGeomSetPosition(cGeomID, 0, x_b - 0.5f * (l)* sinf(angle), 2 + 0.5f * boneRadius + 0.5f * (l)* cosf(angle)); // - tanf(angle) * x_b

			dMatrix3 R;
			dRFromAxisAndAngle(R, 1, 0, 0, angle);
			odeGeomSetRotation(cGeomID, R);

			mENVGeoms.push_back(cGeomID);
			mENVGeomTypes.push_back(BodyType::BodyBox);

			return Vector3(0.0f, 2.0f, PI / 2 - angle);
			break;
		}
		case mDemoTestClimber::DemoLongWall:
		case mDemoTestClimber::DemoRoute1:
		case mDemoTestClimber::DemoRoute2:
		case mDemoTestClimber::DemoRoute3:
		case mDemoTestClimber::DemoJump1:
		case mDemoTestClimber::DemoJump2:
		case mDemoTestClimber::DemoJump3:
		case mDemoTestClimber::DemoJump4:
		case mDemoTestClimber::DemoJump5:
		case mDemoTestClimber::DemoJump6:
		{
			wWidth = 40.0f;
			wHeight = 40.0f;
			createWallFromTo(Vector3(-wWidth / 2, 0, -wHeight / 2), Vector3(wWidth / 2, 0, wHeight / 2));
			break;
		}
		case mDemoTestClimber::DemoPillar:
		{
			cGeomID = odeCreateCapsule(0, r_pilar, wHeight);
			odeGeomSetPosition(cGeomID, 0.0f, y_pilar, wHeight / 2.0f - 1.0f);

			mENVGeoms.push_back(cGeomID);
			mENVGeomTypes.push_back(BodyType::BodyCapsule);

			odeGeomSetCategoryBits(cGeomID, unsigned long(0));
			odeGeomSetCollideBits(cGeomID, unsigned long(0x7FFF));

			cGeomID = odeCreateCapsule(0, r_pilar, wHeight);
			
			dis_pillar = mTools::getRandomBetween(0, 3) + randomf();
			dis_pillar = dis_pillar + 5.0f;
			odeGeomSetPosition(cGeomID, dis_pillar, y_pilar, wHeight / 2.0f - 1.0f);

			mENVGeoms.push_back(cGeomID);
			mENVGeomTypes.push_back(BodyType::BodyCapsule);

			odeGeomSetCategoryBits(cGeomID, unsigned long(0));
			odeGeomSetCollideBits(cGeomID, unsigned long(0x7FFF));

			return Vector3(dis_pillar, y_pilar, r_pilar);
			break;
		}
		case mDemoTestClimber::DemoHorRotatedWall:
			createWallFromTo(Vector3(-wWidth / 2, 0, -wHeight / 2), Vector3(0, 0, wHeight / 2));
			createWallFromTo(Vector3(0, 0, -wHeight / 2), Vector3(wWidth / 2, 0, wHeight / 2));
			break;
		default:
			wWidth = 10.0f;
			wHeight = 40.0f;
			createWallFromTo(Vector3(-wWidth / 2, 0, -wHeight / 2), Vector3(wWidth / 2, 0, wHeight / 2));
		}

		return Vector3(0.0f,0.0f,0.0f);
	}

	void addHoldBodyToWorld(Vector3 cPos, float f_ideal = 1000.0f, Vector3 d_ideal = Vector3(0, 0,-1), float k = 1.0f, float _s = holdSize / 2.0f, int _HoldPushPullMode = 0, float theta_axis = -90.0f)
	{
		float iX = cPos.x(), iY = cPos.y(), iZ = cPos.z();

		int cGeomID = odeCreateSphere(_s);

		odeGeomSetPosition(cGeomID, iX, iY, iZ);

		mENVGeoms.push_back(cGeomID);
		mENVGeomTypes.push_back(BodyType::BodySphere);

		odeGeomSetCollideBits (cGeomID, unsigned long(0x8000)); // do not collide with anything!, but used for rayCasting
		odeGeomSetCategoryBits (cGeomID, unsigned long(0x8000)); // do not collide with anything!, but used for rayCasting

		_s = holdSize / 2.0f;

		holds_body.push_back(holdContent(Vector3(iX, iY, iZ), f_ideal, d_ideal, k, _s, cGeomID, _HoldPushPullMode, theta_axis));

		int rnd_hold_prototype = mTools::getRandomIndex(holds_prototypes.size());
		holds_body[holds_body.size() - 1].setIdealForceDirection(0, holds_prototypes[rnd_hold_prototype].getIdealForceDirection(0));
		holds_body[holds_body.size() - 1].public_prototype_hold_id = holds_prototypes[rnd_hold_prototype].public_prototype_hold_id;
	}

	int createJointType(float pX, float pY, float pZ, int pBodyNum, int cBodyNum, JointType iJ = JointType::BallJoint)
	{
		//float positionSpring = 10000.0f, stopSpring = 20000.0f, damper = 1.0f, maximumForce = 200.0f;
		float positionSpring = 5000.0f, stopSpring = 5000.0f, damper = 1.0f;
		
		float kp = positionSpring; 
        float kd = damper; 

        float erp = timeStep * kp / (timeStep * kp + kd);
        float cfm = 1.0f / (timeStep * kp + kd);

        float stopDamper = 1.0f; //stops best when critically damped
        float stopErp = timeStep * stopSpring / (timeStep * kp + stopDamper);
        float stopCfm = 1.0f / (timeStep * stopSpring + stopDamper);

		int jointID = -1;
		switch (iJ)
		{
			case JointType::BallJoint:
				jointID = odeJointCreateBall();
			break;
			case JointType::ThreeDegreeJoint:
				jointID = odeJointCreateAMotor();
			break;
			case JointType::TwoDegreeJoint:
				jointID = odeJointCreateHinge2();
			break;
			case JointType::OneDegreeJoint:
				jointID = odeJointCreateHinge();
			break;
			case JointType::FixedJoint:
				jointID = odeJointCreateFixed();
			break;
			default:
				break;
		}
		
		if (pBodyNum >= 0)
		{
			if (cBodyNum < BodyNUM && cBodyNum >= 0)
			{
				odeJointAttach(jointID, bodyIDs[pBodyNum], bodyIDs[cBodyNum]);
			}

			if (iJ == JointType::FixedJoint)
				odeJointSetFixed(jointID);
		}
		else
		{
			if (cBodyNum < BodyNUM && cBodyNum >= 0)
			{
				odeJointAttach(jointID, 0, bodyIDs[cBodyNum]);
			}

			if (iJ == JointType::FixedJoint)
				odeJointSetFixed(jointID);
		}

		float angle = 0;
		switch (iJ)
		{
			case JointType::BallJoint:
				odeJointSetBallAnchor(jointID, pX, pY, pZ);
			break;
			case JointType::ThreeDegreeJoint:
				odeJointSetAMotorNumAxes(jointID, 3);

				if (pBodyNum >= 0)
				{
					odeJointSetAMotorAxis(jointID, 0, 1, 0, 1, 0);
					odeJointSetAMotorAxis(jointID, 2, 2, 1, 0, 0);

					if (cBodyNum == BodyName::BodyLeftShoulder || cBodyNum == BodyName::BodyRightShoulder)
					{
						odeJointSetAMotorAxis(jointID, 0, 1, 0, 1, 0);
						odeJointSetAMotorAxis(jointID, 2, 2, 0, 0, 1);
					}
				}
				else
				{

					odeJointSetAMotorAxis(jointID, 0, 0, 1, 0, 0);
					odeJointSetAMotorAxis(jointID, 2, 2, 0, 0, 1);
				}

				odeJointSetAMotorMode(jointID, dAMotorEuler);

				odeJointSetAMotorParam(jointID, dParamFMax1, maximumForce);
				odeJointSetAMotorParam(jointID, dParamFMax2, maximumForce);
				odeJointSetAMotorParam(jointID, dParamFMax3, maximumForce);

				odeJointSetAMotorParam(jointID, dParamVel1, 0);
				odeJointSetAMotorParam(jointID, dParamVel2, 0);
				odeJointSetAMotorParam(jointID, dParamVel3, 0);

				odeJointSetAMotorParam(jointID, dParamCFM1, cfm);
				odeJointSetAMotorParam(jointID, dParamCFM2, cfm);
				odeJointSetAMotorParam(jointID, dParamCFM3, cfm);

				odeJointSetAMotorParam(jointID, dParamERP1, erp);
				odeJointSetAMotorParam(jointID, dParamERP2, erp);
				odeJointSetAMotorParam(jointID, dParamERP3, erp);

				odeJointSetAMotorParam(jointID, dParamStopCFM1, stopCfm);
				odeJointSetAMotorParam(jointID, dParamStopCFM2, stopCfm);
				odeJointSetAMotorParam(jointID, dParamStopCFM3, stopCfm);

				odeJointSetAMotorParam(jointID, dParamStopERP1, stopErp);
				odeJointSetAMotorParam(jointID, dParamStopERP2, stopErp);
				odeJointSetAMotorParam(jointID, dParamStopERP3, stopErp);

				odeJointSetAMotorParam(jointID, dParamFudgeFactor1, -1);
				odeJointSetAMotorParam(jointID, dParamFudgeFactor2, -1);
				odeJointSetAMotorParam(jointID, dParamFudgeFactor3, -1);
				
			break;
			case JointType::TwoDegreeJoint:
				odeJointSetHinge2Anchor(jointID, pX, pY, pZ);
				odeJointSetHinge2Axis(jointID, 1, 0, 1, 0);
				odeJointSetHinge2Axis(jointID, 2, 0, 0, 1);

				odeJointSetHinge2Param(jointID, dParamFMax1, maximumForce);
				odeJointSetHinge2Param(jointID, dParamFMax2, maximumForce);

				odeJointSetHinge2Param(jointID, dParamVel1, 0);
				odeJointSetHinge2Param(jointID, dParamVel2, 0);

				odeJointSetHinge2Param(jointID, dParamCFM1, cfm);
				odeJointSetHinge2Param(jointID, dParamCFM2, cfm);

				odeJointSetHinge2Param(jointID, dParamStopERP1, stopErp);
				odeJointSetHinge2Param(jointID, dParamStopERP2, stopErp);
				odeJointSetHinge2Param(jointID, dParamFudgeFactor1, -1);
				odeJointSetHinge2Param(jointID, dParamFudgeFactor2, -1);
			break;
			case JointType::OneDegreeJoint:
				odeJointSetHingeAnchor(jointID, pX, pY, pZ);
				if (cBodyNum == BodyName::BodyLeftLeg || cBodyNum == BodyName::BodyRightLeg || cBodyNum == BodyName::BodyRightFoot || cBodyNum == BodyName::BodyLeftFoot)
				{
					odeJointSetHingeAxis(jointID, 1, 0, 0);
				}
				else
				{
					odeJointSetHingeAxis(jointID, 0, 0, 1);
				}
				
				odeJointSetHingeParam(jointID, dParamFMax, maximumForce);

				odeJointSetHingeParam(jointID, dParamVel, 0);

				odeJointSetHingeParam(jointID, dParamCFM, cfm);

				odeJointSetHingeParam(jointID, dParamERP, erp);

				odeJointSetHingeParam(jointID, dParamStopCFM, stopCfm);

				odeJointSetHingeParam(jointID, dParamStopERP, stopErp);

				odeJointSetHingeParam(jointID, dParamFudgeFactor, -1);
			break;
			default:
				break;
		}

		return jointID;
	}

	Vector3 getVectorForm(std::vector<float>& _iV)
	{
		return Vector3(_iV[0], _iV[1], _iV[2]);
	}

	void createHumanoidBody(float pX, float pY, float dHeight, float dmass, std::vector<std::vector<float>>& _readFileClimberKinectInfo)
	{
		float current_height = 1.7758f; // with scale of one the created height is calculated in current_height
		float scale = dHeight / current_height;

		float ArmLegWidth = (0.75f * boneRadius) * scale;
		float feetWidth = ArmLegWidth;
		// trunk (root body without parent)
		float trunk_length = (0.1387f + 0.1363f) * scale; // done
		// spine
		float spine_length = (0.1625f + 0.091f) * scale; // done
		// thigh
		float thigh_length = (0.4173f) * scale; // done
		float dis_spine_leg_x = (0.09f) * scale;
		float dis_spine_leg_z = (0.00f) * scale;
		// leg
		float leg_length = (0.39f) * scale; // done
		// foot
		float feet_length = (0.08f + 0.127f + 0.05f) * scale; // done
		// shoulder
		float handShortenAmount = 0.025f;
		float dis_spine_shoulder_x = (0.06f + 0.123f) * scale;
		float dis_spine_shoulder_z = trunk_length - (0.1363f + 0.101f) * scale;
		float shoulder_length = (0.276f + handShortenAmount) * scale; // done
		// arm
		float arm_length = (0.278f) * scale; // done
		// hand
		float handsWidth = 0.9f * ArmLegWidth;
		float hand_length = (0.1f + 0.05f + 0.03f + 0.025f - handShortenAmount) * scale; // done
		// head
		float head_length = (0.21f + 0.04f) * scale; // done
		float dis_spine_head_z = (0.04f) * scale; // done
		float HeadWidth = (0.85f * boneRadius) * scale;

		Vector3 midPos;
		if (_readFileClimberKinectInfo.size() > 0)
		{
			enum mReadFileBoneType{_head = 0, _neck = 1, _spineshoulder = 2,
								   _spinemid = 3, _spinebase = 4, _shoulderright = 5,
								   _shoulderleft = 6, _hipright = 7, _hipleft = 8,
								   _elbowright = 9, _elbowleft = 10, _wristright = 11,
								   _wristleft = 12, _handtipright = 13, _handtipleft = 14,
								   _kneeright = 15, _kneeleft = 16, _ankleright = 17,
								   _ankleleft = 18, _feetright = 19, _feetleft = 20,
								   _groundfeetright = 21, _groundfeetleft = 22};
			trunk_length = (getVectorForm(_readFileClimberKinectInfo[_spineshoulder]) - getVectorForm(_readFileClimberKinectInfo[_spinemid])).norm();
			spine_length = (getVectorForm(_readFileClimberKinectInfo[_spinemid]) - getVectorForm(_readFileClimberKinectInfo[_spinebase])).norm();

			thigh_length = ((getVectorForm(_readFileClimberKinectInfo[_hipright]) - getVectorForm(_readFileClimberKinectInfo[_kneeright])).norm()
							+ (getVectorForm(_readFileClimberKinectInfo[_hipleft]) - getVectorForm(_readFileClimberKinectInfo[_kneeleft])).norm()) / 2.0f;
			leg_length = ((getVectorForm(_readFileClimberKinectInfo[_kneeright]) - getVectorForm(_readFileClimberKinectInfo[_ankleright])).norm()
							+ (getVectorForm(_readFileClimberKinectInfo[_kneeleft]) - getVectorForm(_readFileClimberKinectInfo[_ankleleft])).norm()) / 2.0f;
			//feet_length = 2.5f * (((getVectorForm(_readFileClimberKinectInfo[_groundfeetright]) - getVectorForm(_readFileClimberKinectInfo[_feetright])).norm()
						  //	+ (getVectorForm(_readFileClimberKinectInfo[_groundfeetleft]) - getVectorForm(_readFileClimberKinectInfo[_feetleft])).norm()) / 2.0f);

			shoulder_length = ((getVectorForm(_readFileClimberKinectInfo[_shoulderright]) - getVectorForm(_readFileClimberKinectInfo[_elbowright])).norm()
							+ (getVectorForm(_readFileClimberKinectInfo[_shoulderleft]) - getVectorForm(_readFileClimberKinectInfo[_elbowleft])).norm()) / 2.0f;
			arm_length = ((getVectorForm(_readFileClimberKinectInfo[_elbowright]) - getVectorForm(_readFileClimberKinectInfo[_wristright])).norm()
							+ (getVectorForm(_readFileClimberKinectInfo[_elbowleft]) - getVectorForm(_readFileClimberKinectInfo[_wristleft])).norm()) / 2.0f;
			//hand_length = ((getVectorForm(_readFileClimberKinectInfo[_wristright]) - getVectorForm(_readFileClimberKinectInfo[_handtipright])).norm()
			//				+ (getVectorForm(_readFileClimberKinectInfo[_wristleft]) - getVectorForm(_readFileClimberKinectInfo[_handtipleft])).norm()) / 2.0f;

			midPos = (getVectorForm(_readFileClimberKinectInfo[_shoulderright]) + getVectorForm(_readFileClimberKinectInfo[_shoulderleft])) / 2.0f;
			dis_spine_shoulder_z = -(midPos[1] - _readFileClimberKinectInfo[_spineshoulder][1]);
			dis_spine_shoulder_x = (getVectorForm(_readFileClimberKinectInfo[_shoulderright]) - getVectorForm(_readFileClimberKinectInfo[_shoulderleft])).norm() / 2.0f;

			midPos = (getVectorForm(_readFileClimberKinectInfo[_hipright]) + getVectorForm(_readFileClimberKinectInfo[_hipleft])) / 2.0f;
			dis_spine_leg_z = _readFileClimberKinectInfo[_spinebase][1] - midPos[1];
			dis_spine_leg_x = (getVectorForm(_readFileClimberKinectInfo[_hipright]) - getVectorForm(_readFileClimberKinectInfo[_hipleft])).norm() / 2.0f;

			head_length = (2.0f) * (getVectorForm(_readFileClimberKinectInfo[_head]) - getVectorForm(_readFileClimberKinectInfo[_neck])).norm(); 
			dis_spine_head_z = (getVectorForm(_readFileClimberKinectInfo[_neck]) - getVectorForm(_readFileClimberKinectInfo[_spineshoulder])).norm();

			feetWidth = ((getVectorForm(_readFileClimberKinectInfo[_groundfeetright]) - getVectorForm(_readFileClimberKinectInfo[_ankleright])).norm()
							+ (getVectorForm(_readFileClimberKinectInfo[_groundfeetleft]) - getVectorForm(_readFileClimberKinectInfo[_ankleleft])).norm()) / 2.0f;
		}

		climberRadius = trunk_length + spine_length + leg_length + thigh_length + shoulder_length + arm_length;
		climberLegLegDis = 2 * (leg_length + thigh_length) + (boneRadius * scale);
		climberHandHandDis = 2 * (shoulder_length + arm_length + hand_length) + (1.2f * boneRadius * scale);
		climberHeight = feetWidth + leg_length + thigh_length + dis_spine_leg_z + spine_length + trunk_length + dis_spine_head_z + head_length;

		float pZ = spine_length + thigh_length + dis_spine_leg_z + leg_length + ArmLegWidth;

		float trunkPosX = pX;
		float trunkPosY = pY;
		float trunkPosZ = pZ;
		int cJointID = -1;

		// trunk (root body without parent)
		createBodyi(BodyName::BodyTrunk, (1.2f * boneRadius) * scale, trunk_length);
		odeBodySetPosition(bodyIDs[BodyName::BodyTrunk], pX, pY, pZ + trunk_length / 2);
		fatherBodyIDs[BodyName::BodyTrunk] = -1;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyTrunk)], unsigned long(0x0080));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyTrunk)], unsigned long(0x36FE));

		// spine
		pZ -= (spine_length / 2);
		createBodyi(BodyName::BodySpine, (boneRadius) * scale, spine_length);
		odeBodySetPosition(bodyIDs[BodyName::BodySpine], pX, pY, pZ);
		createJointType(pX, pY, pZ + (spine_length / 2), BodyName::BodyTrunk, BodyName::BodySpine);
		cJointID = createJointType(pX, pY, pZ + (spine_length / 2), BodyName::BodyTrunk, BodyName::BodySpine, JointType::ThreeDegreeJoint);
		std::vector<Vector2> cJointLimits = setAngleLimitations(cJointID, BodyName::BodySpine);
		setJointID(BodyName::BodySpine, cJointID, JointType::ThreeDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodySpine] = BodyName::BodyTrunk;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodySpine)], unsigned long(0x0001));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodySpine)], unsigned long(0x7F6D));

		// left thigh
		pZ = trunkPosZ;
		pX -= (dis_spine_leg_x); 
		pZ -= (spine_length + thigh_length / 2 + dis_spine_leg_z); 
		createBodyi(BodyName::BodyLeftThigh, ArmLegWidth, thigh_length);
		odeBodySetPosition(bodyIDs[BodyName::BodyLeftThigh], pX, pY, pZ);
		createJointType(pX, pY, pZ + (thigh_length / 2), BodyName::BodySpine, BodyName::BodyLeftThigh);
		cJointID = createJointType(pX, pY, pZ + (thigh_length / 2), BodyName::BodySpine, BodyName::BodyLeftThigh, JointType::ThreeDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyLeftThigh);
		setJointID(BodyName::BodyLeftThigh, cJointID, JointType::ThreeDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyLeftThigh] = BodyName::BodyTrunkLower;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyLeftThigh)], unsigned long(0x0010));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyLeftThigh)], unsigned long(0x7FDE));

		// left leg
		pZ -= (thigh_length / 2 + leg_length / 2);
		createBodyi(BodyName::BodyLeftLeg, ArmLegWidth, leg_length);
		odeBodySetPosition(bodyIDs[BodyName::BodyLeftLeg], pX, pY, pZ);
		cJointID = createJointType(pX, pY, pZ + (leg_length / 2), BodyName::BodyLeftThigh, BodyName::BodyLeftLeg, JointType::OneDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyLeftLeg);
		setJointID(BodyName::BodyLeftLeg, cJointID, JointType::OneDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyLeftLeg] = BodyName::BodyLeftThigh;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyLeftLeg)], unsigned long(0x0020));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyLeftLeg)], unsigned long(0x7FAF));

		// left foot end point
		pZ -= (leg_length / 2 + ArmLegWidth / 4.0f);
		pY += (feet_length / 2 - ArmLegWidth / 2.0f);
		createBodyi(BodyName::BodyLeftFoot, ArmLegWidth * 0.9f, feet_length);
		odeBodySetPosition(bodyIDs[BodyName::BodyLeftFoot], pX, pY, pZ);
		cJointID = createJointType(pX, trunkPosY, pZ + ArmLegWidth / 4.0f, BodyName::BodyLeftLeg, BodyName::BodyLeftFoot, JointType::OneDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyLeftFoot);
		setJointID(BodyName::BodyLeftFoot, cJointID, JointType::OneDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyLeftFoot] = BodyName::BodyLeftLeg;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyLeftFoot)], unsigned long(0x0040));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyLeftFoot)], unsigned long(0x7FDF));

//		_feetHandsWidth[0] = ArmLegWidth * 0.9f;

		// right thigh
		pX = trunkPosX;
		pY = trunkPosY;
		pZ = trunkPosZ;

		pX += (dis_spine_leg_x);
		pZ -= (spine_length + thigh_length / 2 + dis_spine_leg_z);
		createBodyi(BodyName::BodyRightThigh, ArmLegWidth, thigh_length);
		odeBodySetPosition(bodyIDs[BodyName::BodyRightThigh], pX, pY, pZ);
		createJointType(pX, pY, pZ + (thigh_length / 2.0f), BodyName::BodySpine, BodyName::BodyRightThigh);
		cJointID = createJointType(pX, pY, pZ + (thigh_length / 2.0f), BodyName::BodySpine, BodyName::BodyRightThigh, JointType::ThreeDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyRightThigh);
		setJointID(BodyName::BodyRightThigh, cJointID, JointType::ThreeDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyRightThigh] = BodyName::BodyTrunkLower;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyRightThigh)], unsigned long(0x0002));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyRightThigh)], unsigned long(0x7FFA));

		// right leg
		pZ -= (thigh_length / 2 + leg_length / 2);
		createBodyi(BodyName::BodyRightLeg, ArmLegWidth, leg_length);
		odeBodySetPosition(bodyIDs[BodyName::BodyRightLeg], pX, pY, pZ);
		cJointID = createJointType(pX, pY, pZ + (leg_length / 2.0f), BodyName::BodyRightThigh, BodyName::BodyRightLeg, JointType::OneDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyRightLeg);
		setJointID(BodyName::BodyRightLeg, cJointID, JointType::OneDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyRightLeg] = BodyName::BodyRightThigh;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyRightLeg)], unsigned long(0x0004));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyRightLeg)], unsigned long(0x7FF5));

		// right foot end point
		pZ -= (leg_length / 2 + ArmLegWidth / 4.0f);
		pY += (feet_length / 2 - ArmLegWidth / 2);
		createBodyi(BodyName::BodyRightFoot, ArmLegWidth * 0.9f, feet_length);
		odeBodySetPosition(bodyIDs[BodyName::BodyRightFoot], pX, pY, pZ);
		cJointID = createJointType(pX, trunkPosY, pZ + ArmLegWidth / 4.0f, BodyName::BodyRightLeg, BodyName::BodyRightFoot, JointType::OneDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyRightFoot);
		setJointID(BodyName::BodyRightFoot, cJointID, JointType::OneDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyRightFoot] = BodyName::BodyRightLeg;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyRightFoot)], unsigned long(0x0008));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyRightFoot)], unsigned long(0x7FFB));

//		_feetHandsWidth[1] = ArmLegWidth * 0.9f;

		// left shoulder
		pX = trunkPosX;
		pY = trunkPosY;
		pZ = trunkPosZ;

		pX -= (shoulder_length / 2.0f + dis_spine_shoulder_x);
		pZ += (trunk_length - dis_spine_shoulder_z);
		createBodyi(BodyName::BodyLeftShoulder, shoulder_length, handsWidth);
		odeBodySetPosition(bodyIDs[BodyName::BodyLeftShoulder], pX, pY, pZ);
		createJointType(pX + (shoulder_length / 2.0f), pY, pZ, BodyName::BodyTrunk, BodyName::BodyLeftShoulder);
		cJointID = createJointType(pX + (shoulder_length / 2.0f), pY, pZ, BodyName::BodyTrunk, BodyName::BodyLeftShoulder, JointType::ThreeDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyLeftShoulder);
		setJointID(BodyName::BodyLeftShoulder, cJointID, JointType::ThreeDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyLeftShoulder] = BodyName::BodyTrunkUpper;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyLeftShoulder)], unsigned long(0x0800));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyLeftShoulder)], unsigned long(0x6F7F));

		// left arm
		pX -= (shoulder_length / 2 + arm_length / 2);
		createBodyi(BodyName::BodyLeftArm, arm_length, handsWidth);
		odeBodySetPosition(bodyIDs[BodyName::BodyLeftArm], pX, pY, pZ);
		cJointID = createJointType(pX + (arm_length / 2.0f), pY, pZ, BodyName::BodyLeftShoulder, BodyName::BodyLeftArm, JointType::OneDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyLeftArm);
		setJointID(BodyName::BodyLeftArm, cJointID, JointType::OneDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyLeftArm] = BodyName::BodyLeftShoulder;

		odeGeomSetCategoryBits(mGeomID[BodyName::BodyLeftArm], unsigned long(0x1000));
		odeGeomSetCollideBits (mGeomID[BodyName::BodyLeftArm], unsigned long(0x57FF));

		// left hand end point
		pX -= (arm_length / 2 + (hand_length / 2.0f));
		createBodyi(BodyName::BodyLeftHand, hand_length, handsWidth);
		odeBodySetPosition(bodyIDs[BodyName::BodyLeftHand], pX, pY, pZ);
		createJointType(pX + (hand_length / 2.0f), pY, pZ, BodyName::BodyLeftArm, BodyName::BodyLeftHand);
		cJointID = createJointType(pX + (hand_length / 2.0f), pY, pZ, BodyName::BodyLeftArm, BodyName::BodyLeftHand, JointType::ThreeDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyLeftHand);
		setJointID(BodyName::BodyLeftHand, cJointID, JointType::ThreeDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyLeftHand] = BodyName::BodyLeftArm;

		odeGeomSetCategoryBits(mGeomID[BodyName::BodyLeftHand], unsigned long(0x2000));
		odeGeomSetCollideBits (mGeomID[BodyName::BodyLeftHand], unsigned long(0x6FFF));

//		_feetHandsWidth[2] = handsWidth;

		// right shoulder
		pX = trunkPosX;
		pZ = trunkPosZ;

		pX += (shoulder_length / 2.0f + dis_spine_shoulder_x);
		pZ += (trunk_length - dis_spine_shoulder_z);
		createBodyi(BodyName::BodyRightShoulder, shoulder_length, handsWidth);
		odeBodySetPosition(bodyIDs[BodyName::BodyRightShoulder], pX, pY, pZ);
		createJointType(pX - (shoulder_length / 2.0f), pY, pZ, BodyName::BodyTrunk, BodyName::BodyRightShoulder);
		cJointID = createJointType(pX - (shoulder_length / 2.0f), pY, pZ, BodyName::BodyTrunk, BodyName::BodyRightShoulder, JointType::ThreeDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyRightShoulder);
		setJointID(BodyName::BodyRightShoulder, cJointID, JointType::ThreeDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyRightShoulder] = BodyName::BodyTrunkUpper;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyRightShoulder)], unsigned long(0x0100));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyRightShoulder)], unsigned long(0x7D7F));

		// right arm
		pX += (shoulder_length / 2 + arm_length / 2);
		createBodyi(BodyName::BodyRightArm, arm_length, handsWidth);
		odeBodySetPosition(bodyIDs[BodyName::BodyRightArm], pX, pY, pZ);
		cJointID = createJointType(pX - (arm_length / 2.0f), pY, pZ, BodyName::BodyRightShoulder, BodyName::BodyRightArm, JointType::OneDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyRightArm);
		setJointID(BodyName::BodyRightArm, cJointID, JointType::OneDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyRightArm] = BodyName::BodyRightShoulder;

		odeGeomSetCategoryBits(mGeomID[BodyName::BodyRightArm], unsigned long(0x0200));
		odeGeomSetCollideBits (mGeomID[BodyName::BodyRightArm], unsigned long(0x7AFF));

		// right hand end point
		pX += (arm_length / 2 + (hand_length / 2.0f));
		createBodyi(BodyName::BodyRightHand, hand_length, handsWidth);
		odeBodySetPosition(bodyIDs[BodyName::BodyRightHand], pX, pY, pZ);
		createJointType(pX - (hand_length / 2.0f), pY, pZ, BodyName::BodyRightArm, BodyName::BodyRightHand);
		cJointID = createJointType(pX - (hand_length / 2.0f), pY, pZ, BodyName::BodyRightArm, BodyName::BodyRightHand, JointType::ThreeDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyRightHand);
		setJointID(BodyName::BodyRightHand, cJointID, JointType::ThreeDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyRightHand] = BodyName::BodyRightArm;

		odeGeomSetCategoryBits(mGeomID[BodyName::BodyRightHand], unsigned long(0x0400));
		odeGeomSetCollideBits (mGeomID[BodyName::BodyRightHand], unsigned long(0x7DFF));

//		_feetHandsWidth[3] = handsWidth;

		// head
		pX = trunkPosX;
		pZ = trunkPosZ;

		pZ += (trunk_length + head_length / 2 + dis_spine_head_z);
		createBodyi(BodyName::BodyHead, HeadWidth, head_length);
		odeBodySetPosition(bodyIDs[BodyName::BodyHead], pX, pY, pZ);
		createJointType(pX, pY, pZ - (head_length / 2.0f), BodyName::BodyTrunk, BodyName::BodyHead);
		cJointID = createJointType(pX, pY, pZ - (head_length / 2.0f), BodyName::BodyTrunk, BodyName::BodyHead, JointType::ThreeDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyHead);
		setJointID(BodyName::BodyHead, cJointID, JointType::ThreeDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyHead] = BodyName::BodyTrunkUpper;

		odeGeomSetCategoryBits(mGeomID[BodyName::BodyHead], unsigned long(0x4000));
		odeGeomSetCollideBits (mGeomID[BodyName::BodyHead], unsigned long(0x7F7F));

		float mass = 0;
		for (int i = 0; i < BodyNUM; i++)
		{
			mass += odeBodyGetMass(bodyIDs[i]);
		}
		float scaleFactor = dmass / mass;
		for (int i = 0; i < BodyNUM; i++)
		{
			float mass_i = odeBodyGetMass(bodyIDs[i]);
			float length,radius;
			odeGeomCapsuleGetParams(mGeomID[i],radius,length);
			odeMassSetCapsuleTotal(bodyIDs[i],mass_i*scaleFactor,radius,length);
		}

		return;
	}

	void setJointID(BodyName iBodyName, int iJointID, int iJointType, std::vector<Vector2>& iJointLimits)
	{
		jointIDs[iBodyName - 1] = iJointID;
		jointTypes[iBodyName - 1] = iJointType;

		if (iJointType == JointType::ThreeDegreeJoint)
		{
			jointIDIndex.push_back(iBodyName - 1);
			jointIDIndex.push_back(iBodyName - 1);
			jointIDIndex.push_back(iBodyName - 1);

			jointAxisIndex.push_back(0);
			jointAxisIndex.push_back(1);
			jointAxisIndex.push_back(2);

			jointLimits.push_back(iJointLimits[0]);
			jointLimits.push_back(iJointLimits[1]);
			jointLimits.push_back(iJointLimits[2]);
		}
		else
		{
			jointIDIndex.push_back(iBodyName - 1);
			jointAxisIndex.push_back(0);
			jointLimits.push_back(iJointLimits[0]);
		}
	}
	
	std::vector<int> createBodyi(int i, float lx, float lz, BodyType bodyType = BodyType::BodyCapsule)
	{
		int bodyID = odeBodyCreate();
		if (i < BodyNUM && i >= 0)
		{
			bodyIDs[i] = bodyID;
			bodyTypes[i] = bodyType;
		}

		float m_body_size = lz;
		float m_body_width = lx;
		if (i == BodyName::BodyLeftArm || i == BodyName::BodyLeftShoulder || i == BodyName::BodyRightArm || i == BodyName::BodyRightShoulder 
			|| i == BodyName::BodyRightHand || i == BodyName::BodyLeftHand)
		{
			m_body_size = lx;
			m_body_width = lz;
		}
		if (i < BodyNUM && i >= 0)
		{
			boneSize[i] = m_body_size;
		}

		int cGeomID = -1;

		if (bodyType == BodyType::BodyBox)
		{
			odeMassSetBox(bodyID, DENSITY, m_body_width, boneRadius, m_body_size);
			cGeomID = odeCreateBox(m_body_width, boneRadius, m_body_size);
		}
		else if (bodyType == BodyType::BodyCapsule)
		{
			m_body_width *= 0.5f;
			
			if (i == BodyName::BodyLeftArm || i == BodyName::BodyLeftShoulder || i == BodyName::BodyRightArm || i == BodyName::BodyRightShoulder 
				|| i == BodyName::BodyRightHand || i == BodyName::BodyLeftHand)
			{
				dMatrix3 R;
				dRFromAxisAndAngle(R, 0, 1, 0, PI /2);
				odeBodySetRotation(bodyID, R);
			}
			if (i == BodyName::BodyRightFoot || i == BodyName::BodyLeftFoot)
			{
				dMatrix3 R;
				dRFromAxisAndAngle(R, 1, 0, 0, PI /2);
				odeBodySetRotation(bodyID, R);
			}
			odeMassSetCapsule(bodyID, DENSITY, m_body_width, m_body_size);
			cGeomID = odeCreateCapsule(0, m_body_width, m_body_size);
		}
		else
		{
			m_body_width *= 0.5;
			odeMassSetSphere(bodyID, DENSITY / 10, m_body_width);
			cGeomID = odeCreateSphere(m_body_width);
		}

		if (i < BodyNUM && i >= 0)
		{
			mGeomID[i] = cGeomID;
		}

		if (i >= 0)
		{
			odeGeomSetCollideBits (cGeomID, 1 << (i + 1)); 
			odeGeomSetCategoryBits (cGeomID, 1 << (i + 1)); 
		}

		odeGeomSetBody(cGeomID, bodyID);

		std::vector<int> ret_val;
		ret_val.push_back(bodyID);
		ret_val.push_back(cGeomID);
		if (i < BodyNUM && i >= 0)
		{
			initialRotations[i]=ode2eigenq(odeBodyGetQuaternion(bodyID));
		}

		return ret_val;
	}

	std::vector<Vector2> setAngleLimitations(int jointID, BodyName iBodyName)
	{
		float hipSwingFwd = convertToRad(130.0f);
        float hipSwingBack = convertToRad(20.0f);
        float hipSwingOutwards = convertToRad(70.0f);
        float hipSwingInwards = convertToRad(15.0f);
        float hipTwistInwards = convertToRad(15.0f);
        float hipTwistOutwards = convertToRad(45.0f);
            
		float shoulderSwingFwd = convertToRad(160.0f);
        float shoulderSwingBack = convertToRad(20.0f);
        float shoulderSwingOutwards = convertToRad(30.0f);
        float shoulderSwingInwards = convertToRad(100.0f);
        float shoulderTwistUp = convertToRad(80.0f);		//in t-pose, think of bending elbow so that hand points forward. This twist direction makes the hand go up
        float shoulderTwistDown = convertToRad(20.0f);

		float spineSwingSideways = convertToRad(20.0f);
        float spineSwingForward = convertToRad(40.0f);
        float spineSwingBack = convertToRad(10.0f);
        float spineTwist = convertToRad(30.0f);
        

		float fwd_limit = 30.0f * (PI / 180.0f);
		float tilt_limit = 10.0f * (PI / 180.0f);
		float twist_limit = 45.0f * (PI / 180.0f);
		/*float wristSwingFwd = 15.0f;
        float wristSwingBack = 15.0f;
        float wristSwingOutwards = 70.0f;
        float wristSwingInwards = 15.0f;
        float wristTwistRange = 30.0f;
        float ankleSwingRange = 30.0f;
        float kneeSwingRange = 140.0f;*/

		std::vector<Vector2> cJointLimits;
		const float elbowStraightLimit=AaltoGames::deg2rad*1.0f;
		const float kneeStraightLimit=AaltoGames::deg2rad*2.0f;
		const float elbowKneeBentLimit=deg2rad*150.0f;
		switch (iBodyName)
		{
		case BodyName::BodySpine:
			odeJointSetAMotorParam(jointID, dParamLoStop1, -spineTwist); // x
			odeJointSetAMotorParam(jointID, dParamHiStop1, spineTwist);

			cJointLimits.push_back(Vector2(-spineTwist,spineTwist));

			odeJointSetAMotorParam(jointID, dParamLoStop2, -spineSwingSideways); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, spineSwingSideways);

			cJointLimits.push_back(Vector2(-spineSwingSideways,spineSwingSideways));

			odeJointSetAMotorParam(jointID, dParamLoStop3, -spineSwingForward); // z
			odeJointSetAMotorParam(jointID, dParamHiStop3, spineSwingBack);

			cJointLimits.push_back(Vector2(-spineSwingForward,spineSwingBack));
			break;
		case BodyName::BodyLeftThigh:
			odeJointSetAMotorParam(jointID, dParamLoStop1, -hipSwingOutwards); // z
			odeJointSetAMotorParam(jointID, dParamHiStop1, hipSwingInwards);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop1),odeJointGetAMotorParam(jointID,dParamHiStop1)));

			odeJointSetAMotorParam(jointID, dParamLoStop2, -hipTwistOutwards); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, hipTwistInwards);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop2),odeJointGetAMotorParam(jointID,dParamHiStop2)));

			odeJointSetAMotorParam(jointID, dParamLoStop3, -hipSwingFwd); // x
			odeJointSetAMotorParam(jointID, dParamHiStop3, hipSwingBack);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop3),odeJointGetAMotorParam(jointID,dParamHiStop3)));
			break;
		case BodyName::BodyRightThigh:
			odeJointSetAMotorParam(jointID, dParamLoStop1, -hipSwingInwards); // z
			odeJointSetAMotorParam(jointID, dParamHiStop1, hipSwingOutwards);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop1),odeJointGetAMotorParam(jointID,dParamHiStop1)));

			odeJointSetAMotorParam(jointID, dParamLoStop2, -hipTwistInwards); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, hipTwistOutwards);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop2),odeJointGetAMotorParam(jointID,dParamHiStop2)));

			odeJointSetAMotorParam(jointID, dParamLoStop3, -hipSwingFwd); // x
			odeJointSetAMotorParam(jointID, dParamHiStop3, hipSwingBack);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop3),odeJointGetAMotorParam(jointID,dParamHiStop3)));
			break;
		case BodyName::BodyLeftShoulder:
			odeJointSetAMotorParam(jointID, dParamLoStop1, -shoulderSwingOutwards); // z
			odeJointSetAMotorParam(jointID, dParamHiStop1, shoulderSwingInwards);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop1),odeJointGetAMotorParam(jointID,dParamHiStop1)));

			odeJointSetAMotorParam(jointID, dParamLoStop2, -shoulderTwistDown); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, shoulderTwistUp);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop2),odeJointGetAMotorParam(jointID,dParamHiStop2)));

			odeJointSetAMotorParam(jointID, dParamLoStop3, -shoulderSwingBack); // x
			odeJointSetAMotorParam(jointID, dParamHiStop3, shoulderSwingFwd);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop3),odeJointGetAMotorParam(jointID,dParamHiStop3)));
			break;
		case BodyName::BodyRightShoulder:
			odeJointSetAMotorParam(jointID, dParamLoStop1, -shoulderSwingInwards); // z
			odeJointSetAMotorParam(jointID, dParamHiStop1, shoulderSwingOutwards);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop1),odeJointGetAMotorParam(jointID,dParamHiStop1)));

			odeJointSetAMotorParam(jointID, dParamLoStop2, -shoulderTwistDown); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, shoulderTwistUp);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop2),odeJointGetAMotorParam(jointID,dParamHiStop2)));

			odeJointSetAMotorParam(jointID, dParamLoStop3, -shoulderSwingFwd); // x
			odeJointSetAMotorParam(jointID, dParamHiStop3, shoulderSwingBack);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop3),odeJointGetAMotorParam(jointID,dParamHiStop3)));
			break;
		case BodyName::BodyLeftLeg:
			odeJointSetHingeParam(jointID, dParamLoStop, kneeStraightLimit);
			odeJointSetHingeParam(jointID, dParamHiStop, elbowKneeBentLimit);

			cJointLimits.push_back(Vector2(kneeStraightLimit,elbowKneeBentLimit));
			break;
		case BodyName::BodyRightLeg:
			odeJointSetHingeParam(jointID, dParamLoStop, kneeStraightLimit);
			odeJointSetHingeParam(jointID, dParamHiStop, elbowKneeBentLimit);

			cJointLimits.push_back(Vector2(kneeStraightLimit,elbowKneeBentLimit));
			break;
		case BodyName::BodyLeftArm:
			odeJointSetHingeParam(jointID, dParamLoStop, elbowStraightLimit);
			odeJointSetHingeParam(jointID, dParamHiStop, elbowKneeBentLimit);

			cJointLimits.push_back(Vector2(elbowStraightLimit,elbowKneeBentLimit));
			break;
		case BodyName::BodyRightArm:
			odeJointSetHingeParam(jointID, dParamLoStop, -elbowKneeBentLimit);
			odeJointSetHingeParam(jointID, dParamHiStop, -elbowStraightLimit);

			cJointLimits.push_back(Vector2(-elbowKneeBentLimit,-elbowStraightLimit));
			break;
		case BodyName::BodyHead:
			

			odeJointSetAMotorParam(jointID, dParamLoStop1, -fwd_limit); // x
			odeJointSetAMotorParam(jointID, dParamHiStop1, fwd_limit);

			cJointLimits.push_back(Vector2(-fwd_limit,fwd_limit));

			

			odeJointSetAMotorParam(jointID, dParamLoStop2, -tilt_limit); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, tilt_limit);

			cJointLimits.push_back(Vector2(-tilt_limit,tilt_limit));

			

			odeJointSetAMotorParam(jointID, dParamLoStop3, -twist_limit); // z
			odeJointSetAMotorParam(jointID, dParamHiStop3, twist_limit);

			cJointLimits.push_back(Vector2(-twist_limit,twist_limit));
			break;
		case BodyName::BodyRightFoot:
			odeJointSetHingeParam(jointID, dParamLoStop, -15 * (PI / 180.0f));
			odeJointSetHingeParam(jointID, dParamHiStop, 45 * (PI / 180.0f));

			//odeJointSetAMotorParam(jointID, dParamLoStop1, -PI / 4); // x
			//odeJointSetAMotorParam(jointID, dParamHiStop1, PI / 4);

			//cJointLimits.push_back(Vector2(-PI / 4,PI / 4));

			//odeJointSetAMotorParam(jointID, dParamLoStop2, -PI / 4); // y
			//odeJointSetAMotorParam(jointID, dParamHiStop2, PI / 4);

			//cJointLimits.push_back(Vector2(-PI / 4,PI / 4));

			//odeJointSetAMotorParam(jointID, dParamLoStop3, -PI / 4); // z
			//odeJointSetAMotorParam(jointID, dParamHiStop3, PI / 4);

			cJointLimits.push_back(Vector2(-15 * (PI / 180.0f),45 * (PI / 180.0f)));
			break;
		case BodyName::BodyLeftHand:
			odeJointSetAMotorParam(jointID, dParamLoStop1, -PI / 4); // x
			odeJointSetAMotorParam(jointID, dParamHiStop1, PI / 4);

			cJointLimits.push_back(Vector2(-PI / 4,PI / 4));

			odeJointSetAMotorParam(jointID, dParamLoStop2, -PI / 4); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, PI / 4);

			cJointLimits.push_back(Vector2(-PI / 4,PI / 4));

			odeJointSetAMotorParam(jointID, dParamLoStop3, -PI / 4); // z
			odeJointSetAMotorParam(jointID, dParamHiStop3, PI / 4);

			cJointLimits.push_back(Vector2(-PI / 4,PI / 4));
			break;
		case BodyName::BodyRightHand:
			odeJointSetAMotorParam(jointID, dParamLoStop1, -PI / 4); // x
			odeJointSetAMotorParam(jointID, dParamHiStop1, PI / 4);

			cJointLimits.push_back(Vector2(-PI / 4,PI / 4));

			odeJointSetAMotorParam(jointID, dParamLoStop2, -PI / 4); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, PI / 4);

			cJointLimits.push_back(Vector2(-PI / 4,PI / 4));

			odeJointSetAMotorParam(jointID, dParamLoStop3, -PI / 4); // z
			odeJointSetAMotorParam(jointID, dParamHiStop3, PI / 4);

			cJointLimits.push_back(Vector2(-PI / 4,PI / 4));
			break;
		case BodyName::BodyLeftFoot:
			odeJointSetHingeParam(jointID, dParamLoStop, -15 * (PI / 180.0f));
			odeJointSetHingeParam(jointID, dParamHiStop, 45 * (PI / 180.0f));
			//odeJointSetAMotorParam(jointID, dParamLoStop1, -PI / 4); // x
			//odeJointSetAMotorParam(jointID, dParamHiStop1, PI / 4);

			//cJointLimits.push_back(Vector2(-PI / 4,PI / 4));

			//odeJointSetAMotorParam(jointID, dParamLoStop2, -PI / 4); // y
			//odeJointSetAMotorParam(jointID, dParamHiStop2, PI / 4);

			//cJointLimits.push_back(Vector2(-PI / 4,PI / 4));

			//odeJointSetAMotorParam(jointID, dParamLoStop3, -PI / 4); // z
			//odeJointSetAMotorParam(jointID, dParamHiStop3, PI / 4);

			cJointLimits.push_back(Vector2(-15 * (PI / 180.0f),45 * (PI / 180.0f)));
			break;
		default:
			odeJointSetAMotorParam(jointID, dParamLoStop2, -PI / 2); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, PI / 2);
		}

		return cJointLimits;
	}

	float convertToRad(float iDegree)
	{
		return iDegree * (PI / 180.0f);
	}

	//////////////////////////////////////////// variables /////////////////////////////////////////////

	int goal_hold_id;

	// for drawing of Env bodies
	int startingHoldGeomsIndex;
	std::vector<int> mENVGeoms;
	std::vector<int> mENVGeomTypes;

	// variables for creating joint in hold positions
	std::vector<std::vector<int>> jointHoldBallIDs; // 4 set of ball joints IDs

	// variable refers to hold pos index
	std::vector<std::vector<int>> holdPosIndex; // only 4 hold IDs for each context

	// for drawing humanoid body
	std::vector<int> bodyIDs;
	std::vector<int> fatherBodyIDs;
	std::vector<int> mGeomID;
	std::vector<int> bodyTypes;
	std::vector<float> mColorBodies;
	// variable for getting end point of the body i
	std::vector<float> boneSize;
	
	std::vector<int> jointIDs;
	std::vector<int> jointTypes;
	int mJointSize;
	std::vector<int> jointIDIndex;
	std::vector<int> jointAxisIndex;
	std::vector<Vector2> jointLimits;

//	std::vector<float> _feetHandsWidth;

	// for testing the humanoid climber
	std::vector<float> desiredAnglesBones; 

	int masterContext;
	int maxNumContexts; 
	int currentFreeSavingStateSlot;

	// climber's info
	float climberRadius; // maximum dis between hand and foot
	float climberLegLegDis; // maximum dis between leg to leg
	float climberHandHandDis; // masimum dis between hand to hand
	float climberHeight; // climber's height from its feet to head

}* mContext;

#define BipedState SimulationContext::BipedState