#include <cmath>
#include "../mathutils.h"
#include "../ProbUtils.hpp"
#include <vector>
#include <Eigen/Geometry>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include "../FileUtils.h"

using namespace std::chrono;
using namespace AaltoGames;
using namespace Eigen;

/*
* See:
* http://www.geometrictools.com/Documentation/KBSplines.pdf
* http://www.gamedev.net/topic/500802-tcb-spline-interpolation/
* */

class RecursiveTCBSpline
{
public:
	RecursiveTCBSpline();
	~RecursiveTCBSpline();
	void setValueAndTangent(float current, float dCurrent);
	void setValue(float current);
	float getValue();
	void step(float timeStep, float p1, float t1, float p2, float t2);
	void save();
	void restore();
	void copyStateFrom(RecursiveTCBSpline src);

	float t = 0;
	float linearMix = 0;
	float current, dCurrent;
private:
	float TCBIncomingTangent(float p0, float t0, float p1, float t1, float p2, float t2);
	float TCBOutgoingTangent(float p0, float t0, float p1, float t1, float p2, float t2);
	float InterpolateHermitian(float p0, float p1, float s, float to0, float ti1, float timeDelta);
	float InterpolateHermitianTangent(float p0, float p1, float s, float to0, float ti1, float timeDelta);

	float savedCurrent, savedDCurrent;
	//Default tcb values 0 => catmull-rom spline
	float c = 0;   //const, because we need to have tangent continuity (incoming = outgoing) for the recursion to work as implemented
	float b = 0;
};

//==============================================================================================

class ScrollingSpline
{
public:
	ScrollingSpline(int nControlPoints);
	~ScrollingSpline();
	bool advanceEvalTime(float timeStep);
	float eval();
	float dEval();
	float evalTime();
	void setState(float current, float dCurrent);
	void resetEvalTime();
	bool shift(float timeStep);

	int nControlPoints;
	std::vector<float> values, times;
	RecursiveTCBSpline *spline;

private:
	float currentTime;
	int controlPointIdx;
};

//==============================================================================================

class SplineSampler
{
public:
	SplineSampler(int nSplines, int nControlPoints, int nDimensions, float horizon, float timeStep, VectorXf controlMin, VectorXf controlMax, VectorXf posePrior);
	~SplineSampler();
	void ResampleSplines(float oldBestResampleRatio);
	void Reset();
	void AddControlPointsToBestSpline(int d);

	std::vector<std::vector<ScrollingSpline *>> splines;
	int oldBestSpline;
	int nDimensions, nSplines, nControlPoints;
	float horizon, timeStep;
private:
	void InitSpline(ScrollingSpline *spline, int d, bool resampleOldBest);
	void LimitSpline(ScrollingSpline *spline, int d, int c = -1);

	const float t = 0;
	const float linearMix = 0;
	VectorXf controlMin, controlMax, controlMean, controlSd;
	float timeStd;
};