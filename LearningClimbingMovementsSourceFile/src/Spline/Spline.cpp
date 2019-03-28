#include "Spline.h"

RecursiveTCBSpline::RecursiveTCBSpline()
{
	current = dCurrent = 0;
}

RecursiveTCBSpline::~RecursiveTCBSpline()
{
}

void RecursiveTCBSpline::setValueAndTangent(float current, float dCurrent)
{
	this->current = current;
	this->dCurrent = dCurrent;
}
float RecursiveTCBSpline::getValue()
{
	return this->current;
}
void RecursiveTCBSpline::setValue(float current)
{
	this->current = current;
}
//incoming tangent at p1, based on three points p0,p1,p2 and their times t0,t1,t2
float RecursiveTCBSpline::TCBIncomingTangent(float p0, float t0, float p1, float t1, float p2, float t2)
{
	return 0.5f*(1 - t)*(1 + c)*(1 - b)*(p2 - p1) / (t2 - t1) + 0.5f*(1 - t)*(1 - c)*(1 + b)*(p1 - p0) / (t1 - t0);
}

float RecursiveTCBSpline::TCBOutgoingTangent(float p0, float t0, float p1, float t1, float p2, float t2)
{
	return 0.5f*(1 - t)*(1 - c)*(1 - b)*(p2 - p1) / (t2 - t1) + 0.5f*(1 - t)*(1 + c)*(1 + b)*(p1 - p0) / (t1 - t0);
}

// p0 y coordinate of the current point
// p1 y coordinate of the next point
// s interpolation value, (t-t0)/timeDelta
// to0 tangent out of (p0)
// ti1 tangent in of (p1)
float RecursiveTCBSpline::InterpolateHermitian(float p0, float p1, float s, float to0, float ti1, float timeDelta)
{
	float s2 = s*s;
	float s3 = s2*s;
	//the tcb formula using a Hermite interpolation basis from Eberly
	return (2 * s3 - 3 * s2 + 1)*p0 + (-2 * s3 + 3 * s2)*p1 + (s3 - 2 * s2 + s)*timeDelta*to0 + (s3 - s2)*timeDelta*ti1;
}

float RecursiveTCBSpline::InterpolateHermitianTangent(float p0, float p1, float s, float to0, float ti1, float timeDelta)
{
	float s2 = s*s;
	return (6 * s2 - 6 * s) * p0 + (-6 * s2 + 6 * s) * p1 + (3 * s2 - 4 * s + 1) * timeDelta * to0 + (3 * s2 - 2 * s) * timeDelta * ti1;
}

//Step the curve forward using the internally stored current value-tangent pair current, dCurrent, and the next two points p1,p2 at times t1,t2.
void RecursiveTCBSpline::step(float timeStep, float p1, float t1, float p2, float t2)
{
	float epsilon = 1e-10f;
	t1 = t1 <= epsilon ? epsilon : t1; //to prevent NaN
	t2 = t2 <= epsilon ? epsilon : t2; //to prevent NaN
	
	float newLinearVal = current + (p1 - current) * timeStep / t1;
	float newLinearTangent = (p1 - current) / t1;
	if (linearMix >= 1)
	{
		current = newLinearVal;
		dCurrent = newLinearTangent;
	}
	else
	{
		float p1IncomingTangent = TCBIncomingTangent(current, 0, p1, t1, p2, t2);
		float newTCBVal = InterpolateHermitian(current, p1, timeStep / t1, dCurrent, p1IncomingTangent, t1);
		float newTCBTangent = InterpolateHermitianTangent(current, p1, timeStep / t1, dCurrent, p1IncomingTangent, t1) / t1;
		current = linearMix * newLinearVal + (1.0f - linearMix) * newTCBVal;
		dCurrent = linearMix * newLinearTangent + (1.0f - linearMix) * newTCBTangent;
	}
}

void RecursiveTCBSpline::save()
{
	savedCurrent = current;
	savedDCurrent = dCurrent;
}
void RecursiveTCBSpline::restore()
{
	current = savedCurrent;
	dCurrent = savedDCurrent;
}
void RecursiveTCBSpline::copyStateFrom(RecursiveTCBSpline src)
{
	current = src.current;
	dCurrent = src.dCurrent;
}

//==============================================================================================

ScrollingSpline::ScrollingSpline(int nControlPoints)
{
	this->nControlPoints = nControlPoints;
	values.resize(nControlPoints);
	times.resize(nControlPoints);
	currentTime = 0;
	controlPointIdx = 1;
	spline = new RecursiveTCBSpline();
}

ScrollingSpline::~ScrollingSpline()
{
	delete spline;
}

bool ScrollingSpline::advanceEvalTime(float timeStep)
{
	
	//float partialStep = std::min(timeToNext, timeStep);

	int index_i = std::min(controlPointIdx, nControlPoints - 1);
	int index_i_1 = std::min(controlPointIdx + 1, nControlPoints - 1);

	float timeToNext = times[index_i] - currentTime;

	if (timeToNext > 0.001f)
	{
		spline->step(timeStep, values[index_i], times[index_i] - currentTime, values[index_i_1], times[index_i_1] + times[index_i] - currentTime);
		currentTime += timeStep;
	}
	else
	{
		currentTime = 0;
		controlPointIdx++;

		index_i = std::min(controlPointIdx, nControlPoints - 1);
		index_i_1 = std::min(controlPointIdx + 1, nControlPoints - 1);

		spline->step(timeStep, values[index_i], times[index_i] - currentTime, values[index_i_1], times[index_i_1] + times[index_i] - currentTime);
	}

	if (controlPointIdx >= nControlPoints - 1)
	{
		return false;
	}

	return true;

	/*float timeToNext = times[controlPointIdx] - currentTime;
	float partialStep = std::min(timeToNext, timeStep);

	int index_i = std::min(controlPointIdx, nControlPoints - 1);
	int index_i_1 = std::min(controlPointIdx + 1, nControlPoints - 1);

	spline->step(partialStep, values[index_i], times[index_i] - currentTime, values[index_i_1], times[index_i_1] + times[index_i] - currentTime);
	currentTime += partialStep;

	if (timeStep >= timeToNext)
	{
		controlPointIdx++;

		index_i = std::min(controlPointIdx, nControlPoints - 1);
		index_i_1 = std::min(controlPointIdx + 1, nControlPoints - 1);

		float remaining = timeStep - partialStep;
		if (remaining > 0.00001f)
		{
			spline->step(remaining, values[index_i], times[index_i], values[index_i_1], times[index_i_1] + times[index_i]);
		}

		if (controlPointIdx == nControlPoints - 1)
		{
			return false;
		}
		currentTime = remaining;
	}
	return true;*/
}
float ScrollingSpline::eval()
{
	return spline->current;
}
float ScrollingSpline::dEval()
{
	return spline->dCurrent;
}
float ScrollingSpline::evalTime()
{
	return currentTime;
}
void ScrollingSpline::setState(float current, float dCurrent)
{
	spline->setValueAndTangent(current, dCurrent);
	spline->save();
}
void ScrollingSpline::resetEvalTime()
{
	spline->restore();
	currentTime = 0;
	controlPointIdx = 1;
}

//This will shift all control points closer in time. If the first control point time becomes <= 0, it is removed and the last control point is duplicated (and true is returned instead of false)
bool ScrollingSpline::shift(float timeStep)
{
	//to make the evaluation work, we first have to make the next evaluation start from one timestep ahead
	resetEvalTime();
	advanceEvalTime(timeStep);
	spline->save();

	//now the actual 
	times[0] -= timeStep;
	bool pointRemoved = false;
	if (times[0] <= 0)
	{
		for (int i = 0; i < nControlPoints - 1; i++)
		{
			times[i] = times[i + 1];
			values[i] = values[i + 1];
		}
		pointRemoved = true;
	}
	return pointRemoved;
}

//==============================================================================================

SplineSampler::SplineSampler(int nSplines, int nControlPoints, int nDimensions, float horizon, float timeStep, VectorXf controlMin, VectorXf controlMax, VectorXf posePrior)
{
	this->nDimensions = nDimensions;
	this->nSplines = nSplines;
	this->nControlPoints = nControlPoints;
	this->horizon = horizon;
	this->timeStep = timeStep;
	this->controlMin = controlMin;
	this->controlMax = controlMax;
	this->controlMean = posePrior;
	this->controlSd = (controlMax - controlMin) / 4.0f;
	this->timeStd = horizon / nControlPoints / 2.0f;

	splines.resize(nDimensions);
	for (int i = 0; i < nDimensions; i++)
	{
		splines[i].resize(nSplines);
		for (int j = 0; j < nSplines; j++)
		{
			splines[i][j] = new ScrollingSpline(nControlPoints);
			splines[i][j]->spline->t = t;
			splines[i][j]->spline->linearMix = linearMix;
			splines[i][j]->setState(0, 0);
		}
	}
	oldBestSpline = -1;
}

SplineSampler::~SplineSampler()
{
	for (int i = 0; i < nDimensions; i++)
		for (int j = 0; j < nSplines; j++)
			delete splines[i][j];
}

void SplineSampler::ResampleSplines(float oldBestResampleRatio)
{
	VectorXf bestResampleCount = VectorXf::Ones(nDimensions) * (int)(oldBestResampleRatio * nSplines);

	if (oldBestSpline < 0) //In the first frame we don't have any old best spline to resample from.
		for (int d = 0; d < nDimensions; d++)
			bestResampleCount[d] = 0;

	for (int d = 0; d < nDimensions; d++)
	{
		for (int s = 0; s < nSplines; s++)
		{
			if (s == oldBestSpline) continue;
			if (bestResampleCount[d] > 0)
			{
				//Resample according to the old best spline
				InitSpline(splines[d][s], d, true);
				bestResampleCount[d]--;
			}
			else
			{
				//Initalize randomly
				InitSpline(splines[d][s], d, false);
			}
		}
	}
	Reset();
}

void SplineSampler::Reset()
{
	oldBestSpline = -1;
	for(int d = 0; d < nDimensions; d++)
		splines[d][0]->setState(0, 0);
}

void SplineSampler::AddControlPointsToBestSpline(int d)
{
	ScrollingSpline *spline = splines[d][oldBestSpline];

	float t = 0;
	BoxMuller(&t, 1);
	spline->times[nControlPoints - 1] = 0.1f + abs(timeStd * t + horizon / (nControlPoints - 1));

	BoxMuller(&t, 1);
	spline->values[nControlPoints - 1] = spline->values[nControlPoints - 1] * controlSd[d];
	//spline->values[nControlPoints - 1] += controlMean[d];
	spline->values[nControlPoints - 1] += spline->values[nControlPoints - 2];
	LimitSpline(spline, d, nControlPoints - 1);
}

void SplineSampler::InitSpline(ScrollingSpline *spline, int d, bool resampleOldBest)
{
	float v = 0, t = 0;
	for (int c = 0; c < nControlPoints; c++)
	{
		BoxMuller(&v, 1);
		BoxMuller(&t, 1);
		if (resampleOldBest)
		{
			//Initalize according to the old best spline
			ScrollingSpline *src = splines[d][oldBestSpline];
			spline->times[c] = (t * timeStd / 2.0f) + src->times[c];

			spline->values[c] = (v * controlSd[d] / 2.0f) + src->values[c];

			src->resetEvalTime();
			spline->setState(src->eval(), src->dEval());
		}
		else
		{
			//Initalize randomly (first frame of a new episode)
			if (c == 0)
			{
				spline->times[0] = 0;
				spline->values[0] = 0;
			}
			else
			{
				spline->times[c] = (t * timeStd) + horizon / (nControlPoints - 1);
				spline->values[c] = (v * controlSd[d]) + controlMean[d];
			}
			spline->setState(0, 0);
		}
		LimitSpline(spline, d, c);
	}
	spline->resetEvalTime();
}

void SplineSampler::LimitSpline(ScrollingSpline *spline, int d, int c)
{
	if (c == -1)
	{
		for (c = 0; c < nControlPoints; c++)
		{
			spline->values[c] = std::min(spline->values[c], controlMax[d]);
			spline->values[c] = std::max(spline->values[c], controlMin[d]);
			spline->times[c] = std::max(spline->times[c], 0.1f);
		}
	}
	else
	{
		spline->values[c] = std::min(spline->values[c], controlMax[d]);
		spline->values[c] = std::max(spline->values[c], controlMin[d]);
		spline->times[c] = std::max(spline->times[c], 0.1f);
	}
}