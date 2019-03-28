class mTools
{
public:
	static float dotProduct(Vector3 v0, Vector3 v1)
	{
		v0.normalize();
		v1.normalize();
		return v0.x() * v1.x() + v0.y() * v1.y() + v0.z() * v1.z();
	}

	static float getAbsAngleBtwVectors(Vector3 v0, Vector3 v1)
	{
		if (v0.norm() > 0)
			v0.normalize();
		if (v1.norm() > 0)
			v1.normalize();

		float _dotProduct = v0.x() * v1.x() + v0.y() * v1.y() + v0.z() * v1.z();
		float angle = 0.0f;
		if (fabs(_dotProduct) < 1)
			angle = acosf(_dotProduct);

		return angle;
	}

	static float getAngleBtwVectorsXZ(Vector3 _v0, Vector3 _v1)
	{
		Vector2 v0(_v0.x(), _v0.z());
		Vector2 v1(_v1.x(), _v1.z());

		v0.normalize();
		v1.normalize();

		float angle = acosf(v0.x() * v1.x() + v0.y() * v1.y());

		float crossproduct = (v0.x() * v1.y() - v0.y() * v1.x()) * 1;// in direction of k

		if (crossproduct < 0)
			angle = -angle;

		return angle;
	}

	static Vector3 getDirectionFromAngles(float theta, float phi)
	{
		Vector3 dir;
		dir[0] = (float) (cosf (theta*DEG_TO_RAD) * cosf (phi*DEG_TO_RAD));
		dir[1] = (float) (sinf (theta*DEG_TO_RAD) * cosf (phi*DEG_TO_RAD));
		dir[2] = (float) (sinf (phi*DEG_TO_RAD));
		dir.normalize();
		return dir;
	}

	static bool isSetAGreaterThan(std::vector<int>& set_a, int f)
	{
		for (unsigned int j = 0; j < set_a.size(); j++)
		{
			if (set_a[j] <= f)
			{
				return false;
			}
		}
		return true;
	}

	static bool isSetAEqualsSetB(const std::vector<int>& set_a, const std::vector<int>& set_b)
	{
		//AALTO_ASSERT1(set_a.size()==4 && set_b.size()==4);
		if (set_a.size() != set_b.size())
			return false;

		if (set_a.size()==4 && set_b.size()==4)
		{
			int diff=0;
			diff+=abs(set_a[0]-set_b[0]);
			diff+=abs(set_a[1]-set_b[1]);
			diff+=abs(set_a[2]-set_b[2]);
			diff+=abs(set_a[3]-set_b[3]);
			return diff==0;
		}
		for (unsigned int i = 0; i < set_a.size(); i++)
		{
			if (set_a[i] != set_b[i])
			{
				return false;
			}
		}

		return true;

	}

	static int getDiffBtwSetASetB(std::vector<int>& set_a, std::vector<int>& set_b)
	{
		int mCount = 0;
		for (unsigned int i = 0; i < set_a.size(); i++)
		{
			if (set_a[i] != set_b[i])
			{
				mCount++;
			}
		}
		return mCount;
	}

	static float getRandomBetween_01()
	{
		return ((float)rand()) / (float)RAND_MAX;
	}
	
	static Vector3 getUniformRandomPointBetween(Vector3 iMin, Vector3 iMax)
	{
		//Random between Min, Max
		float r1 = mTools::getRandomBetween_01();
		float r2 = mTools::getRandomBetween_01();
		float r3 = mTools::getRandomBetween_01();
		Vector3 dis = iMax - iMin;

		return iMin + Vector3(dis.x() * r1, dis.y() * r2, dis.z() * r3); 
	}

	static int getRandomBetween(int a, int b)
	{
		return a + (std::rand() % (b - a + 1));
	}

	static float getRandomBetween(const float& a, const float& b)
	{
		return a + getRandomBetween_01() * (b - a);
	}

	static int getRandomIndex(unsigned int iArraySize)
	{
		if (iArraySize == 0)
			return -1;
		int m_index = rand() % iArraySize;

		if (m_index > (int)(iArraySize - 1))
			m_index = iArraySize - 1;

		return m_index;
	}

	static bool isInSetIDs(int _id, std::vector<int>& iSetIDs)
	{
		for (unsigned int i = 0; i < iSetIDs.size(); i++)
		{
			if (iSetIDs[i] == _id)
			{
				return true;
			}
		}
		return false;
	}
	
	static bool addToSetIDs(int _id, std::vector<int>& iSetIDs)
	{
		if (!isInSetIDs(_id, iSetIDs))
		{
			iSetIDs.push_back(_id);
			return true;
		}
		return false;
	}

	static void removeFromSetIDs(int _id, std::vector<int>& iSetIDs)
	{
		for (unsigned int i = 0; i < iSetIDs.size(); i++)
		{
			if (iSetIDs[i] == _id)
			{
				iSetIDs.erase(iSetIDs.begin() + i);
				return;
			}
		}
	}

	static bool isInSampledStanceSet(std::vector<int>& sample_i, std::vector<std::vector<int>>& nStances)
	{
		for (unsigned int i = 0; i < nStances.size(); i++)
		{
			std::vector<int> t_i = nStances[i];
			bool flag_found_try = isSetAEqualsSetB(t_i, sample_i);
			if (flag_found_try)
			{
				return true;
			}
		}
		return false;
	}
};