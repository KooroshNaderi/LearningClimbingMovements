/*

Part of Aalto University Game Tools. See LICENSE.txt for licensing info. 

*/

#include "IndexRandomizer.h"
#include "MathUtils.h"

namespace AaltoGames
{


	int IndexRandomizer::get()
	{
		int tableIdx=randInt(0,nLeft-1);
		int result=indices[tableIdx];
		_swap(indices[tableIdx],indices[nLeft-1]);
		nLeft--;
		if (nLeft==0)
			init(usedSize);
		return result;
	}

	void IndexRandomizer::init(size_t N)
	{
		if (indices.size()<N)
			indices.resize(N);
		for (size_t i=0; i<N; i++)
		{
			indices[i]=i;
		}
		usedSize=N;
		nLeft=N;
	}

	IndexRandomizer::IndexRandomizer(size_t N)
	{
		init(N);
	}

	IndexRandomizer::IndexRandomizer()
	{

	}
}