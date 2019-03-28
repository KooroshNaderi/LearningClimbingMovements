/*

Part of Aalto University Game Tools. See LICENSE.txt for licensing info. 

*/

#ifndef INDEXRANDOMIZER_H
#define INDEXRANDOMIZER_H
#include <vector>

namespace AaltoGames
{

	class IndexRandomizer
	{
	public:
		IndexRandomizer(size_t N);
		IndexRandomizer();
		void init(size_t N);
		int get();
	private:
		std::vector<int> indices;
		size_t nLeft;
		size_t usedSize;
	};
} //namespace AaltoGames


#endif