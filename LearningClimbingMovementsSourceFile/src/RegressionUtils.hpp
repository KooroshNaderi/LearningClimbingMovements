/*

Part of Aalto University Game Tools. See LICENSE.txt for licensing info. 

*/


#ifndef REGRESSION_UTILS_H
#define REGRESSION_UTILS_H

#include <Eigen/Eigen>
#include <exception>
#include <time.h>
#include <vector>
#include "MiscUtils.hpp"
#include "ProbUtils.hpp"

//The regularization that is used when inverting matrices if they are not invertible
#define REGULARIZATION(Scalar) std::numeric_limits<Scalar>::epsilon()
#define LOG_OF_TWO_TIMES_PI 1.837877066409345484

//template<typename Scalar>
////The vector elements will be inserted as matrix rows
//Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> standardVectorToMatrix(std::vector<Eigen::Matrix<Scalar,Eigen::Dynamic,1> >& input){
//	Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> output = Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Zero(input.size(),input[0].size());
//
//	for (int i = 0; i < input.size(); i++){
//		output.row(i) = input[i];
//	}
//
//	return output;
//
//}


//template <typename Scalar>
////Compute the covariance matrix given the possibly weighted observations in 'input' and the mean of the observations
////'input' has the observations in its rows
//Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> weightedCov(const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& input,const Eigen::Matrix<Scalar,Eigen::Dynamic,1>& means,const Eigen::Matrix<Scalar,Eigen::Dynamic,1>& weights = Eigen::Matrix<Scalar,Eigen::Dynamic,1>::Zero(0),const bool useRegularization=true){
//
//	assert(input.rows() > 1);
//
//	//Normalize the weights
//	Eigen::Matrix<Scalar,Eigen::Dynamic,1> weightsNormalized = weights;
//
//	if (weightsNormalized.size() == 0){
//		weightsNormalized = Eigen::Matrix<Scalar,Eigen::Dynamic,1>::Ones(input.rows());
//	}
//	normalizeProbabilities(weightsNormalized);
//
//	//Compute the distance to the mean
//	Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> inDiff = (input.rowwise() - means.transpose());
//
//	//Compute the distances weighted
//	Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> inDiffWeighted = ((weightsNormalized).asDiagonal()*inDiff).transpose();
//
//	//Compute the covariance matrix
//	Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> cov = inDiffWeighted*inDiff;
//
//	//Normalize the covariance
//	Scalar weightSqSum = weightsNormalized.cwiseAbs2().sum();
//	Scalar normalizationDenominator = (Scalar)1 - weightSqSum;
//	if (normalizationDenominator < std::numeric_limits<Scalar>::epsilon()){
//		normalizationDenominator = std::numeric_limits<Scalar>::epsilon();
//	}
//	cov = cov*( (Scalar)1 / normalizationDenominator );
//
//	assert(isFinite(cov));
//
//	if (useRegularization){
//		cov = eigenvalueClampingRegularization(cov);
//	}
//
//	return cov;
//
//}


template <typename Scalar>
//Compute the covariance matrix given the possibly weighted observations in 'input' and the mean of the observations
//'input' has the observations in its rows
Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> weightedCov(const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& input,const Eigen::Matrix<Scalar,Eigen::Dynamic,1>& means, Eigen::Matrix<Scalar,Eigen::Dynamic,1> weights = Eigen::Matrix<Scalar,Eigen::Dynamic,1>::Zero(0),const bool useRegularization=true){

	assert(input.rows() > 1);

	if (weights.size() == 0){
		weights = Eigen::Matrix<Scalar,Eigen::Dynamic,1>::Ones(input.rows());
	}
	//Normalize the weights
	normalizeProbabilities(weights);

	//Compute the distance to the mean
	Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> inDiff = (input.rowwise() - means.transpose());

	//Compute the covariance matrix
	Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> cov = (((weights).asDiagonal()*inDiff).transpose())*inDiff;

	//Normalize the covariance
	Scalar weightSqSum = weights.cwiseAbs2().sum();
	Scalar normalizationDenominator = (Scalar)1 - weightSqSum;
	if (normalizationDenominator < std::numeric_limits<Scalar>::epsilon()){
		normalizationDenominator = std::numeric_limits<Scalar>::epsilon();
	}
	cov = cov*( (Scalar)1 / normalizationDenominator );

	assert(isFinite(cov));

	if (useRegularization){
		cov = eigenvalueClampingRegularization(cov);
	}

	return cov;

}

template<typename Scalar>
//Compute the variance of each column the matrix.
//The ROWS of 'data' are the data items.
//The data may be weighted by the weights in 'weights'.
Eigen::Matrix<Scalar,Eigen::Dynamic,1> variance(const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& data, Eigen::Matrix<Scalar,Eigen::Dynamic,1> weights = Eigen::Matrix<Scalar,Eigen::Dynamic,1>::Zero(0)){
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> Vector;

	//If no weights are available, use equal weigting
	if (weights.size() == 0){
		weights = Vector::Ones(data.rows());
	}
	//Normalize the weights such that they sum up to one.
	normalizeProbabilities(weights);

	//Initialize a vector where we are going to store the variances.
	Vector var = Vector::Zero(data.cols());

	//Compute the mean of the data
	Matrix mean = weightedMean(data,weights);

	//Compute the variances
	for (int i = 0; i < data.rows();i++){

		//This line handles one data point i.e. all the columns of 'data' at once.
		var += (data.row(i).transpose() - mean).cwiseAbs2()*weights(i);

	}

	//Divide the variances by the effective number of data points
	Scalar weightSqSum = weights.cwiseAbs2().sum();
	Scalar normalizationDenominator = (Scalar)1 - weightSqSum;
	if (normalizationDenominator < std::numeric_limits<Scalar>::epsilon()){
		normalizationDenominator = std::numeric_limits<Scalar>::epsilon();
	}
	var = var*(((Scalar)1)/normalizationDenominator);

	//Check that the variances are finite numbers
	assert(isFinite(var));

	return var;
}

template<typename Scalar>
//Compute the variance of each column the matrix.
//The ROWS of 'data' are the data items.
//The data may be weighted by the weights in 'weights'.
Eigen::Matrix<Scalar,Eigen::Dynamic,1> varianceZeroMean(const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& data, Eigen::Matrix<Scalar,Eigen::Dynamic,1> weights = Eigen::Matrix<Scalar,Eigen::Dynamic,1>::Zero(0)){
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> Vector;

	//If no weights are available, use equal weigting
	if (weights.size() == 0){
		weights = Vector::Ones(data.rows());
	}
	//Normalize the weights such that they sum up to one.
	normalizeProbabilities(weights);

	//Initialize a vector where we are going to store the variances.
	Vector var = Vector::Zero(data.cols());

	//Compute the variances
	for (int i = 0; i < data.rows();i++){

		//This line handles one data point i.e. all the columns of 'data' at once.
		var += (data.row(i).transpose()).cwiseAbs2()*weights(i);

	}

	//Divide the variances by the effective number of data points
	Scalar weightSqSum = weights.cwiseAbs2().sum();
	Scalar normalizationDenominator = (Scalar)1 - weightSqSum;
	if (normalizationDenominator < std::numeric_limits<Scalar>::epsilon()){
		normalizationDenominator = std::numeric_limits<Scalar>::epsilon();
	}
	var = var*(((Scalar)1)/normalizationDenominator);

	//Check that the variances are finite numbers
	assert(isFinite(var));

	return var;
}

template<typename Scalar>
//Compute the value of normal distribution probability density function
//NOTE We use precision matrix instead of the covariance matrix
Scalar normPdfLog(const Eigen::Matrix<Scalar,Eigen::Dynamic,1>& x,const Eigen::Matrix<Scalar,Eigen::Dynamic,1>& mu,const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& precisionMatrix,Scalar normConstLog = 0/0){

	//Difference to the mean
	Eigen::Matrix<Scalar,Eigen::Dynamic,1> diff = x-mu;

	//Argument to the exponential function
	Scalar arg = -0.5*(diff.transpose())*precisionMatrix*diff;

	//Compute the normalization constant of the normal distribution if it is not provided
	if (normConstLog != normConstLog){ //The standard argument yields IND value. Thus if the 
		Scalar tmp = (Scalar)precisionMatrix.rows() / (Scalar)2 * (Scalar)LOG_OF_TWO_TIMES_PI;
		normConstLog = (Scalar)tmp - (Scalar)0.5 * log(std::max(precisionMatrix.determinant(),std::numeric_limits<Scalar>::min())); //NOTE the substraction because of determinant of precision matrix instead of covariance matrix
	}

	assert(((arg - normConstLog) - (arg - normConstLog)) == ((arg - normConstLog) - (arg - normConstLog)));
	return arg - normConstLog;

}

template<typename Matrix>
//Remove the duplicate rows from matrix 'input'
void removeDuplicateRows(Matrix& input){
	for (int i = input.rows()-1;i>=0;i--){
		for(int j = i-1;j>=0;j--){
			if ((input.row(i)-input.row(j)).sum() == 0){
				removeRow(input,(unsigned int)i);
				break;
			}
		}
	}
}

template<typename Scalar>
//Normalize the rows of a matrix such they sum up to one
void normalizeRows(Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& matIn){
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> Vector;
	for (int i = 0;i<matIn.rows();i++){
		Vector tmp = matIn.row(i);
		normalizeProbabilities<Scalar>(tmp);
		matIn.row(i) = tmp;
	}
}

template<typename Scalar>
//Normalize the columns of a matrix such that they sum up to one
void normalizeColumns(Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& matIn){
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> Vector;

	for (int i = 0;i<matIn.cols();i++){
		Vector tmp = matIn.col(i);
		normalizeProbabilities<Scalar>(tmp);
		matIn.col(i) = tmp;
	}
}

template<typename Scalar>
//Clamps the given vector componentwise between the values in minVals and maxVals
//Returns true if the values in vectorToClamp are changed and false otherwise
bool clamp(Eigen::Matrix<Scalar,Eigen::Dynamic,1>& vectorToClamp,const Eigen::Matrix<Scalar,Eigen::Dynamic,1>& minVals,const Eigen::Matrix<Scalar,Eigen::Dynamic,1>& maxVals){
	bool changed = false;
	//Check that the sizes match
	assert(maxVals.size() == minVals.size());
	assert(vectorToClamp.size() == minVals.size());
	//Check each component of the vectorToClamp
	for (int i = 0; i < vectorToClamp.size();i++){
		//Store original value to see if changed
		Scalar orig = vectorToClamp(i);
		//Do the clamping
		vectorToClamp(i) = std::max(minVals(i),std::min(maxVals(i),vectorToClamp(i)));
		//Check if the value changed in clamping
		if (orig != vectorToClamp(i)){
			changed = true;
		}
	}
	return changed;
}


template<typename Matrix>
//Remove the row 'rowToRemove' from matrix 'matrix'
void removeRow(Matrix& matrix, unsigned int rowToRemove)
{
	unsigned int numRows = matrix.rows()-1;
	unsigned int numCols = matrix.cols();

	if( rowToRemove < numRows )
		matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

	matrix.conservativeResize(numRows,numCols);
}

template<typename Matrix>
//Remove the column 'colToRemove' from matrix 'matrix'
void removeColumn(Matrix& matrix, unsigned int colToRemove)
{
	unsigned int numRows = matrix.rows();
	unsigned int numCols = matrix.cols()-1;

	if( colToRemove < numCols )
		matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

	matrix.conservativeResize(numRows,numCols);
}

template<typename Scalar>
//Return the number of IND and INF entries in 'x'
int countInfAndInd(const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& x){
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> Matrix;
	if (isFinite(x)){
		return 0;
	}
	else{
		Matrix infMarked = (x.array().unaryExpr([](Scalar v) { return (v-v == v-v)? 0 : 1; })).matrix();
		return (int)infMarked.sum();
	}
}

template<typename Matrix>
//Return the number of non-zero entries in the matrix 'x'
int nnz(const Matrix& x)
{
	int numNonZero = 0;
	for (int i = 0;i < x.rows();i++){
		for (int j = 0; j < x.cols();j++){
			if(abs(x(i,j)) >= std::numeric_limits<Matrix::Scalar>::min()){
				numNonZero++;
			}
		}
	}
	return numNonZero;
}

template<typename Scalar>
//Determine whether the matrix x has the element element.
bool hasElement(const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& x, Scalar element)
{
	Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> ones = Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Ones(x.rows(),x.cols());
	bool itHas = ((x - ones*element).array() == (x*(Scalar)0).array()).any();
	return itHas;
}

template<typename DataItemType>
//Return true if 'object' is a member of the 'dataSet' else false
bool isMember(std::vector<DataItemType> dataSet,DataItemType object){
	return std::find(dataSet.begin(), dataSet.end(), object) != dataSet.end();
}

template<typename Scalar>
//Change the infinite and nearly infinite values to nearly maximum or minimum representable value
void capInfToMax(Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& matrix, Scalar tolerance = std::numeric_limits<Scalar>::epsilon()){
	assert(tolerance >= 0);
	Scalar maxVal = (1-tolerance)*std::numeric_limits<Scalar>::max();
	Scalar minVal = (1-tolerance)*std::numeric_limits<Scalar>::lowest();
	matrix = matrix.array().unaryExpr([&](Scalar v) { return v >= maxVal ? maxVal : v; });
	matrix = matrix.array().unaryExpr([&](Scalar v) { return v <= minVal ? minVal : v; });
	assert(isFinite(matrix));
}

template<typename Scalar>
//Regularize by Tikhonov regularization, i.e. add sufficiently large elements to the diagonal such that the matrix 'input' becomes invertible. In order for this to work, the matrix 'input' should be square matrix.
Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> regularizeTikhonov(const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& input){
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> Matrix;
	assert(isFinite(input));

	//Create an identity matrix with the same size as the 'input' matrix
	int rows = input.rows();
	int columns = input.cols();
	assert(rows == columns);
	Matrix eye = Matrix::Identity(rows,columns);

	bool invertible = false;
	//This constant will be added to the diagonal. If it isn't sufficiently large, it will be multiplied by 2 until the result is invertible.
	Scalar regularizationConst = (Scalar)REGULARIZATION(Scalar);
	//Store the input matrix to a temporary matrix that will be manipulated.
	Matrix tmp = input;
	Scalar max_element = tmp.maxCoeff();
	tmp /= max_element;
	do{
		//Use the LU-decomposition to check if the matrix is invertible
		Eigen::FullPivLU<Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>> lu(tmp);
		invertible = lu.isInvertible();
		//If the matrix can be inverted, it will be returned.
		if (invertible){
			return max_element*tmp;
		}
		//If the matrix isn't invertible, apply the Tikhonov regularization.
		tmp = input/max_element + regularizationConst*eye;
		//If the Tikhonov regularization fails, the normalization constant that is added to the diagonal must be made larger.
		regularizationConst *= 2;

	} while (!invertible); //This could be a while true loop
	assert(isFinite(tmp));
	return max_element*tmp;
}


template<typename Scalar>
//This function inverts a matrix if invertible. Else it uses the pseudoinverse. As a last resort it will use Tikhonov regularization.
Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> pseudoInverseWithTikhonov(const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& matIn,Scalar deltaForTikhonov = (Scalar)REGULARIZATION(Scalar)){
	
	
	//For full rank matrix return its inverse.
	Eigen::FullPivLU<Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>> lu(matIn);
	if(lu.isInvertible()){
		return matIn.inverse();
	}

	//Pseudoinverse
	Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> normalMatrix = (matIn.adjoint())*matIn;
	lu = Eigen::FullPivLU<Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>>(normalMatrix);
	if(lu.isInvertible()){
		return (normalMatrix.inverse())*(matIn.adjoint());
	}

	int rows = normalMatrix.rows();
	int columns = normalMatrix.cols();

	//Tikhonov regularization
	assert(isFinite(normalMatrix));
	normalMatrix = regularizeTikhonov(normalMatrix);

	assert(isFinite(normalMatrix));
	Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> normalMatrixInv = normalMatrix.inverse();
	return normalMatrixInv*(matIn.adjoint());
}

template<typename Scalar>
//Regularize the matrix 'input' by doing an SVD and restricting the singular values to at least 'tolerance' times the mean of the eigenvalues. If the the singular values still would be less than 'absoluteLowLimitToEigenValues' they will be at least 'absoluteLowLimitToEigenValues'.
Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> eigenvalueClampingRegularization(Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> input,Scalar tolerance = 0.01,Scalar absoluteLowLimitToEigenvalues = std::numeric_limits<Scalar>::epsilon()){
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> Vector;

	//Ensure that the entries in input are not infinite
	capInfToMax(input);
	//The singular value decomposition of the 'input' matrix
	Eigen::JacobiSVD<Matrix> svd(input, Eigen::ComputeFullU | Eigen::ComputeFullV |Eigen::FullPivHouseholderQRPreconditioner );

	//The singular values
	Vector sing = svd.singularValues();

	//The minimum allowed singular value
	Scalar minSing = sing.mean();
	minSing *= tolerance;

	for (int i = 0;i < sing.size();i++){
		sing(i) = std::max(absoluteLowLimitToEigenvalues,std::max(sing(i),minSing));
		sing(i) = std::min(sing(i),std::numeric_limits<Scalar>::max());
	}

	assert(isFinite(sing));

	//Left eigenvectors
	const Matrix& U = svd.matrixU();
	//Right eigenvectors
	const Matrix& V = svd.matrixV();
	//Reconstruct the matrix with the clamped singular values.
	Matrix tmp = sing.asDiagonal()*(V.transpose());
	assert(isFinite(tmp));
	Matrix regReconst = U*tmp;
	assert(isFinite(regReconst));

	//Return the reconstruction.
	return regReconst;


}

template<typename Scalar>
//Perform linear regression.
//Returns: A matrix beta such that regressand = beta*regressor
//The observations (i.e. the data points) are as rows of 'regressor' and 'regressand'.
Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> linearRegression(const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& regressor,const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& regressand){
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> Matrix;

	Matrix regressorInv;
	try{
		regressorInv = pseudoInverseWithTikhonov<Scalar>(regressor);
	}
	catch (Matrix_has_nonfinite_elements e)
	{
		throw e;
	}
	Matrix beta = regressorInv*regressand;
	return beta.transpose();

}

template<typename Scalar>
//Perform affine regression.
//The first column of the return value matrix has the affine vector. I.e. return value beta = [b,A] such that: regressand = A*regressor + b
//The observations (i.e. the data points) are as rows of 'regressor' and 'regressand'.
Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> affineRegression(const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& regressor,const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& regressand){
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> Vector;

	if (regressor.cols()*regressor.rows() == 0 || regressand.cols()*regressand.rows() == 0){
		return Matrix::Zero(0,0);
	}

	//Augment the regressor matrix such that the first column is ones.
	Matrix affineRegressor = Matrix::Zero(regressor.rows(),regressor.cols()+1);
	affineRegressor.block(0,1,regressor.rows(),regressor.cols()) = regressor;
	affineRegressor.col(0) = Vector::Ones(regressor.rows());

	//Perform linear regression to the augmented matrix
	try{
		return linearRegression<Scalar>(affineRegressor,regressand);
	}
	catch (Matrix_has_nonfinite_elements e)
	{
		throw e;
	}

}

template<typename Scalar>
//Perform weighted affine regression
//The first column of the return value matrix has the affine vector. I.e. return value beta = [b,A] such that: regressand = A*regressor + b
//The observations (i.e. the data points) are as rows of 'regressor' and 'regressand'.
Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> weightedAffineRegression(const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& regressor,const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& regressand,Eigen::Matrix<Scalar,Eigen::Dynamic,1> weightsForDataPoints = Eigen::Matrix<Scalar,Eigen::Dynamic,1>::Zero(0)){
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> Vector;

	if (regressor.cols()*regressor.rows() == 0 || regressand.cols()*regressand.rows() == 0){
		return Matrix::Zero(0,0);
	}

	if (weightsForDataPoints.size() == 0){
		try {
			return affineRegression<Scalar>(regressor,regressand);
		}
		catch (Matrix_has_nonfinite_elements e){
			throw e;
		}
	}

	//Scale the weights to sum up to one
	normalizeProbabilities(weightsForDataPoints);

	//Augment the regressor matrix such that the first column is ones.
	Matrix affineRegressor = Matrix::Zero(regressor.rows(),regressor.cols()+1);
	affineRegressor.block(0,1,regressor.rows(),regressor.cols()) = regressor;
	affineRegressor.col(0) = Vector::Ones(regressor.rows());

	Matrix W = weightsForDataPoints.asDiagonal();

	Matrix regressandTmp = affineRegressor.transpose()*W*regressand;
	Matrix tmpNormalMatrix = affineRegressor.transpose()*W*affineRegressor;
	assert(isFinite(tmpNormalMatrix));
	Matrix normInv;
	try{
		normInv = pseudoInverseWithTikhonov<Scalar>(tmpNormalMatrix);
	}
	catch (Matrix_has_nonfinite_elements e){
		throw e;
	}
	assert(isFinite(normInv));

	Matrix beta = normInv*regressandTmp;

	return beta.transpose();
}

template<typename Scalar>
//The data points in 'data' are assumed to be as the rows. This function centers and scales the data such that it has zero mean and unit variance, i.e. the data is shifted by the amount '-shift' and it each column is divided by the corresponding entry in 'scaling'.
void centerAndScaleToUnity(Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& data,Eigen::Matrix<Scalar,Eigen::Dynamic,1>& shift,Eigen::Matrix<Scalar,Eigen::Dynamic,1>& scaling){
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> Vector;

	//Compute the 
	shift = weightedMean<Scalar>(data);
	data = data.rowwise() - shift.transpose();
	scaling = variance<Scalar>(data).cwiseSqrt();
	for (int i = 0; i < data.cols();i++){
		data.col(i) *= (Scalar)1 / std::max(scaling(i),std::numeric_limits<Scalar>::epsilon());
	}

}


template<typename Scalar>
//Compute (origMatrix^(-1) + vector1*vector2^T)^(-1)
Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> shermanMorrison(const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& origMatrix,Eigen::Matrix<Scalar,Eigen::Dynamic,1>& vector1,Eigen::Matrix<Scalar,Eigen::Dynamic,1>& vector2){
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> Vector;

	Matrix result = origMatrix;
	Matrix numerator = origMatrix*(vector1*vector2.adjoint())*origMatrix;
	Scalar denominator = 1 + vector2.adjoint()*origMatrix*vector1;
	assert(std::abs(denominator) > std::numeric_limits<Scalar>::min());

	result -= numerator/denominator;
	return result;

}




#endif