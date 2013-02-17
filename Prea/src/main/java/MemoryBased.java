import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

import prea.data.structure.SparseMatrix;
import prea.data.structure.SparseVector;
import prea.util.EvaluationMetrics;
import prea.util.Sort;



/**
 * This is a class implementing memory-based CF algorithms,
 * including user-based CF and item-based CF (Sarwar et al, UAI 1998).
 * 
 * @author Joonseok Lee
 * @since 2011. 7. 12
 * @version 20110712
 */
public class MemoryBased {
	/*========================================
	 * Method Names
	 *========================================*/
	// algorithm
	/** Algorithm Code for User-based CF */
	public static final int USER_BASED = 1;
	/** Algorithm Code for Item-based CF */
	public static final int ITEM_BASED = 2;
	
	// similarity measure
	/** Similarity Measure Code for Pearson Correlation */
	public static final int PEARSON_CORR = 101;
	/** Similarity Measure Code for Vector Cosine */
	public static final int VECTOR_COS = 102;
	/** Similarity Measure Code for Mean Squared Difference (MSD) */
	public static final int MEAN_SQUARE_DIFF = 103;
	/** Similarity Measure Code for Mean Absolute Difference (MAD) */
	public static final int MEAN_ABS_DIFF = 104;
	/** Similarity Measure Code for Inverse User Frequency */
	public static final int INVERSE_USER_FREQUENCY = 105;
	
	// estimation
	/** Estimation Method Code for Weighted Sum */
	public static final int WEIGHTED_SUM = 201;
	/** Estimation Method Code for Simple Weighted Average */
	public static final int SIMPLE_WEIGHTED_AVG = 202;
	
	
	/*========================================
	 * Common Variables
	 *========================================*/
	/** Rating matrix for each user (row) and item (column) */
	public SparseMatrix rateMatrix;
	/** Rating matrix for test items. Not allowed to refer during training phase. */
	public SparseMatrix testMatrix;
	/** Average of ratings for each user. */
	public SparseVector userRateAverage;
	/** Average of ratings for each item. */
	public SparseVector itemRateAverage;
	/** The number of users. */
	public int userCount;
	/** The number of items. */
	public int itemCount;
	/** The total number of ratings in the rating matrix. */
	public int rateCount;
	/** Maximum value of rating, existing in the dataset. */
	public int maxValue;
	/** Minimum value of rating, existing in the dataset. */
	public int minValue;
	/** The number of neighbors, used for estimation. */
	public int neighborSize;
	/** Indicating whether the pre-calculated user similarity file is used. */
	public boolean userSimilarityPrefetch;
	/** Indicating whether the pre-calculated item similarity file is used. */
	public boolean itemSimilarityPrefetch;
	/** The name of pre-calculated user similarity file, if it is used. */
	public String userSimilarityFileName;
	/** The name of pre-calculated item similarity file, if it is used. */
	public String itemSimilarityFileName;
	/** Indicating whether to use default vote value. */
	public boolean defaultVote;
	/** The default voting value, if used. */
	public double defaultValue;
	/** The method code for similarity measure. */
	public int similarityMethod;
	
	/*========================================
	 * Constructors
	 *========================================*/
	/**
	 * Construct a memory-based model with the given data.
	 * 
	 * @param rm The rating matrix which will be used for training.
	 * @param tm The rating matrix which will be used for testing.
	 * @param ura The average of ratings for each user. 
	 * @param ira The average of ratings for each item.
	 * @param uc The number of users in the dataset.
	 * @param ic The number of items in the dataset.
	 * @param max The maximum rating value in the dataset.
	 * @param min The minimum rating value in the dataset.
	 * @param ns The neighborhood size.
	 * @param usp Whether the pre-calculated user similarity file is used.
	 * @param isp Whether the pre-calculated item similarity file is used.
	 * @param usfn The name of pre-calculated user similarity file, if it is used.
	 * @param isfn The name of pre-calculated item similarity file, if it is used.
	 */
	public MemoryBased(SparseMatrix rm, SparseMatrix tm, SparseVector ura, SparseVector ira,
			int uc, int ic, int max, int min, int ns, boolean usp, boolean isp, String usfn, String isfn,
			int sim, boolean df, double dv) {
		rateMatrix = rm;
		testMatrix = tm;
		userRateAverage = ura;
		itemRateAverage = ira;
		userCount = uc;
		itemCount = ic;
		maxValue = max;
		minValue = min;
		neighborSize = ns;
		userSimilarityPrefetch = usp;
		itemSimilarityPrefetch = isp;
		userSimilarityFileName = usfn;
		itemSimilarityFileName = isfn;
		similarityMethod = sim;
		defaultVote = df;
		defaultValue = dv;
	}
	
	/*========================================
	 * Model Builder
	 *========================================*/
	/**
	 * Predict ratings for a given user regarding given set of items, by user-based CF algorithm.
	 * 
	 * @param userNo The user ID.
	 * @param testItemIndex The list of items whose ratings will be predicted.
	 * @param k The neighborhood size.
	 * @param userSim The similarity vector between the target user and all the other users.
	 * @return The predicted ratings for each item.
	 */
	public SparseVector userBased(int userNo, int[] testItemIndex, int k, double[] userSim) {
		if (testItemIndex == null)
			return null;
		
		double[][] sim = new double[testItemIndex.length][userCount];
		int[][] index = new int[testItemIndex.length][userCount];
		SparseVector a = rateMatrix.getRow(userNo);
		SparseVector c = new SparseVector(itemCount+1);
		double a_avg = a.average();
		
		// calculate similarity with every user:
		int[] tmpIdx = new int[testItemIndex.length];
		for (int u = 1; u <= userCount; u++) {
			SparseVector b = rateMatrix.getRowRef(u);
			double similarityMeasure;
			
			if (userSimilarityPrefetch) {
				similarityMeasure = userSim[u];
			}
			else {
				similarityMeasure = similarity (true, a, b, a_avg, userRateAverage.getValue(u), similarityMethod);
			}
			
			if (similarityMeasure > 0.0) {
				for (int t = 0; t < testItemIndex.length; t++) {
					if (b.getValue(testItemIndex[t]) > 0.0) {
						sim[t][tmpIdx[t]] = similarityMeasure;
						index[t][tmpIdx[t]] = u;
						tmpIdx[t]++;
					}
				}
			}
		}
		
		// Estimate rating for items in test set:
		for (int t = 0; t < testItemIndex.length; t++) {
			// find k most similar users:
			Sort.kLargest(sim[t], index[t], 0, tmpIdx[t]-1, neighborSize);
			
			int[] similarUsers = new int[k];
			int similarUserCount = 0;
			for (int i = 0; i < k; i++) {
				if (sim[t][i] > 0) { // sim[t][i] is already sorted!
					similarUsers[i] = index[t][i];
					similarUserCount++;
				}
			}
			
			int i = testItemIndex[t];
			if (similarUserCount > 0) {
				double estimated = estimation(true, userNo, i, similarUsers, similarUserCount, sim[t], WEIGHTED_SUM);
				
				// NaN check: it happens that no similar user has rated on item i, then the estimate is NaN.
				if (!Double.isNaN(estimated)) {
					c.setValue(i, estimated);
				}
				else {
					c.setValue(i, (maxValue + minValue) / 2);
				}
			}
			else {
				c.setValue(i, (maxValue + minValue) / 2);
			}
		}
		
		return c;
	}
	
	/**
	 * Predict ratings for a given user regarding given set of items, by item-based CF algorithm.
	 * 
	 * @param userNo The user ID.
	 * @param testItemIndex The list of items whose ratings will be predicted.
	 * @param k The neighborhood size.
	 * @param itemSim The similarity matrix containing similarity between every two-item-pair.
	 * @return The predicted ratings for each item.
	 */
	public SparseVector itemBased(int userNo, int[] testItemIndex, int k, SparseMatrix itemSim) {
		if (testItemIndex == null)
			return null;
		
		SparseVector c = new SparseVector(itemCount+1);

		for (int i : testItemIndex) {
			SparseVector a = rateMatrix.getColRef(i);
			
			// calculate similarity of every item to item i:
			double[] sim = new double[itemCount];
			int[] index = new int[itemCount];
			int[] similarItems = new int[k];
			int tmpIdx = 0;
			
			for (int j = 1; j <= itemCount; j++) {
				if (rateMatrix.getValue(userNo, j) > 0.0) {
					double similarityMeasure;

					if (itemSimilarityPrefetch) {
						if (i < j)
							similarityMeasure = itemSim.getValue(i, j);
						else
							similarityMeasure = itemSim.getValue(j, i);
					}
					else {
						SparseVector b = rateMatrix.getColRef(j);
						similarityMeasure = similarity (false, a, b, itemRateAverage.getValue(i), itemRateAverage.getValue(j), similarityMethod);
					}
					
					if (similarityMeasure > 0.0) {
						sim[tmpIdx] = similarityMeasure;
						index[tmpIdx] = j;
						tmpIdx++;
					}
				}
			}
			
			// find k most similar items:
			Sort.kLargest(sim, index, 0, tmpIdx-1, neighborSize);
			
			int similarItemCount = 0;
			for (int j = 0; j < k; j++) {
				if (sim[j] > 0) { // sim[j] is already sorted!
					similarItems[j] = index[j];
					similarItemCount++;
				}
			}
			
			if (similarItemCount > 0) {
				// estimate preference of item i by user a:
				double estimated = estimation(false, i, userNo, similarItems, similarItemCount, sim, WEIGHTED_SUM);
				
				// NaN check: it happens that no similar user has rated on item i, then the estimate is NaN.
				if (!Double.isNaN(estimated)) {
					c.setValue(i, estimated);
				}
				else {
					c.setValue(i, (maxValue + minValue) / 2);
				}
			}
			else {
				c.setValue(i, (maxValue + minValue) / 2);
			}
		}
		
		return c;
	}
	
	/*========================================
	 * Prediction
	 *========================================*/
	/**
	 * Evaluate the designated algorithm with the given test data.
	 * 
	 * @param method The code of algorithm to be tested. It can have one of the following:
	 * USER_BASED or ITEM_BASED.
	 * @return The result of evaluation, such as MAE, RMSE, and rank-score.
	 */
	public EvaluationMetrics evaluate(int method) {
		if (method == USER_BASED) {
			return userBasedEvaluate();
		}
		else if (method == ITEM_BASED) {
			return itemBasedEvaluate();
		}
		else {
			return null;
		}
	}
	
	/**
	 * Evaluate the user-based CF algorithm with the given test data.
	 * 
	 * @return The result of evaluation, such as MAE, RMSE, and rank-score.
	 */
	private EvaluationMetrics userBasedEvaluate() {
		SparseMatrix predicted = new SparseMatrix(userCount+1, itemCount+1);
		
		if (userSimilarityPrefetch) {
			try {
				FileInputStream stream = new FileInputStream(userSimilarityFileName);
				InputStreamReader reader = new InputStreamReader(stream);
				BufferedReader buffer = new BufferedReader(reader);
				
				String line;
				for (int u = 1; u <= userCount; u++) {
					line = buffer.readLine();
					int[] testItems = testMatrix.getRowRef(u).indexList();
					
					if (testItems != null) {
						// Parse similarity
						double[] userSim = new double[userCount+1];
						StringTokenizer st = new StringTokenizer (line);
						int idx = 1;
						while (st.hasMoreTokens()) {
							double sim = Double.parseDouble(st.nextToken()) / 10000;
							
							if (sim != 0.0) {
								userSim[idx] = sim;
							}
							
							idx++;
						}
						
						// Prediction
						SparseVector predictedForUser = userBased(u, testItems, neighborSize, userSim);
						
						if (predictedForUser != null) {
							for (int i : predictedForUser.indexList()) {
								predicted.setValue(u, i, predictedForUser.getValue(i));
							}
						}
					}
				}
				
				stream.close();
			}
			catch (IOException ioe) {
				System.out.println ("No such file.");
			}
		}
		else {
			for (int u = 1; u <= userCount; u++) {
				int[] testItems = testMatrix.getRowRef(u).indexList();
				
				if (testItems != null) {
					SparseVector predictedForUser = userBased(u, testItems, neighborSize, null);
					
					if (predictedForUser != null) {
						for (int i : predictedForUser.indexList()) {
							predicted.setValue(u, i, predictedForUser.getValue(i));
						}
					}
				}
			}
		}
		
		return new EvaluationMetrics(testMatrix, predicted, maxValue, minValue);
	}
	
	/**
	 * Evaluate the item-based CF algorithm with the given test data.
	 * 
	 * @return The result of evaluation, such as MAE, RMSE, and rank-score.
	 */
	private EvaluationMetrics itemBasedEvaluate() {
		SparseMatrix predicted = new SparseMatrix(userCount+1, itemCount+1);
		
		for (int u = 1; u <= userCount; u++) {
			int[] testItems = testMatrix.getRowRef(u).indexList();
			
			if (testItems != null) {
				SparseVector predictedForUser;
				
				if (itemSimilarityPrefetch) {
					SparseMatrix itemSim = readItemSimData(testItems);
					predictedForUser = itemBased(u, testItems, neighborSize, itemSim);
				}
				else {
					predictedForUser = itemBased(u, testItems, neighborSize, null);
				}
				
				if (predictedForUser != null) {
					for (int i : predictedForUser.indexList()) {
						predicted.setValue(u, i, predictedForUser.getValue(i));
					}
				}
			}
		}
		
		return new EvaluationMetrics(testMatrix, predicted, maxValue, minValue);
	}
	
	/*========================================
	 * Possible options
	 *========================================*/
	/**
	 * Calculate similarity between two given vectors.
	 * 
	 * @param rowOriented Use true if user-based, false if item-based.
	 * @param i1 The first vector to calculate similarity.
	 * @param i2 The second vector to calculate similarity.
	 * @param i1Avg The average of elements in the first vector.
	 * @param i2Avg The average of elements in the second vector.
	 * @param method The code of similarity measure to be used.
	 * It can be one of the following: PEARSON_CORR, VECTOR_COS,
	 * MEAN_SQUARE_DIFF, MEAN_ABS_DIFF, or INVERSE_USER_FREQUENCY.
	 * @return The similarity value between two vectors i1 and i2.
	 */
	public double similarity(boolean rowOriented, SparseVector i1, SparseVector i2, double i1Avg, double i2Avg, int method) {
		double result = 0.0;
		SparseVector v1, v2;
		
		if (defaultVote) {
			int[] i1ItemList = i1.indexList();
			int[] i2ItemList = i2.indexList();
			v1 = new SparseVector(i1.length());
			v2 = new SparseVector(i2.length());
			
			if (i1ItemList != null) {
				for (int t = 0; t < i1ItemList.length; t++) {
					v1.setValue(i1ItemList[t], i1.getValue(i1ItemList[t]));
					if (i2.getValue(i1ItemList[t]) == 0.0) {
						v2.setValue(i1ItemList[t], defaultValue);
					}
				}
			}
			
			if (i2ItemList != null) {
				for (int t = 0; t < i2ItemList.length; t++) {
					v2.setValue(i2ItemList[t], i2.getValue(i2ItemList[t]));
					if (i1.getValue(i2ItemList[t]) == 0.0) {
						v1.setValue(i2ItemList[t], defaultValue);
					}
				}
			}
		}
		else {
			v1 = i1;
			v2 = i2;
		}
		
		if (method == PEARSON_CORR) { // Pearson correlation
			SparseVector a = v1.sub(i1Avg);
			SparseVector b = v2.sub(i2Avg);
			
			result = a.innerProduct(b) / (a.norm() * b.norm());
		}
		else if (method == VECTOR_COS) { // Vector cosine
			result = v1.innerProduct(v2) / (v1.norm() * v2.norm());
		}
		else if (method == MEAN_SQUARE_DIFF) { // Mean Square Difference
			SparseVector a = v1.commonMinus(v2);
			a = a.power(2);
			result = a.sum() / a.itemCount();
		}
		else if (method == MEAN_ABS_DIFF) { // Mean Absolute Difference
			SparseVector a = v1.commonMinus(v2);
			result = a.absoluteSum() / a.itemCount();
		}
		else if (method == INVERSE_USER_FREQUENCY) {
			SparseVector a = v1.commonMinus(v2);
			int[] commonItemList = a.indexList();
			
			if (commonItemList == null)
				return 0.0;
			
			double invFreqSum = 0.0;
			double invFreqUser1Sum = 0.0;
			double invFreqUser2Sum = 0.0;
			double invFreqUser11Sum = 0.0;
			double invFreqUser22Sum = 0.0;
			double invFreqUser12Sum = 0.0;
			
			for (int t = 0; t < commonItemList.length; t++) {
				double invFreq = Math.log(userCount / rateMatrix.getColRef(commonItemList[t]).itemCount());
				
				invFreqSum += invFreq;
				invFreqUser1Sum += (invFreq * v1.getValue(commonItemList[t]));
				invFreqUser2Sum += (invFreq * v2.getValue(commonItemList[t]));
				invFreqUser11Sum += (invFreq * v1.getValue(commonItemList[t]) * v1.getValue(commonItemList[t]));
				invFreqUser22Sum += (invFreq * v1.getValue(commonItemList[t]) * v2.getValue(commonItemList[t]));
				invFreqUser12Sum += (invFreq * v1.getValue(commonItemList[t]) * v2.getValue(commonItemList[t]));
			}
			
			result = (invFreqSum * invFreqUser12Sum - invFreqUser1Sum * invFreqUser2Sum)
					/ Math.sqrt(invFreqSum * (invFreqUser11Sum - invFreqUser1Sum * invFreqUser1Sum)
											* (invFreqUser22Sum - invFreqUser2Sum * invFreqUser2Sum));
		}
		
		return result;
	}
	
	/**
	 * Estimate a rating based on neighborhood data.
	 * 
	 * @param rowOriented Use true if user-based, false if item-based.
	 * @param activeIndex The active user index for user-based CF; The item index for item-based CF.
	 * @param targetIndex The target item index for user-based CF; The user index for item-based CF.
	 * @param ref The indices of neighborhood, which will be used for estimation.
	 * @param refCount The number of neighborhood, which will be used for estimation.
	 * @param refWeight The weight of each neighborhood.
	 * @param method The code of estimation method. It can be one of the following:
	 * WEIGHTED_SUM or SIMPLE_WEIGHTED_AVG.
	 * @return The estimated rating value.
	 */
	public double estimation(boolean rowOriented, int activeIndex, int targetIndex, int[] ref, int refCount, double[] refWeight, int method) {
		double sum = 0.0;
		double weightSum = 0.0;
		double result = 0.0;
		
		if (method == WEIGHTED_SUM) { // Weighted Sum of Others' rating
			double activeAvg;
			if (rowOriented) {
				activeAvg = userRateAverage.getValue(activeIndex);
			}
			else {
				activeAvg = itemRateAverage.getValue(activeIndex);
			}
			
			for (int u = 0; u < refCount; u++) {
				double refAvg, ratedValue;
				if (rowOriented) {
					refAvg = userRateAverage.getValue(ref[u]);
					ratedValue = rateMatrix.getValue(ref[u], targetIndex);
				}
				else {
					refAvg = itemRateAverage.getValue(ref[u]);
					ratedValue = rateMatrix.getValue(targetIndex, ref[u]);
				}
				
				if (ratedValue > 0.0) {
					sum += ((ratedValue - refAvg) * refWeight[u]);
					weightSum += refWeight[u];
				}
			}
			
			result = activeAvg + sum / weightSum;
		}
		else if (method == SIMPLE_WEIGHTED_AVG) { // Simple Weighted Average
			for (int u = 0; u < refCount; u++) {
				double ratedValue;
				if (rowOriented) {
					ratedValue = rateMatrix.getValue(ref[u], targetIndex);
				}
				else {
					ratedValue = rateMatrix.getValue(targetIndex, ref[u]);
				}
				
				if (ratedValue > 0.0) {
					sum += (ratedValue * refWeight[u]);
					weightSum += refWeight[u];
				}
			}
			
			result = sum / weightSum;
		}
		
		// rating should be located between minValue and maxValue:
		if (result < minValue)
			result = minValue;
		else if (result > maxValue)
			result = maxValue;
		
		return result;
	}
	
	/*========================================
	 * File I/O
	 *========================================*/
	/**
	 * Read the pre-calculated item similarity data file.
	 * Make sure that the similarity file is compatible with the split file you are using,
	 * for a fair comparison.
	 * 
	 * @param validationItemSet The list of items which will be used for validation.
	 * @return The item similarity matrix.
	 */
	private SparseMatrix readItemSimData(int[] validationItemSet) {
		SparseMatrix itemSimilarity = new SparseMatrix (itemCount+1, itemCount+1);
		
		try {
			FileInputStream stream = new FileInputStream(itemSimilarityFileName);
			InputStreamReader reader = new InputStreamReader(stream);
			BufferedReader buffer = new BufferedReader(reader);
			
			// validationItemSet needs to be sorted at here!!
			Sort.quickSort(validationItemSet, 0, validationItemSet.length - 1, true);

			String line;
			int lineNo = 1;
			int validIdx = 0; // validation item index
			while((line = buffer.readLine()) != null && !line.equals("TT_EOF")) {
				int itemIdx = validationItemSet[validIdx];
				
				if (lineNo == itemIdx) {
					StringTokenizer st = new StringTokenizer (line);
					
					int idx = 1;
					while (st.hasMoreTokens()) {
						double sim = Double.parseDouble(st.nextToken()) / 10000;
						
						if (sim != 0.0) {
							itemSimilarity.setValue(itemIdx, idx, sim);
						}
						
						idx++;
					}
					
					validIdx++;
					
					if (validIdx >= validationItemSet.length)
						break;
				}
				
				lineNo++;
			}
			
			stream.close();
		}
		catch (IOException ioe) {
			System.out.println ("No such file.");
		}
		
		return itemSimilarity;
	}
}