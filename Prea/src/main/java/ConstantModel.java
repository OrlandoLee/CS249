import prea.data.structure.SparseMatrix;
import prea.data.structure.SparseVector;
import prea.util.EvaluationMetrics;

/**
 * This is a class implementing three baselines, including constant model,
 * user average, and item average.
 * 
 * @author Joonseok Lee
 * @since 2011. 7. 12
 * @version 20110712
 */
public class ConstantModel {
	/*========================================
	 * Method Names
	 *========================================*/
	// algorithm
	/** Algorithm Code for Constant Model */
	public static final int MEDIAN = 91;
	/** Algorithm Code for User Average */
	public static final int USER_AVG = 92;
	/** Algorithm Code for Item Average */
	public static final int ITEM_AVG = 93;
	/** Algorithm Code for Random */
	public static final int RANDOM = 94;
	
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
	
	/*========================================
	 * Constructors
	 *========================================*/
	/**
	 * Construct a constant model with the given data.
	 * 
	 * @param rm The rating matrix which will be used for training.
	 * @param tm The rating matrix which will be used for testing.
	 * @param ura The average of ratings for each user. 
	 * @param ira The average of ratings for each item.
	 * @param uc The number of users in the dataset.
	 * @param ic The number of items in the dataset.
	 * @param max The maximum rating value in the dataset.
	 * @param min The minimum rating value in the dataset.
	 */
	public ConstantModel(SparseMatrix rm, SparseMatrix tm, SparseVector ura, SparseVector ira,
			int uc, int ic, int max, int min) {
		rateMatrix = rm;
		testMatrix = tm;
		userRateAverage = ura;
		itemRateAverage = ira;
		userCount = uc;
		itemCount = ic;
		maxValue = max;
		minValue = min;
	}
	
	/*========================================
	 * Prediction
	 *========================================*/
	/**
	 * Evaluate the designated algorithm with the given test data.
	 * 
	 * @param method The code of algorithm to be tested. It can have one of the following:
	 * MEDIAN, USER_AVG, or ITEM_AVG.
	 * @return The result of evaluation, such as MAE, RMSE, and rank-score.
	 */
	public EvaluationMetrics evaluate(int method) {
		SparseMatrix predicted = new SparseMatrix(userCount+1, itemCount+1);
		for (int u = 1; u <= userCount; u++) {
			int[] testItems = testMatrix.getRowRef(u).indexList();
			
			if (testItems != null) {
				SparseVector predictedForUser = null;
				predictedForUser = new SparseVector(itemCount+1);
				
				// Base line 1: estimate by median value
				if (method == MEDIAN) {
					predictedForUser.initialize(testItems, (maxValue + minValue) / 2);
				}
				
				// Base line 2: estimate by user average
				else if (method == USER_AVG) {
					predictedForUser.initialize(testItems, userRateAverage.getValue(u));
				}
				
				// Base line 3: estimate by item average
				else if (method == ITEM_AVG) {
					for (int t = 0; t < testItems.length; t++) {
						predictedForUser.setValue(testItems[t], itemRateAverage.getValue(testItems[t]));
					}
				}
				
				// Base line 4: estimate randomly
				else if (method == RANDOM) {
					for (int t = 0; t < testItems.length; t++) {
						double rdm = Math.random() * (maxValue - minValue) + minValue;
						predictedForUser.setValue(testItems[t], rdm);
					}
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
}