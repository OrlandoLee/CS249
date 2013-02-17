package prea.util;
import prea.data.structure.SparseMatrix;
import prea.data.structure.SparseVector;

/**
 * This is a unified class providing evaluation metrics,
 * including comparison of predicted ratings and rank-based metrics, etc.
 * 
 * @author Joonseok Lee
 * @author Mingxuan Sun
 * @since 2012. 4. 20
 * @version 1.1
 */
public class EvaluationMetrics {
	/** Real ratings for test items. */
	private SparseMatrix testMatrix;
	/** Predicted ratings by CF algorithms for test items. */
	private SparseMatrix predicted;
	/** Maximum value of rating, existing in the dataset. */
	private double maxValue;
	/** Minimum value of rating, existing in the dataset. */
	private double minValue;
	/** The number of items to recommend, in rank-based metrics */
	private int recommendCount;
	/** Half-life in rank-based metrics */
	private int halflife;
 

    /** Mean Absoulte Error (MAE) */
    private double mae;
    /** Mean Squared Error (MSE) */
    private double mse;
    /** Rank-based Half-Life Utility (HLU) */
    private double hlu;
    /** Rank-based Normalized Discounted Cumulative Gain (NDCG) */
    private double ndcg;
    /** Rank-based Kendall's Tau */
    private double kendallsTau;
    /** Rank-based Spear */
    private double spearman;
    /** Asymmetric Loss */
    private double asymmetricLoss;
	
	/**
	 * Standard constructor for EvaluationMetrics class.
	 * 
	 * @param tm Real ratings of test items.
	 * @param p Predicted ratings of test items.
	 * @param max Maximum value of rating, existing in the dataset.
	 * @param min Minimum value of rating, existing in the dataset.
	 *
	 */
	public EvaluationMetrics(SparseMatrix tm, SparseMatrix p, double max, double min) {
		testMatrix = tm;
		predicted = p;
		maxValue = max;
		minValue = min;
		recommendCount = 10;
		halflife = 5;
		
		build();
	}
	
	/**
	 * Getter method for Mean Absolute Error (MAE)
	 * 
	 * @return Mean Absolute Error (MAE)
	 */
	public double getMAE() {
		return mae;
	}
	
	/**
	 * Getter method for Normalized Mean Absolute Error (NMAE)
	 * 
	 * @return Normalized Mean Absolute Error (NMAE)
	 */
	public double getNMAE() {
		return mae / (maxValue - minValue);
	}
	
	/**
	 * Getter method for Mean Squared Error (MSE)
	 * 
	 * @return Mean Squared Error (MSE)
	 */
	public double getMSE() {
		return mse;
	}
	
	/**
	 * Getter method for Root of Mean Squared Error (RMSE)
	 * 
	 * @return Root of Mean Squared Error (RMSE)
	 */
	public double getRMSE() {
		return Math.sqrt(mse);
	}
	
	/**
	 * Getter method for Rank-based Half-life score
	 * 
	 * @return Rank-based Half-life score
	 */
	public double getHLU() {
		return hlu;
	}
	/**
	 * Getter method for Rank-based NDCG
	 * 
	 * @return Rank-based NDCG score
	 */
	public double getNDCG() {
		return ndcg;
	}
	/**
	 * Getter method for Rank-based Kendall's Tau
	 * 
	 * @return Rank-based Kendall's Tau score
	 */
	public double getKendall() {
		return kendallsTau;
	}
	/**
	 * Getter method for Rank-based Spearman score
	 * 
	 * @return Rank-based Spearman score
	 */
	public double getSpearman() {
		return spearman;
	}

	/**
	 * Getter method for Asymmetric loss
	 * 
	 * @return Asymmetric loss
	 */
	public double getAsymmetricLoss() {
		return asymmetricLoss;
	}
		
	/** Calculate all evaluation metrics with given real and predicted rating matrices. */
	private void build() {
		int userCount = (testMatrix.length())[0] - 1;
		int testUserCount = 0;
		int testItemCount = 0;
		double rScoreSum = 0.0;
		double rMaxSum = 0;
		
		for (int u = 1; u <= userCount; u++) {
			testUserCount++;
			
			SparseVector realRateList = testMatrix.getRowRef(u);
			SparseVector predictedRateList = predicted.getRowRef(u);
			
			if (realRateList.itemCount() != predictedRateList.itemCount()) {
				System.out.print("Error: The number of test items and predicted items does not match!");
				continue;
			}
			
			if (realRateList.itemCount() > 0) {
				int[] realRateIndex = realRateList.indexList();
				double[] realRateValue = realRateList.valueList();
				int[] predictedRateIndex = predictedRateList.indexList();
				double[] predictedRateValue = predictedRateList.valueList();

				// k-largest rating value arrays are sorted here:
				Sort.kLargest(predictedRateValue, predictedRateIndex, 0, predictedRateIndex.length-1, recommendCount);
				Sort.kLargest(realRateValue, realRateIndex, 0, predictedRateIndex.length-1, recommendCount);

				int r = 1;
				double rScore = 0.0;
				for (int i : predictedRateIndex) {
					double realRate = testMatrix.getValue(u, i);
					double predictedRate = predicted.getValue(u, i);
					
					// Accuracy calculation:
					mae += Math.abs(realRate - predictedRate);
					mse += Math.pow(realRate - predictedRate, 2);
					asymmetricLoss += Loss.asymmetricLoss(realRate, predictedRate, minValue, maxValue);
					testItemCount++;
					
					// Half-life evaluation:
					if (r <= recommendCount) {
						rScore += Math.max(realRate - (double) (maxValue + minValue) / 2.0, 0.0) 
									/ Math.pow(2.0, (double) (r-1) / (double) (halflife-1));
						
						r++;
					}
				}

				// calculate R_Max here, and divide rScore by it.
				int rr = 1;
				double rMax = 0.0;
				for (int i : realRateIndex) {
					if (rr < r) {
						double realRate = testMatrix.getValue(u, i);
						rMax += Math.max(realRate - (double) (maxValue + minValue) / 2.0, 0.0) 
								/ Math.pow(2.0, (double) (rr-1) / (double) (halflife-1));
						
						rr++;
					}
				}
				
				rScoreSum += rScore * Math.min(realRateIndex.length, recommendCount);
				rMaxSum += rMax * Math.min(realRateIndex.length, recommendCount);
				
				// Rank-based metrics:
				ndcg += Distance.distanceNDCG(realRateList.indexList(), realRateList.valueList(), predictedRateList.indexList(), predictedRateList.valueList());
				kendallsTau += Distance.distanceKendall(realRateList.indexList(), realRateList.valueList(), predictedRateList.indexList(), predictedRateList.valueList(), realRateList.itemCount());
				spearman += Distance.distanceSpearman(realRateList.indexList(), realRateList.valueList(), predictedRateList.indexList(), predictedRateList.valueList(), realRateList.itemCount());
			}
		}
		
		mae /= (double) testItemCount;
		mse /= (double) testItemCount;
		hlu = rScoreSum / rMaxSum;
		ndcg /= (double) testUserCount;
		kendallsTau /= (double) testUserCount;
		spearman /= (double) testUserCount;
		asymmetricLoss /= (double) testItemCount;
	}

	
	public String printMultiLine() {
		return	"MAE\t" + this.getMAE() + "\r\n" +
				"RMSE\t" + this.getRMSE() + "\r\n" +
				"Asymm\t" + this.getAsymmetricLoss() + "\r\n" +
				"HLU\t" + this.getHLU() + "\r\n" +
				"NDCG\t" + this.getNDCG() + "\r\n" +
				"Kendall\t" + this.getKendall() + "\r\n" +
				"Spear\t" + this.getSpearman();
	}
	
	public String printOneLine() {
		return String.format("%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f",
				this.getMAE(),
				this.getRMSE(),
				this.getAsymmetricLoss(),
				this.getHLU(),
				this.getNDCG(),
				this.getKendall(),
				this.getSpearman());
	}
	
	public static String printTitle() {
		return "==============================================================================================\r\nName\tMAE\tRMSE\tAsymm\tHLU\tNDCG\tKendall\tSpear";
	}
}