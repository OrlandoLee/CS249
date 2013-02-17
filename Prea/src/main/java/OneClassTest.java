import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.StringTokenizer;

import prea.data.structure.SparseMatrix;
import prea.data.structure.SparseVector;
import prea.recommender.etc.FastNPCA;
import prea.recommender.etc.RankBased;
import prea.recommender.etc.SlopeOne;
import prea.util.EvaluationMetrics;
import prea.util.Loss;
import prea.util.Printer;



/**
 * This is an One-class Collaborative Filtering Test main file.
 * 
 * @author Joonseok Lee
 * @since 2011. 7. 12
 * @version 20110712
 */
public class OneClassTest {
	/*========================================
	 * Parameters
	 *========================================*/
	/** Proportion of items which will be used for test purpose. */
	public static final double TEST_RATIO = 0.2;
	/** The number of similar users/items to be used for estimation in neighborhood-based methods. */
	public static final int NEIGHBOR_SIZE = 50;
	/** Indicating whether loading split file or not */
	public static boolean SPLIT_PREFETCH = false;
	/** Indicating whether loading pre-calculated user similarity file or not */
	public static boolean USER_SIM_PREFETCH = true;
	/** Indicating whether loading pre-calculated item similarity file or not */
	public static boolean ITEM_SIM_PREFETCH = true;
	
	public static int UNIFORM_RANDOM = 8001;
	public static int USER_ORIENTED = 8002;
	public static int ITEM_ORIENTED = 8003;
	public static int MAX_DIFF = 9001;
	public static int THRESHOLD_UNIFORM = 9002;
	public static int THRESHOLD_ROULETTE = 9003;
	public static int SCORE_REFLECT = 9004;
	
	/*========================================
	 * Common Variables
	 *========================================*/
	/** Rating matrix for each user (row) and item (column) */
	public static SparseMatrix rateMatrix;
	/** Rating matrix for test items. Not allowed to refer during training and validation phase. */
	public static SparseMatrix testMatrix;
	/** Average of ratings for each user. */
	public static SparseVector userRateAverage;
	/** Average of ratings for each item. */
	public static SparseVector itemRateAverage;
	/** The number of users. */
	public static int userCount;
	/** The number of items. */
	public static int itemCount;
	/** Maximum value of rating, existing in the dataset. */
	public static int maxValue;
	/** Minimum value of rating, existing in the dataset. */
	public static int minValue;
	
	/** The list of item names, provided with the dataset. */
	public static String[] columnName;
	/** The name of data file used for test. */
	public static String dataFileName;
	/** The name of pre-calculated item similarity file, if it is used. */
	public static String itemSimilarityFileName;
	/** Item Similarity */
	public static double[][] itemSimilarity;
	
	/**
	 * Test examples for every algorithm. Also includes parsing the given parameters.
	 * 
	 * @param argv The argument list. Each element is separated by an empty space.
	 * First element is the data file name, and second one is the algorithm name.
	 * Third and later includes parameters for the chosen algorithm.
	 * Please refer to our web site for detailed syntax.
	 */
	public static void main(String argv[]) {
		// Read arff file:
		if (argv.length > 0) {
			dataFileName = argv[0];
		}
		else {
			dataFileName = "netflix_10m1k20r";
		}
		readArff (dataFileName + ".arff");
		
		// Train/test data split:
		if (SPLIT_PREFETCH || USER_SIM_PREFETCH || ITEM_SIM_PREFETCH) {
			readSplitData(dataFileName + "_split.txt");
		}
		else {
			split(TEST_RATIO);
		}
		
		// Read Item Similarity file:
		itemSimilarityFileName = dataFileName + "_itemSim.txt";
		readItemSimData();
		
		
		
		
		for (int u = 1; u <= userCount; u++) {
			SparseVector items = rateMatrix.getRowRef(u);
			int[] itemList = items.indexList();
			double minSuitability = Double.MAX_VALUE;
			double minRealRate = 0.0;
			double avg = items.average();
			double stdev = items.stdev();
			
			int[] itemList2 = new int[itemList.length];
			for (int i : itemList) {
				int idx = 0;
				
				for (int ii : itemList) {
					double rate = rateMatrix.getValue(u, ii);
					double zScore = (rate - avg) / stdev;
					
					if (i != ii && zScore >= 1.0 /* rate == 5.0 */) {
						itemList2[idx] = ii;
						idx++;
					}
				}
				
				double itemSuitability = getItemSuitability(itemList2, i);
				if (itemSuitability < minSuitability) {
					minSuitability = itemSuitability;
					minRealRate = rateMatrix.getValue(u, i);
				}
				//System.out.println (itemList2.length + "\t" + rateMatrix.getValue(u, i) + "\t" + itemSuitability);
			}
			
			// ref_group_size, min_suitability, real_rating, user_avg, user_std
			System.out.println(itemList2.length + "\t" + minSuitability + "\t" + minRealRate + "\t" + avg + "\t" + stdev);
		}
		
		
		
		// Hide positive votes from rating matrix:
		//hideRatingsAbs(1.0, 3.0);
		hideRatingsTic(0.75);
		calculateAverage();
		
		System.out.println("Remaining Ratings: " + rateMatrix.itemCount());
		
		
		// Sample plausible negative votes:
		sampleNegative(rateMatrix.itemCount() / 2, MAX_DIFF);
		
		
		// Actual test:
		if (argv.length > 0) {
			if (argv[1].toLowerCase().equals("median")) {
				System.out.println(EvaluationMetrics.printTitle() + "\tLearn\tEval");
				System.out.print("MEDIAN\t");
				constantModelTest(ConstantModel.MEDIAN);
			}
			else if (argv[1].toLowerCase().equals("useravg")) {
				System.out.print("USER_AVG\t");
				constantModelTest(ConstantModel.USER_AVG);
			}
			else if (argv[1].toLowerCase().equals("itemavg")) {
				System.out.print("ITEM_AVG\t");
				constantModelTest(ConstantModel.ITEM_AVG);
			}
			else if (argv[1].toLowerCase().equals("random")) {
				System.out.print("RANDOM\t");
				constantModelTest(ConstantModel.RANDOM);
			}
			else if (argv[1].toLowerCase().equals("userbased")) {
				int similarityMethod;
				
				if (argv[3].equals("pearson")) similarityMethod = MemoryBased.PEARSON_CORR;
				else if (argv[3].equals("cosine")) similarityMethod = MemoryBased.VECTOR_COS;
				else if (argv[3].equals("msd")) similarityMethod = MemoryBased.MEAN_SQUARE_DIFF;
				else if (argv[3].equals("mad")) similarityMethod = MemoryBased.MEAN_ABS_DIFF;
				else if (argv[3].equals("invuserfreq")) similarityMethod = MemoryBased.INVERSE_USER_FREQUENCY;
				else similarityMethod = MemoryBased.PEARSON_CORR;
				
				if (argv.length > 4 && argv[4].equals("default")) {
					System.out.print("USER_BASED_" + argv[3].toUpperCase() + "_DEFAULT\t");
					memoryBasedTest(Integer.parseInt(argv[2]), MemoryBased.USER_BASED, similarityMethod, true, Double.parseDouble(argv[5]));
				}
				else {
					System.out.print("USER_BASED_" + argv[3].toUpperCase() + "\t");
					memoryBasedTest(Integer.parseInt(argv[2]), MemoryBased.USER_BASED, similarityMethod, false, 0.0);
				}
			}
			else if (argv[1].toLowerCase().equals("itembased")) {
				int similarityMethod;
				
				if (argv[3].equals("pearson")) similarityMethod = MemoryBased.PEARSON_CORR;
				else if (argv[3].equals("cosine")) similarityMethod = MemoryBased.VECTOR_COS;
				else if (argv[3].equals("msd")) similarityMethod = MemoryBased.MEAN_SQUARE_DIFF;
				else if (argv[3].equals("mad")) similarityMethod = MemoryBased.MEAN_ABS_DIFF;
				else if (argv[3].equals("invuserfreq")) similarityMethod = MemoryBased.INVERSE_USER_FREQUENCY;
				else similarityMethod = MemoryBased.PEARSON_CORR;
				
				if (argv.length > 4 && argv[4].equals("default")) {
					System.out.print("ITEM_BASED" + argv[3].toUpperCase() + "_DEFAULT\t");
					memoryBasedTest(Integer.parseInt(argv[2]), MemoryBased.ITEM_BASED, similarityMethod, true, Double.parseDouble(argv[5]));
				}
				else {
					System.out.print("ITEM_BASED_" + argv[3].toUpperCase() + "\t");
					memoryBasedTest(Integer.parseInt(argv[2]), MemoryBased.ITEM_BASED, similarityMethod, false, 0.0);
				}
			}
			else if (argv[1].toLowerCase().equals("slopeone")) {
				System.out.print("SLOPE_ONE\t");
				slopeOneTest();
			}
			else if (argv[1].toLowerCase().equals("regsvd")) {
				System.out.print("REG_SVD\t");
				matrixFactorizationTest(MatrixFactorization.REGULARIZED_SVD, Integer.parseInt(argv[2]), Double.parseDouble(argv[3]), Double.parseDouble(argv[4]), 0, Integer.parseInt(argv[5]));
			}
			else if (argv[1].toLowerCase().equals("nmf")) {
				System.out.print("NMF\t");
				matrixFactorizationTest(MatrixFactorization.NON_NEGATIVE_MF_FROB, Integer.parseInt(argv[2]), 0, Double.parseDouble(argv[3]), 0, Integer.parseInt(argv[4]));
			}
			else if (argv[1].toLowerCase().equals("pmf")) {
				System.out.print("PMF\t");
				matrixFactorizationTest(MatrixFactorization.PROBABLISTIC_MF, Integer.parseInt(argv[2]), Integer.parseInt(argv[3]), Double.parseDouble(argv[4]), Double.parseDouble(argv[5]), Integer.parseInt(argv[6]));
			}
			else if (argv[1].toLowerCase().equals("bpmf")) {
				System.out.print("BPMF\t");
				matrixFactorizationTest(MatrixFactorization.BAYESIAN_PROBABLISTIC_MF, Integer.parseInt(argv[2]), 0, 0, 0, Integer.parseInt(argv[3]));
			}
			else if (argv[1].toLowerCase().equals("npca")) {
				System.out.print("NPCA\t");
				fastNPCATest(Double.parseDouble(argv[2]), Integer.parseInt(argv[3]));
			}
			else if (argv[1].toLowerCase().equals("rank")) {
				System.out.print("RANK_BASED\t");
				rankBasedTest(Double.parseDouble(argv[2]));
			}
		}
		else {
			System.out.println(EvaluationMetrics.printTitle() + "\tLearn\tEval");
			
//			System.out.print("MEDIAN\t");
//			constantModelTest(ConstantModel.MEDIAN);
//			
//			System.out.print("USE_AVG\t");
//			constantModelTest(ConstantModel.USER_AVG);
//			
//			System.out.print("ITEM_AVG\t");
//			constantModelTest(ConstantModel.ITEM_AVG);
//			
//			System.out.print("RANDOM\t");
//			constantModelTest(ConstantModel.RANDOM);
//			
			System.out.print("USER_BASED\t");
			memoryBasedTest(50, MemoryBased.USER_BASED, MemoryBased.VECTOR_COS, false, 0.0);
			
//			System.out.print("USER_BASED_DEFAULT\t");
//			memoryBasedTest(50, MemoryBased.USER_BASED, MemoryBased.VECTOR_COS, true, (maxValue + minValue) / 2);
//			
			System.out.print("ITEM_BASED\t");
			memoryBasedTest(50, MemoryBased.ITEM_BASED, MemoryBased.VECTOR_COS, false, 0.0);
			
//			System.out.print("ITEM_BASED_DEFAULT\t");
//			memoryBasedTest(50, MemoryBased.ITEM_BASED, MemoryBased.VECTOR_COS, true, (maxValue + minValue) / 2);
//			
			System.out.print("SLOPE_ONE\t");
			slopeOneTest();
			
			System.out.print("REG_SVD\t");
			matrixFactorizationTest(MatrixFactorization.REGULARIZED_SVD, 60, 0.005, 0.1, 0, 200);
			
			System.out.print("NMF\t");
			matrixFactorizationTest(MatrixFactorization.NON_NEGATIVE_MF_FROB, 100, 0, 0.0001, 0, 50);
			
			System.out.print("PMF\t");
			matrixFactorizationTest(MatrixFactorization.PROBABLISTIC_MF, 10, 50, 0.4, 0.8, 200);
			
			System.out.print("BPMF\t");
			matrixFactorizationTest(MatrixFactorization.BAYESIAN_PROBABLISTIC_MF, 2, 0, 0, 0, 20);
			
			System.out.print("NPCA\t");
			fastNPCATest(0.15, 50);

			System.out.print("RANK_BASED\t");
			rankBasedTest(1.0);
		}
	}
	
	/**
	 * Test interface for fast Constant Model baselines.
	 * Print MAE, RMSE, and rank-based half-life score for given test data.
	 * 
	 * @param method The code for algorithm to be tested.
	 */
	public static void constantModelTest(int method) {
		ConstantModel cm = new ConstantModel(rateMatrix, testMatrix, userRateAverage, itemRateAverage, 
				userCount, itemCount, maxValue, minValue);
		
		long evalStart = System.currentTimeMillis();
		EvaluationMetrics CMresult = cm.evaluate(method);
		long evalEnd = System.currentTimeMillis();

		System.out.println(CMresult.printOneLine() + "\t" + Printer.printTime(0) + "\t" + Printer.printTime(evalEnd - evalStart));
//		System.out.println(CMresult.printOneLine());
//		System.out.println("Learn\t" + CFUtils.printTime(0));
//		System.out.println("Eval\t" + CFUtils.printTime(evalEnd - evalStart));
	}
	
	/**
	 * Test interface for Memory-based algorithms.
	 * Print MAE, RMSE, and rank-based half-life score for given test data.
	 * 
	 * @param k The neighborhood size.
	 * @param method The code for algorithm to be tested.
	 */
	public static void memoryBasedTest(int k, int method, int similarityMethod, boolean defaultUse, double defaultValue) {
		MemoryBased mb = new MemoryBased(rateMatrix, testMatrix, userRateAverage, itemRateAverage, 
				userCount, itemCount, maxValue, minValue, k, USER_SIM_PREFETCH, ITEM_SIM_PREFETCH,
				dataFileName + "_userSim.txt", dataFileName + "_itemSim.txt", similarityMethod, defaultUse, defaultValue);
		
		long evalStart = System.currentTimeMillis();
		EvaluationMetrics MEMresult = mb.evaluate(method);
		long evalEnd = System.currentTimeMillis();
		
		System.out.println(MEMresult.printOneLine() + "\t" + Printer.printTime(0) + "\t" + Printer.printTime(evalEnd - evalStart));
//		System.out.println(MEMresult.printOneLine());
//		System.out.println("Learn\t" + CFUtils.printTime(0));
//		System.out.println("Eval\t" + CFUtils.printTime(evalEnd - evalStart));
	}
	
	/**
	 * Test interface for slope-one algorithm.
	 * Builds a model with given data, and print MAE, RMSE, and rank-based half-life score.
	 */
	public static void slopeOneTest() {
		SlopeOne so = new SlopeOne(userCount, itemCount, maxValue, minValue);
		
		long learnStart = System.currentTimeMillis();
		so.buildModel(rateMatrix);
		long learnEnd = System.currentTimeMillis();
		
		long evalStart = System.currentTimeMillis();
		EvaluationMetrics SOresult = so.evaluate(testMatrix);
		long evalEnd = System.currentTimeMillis();
		
		System.out.println(SOresult.printOneLine() + "\t" + Printer.printTime(learnEnd - learnStart) + "\t" + Printer.printTime(evalEnd - evalStart));
//		System.out.println(SOresult.printOneLine());
//		System.out.println("Learn\t" + CFUtils.printTime(learnEnd - learnStart));
//		System.out.println("Eval\t" + CFUtils.printTime(evalEnd - evalStart));
	}
	
	/**
	 * Test interface for fast Matrix-Factorization-based algorithms.
	 * Builds a model with given data, and print MAE, RMSE, and rank-based half-life score. 
	 * 
	 * @param method The code for algorithm to be tested.
	 * @param features The number of features in low-rank matrix representation.
	 * @param learningRate The learning rate for gradient-descent. 
	 * @param regularizer The regularization parameter.
	 * @param momentum The momentum parameter.
	 * @param maxIter Maximum The number of iteration.
	 * 
	 */
	public static void matrixFactorizationTest(int method, int features, double learningRate, double regularizer, double momentum, int maxIter) {
		MatrixFactorization mf = new MatrixFactorization(rateMatrix, testMatrix, userCount, itemCount, 
				maxValue, minValue, features, learningRate, regularizer, momentum, maxIter);
		
		long learnStart = System.currentTimeMillis();
		mf.buildModel(method);
		long learnEnd = System.currentTimeMillis();
		
		long evalStart = System.currentTimeMillis();
		EvaluationMetrics MFresult = mf.evaluate(method);
		long evalEnd = System.currentTimeMillis();
		
		System.out.println(MFresult.printOneLine() + "\t" + Printer.printTime(learnEnd - learnStart) + "\t" + Printer.printTime(evalEnd - evalStart));
//		System.out.println(MFresult.printOneLine());
//		System.out.println("Learn\t" + CFUtils.printTime(learnEnd - learnStart));
//		System.out.println("Eval\t" + CFUtils.printTime(evalEnd - evalStart));
	}
	
	/**
	 * Test interface for fast NPCA.
	 * Builds a model with given data, and print MAE, RMSE, and rank-based half-life score. 
	 * 
	 * @param validationRatio Fraction of items which will be used for validation.
	 * @param maxIter maximum The number of iteration.
	 * 
	 */
	public static void fastNPCATest(double validationRatio, int maxIter) {
		FastNPCA fnpca = new FastNPCA(userCount, itemCount, maxValue, minValue, validationRatio, maxIter);
		
		long learnStart = System.currentTimeMillis();
		fnpca.buildModel(rateMatrix);
		long learnEnd = System.currentTimeMillis();
		
		long evalStart = System.currentTimeMillis();
		EvaluationMetrics NPCAresult = fnpca.evaluate(testMatrix);
		long evalEnd = System.currentTimeMillis();
		
		System.out.println(NPCAresult.printOneLine() + "\t" + Printer.printTime(learnEnd - learnStart) + "\t" + Printer.printTime(evalEnd - evalStart));
//		System.out.println(NPCAresult.printOneLine());
//		System.out.println("Learn\t" + CFUtils.printTime(learnEnd - learnStart));
//		System.out.println("Eval\t" + CFUtils.printTime(evalEnd - evalStart));
	}

	/**
	 * Test interface for Mingxuan's Rank-based algorithm.
	 * Builds a model with given data, and print MAE, RMSE, and rank-based half-life score.
	 * @param kernelWidth The kernel bandwidth.
	 */
	public static void rankBasedTest(double kernelWidth) {
		RankBased rb1 = new RankBased(userCount, itemCount, maxValue, minValue, kernelWidth, RankBased.MEAN_LOSS);

		long learnStart = System.currentTimeMillis();
		rb1.buildModel(rateMatrix);
		long learnEnd = System.currentTimeMillis();
		
		long evalStart = System.currentTimeMillis();
		EvaluationMetrics Rankresult1 = rb1.evaluate(testMatrix);
		long evalEnd = System.currentTimeMillis();
			
		System.out.println(Rankresult1.printOneLine() + "\t" + Printer.printTime(learnEnd - learnStart) + "\t" + Printer.printTime(evalEnd - evalStart));
		
		
		RankBased rb2 = new RankBased(userCount, itemCount, maxValue, minValue, kernelWidth, RankBased.ASYMM_LOSS);

		learnStart = System.currentTimeMillis();
		rb2.buildModel(rateMatrix);
		learnEnd = System.currentTimeMillis();
				
		evalStart = System.currentTimeMillis();
		EvaluationMetrics Rankresult2 = rb2.evaluate(testMatrix);
		evalEnd = System.currentTimeMillis();
					
		System.out.println(Rankresult2.printOneLine() + "\t" + Printer.printTime(learnEnd - learnStart) + "\t" + Printer.printTime(evalEnd - evalStart));
//		System.out.println(Rankresult.printOneLine());
//		System.out.println("Learn\t" + CFUtils.printTime(learnEnd - learnStart));
//		System.out.println("Eval\t" + CFUtils.printTime(evalEnd - evalStart));
	}
	
	/*========================================
	 * Train/Validation set management
	 *========================================*/
	/**
	 * Items which will be used for test purpose are moved from rateMatrix to testMatrix.
	 * 
	 *  @param testRatio proportion of items which will be used for test purpose. 
	 *  
	 */
	private static void split(double testRatio) {
		recoverTestItems();
		
		// Record and remove test items:
		for (int u = 1; u <= userCount; u++) {
			int[] itemList = rateMatrix.getRowRef(u).indexList();
			
			if (itemList != null) {
				for (int i : itemList) {
					double rdm = Math.random();
					
					if (rdm < testRatio) {
						testMatrix.setValue(u, i, rateMatrix.getValue(u, i));
						rateMatrix.setValue(u, i, 0.0);						
					}
				}
			}
		}
	}
	
	/** Items in testMatrix are moved to original rateMatrix. */
	private static void recoverTestItems() {
		for (int u = 1; u <= userCount; u++) {
			int[] itemList = testMatrix.getRowRef(u).indexList();
			
			if (itemList != null) {
				for (int i : itemList) {
					rateMatrix.setValue(u, i, testMatrix.getValue(u, i));
					//testMatrix.setValue(u, i, 0.0);
				}
			}
		}
		testMatrix = new SparseMatrix(userCount+1, itemCount+1);
	}
	
	/** 
	 * Remove ratings from the rateMatrix, between min and max, inclusive.
	 * 
	 * @param min Minimum rating value to be deleted.
	 * @param max Maximum rating value to be deleted.
	 */
	private static void hideRatingsAbs(double min, double max) {
		for (int u = 1; u <= userCount; u++) {
			int[] itemList = rateMatrix.getRowRef(u).indexList();
			
			if (itemList != null) {
				for (int i : itemList) {
					double rate = rateMatrix.getValue(u, i);
					if (rate >= min && rate <= max) {
						rateMatrix.setValue(u, i, 0.0);
					}
				}
			}
		}
	}
	
	private static void hideRatingsTic(double tic) {
		for (int u = 1; u <= userCount; u++) {
			SparseVector ratedItems = rateMatrix.getRowRef(u);
			int[] itemList = ratedItems.indexList();
			
			if (itemList != null) {
				double average = ratedItems.average();
				double stdev = ratedItems.stdev();
				
				for (int i : itemList) {
					double rate = rateMatrix.getValue(u, i);
					double zScore = (rate - average) / stdev;
					
					if (zScore < tic) {
						rateMatrix.setValue(u, i, 0.0);
					}
				}
			}
		}
	}
	
	private static void sampleNegative(int sampleCount, int method) {
		if (method == UNIFORM_RANDOM) {
			int s = 0;
			while (s < sampleCount) {
				int u = (int) (Math.random() * userCount + 1);
				int i = (int) (Math.random() * itemCount + 1);
				
				if (rateMatrix.getValue(u, i) == 0.0) {
					rateMatrix.setValue(u, i, 1.0);
					s++;
				}
			}
		}
		else if (method == USER_ORIENTED) {
			double[] probDist = new double[userCount+1];
			
			// Making probability distribution:
			for (int u = 1; u <= userCount; u++) {
				probDist[u] = (double) rateMatrix.getRowRef(u).itemCount() / (double) rateMatrix.itemCount();
				
				if (u >= 2) { // cumulative
					probDist[u] += probDist[u-1];
				}
			}
			
			// Sampling:
			for (int s = 0; s < sampleCount; s++) {
				double rdm = Math.random();
				int userId = locateInProbDist(probDist, rdm, 0, userCount);
				
				// Uniformly randomly select an item without its rating:
				int itemId;
				do {
					itemId = (int) (Math.random() * itemCount + 1);
				} while (rateMatrix.getValue(userId, itemId) > 0);
				
				rateMatrix.setValue(userId, itemId, 1.0);
			}
		}
		else if (method == ITEM_ORIENTED) {
			double[] probDist = new double[itemCount+1];
			double sum = 0.0;
			
			// Making probability distribution:
			for (int i = 1; i <= itemCount; i++) {
				probDist[i] = 1 / (double) (rateMatrix.getColRef(i).itemCount() + 1);
				sum += probDist[i];
			}
			
			// Normalization:
			for (int i = 1; i <= itemCount; i++) {
				probDist[i] /= sum;
				
				if (i >= 2) { // cumulative
					probDist[i] += probDist[i-1];
				}
			}
			
			// Sampling:
			for (int s = 0; s < sampleCount; s++) {
				double rdm = Math.random();
				int itemId = locateInProbDist(probDist, rdm, 0, itemCount);
				
				// Uniformly randomly select an user without its rating:
				int userId;
				do {
					userId = (int) (Math.random() * userCount + 1);
				} while (rateMatrix.getValue(userId, itemId) > 0);
				
				rateMatrix.setValue(userId, itemId, 1.0);
			}
		}
		else if (method == MAX_DIFF) {
			int s = 0;
			while (s < sampleCount) {
				int u = (int) (Math.random() * userCount + 1);
				SparseVector items = rateMatrix.getRowRef(u);
				int[] itemList = items.indexList();
				double smallestSuitability = Double.MAX_VALUE;
				int smallestIndex = 0;
				
				for (int i = 1; i <= itemCount; i++) {
					double itemSuitability = getItemSuitability(itemList, i);
					
					if (itemSuitability < smallestSuitability && rateMatrix.getValue(u, i) == 0) {
						smallestSuitability = itemSuitability;
						smallestIndex = i;
					}
				}
				
				if (smallestSuitability < Double.MAX_VALUE) {
					rateMatrix.setValue(u, smallestIndex, 1.0);
					s++;
				}
			}
		}
		else if (method == THRESHOLD_UNIFORM) {
			int s = 0;
			while (s < sampleCount) {
				int u = (int) (Math.random() * userCount + 1);
				int i = (int) (Math.random() * itemCount + 1);
				
				if (rateMatrix.getValue(u, i) == 0.0) {
					// List of favorite items (assuming that other items are hidden at this point.)
					SparseVector items = rateMatrix.getRowRef(u);
					int[] itemList = items.indexList();
					
					// Calculate item suitability, based on item similarity:
					if (itemList != null) {
						double itemSuitability = getItemSuitability(itemList, i);
					
						if (itemSuitability < 0.01) {
							rateMatrix.setValue(u, i, 1.0);
							s++;
						}
					}
				}
			}
		}
		else if (method == THRESHOLD_ROULETTE) {
			
			// UNDER CONSTRUCTION here....
			
			int s = 0;
			while (s < sampleCount) {
				int u = (int) (Math.random() * userCount + 1);
				SparseVector items = rateMatrix.getRowRef(u);
				int[] itemList = items.indexList();
				double smallestSuitability = Double.MAX_VALUE;
				int smallestIndex = 0;
				
				for (int i = 1; i <= itemCount; i++) {
					double itemSuitability = getItemSuitability(itemList, i);
//System.out.println(itemSuitability);
					
					if (itemSuitability < smallestSuitability && rateMatrix.getValue(u, i) == 0) {
						smallestSuitability = itemSuitability;
						smallestIndex = i;
					}
				}
//System.out.println();
				
				if (smallestSuitability < Double.MAX_VALUE) {
					rateMatrix.setValue(u, smallestIndex, 1.0);
					s++;
				}
			}
		}
		else if (method == SCORE_REFLECT) {
			
		}
	}
	
	/** 
	 * Return the index of given data from a cumulative probability distribution.
	 * 
	 * @param distribution The cumulative probability distribution
	 * @param value The data value to locate
	 * @param min The minimum index to deal with
	 * @param max The maximum index to deal with
	 * @return The index in the probability distribution
	 */
	private static int locateInProbDist(double[] distribution, double value, int min, int max) {
		if (min > max) {
			return max;
		}
		else if (min == max) {
			return max;
		}
		else {
			int mid = (min + max) / 2;
			if (value < distribution[mid]) {
				return locateInProbDist(distribution, value, min, mid);
			}
			else {
				return locateInProbDist(distribution, value, mid+1, max);
			}
		}
	}
	
	/** 
	 * Calculate the suitability of an item for a specific user,
	 * assuming that the user likes items in the itemList.
	 * 
	 * @param itemList The list of items which are liked by an user.
	 * @param targetItem The target item which wants to know suitability for the user.
	 * @return The suitability ranging from 0 (not relevant) to 1 (perfectly suited).
	 */
	private static double getItemSuitability(int[] itemList, int targetItem) {
		// Suitability by maximum
//		double suitability = Double.MIN_VALUE;
//		
//		if (itemList != null) {
//			for (int i : itemList) {
//				if (itemSimilarity[i][targetItem] > suitability) {
//					suitability = itemSimilarity[i][targetItem];
//				}
//			}
//		}
//		
//		return suitability;
		
		
		// Suitability by average
		double sum = 0.0;
		int count = 0;

		if (itemList != null) {
			for (int i : itemList) {
				sum += itemSimilarity[i][targetItem];
				count++;
			}
		}
		
		return sum / (double) count;
		
		
		// Suitability by minimum
//		double suitability = Double.MAX_VALUE;
//		
//		if (itemList != null) {
//			for (int i : itemList) {
//				if (itemSimilarity[i][targetItem] < suitability) {
//					suitability = itemSimilarity[i][targetItem];
//				}
//			}
//		}
//		
//		return suitability;
	}
	
	/**
	 * Read the pre-calculated item similarity data file.
	 * Make sure that the similarity file is compatible with the split file you are using,
	 * for a fair comparison.
	 */
	private static void readItemSimData() {
		itemSimilarity = new double[itemCount+1][itemCount+1];
		
		try {
			FileInputStream stream = new FileInputStream(itemSimilarityFileName);
			InputStreamReader reader = new InputStreamReader(stream);
			BufferedReader buffer = new BufferedReader(reader);
			
			String line;
			int lineNo = 1;
			while((line = buffer.readLine()) != null && !line.equals("TT_EOF")) {
				StringTokenizer st = new StringTokenizer (line);
					
				int idx = 1;
				while (st.hasMoreTokens()) {
					double sim = Double.parseDouble(st.nextToken()) / 10000;
					itemSimilarity[lineNo][idx] = sim;

					idx++;
				}
				
				lineNo++;
			}
			
			stream.close();
		}
		catch (IOException ioe) {
			System.out.println ("No such file.");
		}
	}
	
	/*========================================
	 * File I/O
	 *========================================*/
	/**
	 * Read the data file in ARFF format, and store it in rating matrix.
	 * Peripheral information such as max/min values, user/item count are also set in this method.
	 * 
	 * @param fileName The name of data file.
	 * 
	 */
	private static void readArff(String fileName) {
		try {
			FileInputStream stream = new FileInputStream(fileName);
			InputStreamReader reader = new InputStreamReader(stream);
			BufferedReader buffer = new BufferedReader(reader);
			
			ArrayList<String> tmpColumnName = new ArrayList<String>();
			
			String line;
			int userNo = 0; // sequence number of each user
			int attributeCount = 0;
			
			maxValue = -1;
			minValue = 99999;
			
			// Read attributes:
			while((line = buffer.readLine()) != null && !line.equals("TT_EOF")) {
				if (line.contains("@ATTRIBUTE")) {
					String name;
					//String type;
					
					line = line.substring(10).trim();
					if (line.charAt(0) == '\'') {
						int idx = line.substring(1).indexOf('\'');
						name = line.substring(1, idx+1);
						//type = line.substring(idx+2).trim();
					}
					else {
						int idx = line.substring(1).indexOf(' ');
						name = line.substring(0, idx+1).trim();
						//type = line.substring(idx+2).trim();
					}
					
					//columnName[lineNo] = name;
					tmpColumnName.add(name);
					attributeCount++;
				}
				else if (line.contains("@RELATION")) {
					// do nothing
				}
				else if (line.contains("@DATA")) {
					// This is the end of attribute section!
					break;
				}
				else if (line.length() <= 0) {
					// do nothing
				}
			}
			
			// Set item count to data structures:
			itemCount = (attributeCount - 1)/2;
			columnName = new String[attributeCount];
			tmpColumnName.toArray(columnName);
			
			int[] itemRateCount = new int[itemCount+1];
			rateMatrix = new SparseMatrix(500000, itemCount+1); // max 480189, 17770
			testMatrix = new SparseMatrix(500000, itemCount+1);
			userRateAverage = new SparseVector(500000);
			itemRateAverage = new SparseVector(itemCount+1);
			
			// Read data:
			while((line = buffer.readLine()) != null && !line.equals("TT_EOF")) {
				if (line.length() > 0) {
					line = line.substring(1, line.length() - 1);
					
					StringTokenizer st = new StringTokenizer (line, ",");
					
					//int userID = 0;
					double rateSum = 0.0;
					int rateCount = 0;
					
					while (st.hasMoreTokens()) {
						String token = st.nextToken().trim();
						int i = token.indexOf(" ");
						
						int movieID, rate;
						int index = Integer.parseInt(token.substring(0, i));
						String data = token.substring(i+1);
						
						if (index == 0) { // User ID
							//userID = Integer.parseInt(data);
							
							rateSum = 0.0;
							rateCount = 0;
							
							userNo++;
						}
						else if (data.length() == 1) { // Rate
							movieID = index;
							rate = Integer.parseInt(data);
							
							if (rate > maxValue) {
								maxValue = rate;
							}
							else if (rate < minValue) {
								minValue = rate;
							}
							
							rateSum += rate;
							rateCount++;
							
							userRateAverage.setValue(userNo, rateSum / rateCount);
							itemRateAverage.setValue(movieID, itemRateAverage.getValue(movieID) + rate);
							(itemRateCount[movieID])++;
							
							rateMatrix.setValue(userNo, movieID, rate);
						}
						else { // Date
							// Not implement yet
						}
					}
				}
			}
			
			// Item average calculation:
			for (int i = 0; i < itemCount; i++) {
				itemRateAverage.setValue(i, itemRateAverage.getValue(i) / itemRateCount[i]);
			}
			
			userCount = userNo;
			
			// Reset user vector length:
			rateMatrix.setSize(userCount+1, itemCount+1);
			testMatrix.setSize(userCount+1, itemCount+1);
			for (int i = 1; i <= itemCount; i++) {
				rateMatrix.getColRef(i).setLength(userCount+1);
				testMatrix.getColRef(i).setLength(userCount+1);
			}
			userRateAverage.setLength(userCount+1);
			
			System.out.println ("User Count: " + userCount);
			System.out.println ("Item Count: " + itemCount);
			System.out.println ("Rating Count: " + rateMatrix.itemCount());
			
			stream.close();
		}
		catch (IOException ioe) {
			System.out.println ("No such file: " + ioe);
			System.exit(0);
		}
	}
	
	/**
	 * Split the rating matrix into train and test set, by given split data file.
	 * 
	 * @param fileName the name of split data file. 
	 * 
	 **/
	private static void readSplitData(String fileName) {
		recoverTestItems();
		
		try {
			FileInputStream stream = new FileInputStream(fileName);
			InputStreamReader reader = new InputStreamReader(stream);
			BufferedReader buffer = new BufferedReader(reader);
			
			// Read Train/Test user/item data:
			String line;
			int testUserCount = userCount;
			boolean[] isTestUser = new boolean[userCount+1];
			int[] testUserList = new int[testUserCount];
			
			for (int u = 0; u < userCount; u++) {
				isTestUser[u+1] = true;
				testUserList[u] = u+1;
			}
			
			while((line = buffer.readLine()) != null && !line.equals("TT_EOF")) {
				StringTokenizer st = new StringTokenizer (line);
				int userNo = Integer.parseInt(st.nextToken());
				int itemNo = Integer.parseInt(st.nextToken());
				isTestUser[userNo] = true;
				
				testMatrix.setValue(userNo, itemNo, rateMatrix.getValue(userNo, itemNo));
				rateMatrix.setValue(userNo, itemNo, 0.0);
			}
			
			stream.close();
		}
		catch (IOException ioe) {
			System.out.println ("No such file.");
		}
	}
	
	/**
	 * Calculate average of ratings for each user and each item.
	 * Calculated results are stored in two arrays, userRateAverage and itemRateAverage.
	 **/
	private static void calculateAverage() {
		// We assume that rateMatrix and testMatrix are already split.
		
		// User Rate Average
		for (int u = 1; u <= userCount; u++) {
			SparseVector v = rateMatrix.getRowRef(u);
			double avg = v.average();
			if (Double.isNaN(avg)) {
				avg = (double) (maxValue + minValue) / 2.0;
			}
			userRateAverage.setValue(u, avg);
		}
		
		// Item Rate Average
		for (int i = 1; i <= itemCount; i++) {
			SparseVector j = rateMatrix.getColRef(i);
			double avg = j.average();
			if (Double.isNaN(avg)) {
				avg = (double) (maxValue + minValue) / 2.0;
			}
			if (j.itemCount() > 0) {
				itemRateAverage.setValue(i, avg);
			}
		}
	}
}