package prea.recommender;

import java.io.BufferedReader;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.StringTokenizer;
import java.util.HashMap;

import prea.data.structure.SparseMatrix;
import prea.data.structure.SparseVector;
import prea.util.EvaluationMetrics;

/**
 * This is a class defines an SVD based recommender.  The recommender takes the
 * arff formatted dataset as input and conducts a count of frequent items.  For
 * items that are not frequent, the aggregated frequency count for that item's category
 * is checked to see if the category as a whole met the frequency count threshold, and if
 * so, then the item is represented by its category in the item-user rating matrix.  The
 * category has three seperate levels, and the bottom two of the categories are considered
 * when trying to substitute a frequent category for its infrequent singleton item.  This
 * modified user-item matrix is transformed using a SVD, the results of which are subsequently
 * used as the basis for recommendations.
 * 
 * @author Joseph Korpela
 * @since 2013. 2. 17
 * @version 0.1
 */
public class TafengSVD implements Recommender {
	/*========================================
	 * Common Variables
	 *========================================*/
	/** Rating matrix for each user (row) and item (column) */
	public SparseMatrix rateMatrix;
	
	/* =====================
	 *  COMMENT FOR AUTHORS
	 * =====================
	 * These variables may be commonly used in most recommendation systems.
	 * You can freely add new variables if needed.
	 * Note that do not delete these since they are used in evaluation method.
	 */
	/** The number of users. */
	public int userCount;
	/** The number of items. */
	public int itemCount;
	/** Maximum value of rating, existing in the dataset. */
	public double maxValue;
	/** Minimum value of rating, existing in the dataset. */
	public double minValue;
	
	/*========================================
	 * Constructors
	 *========================================*/
	/**
	 * Construct a customized recommender model with the given data.
	 * 
	 * @param uc The number of users in the dataset.
	 * @param ic The number of items in the dataset.
	 * @param max The maximum rating value in the dataset.
	 * @param min The minimum rating value in the dataset.
	 */
	public TafengSVD(int uc, int ic, double max, double min) {
		/* =====================
		 *  COMMENT FOR AUTHORS
		 * =====================
		 * Please make sure that all your custom member variables
		 * are correctly initialized in this stage.
		 * If you added new variables in "Common Variables" section above,
		 * you should initialize them properly here.
		 * You may add parameters as well.
		 */
		userCount = uc;
		itemCount = ic;
		maxValue = max;
		minValue = min;
	}
	
	/*========================================
	 * Model Builder
	 *========================================*/
	/** 
	 * Build a model with the given data and algorithm.
	 * 
	 * @param rm The rating matrix with train data.
	 */
	public void buildModel(SparseMatrix rm) {
		
		//read in the actual data and recreate a modified rateMatrix
		rateMatrix = rm;
		

		
		/* =====================
		 *  COMMENT FOR AUTHORS
		 * =====================
		 * Using the training data in "rm", you are supposed to write codes to learn your model here.
		 * If your method is memory-based one, you may leave the model as rateMatrix itself, simply by "rateMatrix = rm;".
		 * If your method is model-based algorithm, you may not need a reference to rateMatrix.
		 * (In this case, you may remove the variable "rateMatrix", just as matrix-factorization-based methods do in this toolkit.)
		 * Note that in any case train data in "rateMatrix" are read-only. You should not alter any value in it to guarantee proper operation henceforth.
		 */
	}
	
	/*========================================
	 * Prediction
	 *========================================*/
	/**
	 * Evaluate the designated algorithm with the given test data.
	 * 
	 * @param testMatrix The rating matrix with test data.
	 * 
	 * @return The result of evaluation, such as MAE, RMSE, and rank-score.
	 */
	public EvaluationMetrics evaluate(SparseMatrix testMatrix) {
		SparseMatrix predicted = new SparseMatrix(userCount+1, itemCount+1);
		for (int u = 1; u <= userCount; u++) {
			int[] testItems = testMatrix.getRowRef(u).indexList();
			
			if (testItems != null) {
				SparseVector predictedForUser = new SparseVector(itemCount);
				
				for (int i : testItems) {
					/* =====================
					 *  COMMENT FOR AUTHORS
					 * =====================
					 * From your model (model-based) or with your estimation method (memory-based) from rating matrix,
					 * you are supposed to estimate an unseen rating for an item "i" by a user "u" this point.
					 * Please store your estimation for this (u, i) pair in the variable "estimate" below.
					 * If the estimation is not simple, you may make private methods to help the decision.
					 * Obviously again, you should not alter/add/remove any value in testMatrix during the evaluation process.
					 */
					double estimate = 0.0;
										
					/* =====================
					 *  COMMENT FOR AUTHORS
					 * =====================
					 * This part ensures that your algorithm always produces a valid estimation.
					 * You may be freely remove this part under your judge, but you should make sure that
					 * your algorithm does not estimate ratings outside of legal range in the domain.
					 */
					if (estimate < minValue)
						estimate = minValue;
					else if (estimate > maxValue)
						estimate = maxValue;
					
					predictedForUser.setValue(i, estimate);
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

	
	/*=======================================
	 * Frequent Item Count
	 * ======================================*/
	/**
	 * Do a frequency count of all the items, categories, and subcategories
	 * in the dataset to determine which items should be collapsed into their
	 * parent category.
	 */
	private static void countFrequency(String fileName) {

		FileInputStream stream = new FileInputStream(fileName);
		InputStreamReader reader = new InputStreamReader(stream);
		BufferedReader buffer = new BufferedReader(reader);
		String line;

		//infrequent holds the counts for all the items that haven't yet been found frequent
		//frequent holds the counts for all items that have been found frequent
		HashMap infrequent = new HashMap();
		HashMap frequent = new HashMap();
		int minSupport = 10; //arbitrarily assigned minimum item support to be frequent
		
		//Holds the different levels of keys for an item (cat/subcat/itemid)
		Integer[] countKeys = new Integer[3];

		while((line = buffer.readLine()) != null && !line.equals("TT_EOF")) {
			if (line.contains("@DATA")){
				// skip ahead to data section
				break;
			}
		}

		// Read data:
		while((line = buffer.readLine()) != null && !line.equals("TT_EOF")) {
			if (line.length() > 0) {
				line = line.substring(1, line.length() - 1);
				
				StringTokenizer st = new StringTokenizer (line, ",");
				
				int userID = 0;
				
				while (st.hasMoreTokens()) {
					String token = st.nextToken().trim();
					int i = token.indexOf(" ");
					
					int itemID, rate;
					int index = Integer.parseInt(token.substring(0, i));
					String data = token.substring(i+1);
					
					if (index == 0) { // User ID
						userID = Integer.parseInt(data);
						
						rateSum = 0.0;
						rateCount = 0;
						
						userNo++;
					}
					else if (data.length() == 1) { // Rate
						itemID = index;
						rate = Integer.parseInt(data);
						
						if (rate > maxValue) {
							maxValue = rate;
						}
						else if (rate < minValue) {
							minValue = rate;
						}
						
						rateSum += rate;
						rateCount++;
						(itemRateCount[itemID])++;
						rateMatrix.setValue(userID, itemID, rate);
					}
					else { // Date
						// Do not use
					}
				}
			}
		}
	
		
		
		
		
		countKeys[0] = Integer.valueOf(itemID);
		countKeys[1] = Integer.valueOf();
		countKeys[2] = Integer.valueOf();
		
		//For each of the values that can be used as a key in the hashmap:
		//Checks if the key is already considered to be frequent, if so, then
		//increments that count.  If not frequent, checks if the newly incremented
		//count will be frequent and if so, moves that key from the infrequent mapping
		//to the frequent mapping.  If it will not be frequent, then just increments
		//the count for the key in the infrequent mapping.
		for (Integer itemKey : countKeys) {
			if (frequent.containsKey(itemKey)) {
				frequent.put(itemKey, frequent.get(itemKey) + 1);
			} else if (infrequent.containsKey(itemKey)){
				if (infrequent.get(itemKey) + 1 >= minSupport) {
					frequent.put(itemKey, 1);
					infrequent.remove(itemKey);
				} else {
					infrequent.put(itemKey, infrequent.get(itemKey) + 1);
				}
			} else {
				infrequent.put(itemKey, 1);
			}
		}

		stream.close();
	}
	
	/*========================================
	 * File I/O
	 *========================================*/
	/**
	 * Read the data file in ARFF format, conduct count of the frequent items, and combine any infrequent
	 * items into their parent category.  Store the representation into the rating matrix.
	 * Peripheral information such as max/min values, user/item count are also set in this method.
	 * 
	 * @param fileName The name of data file.
	 */
	private static void readArff(String fileName) {
		try {
			countFrequency(fileName);
			
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
			
			// Read data:
			while((line = buffer.readLine()) != null && !line.equals("TT_EOF")) {
				if (line.length() > 0) {
					line = line.substring(1, line.length() - 1);
					
					StringTokenizer st = new StringTokenizer (line, ",");
					
					double rateSum = 0.0;
					int rateCount = 0;
					int userID = 0;
					
					while (st.hasMoreTokens()) {
						String token = st.nextToken().trim();
						int i = token.indexOf(" ");
						
						int itemID, rate;
						int index = Integer.parseInt(token.substring(0, i));
						String data = token.substring(i+1);
						
						if (index == 0) { // User ID
							userID = Integer.parseInt(data);
							
							rateSum = 0.0;
							rateCount = 0;
							
							userNo++;
						}
						else if (data.length() == 1) { // Rate
							itemID = index;
							rate = Integer.parseInt(data);
							
							if (rate > maxValue) {
								maxValue = rate;
							}
							else if (rate < minValue) {
								minValue = rate;
							}
							
							rateSum += rate;
							rateCount++;
							(itemRateCount[itemID])++;
							rateMatrix.setValue(userID, itemID, rate);
						}
						else { // Date
							// Do not use
						}
					}
				}
			}
			
			userCount = userNo;
			
			// Reset user vector length:
			rateMatrix.setSize(userCount+1, itemCount+1);
			for (int i = 1; i <= itemCount; i++) {
				rateMatrix.getColRef(i).setLength(userCount+1);
			}
			
			System.out.println ("Data File\t" + dataFileName);
			System.out.println ("User Count\t" + userCount);
			System.out.println ("Item Count\t" + itemCount);
			System.out.println ("Rating Count\t" + rateMatrix.itemCount());
			System.out.println ("Rating Density\t" + String.format("%.2f", ((double) rateMatrix.itemCount() / (double) userCount / (double) itemCount * 100.0)) + "%");
			
			stream.close();
		}
		catch (IOException ioe) {
			System.out.println ("No such file: " + ioe);
			System.exit(0);
		}
	}
}

/* =====================
 *  COMMENT FOR AUTHORS
 * =====================
 * How to run your algorithm:
 * In the main method, you can easily test your algorithm by the following two steps.
 * 
 * First, make an instance of your recommender with a constructor you implemented.
 *  Ex) CustomRecommender myRecommender = new CustomRecommender(2000, 1000, 5.0, 1.0);
 * 
 * Second, call "testRecommender" method in main method.
 * This returns a String which contains evaluation results.
 * The first argument is the name to be printed, and the second one is the instance you created previously.
 *  Ex) System.out.println(testRecommender("MyRec", myRecommender));
 * 
 * 
 * We provide a unit test module to help you verifying whether your implementation is correct.
 * In the main method, you can make an instance of unit test module with your recommender by
 *  Ex) UnitTest u = new UnitTest(myRecommender, rateMatrix, testMatrix);
 * 
 * After you make the instance, simply call "check" method by
 *  Ex) u.check();
 * 
 * The unit test module may print some warnings or errors based on verification result.
 * If you get some errors, they should be fixed since they imply your implementation is illegal or incorrect.
 * If you get some warnings, you may concern them and we recommend to investigate your code.
 * If the unit test module does not find any problem, it will say so.
 * We recommend to rerun with various parameters since some problems may occur occasionally. 
 */


/* How to call in main method:
 * 
 * 	CustomRecommender myRecommender = new CustomRecommender(userCount, itemCount, maxValue, minValue);
 *	UnitTest u = new UnitTest(myRecommender, rateMatrix, testMatrix);
 *	u.check();
 */
