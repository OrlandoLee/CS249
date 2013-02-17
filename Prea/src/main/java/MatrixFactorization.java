import prea.data.structure.SparseMatrix;
import prea.data.structure.SparseVector;
import prea.util.Distribution;
import prea.util.EvaluationMetrics;

/**
 * This is a class implementing matrix-factorization-based CF algorithms,
 * including regularized SVD, NMF (Lee and Seung, NIPS 2001), PMF (NIPS 2008),
 * and Bayesian PMF (ICML 2008).
 * 
 * @author Joonseok Lee
 * @since 2011. 7. 12
 * @version 20110712
 */
public class MatrixFactorization {
	/*========================================
	 * Method Names
	 *========================================*/
	/** Algorithm Code for Regularized SVD */
	public static final int REGULARIZED_SVD = 1001;
	/** Algorithm Code for NMF, optimizing Frobenius Norm */
	public static final int NON_NEGATIVE_MF_FROB = 1002;
	/** Algorithm Code for NMF, optimizing KL Divergence */
	public static final int NON_NEGATIVE_MF_KLD = 1003;
	/** Algorithm Code for PMF */
	public static final int PROBABLISTIC_MF = 1004;
	/** Algorithm Code for Bayesian PMF */
	public static final int BAYESIAN_PROBABLISTIC_MF = 1005;
	
	/*========================================
	 * Common Variables
	 *========================================*/
	/** Rating matrix for each user (row) and item (column) */
	public SparseMatrix rateMatrix;
	/** Rating matrix for test items. Not allowed to refer during training and validation phase. */
	public SparseMatrix testMatrix;
	/** Rating matrix for items which will be used during the validation phase.
	 * Not allowed to refer during training phase. */
	private SparseMatrix validationMatrix;
	/** The number of features. */
	public int featureCount;
	/** The number of users. */
	public int userCount;
	/** The number of items. */
	public int itemCount;
	/** Maximum value of rating, existing in the dataset. */
	public int maxValue;
	/** Minimum value of rating, existing in the dataset. */
	public int minValue;
	/** Learning rate parameter. */
	public double learningRate;
	/** Regularization factor parameter. */
	public double regularizer;
	/** Momentum parameter. */
	public double momentum;
	/** Maximum number of iteration. */
	public int maxIter;
	/** Offset to rating estimation. Usually this is the average of ratings. */
	public double offset;
	/** Proportion of dataset, using for validation purpose. */
	public double validationRatio;
	
	/** User profile in low-rank matrix form. */
	public SparseMatrix userFeatures;
	/** Item profile in low-rank matrix form. */
	public SparseMatrix itemFeatures;
	
	/** Indicator whether to show progress of iteration. */
	public boolean showProgress = false;
	
	/*========================================
	 * Constructors
	 *========================================*/
	/**
	 * Construct a matrix-factorization model with the given data.
	 * 
	 * @param rm The rating matrix which will be used for training.
	 * @param tm The rating matrix which will be used for testing.
	 * @param uc The number of users in the dataset.
	 * @param ic The number of items in the dataset.
	 * @param max The maximum rating value in the dataset.
	 * @param min The minimum rating value in the dataset.
	 * @param fc The number of features in low-rank factorized matrix.
	 * @param lr The learning rate for gradient-descent method.
	 * @param r The regularization factor.
	 * @param m The momentum parameter.
	 * @param iter The maximum number of iteration.
	 */
	public MatrixFactorization(SparseMatrix rm, SparseMatrix tm, int uc, int ic,
			int max, int min, int fc, double lr, double r, double m, int iter) {
		rateMatrix = rm;
		testMatrix = tm;
		userCount = uc;
		itemCount = ic;
		maxValue = max;
		minValue = min;
		featureCount = fc;
		learningRate = lr;
		regularizer = r;
		momentum = m;
		maxIter = iter;
		offset = 0.0;
		validationRatio = 0.2;
		
		userFeatures = new SparseMatrix(userCount+1, featureCount);
		itemFeatures = new SparseMatrix(featureCount, itemCount+1);
	}
	
	/*========================================
	 * Model Builder
	 *========================================*/
	/**
	 * Build a model with the given data and algorithm.
	 * 
	 * @param method The code of algorithm to be tested.
	 * It can have one of the following: REGULARIZED_SVD, NON_NEGATIVE_MF_FROB,
	 * NON_NEGATIVE_MF_KLD, PROBABLISTIC_MF, and BAYESIAN_PROBABLISTIC_MF.
	 */
	public void buildModel(int method) {
		//makeValidationSet(validationRatio);
		
		// Initialize user/item features:
		for (int u = 1; u <= userCount; u++) {
			for (int f = 0; f < featureCount; f++) {
				double rdm = Math.random() / featureCount;
				userFeatures.setValue(u, f, rdm);
			}
		}
		for (int i = 1; i <= itemCount; i++) {
			for (int f = 0; f < featureCount; f++) {
				double rdm = Math.random() / featureCount;
				itemFeatures.setValue(f, i, rdm);
			}
		}
		
		if (method == REGULARIZED_SVD) {
			// Gradient Descent:
			int round = 0;
			int rateCount = rateMatrix.itemCount();
			double prevErr = 99999;
			double currErr = 9999;
			
			while (Math.abs(prevErr - currErr) > 0.0001 && round < maxIter) {
				double sum = 0.0;
				for (int u = 1; u <= userCount; u++) {
					SparseVector items = rateMatrix.getRowRef(u);
					int[] itemIndexList = items.indexList();
					
					if (itemIndexList != null) {
						for (int i : itemIndexList) {
							// Avoid retrieving test data:
							//if (isTestUser[u] && (isTestItem.getValue(u, i) > 0.0)) {
							//	continue;
							//}
							
							SparseVector Fu = userFeatures.getRowRef(u);
							SparseVector Gi = itemFeatures.getColRef(i);
							
							double AuiEst = Fu.innerProduct(Gi);
							double AuiReal = rateMatrix.getValue(u, i);
							double err = AuiReal - AuiEst;
							sum += Math.abs(err);
							
							for (int s = 0; s < featureCount; s++) {
								double Fus = userFeatures.getValue(u, s);
								double Gis = itemFeatures.getValue(s, i);
								userFeatures.setValue(u, s, Fus + learningRate*(err*Gis - regularizer*Fus));
								itemFeatures.setValue(s, i, Gis + learningRate*(err*Fus - regularizer*Gis));
							}
						}
					}
				}
			
				prevErr = currErr;
				currErr = sum/rateCount;
			
				round++;
				
				// Show progress:
				if (showProgress)
					System.out.println(round + "\t" + currErr);
			}
		}
		
		// NMF, Based on Frobenius Norm ||V - WH||
		else if (method == NON_NEGATIVE_MF_FROB) {
			makeValidationSet(validationRatio);
			
			int round = 0;
			double prevErr = 99999;
			double currErr = 9999;
			
			while (prevErr > currErr && round < maxIter) {
				// User features update:
				for (int u = 1; u <= userCount; u++) {
					int[] itemList = rateMatrix.getRowRef(u).indexList();
					
					if (itemList != null) {
						SparseVector ratedItems = new SparseVector(itemCount+1);
						for (int i : itemList) {
							double estimate = userFeatures.getRowRef(u).innerProduct(itemFeatures.getColRef(i));
							ratedItems.setValue(i, estimate);
						}
						
						for (int f = 0; f < featureCount; f++) {
							double estimatedProfile = ratedItems.innerProduct(itemFeatures.getRowRef(f));
							double realProfile = rateMatrix.getRowRef(u).innerProduct(itemFeatures.getRowRef(f));
							double ratio = Math.max(realProfile - regularizer, 1E-9) / (estimatedProfile + 1E-9);
							
							userFeatures.setValue(u, f, userFeatures.getValue(u, f) * ratio);
						}
					}
				}
				
				// Item features update:
				for (int i = 1; i <= itemCount; i++) {
					int[] userList = rateMatrix.getColRef(i).indexList();
					
					if (userList != null) {
						SparseVector ratedUsers = new SparseVector(userCount+1);
						for (int u : userList) {
							double estimate = userFeatures.getRowRef(u).innerProduct(itemFeatures.getColRef(i));
							ratedUsers.setValue(u, estimate);
						}
						
						for (int f = 0; f < featureCount; f++) {
							double estimatedProfile = userFeatures.getColRef(f).innerProduct(ratedUsers);
							double realProfile = userFeatures.getColRef(f).innerProduct(rateMatrix.getColRef(i));
							double ratio = Math.max(realProfile - regularizer, 1E-9) / (estimatedProfile + 1E-9);
							
							itemFeatures.setValue(f, i, itemFeatures.getValue(f, i) * ratio);
						}
					}
				}
				
				round++;
				
				// show progress:
				double err = 0.0;
				
				for (int u = 1; u <= userCount; u++) {
					int[] itemList = validationMatrix.getRowRef(u).indexList();
					
					if (itemList != null) {
						for (int i : itemList) {
							double Aij = validationMatrix.getValue(u, i);
							double Bij = userFeatures.getRowRef(u).innerProduct(itemFeatures.getColRef(i));
							err += Math.pow(Aij - Bij, 2);
						}
					}
				}
				
				prevErr = currErr;
				currErr = err/validationMatrix.itemCount();
				
				if (showProgress)
					System.out.println(round + "\t" + Math.sqrt(currErr));
			}
			
			restoreValidationSet();
		}
		
		// NMF, Based on KL Divergence D(V||WH)
		else if (method == NON_NEGATIVE_MF_KLD) {
			int round = 0;
			int rateCount = rateMatrix.itemCount();
			
			while (round < maxIter) {

				// Update User Profile:
				for (int u = 1; u <= userCount; u++) {
					int[] itemList = rateMatrix.getRowRef(u).indexList();
					SparseVector updateProfile = new SparseVector(featureCount);
					
					if (itemList != null) {
						for (int i : itemList) {
							double realRate = rateMatrix.getValue(u, i);
							double estimatedRate = userFeatures.getRowRef(u).innerProduct(itemFeatures.getColRef(i));
							
							SparseVector item = itemFeatures.getCol(i);
							item = item.scale(realRate / estimatedRate);
							updateProfile = updateProfile.plus(item);
						}
					}
					
					for (int f = 0; f < featureCount; f++) {
						userFeatures.setValue(u, f, userFeatures.getValue(u, f) * updateProfile.getValue(f));
					}
				}
				
				// Normalize user profile:
				for (int f = 0; f < featureCount; f++) {
					SparseVector featureVector = userFeatures.getColRef(f);
					double featureSum = featureVector.sum();
					
					for (int u : featureVector.indexList())
						userFeatures.setValue(u, f, featureVector.getValue(u) / featureSum);
				}
				
				// Update Item Profile:
				for (int i = 1; i <= itemCount; i++) {
					int[] userList = rateMatrix.getColRef(i).indexList();
					SparseVector updateProfile = new SparseVector(featureCount);
					
					if (userList != null) {
						for (int u : userList) {
							double realRate = rateMatrix.getValue(u, i);
							double estimatedRate = userFeatures.getRowRef(u).innerProduct(itemFeatures.getColRef(i));
							
							SparseVector user = userFeatures.getRow(u);
							user = user.scale(realRate / estimatedRate);
							updateProfile = updateProfile.plus(user);
						}
					}
					
					for (int f = 0; f < featureCount; f++) {
						itemFeatures.setValue(f, i, itemFeatures.getValue(f, i) * updateProfile.getValue(f));
					}
				}

				round++;
				
				// show progress:
				double err = 0.0;
				for (int u = 1; u <= userCount; u++) {
					int[] itemList = rateMatrix.getRowRef(u).indexList();
					
					if (itemList != null) {
						for (int i : itemList) {
							double Aij = rateMatrix.getValue(u, i);
							double Bij = userFeatures.getRowRef(u).innerProduct(itemFeatures.getColRef(i));
							err += Aij * Math.log(Aij/Bij) - Aij + Bij;
						}
					}
				}
				
				if (showProgress)
					System.out.println(round + "\t" + (err/rateCount));
			}
		}
		
		else if (method == PROBABLISTIC_MF) {
			int round = 0;
			double prevErr = 99999;
			double currErr = 9999;
			
			int rateCount = rateMatrix.itemCount();
			double mean_rating = rateMatrix.average();
			this.offset = mean_rating;
			
			SparseMatrix userFeaturesInc = new SparseMatrix(userCount+1, featureCount);
			SparseMatrix itemFeaturesInc = new SparseMatrix(featureCount, itemCount+1);
			
			// Initialize with random values:
			for (int f = 0; f < featureCount; f++) {
				for (int u = 1; u <= userCount; u++) {
					userFeatures.setValue(u, f, 0.1 * Distribution.normalRandom(0, 1));
				}
				for (int i = 1; i <= itemCount; i++) {
					itemFeatures.setValue(f, i, 0.1 * Distribution.normalRandom(0, 1));
				}
			}
			
			// Iteration:
			while (prevErr > currErr && round < maxIter) {
				double errSum = 0.0;
				SparseMatrix userDelta = new SparseMatrix (userCount+1, featureCount);
				SparseMatrix itemDelta = new SparseMatrix (featureCount, itemCount+1);
				
				for (int u = 1; u <= userCount; u++) {
					int[] itemList = rateMatrix.getRowRef(u).indexList();
					if (itemList == null) continue;
					
					for (int i : itemList) {
						// Compute predictions:
						double realRating = rateMatrix.getValue(u, i) - mean_rating;
						double prediction = userFeatures.getRowRef(u).innerProduct(itemFeatures.getColRef(i));
						double userFeatureSum = userFeatures.getRowRef(u).sum();
						double itemFeatureSum = itemFeatures.getColRef(i).sum();
						double err = Math.pow(prediction - realRating, 2) + 0.5 * regularizer * (Math.pow(userFeatureSum, 2) + Math.pow(itemFeatureSum, 2));
						errSum += err;
						
						// Compute gradients:
						double repmatValue = 2 * (prediction - realRating);
						for (int f = 0; f < featureCount; f++) {
							double Ix_p = repmatValue * itemFeatures.getValue(f, i) + regularizer * userFeatures.getValue(u, f);
							double Ix_m = repmatValue * userFeatures.getValue(u, f) + regularizer * itemFeatures.getValue(f, i);
							
							userDelta.setValue(u, f, userDelta.getValue(u, f) + Ix_p);
							itemDelta.setValue(f, i, itemDelta.getValue(f, i) + Ix_m);
						}
					}
				}
				
				// Update user and item features:
				userFeaturesInc = userFeaturesInc.scale(momentum).plus(userDelta.scale(learningRate / rateCount));
			    userFeatures = userFeatures.plus(userFeaturesInc.scale(-1));
			    
			    itemFeaturesInc = itemFeaturesInc.scale(momentum).plus(itemDelta.scale(learningRate / rateCount));
			    itemFeatures = itemFeatures.plus(itemFeaturesInc.scale(-1));
			    
			    
			    round++;
			    
			    // show progress:
			    prevErr = currErr;
			    currErr = errSum/rateCount;
			    
			    if (showProgress)
			    	System.out.println(round + "\t" + currErr);
			}
		}
		
		else if (method == BAYESIAN_PROBABLISTIC_MF) {
			double prevErr = 99999999;
			
			// Initialize hierarchical priors:
			int beta = 2; // observation noise (precision)
			SparseVector mu_u = new SparseVector(featureCount);
			SparseVector mu_m = new SparseVector(featureCount);
			SparseMatrix alpha_u = SparseMatrix.makeIdentity(featureCount);
			SparseMatrix alpha_m = SparseMatrix.makeIdentity(featureCount);
			
			// parameters of Inv-Whishart distribution:
			SparseMatrix WI_u = SparseMatrix.makeIdentity(featureCount);
			int b0_u = 2;
			int df_u = featureCount;
			SparseVector mu0_u = new SparseVector(featureCount);
			
			SparseMatrix WI_m = SparseMatrix.makeIdentity(featureCount);
			int b0_m = 2;
			int df_m = featureCount;
			SparseVector mu0_m = new SparseVector(featureCount);
			
			double mean_rating = rateMatrix.average();
			this.offset = mean_rating;
			
			// Initialization using MAP solution found by PMF:
			for (int f = 0; f < featureCount; f++) {
				for (int u = 1; u <= userCount; u++) {
					userFeatures.setValue(u, f, Distribution.normalRandom(0, 1));
				}
				for (int i = 1; i <= itemCount; i++) {
					itemFeatures.setValue(f, i, Distribution.normalRandom(0, 1));
				}
			}
			
			for (int f = 0; f < featureCount; f++) {
				mu_u.setValue(f, userFeatures.getColRef(f).average());
				mu_m.setValue(f, itemFeatures.getRowRef(f).average());
			}
			alpha_u = (userFeatures.covariance()).inverse();
			alpha_m = (itemFeatures.transpose().covariance()).inverse();


			// Iteration:
			SparseVector x_bar = new SparseVector(featureCount);
			SparseVector normalRdn = new SparseVector(featureCount);
			SparseMatrix S_bar, WI_post, lam;
			SparseVector mu_temp;
			double df_upost, df_mpost;
			
			for (int round = 1; round <= maxIter; round++) {
				// Sample from user hyper parameters:
				int M = userCount;
				
				for (int f = 0; f < featureCount; f++) {
					x_bar.setValue(f, userFeatures.getColRef(f).average());
				}
				S_bar = userFeatures.covariance();

				//WI_post = (WI_u.inverse().plus(S_bar.scale(N)).plus((mu0_u.minus(x_bar)).outerProduct(mu0_u.minus(x_bar)).scale(N * b0_u / (b0_u + N))).inverse());
				SparseVector mu0_u_x_bar = mu0_u.minus(x_bar);
				SparseMatrix e1e2 = mu0_u_x_bar.outerProduct(mu0_u_x_bar).scale((double) M * (double) b0_u / (double) (b0_u + M));
				WI_post = WI_u.inverse().plus(S_bar.scale(M)).plus(e1e2);
				WI_post = WI_post.inverse();
				WI_post = (WI_post.plus(WI_post.transpose())).scale(0.5);
				
				df_upost = df_u + M;
				SparseMatrix wishrnd_u = Distribution.wishartRandom(WI_post, df_upost);
				if (wishrnd_u != null)
					alpha_u = wishrnd_u; 
				mu_temp = ((mu0_u.scale(b0_u)).plus(x_bar.scale(M))).scale(1 / ((double) b0_u + (double) M));
				lam = alpha_u.scale(b0_u + M).inverse().cholesky();
				
				if (lam != null) {
					lam = lam.transpose();
					
					normalRdn = new SparseVector(featureCount);
					for (int f = 0; f < featureCount; f++) {
						normalRdn.setValue(f, Distribution.normalRandom(0, 1));
					}
					
					mu_u = lam.times(normalRdn).plus(mu_temp);
				}
				
				//Sample from item hyper parameters:  
				int N = itemCount;
				
				for (int f = 0; f < featureCount; f++) {
					x_bar.setValue(f, itemFeatures.getRowRef(f).average());
				}
				S_bar = itemFeatures.transpose().covariance();

				//WI_post = (WI_m.inverse().plus(S_bar.scale(N)).plus((mu0_m.minus(x_bar)).outerProduct(mu0_m.minus(x_bar)).scale(N * b0_m / (b0_m + N))).inverse());
				SparseVector mu0_m_x_bar = mu0_m.minus(x_bar);
				SparseMatrix e3e4 = mu0_m_x_bar.outerProduct(mu0_m_x_bar).scale((double) N * (double) b0_m / (double) (b0_m + N));
				WI_post = WI_m.inverse().plus(S_bar.scale(N)).plus(e3e4);
				WI_post = WI_post.inverse();
				WI_post = (WI_post.plus(WI_post.transpose())).scale(0.5);
				
				df_mpost = df_m + N;
				SparseMatrix wishrnd_m = Distribution.wishartRandom(WI_post, df_mpost);
				if (wishrnd_m != null)
					alpha_m = wishrnd_m;
				mu_temp = ((mu0_m.scale(b0_m)).plus(x_bar.scale(N))).scale(1 / ((double) b0_m + (double) N));
				lam = alpha_m.scale(b0_m + N).inverse().cholesky();
				
				if (lam != null) {
					lam = lam.transpose();
				
					normalRdn = new SparseVector(featureCount);
					for (int f = 0; f < featureCount; f++) {
						normalRdn.setValue(f, Distribution.normalRandom(0, 1));
					}
					
					mu_m = lam.times(normalRdn).plus(mu_temp);
				}
				
				// Gibbs updates over user and item feature vectors given hyper parameters:
				for (int gibbs = 1; gibbs < 2; gibbs++) {
					// Infer posterior distribution over all user feature vectors 
					for (int uu = 1; uu <= userCount; uu++) {
						// list of items rated by user uu:
						int[] ff = rateMatrix.getRowRef(uu).indexList();
						
						if (ff == null)
							continue;
						
						// Avoid retrieving test data:
						int ff_idx = 0;
						for (int t = 0; t < ff.length; t++) {
							int i = ff[t];
							if (testMatrix.getValue(uu, i) > 0.0) {
								continue;
							}
							else {
								ff[ff_idx] = ff[t];
								ff_idx++;
							}
						}
						
						// features of items rated by user uu:
						SparseMatrix MM = new SparseMatrix(ff_idx, featureCount);
						SparseVector rr = new SparseVector(ff_idx);
						int idx = 0;
						for (int t = 0; t < ff_idx; t++) {
							int i = ff[t];
							rr.setValue(idx, rateMatrix.getValue(uu, i) - mean_rating);
							for (int f = 0; f < featureCount; f++) {
								MM.setValue(idx, f, itemFeatures.getValue(f, i));
							}
							idx++;
						}
						
						SparseMatrix covar = (alpha_u.plus((MM.transpose().times(MM)).scale(beta))).inverse();
						SparseVector a = MM.transpose().times(rr).scale(beta);
						SparseVector b = alpha_u.times(mu_u);
						SparseVector mean_u = covar.times(a.plus(b));
						lam = covar.cholesky();
						
						if (lam != null) {
							lam = lam.transpose();
							for (int f = 0; f < featureCount; f++) {
								normalRdn.setValue(f, Distribution.normalRandom(0, 1));
							}
							
							SparseVector w1_P1_uu = lam.times(normalRdn).plus(mean_u);
							
							for (int f = 0; f < featureCount; f++) {
								userFeatures.setValue(uu, f, w1_P1_uu.getValue(f));
							}
						}
					}
					
					// Infer posterior distribution over all movie feature vectors 
					for (int ii = 1; ii <= itemCount; ii++) {
						// list of users who rated item ii:
						int[] ff = rateMatrix.getColRef(ii).indexList();
						
						if (ff == null)
							continue;
						
						// Avoid retrieving test data:
						int ff_idx = 0;
						for (int t = 0; t < ff.length; t++) {
							int u = ff[t];
							if (testMatrix.getValue(u, ii) > 0.0) {
								continue;
							}
							else {
								ff[ff_idx] = ff[t];
								ff_idx++;
							}
						}
						
						// features of users who rated item ii:
						SparseMatrix MM = new SparseMatrix(ff_idx, featureCount);
						SparseVector rr = new SparseVector(ff_idx);
						int idx = 0;
						for (int t = 0; t < ff_idx; t++) {
							int u = ff[t];
							rr.setValue(idx, rateMatrix.getValue(u, ii) - mean_rating);
							for (int f = 0; f < featureCount; f++) {
								MM.setValue(idx, f, userFeatures.getValue(u, f));
							}
							idx++;
						}
						
						SparseMatrix covar = (alpha_m.plus((MM.transpose().times(MM)).scale(beta))).inverse();
						SparseVector a = MM.transpose().times(rr).scale(beta);
						SparseVector b = alpha_m.times(mu_m);
						SparseVector mean_m = covar.times(a.plus(b));
						lam = covar.cholesky();
						
						if (lam != null) {
							lam = lam.transpose();
							for (int f = 0; f < featureCount; f++) {
								normalRdn.setValue(f, Distribution.normalRandom(0, 1));
							}
							
							SparseVector w1_M1_ii = lam.times(normalRdn).plus(mean_m);
							
							for (int f = 0; f < featureCount; f++) {
								itemFeatures.setValue(f, ii, w1_M1_ii.getValue(f));
							}
						}
					}
				}
				
				
				// show progress:
				double err = 0.0;
				for (int u = 1; u <= userCount; u++) {
					int[] itemList = rateMatrix.getRowRef(u).indexList();
					
					if (itemList != null) {
						for (int i : itemList) {
							double Aij = rateMatrix.getValue(u, i) - mean_rating;
							double Bij = userFeatures.getRowRef(u).innerProduct(itemFeatures.getColRef(i));
							err += Math.pow(Aij - Bij, 2);
						}
					}
				}
				
				if (showProgress)
					System.out.println(round + "\t" + (err / rateMatrix.itemCount()));
				
				if (prevErr < err) {
					break;
				}
				else {
					prevErr = err;
				}
			}
		}
		
		//restoreValidationSet();
	}
	
	/*========================================
	 * Prediction
	 *========================================*/
	/**
	 * Evaluate the designated algorithm with the given test data.
	 * 
	 * @param method The code of algorithm to be tested.
	 * It can have one of the following: REGULARIZED_SVD, NON_NEGATIVE_MF_FROB,
	 * NON_NEGATIVE_MF_KLD, PROBABLISTIC_MF, and BAYESIAN_PROBABLISTIC_MF.
	 * @return The result of evaluation, such as MAE, RMSE, and rank-score.
	 */
	public EvaluationMetrics evaluate(int method) {
		SparseMatrix predicted = new SparseMatrix(userCount+1, itemCount+1);
		for (int u = 1; u <= userCount; u++) {
			int[] testItems = testMatrix.getRowRef(u).indexList();
			
			if (testItems != null) {
				SparseVector predictedForUser = new SparseVector(itemCount);
				
				for (int t = 0; t < testItems.length; t++) {
					double estimate = this.offset + userFeatures.getRowRef(u).innerProduct(itemFeatures.getColRef(testItems[t]));
					
					// rating should be located between minValue and maxValue:
					if (estimate < minValue)
						estimate = minValue;
					else if (estimate > maxValue)
						estimate = maxValue;
					
					predictedForUser.setValue(testItems[t], estimate);
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
	 * Train/Validation set management
	 *========================================*/
	/**
	 * Items which will be used for validation purpose are moved from rateMatrix to validationMatrix.
	 * 
	 * @param validationRatio Proportion of dataset, using for validation purpose.
	 */
	private void makeValidationSet(double validationRatio) {
		validationMatrix = new SparseMatrix(userCount+1, itemCount+1);
		
		int validationCount = (int) (rateMatrix.itemCount() * validationRatio);
		while (validationCount > 0) {
			int index = (int) (Math.random() * userCount) + 1;
			SparseVector row = rateMatrix.getRowRef(index);
			int[] itemList = row.indexList();
			
			if (itemList != null && itemList.length > 5) {
				int index2 = (int) (Math.random() * itemList.length);
				validationMatrix.setValue(index, itemList[index2], rateMatrix.getValue(index, itemList[index2]));
				rateMatrix.setValue(index, itemList[index2], 0.0);
				
				validationCount--;
			}
		}
	}
	
	/** Items in validationMatrix are moved to original rateMatrix. */
	private void restoreValidationSet() {
		for (int i = 1; i <= userCount; i++) {
			SparseVector row = validationMatrix.getRowRef(i);
			int[] itemList = row.indexList();
			
			if (itemList != null) {
				for (int j : itemList) {
					rateMatrix.setValue(i, j, validationMatrix.getValue(i, j));
					//validationMatrix.setValue(i, j, 0.0);
				}
			}
		}
	}
}