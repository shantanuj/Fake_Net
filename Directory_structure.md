Our code is stored as ipython notebooks (.html, .ipynb) and .py files which can be viewed as html documents or run as jupyter notebooks.

The directory is arranged as follows:

1) Model 1: Predict fakeness by mining text features for a specific dataset

Dataset collection: a) Dataset_preprocessing and Scraper directories consists of the preprocessing code and custom scraper we built. b) final_combined_kaggle_scraped_pruned.csv consists of the final csv dataset. It has been zipped for space issues. Model development: a) The models are available in Model1_Predict_fake_dataset_specific and code is viewable as an html file

2) Model 2: Stance detection All related files for our stance model is present in Stance_Detection directory.

Dataset: a) data_stance directory contains the processed training and testing set collected from fakenewschallenge.com Model development: a) The final model code used is available at train_final_stance_model_tfidf_bilstm.py Since each model requires GPU access to run, we have stored parameters of our best working model. b) Additionaly, model parameters are stored in best_stance_model_parameters c) Our_older_models consists of two other models that we used which had a lower accuracy. Model predictions: a) The trained model can be loaded to make predictions as shown in Predict_stance_using_trained_models file. b) The predictions run on the kaggle dataset (previous) is shown in the same directory. Competition evaluation: a) Our final trained model can be evaluated for the competition by running the comp_scorer.py file provided by fakenewschallenge

3) Model 3: Computing domain trustability, finding similar headlines, and computing final score.

Dataset: a) We make use of the same final_combined_kaggle dataset as source dataset

Model: a) The model is a simple frequency calculator available in Model3_Domain_weighted_stance_combined b) We use LDA to detect similar topics c) Compute final weighted fake prediction using 0.3 of dataset specific score and 0.7 of domain weighted stance score.
