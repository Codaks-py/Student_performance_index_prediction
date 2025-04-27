A simple model that predicts the student performance index based in variables that include 'Hours Studied', 'Previous Scores', 'Extracurricular Activities',
       'Sleep Hours', 'Sample Question Papers Practiced'.
Firstly, I imported the dataset, after checking for the correlation without the 'Extracurricular Activities' column, I applied the LabelEncoder function to convert categorical values to numerical values.
Next major was using the OLS to check for Multicollinerity and there was none
I applied the train_test_split to divide the dataset into train and test to avoid the model knowing all about the data.
I checked the MSE of both the train and test dataset and calcuate the percentage error which gave  a 0.1% percentage error
