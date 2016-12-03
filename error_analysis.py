import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

df_forest = pd.DataFrame.from_csv("ResultForest.csv", sep=',', header=None)
df_linear = pd.DataFrame.from_csv("Resultlinear.csv", sep=',', header=None)
df_svm = pd.DataFrame.from_csv("ResultSVM.csv", sep=',', header=None)
df_result = pd.DataFrame.from_csv("valid_sample_submission_5_column.csv", sep=',', header=0)
np1 = np.loadtxt("output_linear_wordvec.csv")
word_linear_pred = np.reshape(np1, (len(np1), 1))
np2 = np.loadtxt("output_forest_wordvec.csv")
word_forest_pred = np.reshape(np2, (len(np2), 1))
np3 = np.loadtxt("output_svmrbf_wordvec.csv")
word_rbf_pred = np.reshape(np3, (len(np3), 1))

forest_pred = np.array(df_forest[1])
linear_pred = np.array(df_linear[1])
svm_pred = np.array(df_svm[1])
result = np.array(df_result['predicted_score'])

forest_error = mean_squared_error(result, forest_pred)
print("Forest error %f"%forest_error)
linear_error = mean_squared_error(result, linear_pred)
print("Linear error %f"%linear_error)
svm_error = mean_squared_error(result, svm_pred)
print("SVM error %f"%svm_error)


print("Error analysis using word2Vec:")
word_linear_error = mean_squared_error(result, word_linear_pred)
print("Linear Regressor error %f"%word_linear_error)
word_forest_error = mean_squared_error(result, word_forest_pred)
print("Forest Regressor error %f"%word_forest_error)
word_svm_error = mean_squared_error(result, word_rbf_pred)
print("SVM RBF kernel error %f"%word_svm_error)

# df = pd.DataFrame.from_csv("training_set_rel3.tsv", sep='\t', header=0)
