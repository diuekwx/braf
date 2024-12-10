# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
import shap

# %%
# Read in the prepped dataset
df = pd.read_csv ('chemical_compounds.csv')


# %%
# df


# %%
# Obtain general information about the dataset
# df.describe


# %%
# df.head(5)


# %%
# print(df.columns)


# %%
# print(df.info())


# %%
# print(df.isnull().sum())



# %%
# Separate coordinates into their respective columns
df[['Coordinate_1', 'Coordinate_2', 'Coordinate_3']] = df['PUBCHEM_COORDINATE_TYPE'].str.split(' ', expand=True)




# %%
# Use pd to convert all values into numeric (can also opt to use coerce)
df['Coordinate_1'] = pd.to_numeric(df['Coordinate_1'])
df['Coordinate_2'] = pd.to_numeric(df['Coordinate_2'])
df['Coordinate_3'] = pd.to_numeric(df['Coordinate_3'])


# %%
# Remove original combined coordinate column
df.drop('PUBCHEM_COORDINATE_TYPE', axis= 1, inplace = True)

#drop columns after we found out estrain importance
corr_matrix = df.corr()
e_strain_correlations = corr_matrix["E_strain"].sort_values(ascending=False)
strong_correlations = e_strain_correlations[abs(e_strain_correlations) > 0.8]
print(strong_correlations)
df.drop(['E', 'E_str'], axis= 1, inplace = True)


# Create heatmap displaying correlation between features of the dataset
# corr_matrix = df.corr()
# sns.heatmap(corr_matrix, cannot=False, cmap="coolwarm")
# plt.show()


# %%
# Assign target and features
X = df.drop(columns=['Class'])
Y = df['Class']


# %%
# Impute missing values in order to maintain dataset consistency
imputer = SimpleImputer(strategy='median') #maybe median
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)


# %%
# Create relevant train/test sets
X_temp, X_test, y_temp, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y  
)


X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp  
)


# Verify integrity of training/test sets
print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
print(f"Training labels: {y_train.shape}, Validation labels: {y_val.shape}, Test labels: {y_test.shape}")


# %%
# Scale sets for regularization and to prevent feature bias or unfair weighting
scaler = StandardScaler()


# Scaling performed to sets
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


#REMOVING LOW VARIANCE
threshold = 0.1  
variance_filter = VarianceThreshold(threshold=threshold)
X_reduced = variance_filter.fit_transform(X_train_scaled)


low_variance_features = X_train.columns[variance_filter.get_support()]
print("removed low-variance features:", low_variance_features)

remaining_features = X_train.columns[variance_filter.get_support()]

#df w reduced 
X_reduced = pd.DataFrame(X_reduced, columns=remaining_features)





# %%
# Check for bias from the mean and std error
# print(f"Mean of X_train_scaled: {X_train_scaled.mean(axis=0)}")
# print(f"Std of X_train_scaled: {X_train_scaled.std(axis=0)}")

# %%
# Mutual_info_classif for finding most relevant features
# Calculate the mutual information between each feature and the target
mi = mutual_info_classif(X_train_scaled, y_train)

# DataFrame to store the features and their scores
mi_df = pd.DataFrame({'Feature': X_train.columns, 'Mutual Information': mi})

# Sort the features by score
mi_sorted = mi_df.sort_values(by='Mutual Information', ascending=False)

print("Top Features Based On Mutual Information:")
print(mi_sorted.head(10))

# %%
# Create our SVM classifier model
svm_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42) #defaults


# Train on our scaled sets ; svm_model.fit(X_train_scaled, y_train)
# CHANGED TO FIT ON REDUCED
svm_model.fit(X_reduced, y_train)

X_test_reduced = variance_filter.transform(X_test_scaled)
X_val_reduced = variance_filter.transform(X_val_scaled)
X_test_reduced = pd.DataFrame(X_test_reduced, columns=remaining_features)
X_val_reduced = pd.DataFrame(X_val_reduced, columns=remaining_features)

#reduced test set preidctions
y_reduced_pred = svm_model.predict(X_test_reduced)
accuracy = accuracy_score(y_test, y_reduced_pred)
print(f"Accuracy: {accuracy:.2f}")

# END OF TEST FOR REDUCED COLUMNS 

# %%
# Create predictions using our SVM model
y_pred = svm_model.predict(X_test_reduced)


# Report the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# View additional relevant scores such as precision, recall, F1-score, and support
print(classification_report(y_test, y_pred))


# %%
# View the accuracy of the model on the train/test sets
train_accuracy = svm_model.score(X_reduced, y_train)
test_accuracy = svm_model.score(X_test_reduced, y_test)


print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")


# %%
# View the accuracy of the model on the validation set
y_val_pred = svm_model.predict(X_val_reduced)


print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.2f}")

# %%
# SHAP analysis of features

# Explainer for the trained SVM model (100 rows)
# explainer = shap.KernelExplainer(svm_model.predict, X_train_scaled[:100])

# SHAP values for the test set
# shap_values = explainer.shap_values(X_test_scaled)

# shap.summary_plot(shap_values, X_test_scaled, feature_names=X_test.columns, plot_type="bar")

# plt.show(block=False)

# %% [markdown]
# Cross-validation


# %%
# Perform cross-validation to further analyze the model regarding generalization and overfitting
cv_scores = cross_val_score(svm_model, X_reduced, y_train, cv=5, scoring='accuracy')


print("Cross-validation scores:", cv_scores)


print(f"Mean Cross-validation Accuracy: {np.mean(cv_scores):.2f}")
print(f"Standard Deviation Of Cross-validation Accuracy: {np.std(cv_scores):.2f}")


# %%
# View the classification report in a visualization
# ConfusionMatrixDisplay.from_estimator(svm_model, X_test_reduced, y_test)


# %% [markdown]
# Learning curve to analyze overfitting


# %%
# Learning curve
# train_sizes, train_scores, test_scores = learning_curve(
#     svm_model, X_reduced, y_train, cv=3, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
# )


# train_mean = train_scores.mean(axis=1)
# test_mean = test_scores.mean(axis=1)

# # Plots and displays learning curve
# plt.figure(figsize=(10, 6))
# plt.plot(train_sizes, train_mean, label='Training Score')
# plt.plot(train_sizes, test_mean, label='Test Score')
# plt.xlabel('Training Set Size')
# plt.ylabel('Score')
# plt.legend(loc='best')
# plt.title('Learning Curves For SVM')
# plt.tight_layout()
# plt.show()


# %% [markdown]
# Most important features


# %%
# Calculates the most important features with permutation importance
result = permutation_importance(svm_model, X_reduced, y_train, n_repeats=10, random_state=69)

importances = result.importances_mean
# Stores features along with their importance scores
features = pd.DataFrame({'Feature': X_reduced.columns, 'Importance': importances})

features_sorted = features.sort_values(by='Importance', ascending=False)

# Change n accordingly
n = 10
# Outputs the top n features that are sorted based off of their importance score
print("Top Features By Importance:")
print(features_sorted.head(n))

# %%
# Prediction
# Use dataset but can expand to reference greater range with PubChem API
data = pd.read_csv("chemical_compounds.csv")

# Keep 'CID' column for access
cids = data["CID"]

# Fix coordinate columns
data[['Coordinate_1', 'Coordinate_2', 'Coordinate_3']] = data['PUBCHEM_COORDINATE_TYPE'].str.split(' ', expand=True)

data['Coordinate_1'] = pd.to_numeric(data['Coordinate_1'])
data['Coordinate_2'] = pd.to_numeric(data['Coordinate_2'])
data['Coordinate_3'] = pd.to_numeric(data['Coordinate_3'])

data.drop('PUBCHEM_COORDINATE_TYPE', axis= 1, inplace = True)

# Keep relevant columns
data = data[remaining_features]

# Scale
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Revert to pandas
data = pd.DataFrame(data, columns=remaining_features)

# Read input.txt
with open("input.txt", "r") as input_file:
    input = [line.strip() for line in input_file.readlines()]

# Prediction in output.txt
with open("output.txt", "w") as output_file:
    # Describing output
    output_file.write("CID, Classification\n")

    for cid in input:
        # Find entry to map features according to CSV
        entry = data[cids == int(cid)]

        # Predict with model
        try:
            prediction = svm_model.predict(entry)[0]
            # Write to output file
            output_file.write(f"{cid},{prediction}\n")
        except Exception as e:
            print(f"Error: {e}")