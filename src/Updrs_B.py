import pandas as pd
import numpy as np
from sdv.tabular import GaussianCopula
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib as plt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFECV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import sklearn.datasets
from sklearn.kernel_ridge import KernelRidge
import seaborn as sns
import warnings


class UpdrsB:
    def __init__(self):
        # DataFrame Initiation Procedure
        data_raw = pd.read_excel('data/Updrs_B.xlsx')

        # Selected Rows & attribute quantifier for the dataset
        row_select = [1, 17]
        self.num_attribute = row_select[1] - row_select[0] - 1

        # Establishing a base Data frame
        self.data_frame = data_raw.iloc[:, row_select[0]:row_select[1]]

        # Attributes List
        self.columns = ['AC', 'NTH', 'HTN', 'Median pitch', 'Mean pitch', 'Standard deviation',
                        'Minimum pitch', 'Maximum pitch', 'Number of pulses', 'Number of periods',
                        'Mean period', 'Standard deviation of period', 'Fraction of locally unvoiced frames',
                        'Number of voice breaks', 'Degree of voice breaks']

        # Initialization of variables associated with Data Processing
        self.x_data = ""
        self.y_data = ""
        self.scale_y = ""

        self.x_train = ""
        self.x_test = ""
        self.y_train = ""
        self.y_test = ""

    def data_synthesis(self, synth_size):
        # Employ Synthetic Data Vault to extend the range of samples for the dataset
        synth_sample = GaussianCopula()
        synth_sample.fit(self.data_frame)
        data_synth = synth_sample.sample(synth_size)

        # DataFrame Concatenation
        self.data_frame = pd.concat([self.data_frame, data_synth], axis=0)

    def standardization(self):
        # Separate Attributes & Label for the dataset
        attributes = self.data_frame.iloc[:, 0:self.num_attribute]
        label = self.data_frame.iloc[:, self.num_attribute:self.num_attribute + 1]
        label = np.array(label).reshape(-1, 1)

        # Standardize data A & B
        scale_x = StandardScaler()
        self.scale_y = StandardScaler()
        x_data = scale_x.fit_transform(attributes.values.reshape(-1, self.num_attribute))

        # Note: Reshape will give the values in a numpy array (shape: (n,1))
        self.y_data = self.scale_y.fit_transform(label.reshape(-1, 1))

        # Convert Scaled Attributes into a Data frame
        self.x_data = pd.DataFrame(x_data, columns=self.columns)

    def bar_chart(self, importance):
        # Set Up Fonts & Customizability
        fig = plt.figure(figsize=(12, 8))
        font = FontProperties()
        text = "Dataset B"
        color = '#2286AA'
        font.set_family('serif bold')
        font.set_style('oblique')
        font.set_weight('bold')
        csfont = {'fontname': 'Helvetica'}
        hfont = {'fontname': 'Comic Sans MS'}

        # Plot the data
        ax = plt.axes()
        # Setting the background color of the plot using set_facecolor() method
        ax.set_facecolor("maroon")
        fig.patch.set_facecolor('maroon')
        plt.title('Feature Importance ' + str(text), **csfont)
        plt.barh(self.columns, importance, color=color, align='center')
        plt.xlabel('Relative Importance ' + str(text), **hfont)
        plt.show()

    def scatter_plot(self, y_predict):
        y_true = self.scale_y.inverse_transform(self.y_test.reshape(-1, 1))
        y_predict = self.scale_y.inverse_transform(y_predict.reshape(-1, 1))

        # Set Up Fonts & Customizability
        plt.figure(figsize=(10, 10))
        font = FontProperties()
        font.set_family('serif bold')
        font.set_style('oblique')
        font.set_weight('bold')

        # Checking the relationship between two variables
        plt.scatter(y_true, y_predict, color='#2286AA')

        # Primary Discrepencies
        p1 = max(max(y_predict), max(y_true))
        p2 = min(min(y_predict), min(y_true))
        plt.plot([p1, p2], [p1, p2], 'black')

        plt.xlabel('True Values', fontsize=15)
        plt.ylabel('Predictions', fontsize=15)
        plt.axis('equal')
        plt.show()

    def feature_subset(self, feature_selection, model_type, display_graph):
        # Initiate Permutation Feature Importance
        if feature_selection == "PFI":
            # Selection of Algorithms
            if model_type == "CART":
                model = DecisionTreeRegressor(max_depth=4)

            elif model_type == "RF":
                model = RandomForestRegressor(n_estimators=50)

            # Note: Ravel will convert that array shape to (n, ) (i.e. flatten it)
            pfi_model = model
            pfi_model.fit(self.x_data, self.y_data.ravel())

            # Perform Permutation Feature Importance to assign Importance score
            results = permutation_importance(pfi_model, self.x_data, self.y_data,
                                             scoring='neg_mean_squared_error', random_state=1)

            # Store the scores of each Attribute
            importance = results.importances_mean
            print("Attribute Score (PFI): ", importance)

        # Recursive Feature Eliminator Cross-Validation
        elif feature_selection == "RFECV":
            # Selection of Algorithms
            if model_type == "CART":
                model = DecisionTreeRegressor(max_depth=4)

            elif model_type == "RF":
                model = RandomForestRegressor(n_estimators=50)

            # Initiate Recursive Feature Eliminator Cross Validation
            rfecv_model = RFECV(model)

            # Note: Ravel will convert that array shape to (n, ) (i.e. flatten it)
            rfecv_model.fit(self.x_data, self.y_data.ravel())

            # Readjust the Columns with Priority Features after Truncating
            self.x_data = self.x_data[self.x_data.columns[rfecv_model.support_]]
            print("Using RFECV, the following Columns are Auto-Selected:", self.x_data.columns)

            # Extract & Invert the Scores to be presentable in Ascending Order, then Print Values
            ranking = rfecv_model.ranking_
            importance = [0] * len(self.columns)

            for name, x in zip(self.columns, range(len(ranking))):
                importance[x] = 1 - (ranking[x] / 10)
                print(name, "=", ranking[x])

        # Display Graph Section
        print(importance)
        if display_graph == 1:
            self.bar_chart(importance)

    def data_segmentation(self, test_size, random_state):
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.x_data, self.y_data, test_size=test_size, random_state=random_state)

    def evaluation_mse(self, y_predict):
        return mean_squared_error(self.y_test, y_predict, squared=True)

    def evaluation_rsquared(self, y_predict):
        score = sklearn.metrics.r2_score(self.y_test, y_predict) * 100
        return score

    def model_training(self, algorithm):
        # SVR (Support Vector Regression)
        if algorithm == "SVR":
            # Hyper-parameter: kernel = Radial Basis Function, Linear
            model = SVR(kernel='rbf')

        elif algorithm == "Kernel":
            model = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

        # Model Training Procedure
        model.fit(self.x_train, self.y_train.ravel())
        y_predict = model.predict(self.x_test)

        # Plot Scatter Diagram
        self.scatter_plot(y_predict)

        # Display Model Performance
        print('Mean Squared Error ('+algorithm+f') = {self.evaluation_mse(y_predict)} %')
        print('R-Squared Accuracy ('+algorithm+f') = {self.evaluation_rsquared(y_predict)} %')


updrs = UpdrsB()

# Size of Synthetic Sample
synth_size = 825

# PFI (Feature Importance),  RFECV (Feature Extractor)
feature_selection = "RFECV"

# CART (Decision Tree), RF (Random Forest)
model_type = "RF"
display_graph = 1

# Test Size
test_size = 0.2

# Controls the Arbitrariness of the data to get same results every simulation
random_state = 42

algorithm = "SVR"  # SVR,  Kernel

# Function Initiations
updrs.data_synthesis(synth_size)     # Toggleable
updrs.standardization()              # Toggleable
updrs.feature_subset(feature_selection, model_type, display_graph)
updrs.data_segmentation(test_size, random_state)
updrs.model_training(algorithm)

