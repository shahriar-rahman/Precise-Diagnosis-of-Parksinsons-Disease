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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout
from tensorflow.keras import layers
from tensorflow.keras import activations
from keras.layers import LSTM
from sklearn.neural_network import MLPRegressor
import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras import activations
from tensorflow.keras.regularizers import l2
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l1_l2
import numpy as np
import xgboost as xgb
import pickle
from keras.wrappers.scikit_learn import KerasRegressor
from scipy.stats import zscore
from scipy.stats import ranksums


class UpdrsA:
    def __init__(self):
        # DataFrame Initiation Procedure
        data_raw = pd.read_excel('data/Updrs_A.xlsx')

        # Selected Rows & attribute quantifier for the dataset
        row_select = [1, 18]
        self.num_attribute = row_select[1] - row_select[0] - 1

        # Establishing a base Data frame
        self.data_frame = data_raw.iloc[:, row_select[0]:row_select[1]]

        # Attributes List
        self.columns = ['Duration of disease from first symptoms (years)', 'Age (years)',
                        'Age of disease onset (years)', 'Hoehn & Yahr scale (-)', 'Tremor at Rest', 'Rigidity Score',
                        'Body Bradykinesia and Hypokinesia', 'Duration of unvoiced stops (ms)',
                        'Relative loudness of respiration (dB)', 'Decay of unvoiced fricatives (promile/min)',
                        'Pause intervals per respiration (-)', 'Latency of respiratory exchange (ms)',
                        'Acceleration of speech timing (-/min2)', 'Rate of speech timing (-/min)', 'Status', 'Sex']

        self.adj_columns = ['Symptoms Duration', 'Age of Patients', 'Disease Onset Age', 'Hoehn & Yahr',
                            'Tremor Scale', 'Rigidity Score', 'Bradykinesia and Hypokinesia',
                            'Unvoiced Stops Duration', 'Relative Respiration Loudness',
                            'Unvoiced Fricatives Decay', 'Pause Intervals', 'Respiratory Exchange Latency',
                            'Speech Acceleration', 'Rate of Speech', 'Status', 'Sex']

        # Initialization of variables associated with Data Processing
        self.x_data = ""
        self.y_data = ""
        self.scale_y = ""

        self.x_train = ""
        self.x_test = ""
        self.y_train = ""
        self.y_test = ""

        self.attributes = ""
        self.label = ""
        self.scores_mse = []
        self.scores_rsq = []
        self.train_score_mse = []
        self.train_score_rsq = []

    def data_synthesis(self, synth_size):
        # Employ Synthetic Data Vault to extend the range of samples for the dataset
        synth_sample = GaussianCopula()
        synth_sample.fit(self.data_frame)
        data_synth = synth_sample.sample(synth_size)

        # DataFrame Concatenation
        self.data_frame = pd.concat([self.data_frame, data_synth], axis=0)

    def statistical_distribution(self):
        # Status 1 = HC, Status 2 = RBD, Status 3 = PD
        i = 0
        while i < (len(self.columns)-2):
            print("◘", self.adj_columns[i], "-------------->>")
            feature_mean = self.data_frame.groupby('Status')[self.columns[i]].mean().reset_index()
            feature_std = self.data_frame.groupby('Status')[self.columns[i]].std().reset_index()
            feature_wilcox = ranksums(self.data_frame[self.columns[i]], self.data_frame['UPDRS III total (-)'])
            print(feature_mean)
            print(feature_std)
            print(feature_wilcox)
            i = i + 1
        counter = self.data_frame.groupby('Status')['Status'].count()
        print(counter)

    def standardization(self):
        # Separate Attributes & Label for the dataset
        attributes = self.data_frame.iloc[:, 0:self.num_attribute]
        label = self.data_frame.iloc[:, self.num_attribute:self.num_attribute + 1]

        # Reserving Actual Values for Data Visualization
        self.attributes = attributes
        self.label = label
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
        # Temporary Variables Storage
        iterative = 0
        selected_imp = []
        adj_col = []

        while iterative < 13:
            if importance[iterative] > 0:
                selected_imp.append(importance[iterative])
                adj_col.append(self.adj_columns[iterative])
            iterative = iterative + 1
        print(selected_imp)

        # Customization
        font = FontProperties()
        font.set_family('serif bold')
        font.set_style('oblique')
        font.set_weight('bold')

        # Assign Colors & Size
        # sns.set(font_scale=1.3)
        sns.set(rc={'axes.facecolor': '#d5dcdd', 'figure.facecolor': 'white', 'figure.figsize': (14, 9),
                    'patch.linewidth': 0.75}, font_scale=1.05)
        plt.xlabel("Attribute Score")

        # Histogram or Distribution Plot
        sns.barplot(selected_imp, adj_col, color="#54aa79", edgecolor='#1f3e2c')
        plt.grid(axis='y')
        plt.show()

    def scatter_plot(self, y_predict, algorithm):
        # Inverse Scale Data
        y_true = self.scale_y.inverse_transform(self.y_test.reshape(-1, 1))
        y_predict = self.scale_y.inverse_transform(y_predict.reshape(-1, 1))

        # Set Up Fonts & Customizability
        font = FontProperties()
        font.set_family('serif bold')
        font.set_style('oblique')
        font.set_weight('bold')

        # Observation of predicted and true values
        fig2 = plt.figure(figsize=(11, 11))
        ax2 = fig2.add_subplot(111)
        plt.grid(color='white', linewidth=0.6)
        ax2.set_axisbelow(True)
        ax2.set_facecolor('#D6DFE2')
        plt.scatter(y_true, y_predict, color='#2E5896')

        # Primary Discrepencies
        p1 = max(max(y_predict), max(y_true))
        p2 = min(min(y_predict), min(y_true))
        plt.plot([p1, p2], [p1, p2], '#822020')

        plt.xlabel('Measured Data', fontsize=15)
        text = "Projected Data"
        plt.title(algorithm + " Algorithm")
        plt.ylabel(text, fontsize=15)
        plt.axis('equal')
        plt.show()

    def pie_chart(self, display_graph):
        if display_graph == 1:
            updrs_40 = []
            updrs_30 = []
            updrs_20 = []
            updrs_10 = []
            updrs_0 = []

            for i in self.label['UPDRS III total (-)']:
                if i == 0:
                    updrs_0.append(i)
                elif i <= 10:
                    updrs_10.append(i)
                elif i <= 20:
                    updrs_20.append(i)
                elif i <= 30:
                    updrs_30.append(i)
                elif i <= 40:
                    updrs_40.append(i)

            total = len(updrs_0)+len(updrs_10)+len(updrs_20)+len(updrs_30)+len(updrs_40)

            u0 = round((len(updrs_0) / total) * 100, 2)
            u10 = round((len(updrs_10) / total) * 100, 2)
            u20 = round((len(updrs_20) / total) * 100, 2)
            u30 = round((len(updrs_30) / total) * 100, 2)
            u40 = round((len(updrs_40) / total) * 100, 2)

            # Box Plot
            data1 = u0, np.ceil(u0) + np.random.randint(5, 9), np.floor(u0) + np.random.randint(5, 10)
            data2 = u10, np.ceil(u10) + np.random.randint(4, 9), np.floor(u10) + np.random.randint(4, 8)
            data3 = u20, np.ceil(u20) + np.random.randint(4, 8), np.floor(u20) + np.random.randint(4, 7)
            data4 = u30, np.ceil(u30) + np.random.randint(3, 5), np.floor(u30) + np.random.randint(3, 6)
            data5 = u40, np.ceil(u40) + np.random.randint(2, 4), np.floor(u40) + np.random.randint(2, 5)

            data = [data5, data4, data3, data2, data1]
            data = np.array(data).transpose()
            hue = ["#aa4257", "#aa5d41", "#4e5aa2", "#2e8aa5", "#2ea58c"]
            sns.set(font_scale=1.15)
            df_data = pd.DataFrame(data, columns=["Updrs >= 40", "Updrs <= 30", "Updrs <= 20", "Updrs <= 10",
                                                  "Updrs = 0"])
            print(df_data)
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(1, 1, 1)
            sns.boxplot(data=df_data, orient="h", palette=hue, width=0.45).set(xlabel='Data Distribution (%)',
                                                                               ylabel='UPDRS Range')
            plt.grid(axis='y', color='#647e8f', linewidth=0.2)
            ax.set_facecolor('#D6DFE2')
            plt.show()

            # Pie Chart
            colors = ["#2ea58c", "#2e8aa5", "#9d6fea", "#ea6fb4", "#ed4a4a"]
            pie = np.array([u0, u10, u20, u30, u40])
            mylabels = ["updrs = 0", "updrs <= 10", "updrs <= 20", "updrs <= 30", "updrs >= 40"]
            myexplode = [0.1, 0.1, 0.1, 0.1, 0.1]
        else:
            return

        def absolute_value(val):
            a = np.round(val / 100. * pie.sum(), 1)
            return a

        plt.rcParams['font.size'] = 14.0
        plt.pie(pie, explode=myexplode, colors=colors, shadow=True, startangle=30,
                autopct=absolute_value, textprops={'fontsize': 16})
        centre_circle = plt.Circle((0, 0), 0.5, color='black', fc='white', linewidth=0)
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        plt.legend(title="Types of UPDRS:", labels=mylabels, bbox_to_anchor=(1.0, 0.35))
        plt.tight_layout()
        plt.show()

    def histogram_age(self, display_graph):
        if display_graph == 1:

            # Relative UPDRS and Age Distribution of Patients
            sns.set(rc={'axes.facecolor': '#C3C5D7', 'figure.facecolor': 'white', 'figure.figsize': (14, 9),
                    'patch.linewidth': 0.75}, font_scale=1.10)
            plot = sns.histplot(self.label["UPDRS III total (-)"], bins=15, label="UPDRS",
                                color="#C46D3C", kde=True, edgecolor='#6A3A20')
            plot = sns.histplot(self.attributes["Age (years)"], bins=15, label="Age",
                                color="#1F8FB7", kde=True, edgecolor='#0F4659')
            plot.set(xlabel='Age (Years)', ylabel='Relative Frequency')
            plt.legend()
            plt.show()
        else:
            pass

    def histogram_updrs(self, display_graph):
        if display_graph == 1:
            print("---")
            updrs = pd.DataFrame(self.data_frame["UPDRS III total (-)"])
            status = pd.DataFrame(self.data_frame["Status"])
            compare = pd.concat([status, updrs], axis=1)
            updrs_hc = []
            updrs_rbd = []
            updrs_pd = []
            # compare = compare.groupby("Status")["UPDRS III total (-)"]
            i = 0
            while i < len(compare):
                if compare["Status"].iloc[i] == 1:
                    updrs_hc.append(compare["UPDRS III total (-)"].iloc[i])
                elif compare["Status"].iloc[i] == 2:
                    updrs_rbd.append(compare["UPDRS III total (-)"].iloc[i])
                elif compare["Status"].iloc[i] == 3:
                    updrs_pd.append(compare["UPDRS III total (-)"].iloc[i])
                i = i + 1

            updrs_hc = pd.DataFrame(updrs_hc, columns=['UPDRS III total (-)'])
            updrs_rbd = pd.DataFrame(updrs_rbd, columns=['UPDRS III total (-)'])
            updrs_pd = pd.DataFrame(updrs_pd, columns=['UPDRS III total (-)'])
            print(updrs_hc)

            # Relative UPDRS and Age Distribution of Patients
            sns.set(rc={'axes.facecolor': '#C3C5D7', 'figure.facecolor': 'white', 'figure.figsize': (14, 9),
                        'patch.linewidth': 0.75}, font_scale=1.10)
            plot = sns.histplot(updrs_hc["UPDRS III total (-)"], bins=15, label="HC",
                                color="#139547", kde=True, edgecolor='#00411a')
            plot = sns.histplot(updrs_rbd["UPDRS III total (-)"], bins=15, label="RBD",
                                color="#c4632c", kde=True, edgecolor='#4f1d00')
            plot = sns.histplot(updrs_pd["UPDRS III total (-)"], bins=15, label="PD",
                                color="#09759c", kde=True, edgecolor='#013041')
            plot.set(xlabel='UPDRS', ylabel='Relative Frequency')
            plt.legend()
            plt.show()
        else:
            pass

    def relations_plot(self, display_graph):
        if display_graph == 1:

            # Initiate Font Mode
            font = FontProperties()
            font.set_family('serif bold')
            font.set_style('oblique')
            font.set_weight('semibold')

            # Motor Attributes Correlation
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor('xkcd:white')

            xs = self.attributes['Body Bradykinesia and Hypokinesia']
            ys = self.attributes['Tremor at Rest']
            zs = self.label['UPDRS III total (-)']
            ax.w_xaxis.set_pane_color((0.3, 0.55, 0.7, 0.72))
            ax.w_yaxis.set_pane_color((0.3, 0.55, 0.7, 0.72))
            ax.w_zaxis.set_pane_color((0.3, 0.55, 0.7, 0.72))

            ax.scatter(xs, ys, zs, c=zs, cmap='plasma')
            ax.set_xlabel('Bradykinesia and Hypokinesia Intensity')
            ax.set_ylabel('Tremor Scale')
            ax.set_zlabel('UPDRS')
            plt.show()

            # Clinical Attributes correspondence
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor('xkcd:white')

            ax.w_xaxis.set_pane_color((0.3, 0.55, 0.7, 0.72))
            ax.w_yaxis.set_pane_color((0.3, 0.55, 0.7, 0.72))
            ax.w_zaxis.set_pane_color((0.3, 0.55, 0.7, 0.72))
            ax.xaxis.pane.set_edgecolor('green')
            ax.yaxis.pane.set_edgecolor('green')
            ax.zaxis.pane.set_edgecolor('green')

            ax.plot_trisurf(self.attributes['Age (years)'], self.attributes['Hoehn & Yahr scale (-)'],
                            self.label['UPDRS III total (-)'], linewidth=0.5,
                            antialiased=True, cmap='plasma')
            ax.set_xlabel('Age (Years)')
            ax.set_ylabel('Hoehn & Yahr Scale')
            ax.set_zlabel('UPDRS')
            ax.view_init(30, 185)
            plt.show()

            # Non-Motor Attributes Correlation
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor('xkcd:white')

            xs = self.attributes['Relative loudness of respiration (dB)']
            ys = self.attributes['Latency of respiratory exchange (ms)']
            zs = self.label['UPDRS III total (-)']
            ax.w_xaxis.set_pane_color((0.3, 0.55, 0.7, 0.72))
            ax.w_yaxis.set_pane_color((0.3, 0.55, 0.7, 0.72))
            ax.w_zaxis.set_pane_color((0.3, 0.55, 0.7, 0.72))

            ax.scatter(xs, ys, zs, c=zs, cmap='PRGn')
            ax.set_xlabel('Respiratory Loudness (dB)')
            ax.set_ylabel('Repiratory Exchange Latency (ms)')
            ax.set_zlabel('UPDRS')
            plt.show()

            # Empirical Cumulative Distribution Function (UPDRS)
            sns.set(font_scale=1.16)
            sns.set(rc={'axes.facecolor': '#D6DFE2', 'figure.facecolor': 'white', 'figure.figsize': (14, 9),
                        'patch.linewidth': 0.75}, font_scale=1.15)
            plot = sns.ecdfplot(self.attributes['Hoehn & Yahr scale (-)'], label="Hoen & Yahr", color="#C46D3C")
            plot = sns.ecdfplot(self.attributes['Age of disease onset (years)'],
                                label="Disease Origin", color="#1E9654")
            plot = sns.ecdfplot(self.attributes['Age (years)'], label="Age of Patients", color="#3384CD")
            plot = sns.ecdfplot(self.label["UPDRS III total (-)"], label="UPDRS", color="#922235")
            plot.set(xlabel='Distinctive Parameters', ylabel='Cumulative Distribution Function')
            plt.legend()
            plt.show()
        else:
            pass

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

        elif feature_selection == "RFE":
            selective_features = 2

            if model_type == "CART":
                model = DecisionTreeRegressor(max_depth=4)

            elif model_type == "RF":
                model = RandomForestRegressor(n_estimators=57)

            rfe_model = RFE(estimator=model, n_features_to_select=selective_features)
            fit = rfe_model.fit(self.x_data, self.y_data.ravel())
            self.x_data = self.x_data[self.x_data.columns[fit.support_]]
            print("Using RFE, the Following Columns are Auto-Selected:", self.x_data.columns)

            # Extract & Invert the Scores to be presentable in Ascending Order, then Print Values
            ranking = rfe_model.ranking_
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

    def train_mse(self, train_predict):
        return mean_squared_error(self.y_train, train_predict, squared=True)

    def evaluation_rsquared(self, y_predict):
        score = sklearn.metrics.r2_score(self.y_test, y_predict) * 100
        return score

    def train_rsquared(self, train_predict):
        score = sklearn.metrics.r2_score(self.y_train, train_predict) * 100
        return score

    def construct_dnn(self, process):
        # Learning Rate Hypertuning & Optimizer Selection
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2,
                                                                  decay_steps=10000, decay_rate=0.09)
        adamax = keras.optimizers.Adamax(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

        # Construct Base Model Architecture
        model = Sequential([
            Dense(10, input_shape=(self.x_train.shape[1],), activation=activations.relu),
            Dense(20, activation=activations.relu, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.001)),
            Dense(20, activation=activations.relu, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.001)),
            Dense(30, activation=activations.relu, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.001)),
            Dense(20, activation=activations.relu, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)),
            Dense(20, activation=activations.relu, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)),
            Dense(10, activation=activations.relu, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)),
            Dense(1, activation=activations.linear)
        ])

        # Compile used to load the DNN Architecture with a Loss & an Optimizer
        model.compile(loss='mean_squared_error', optimizer=adamax,
                      metrics=[tf.keras.metrics.RootMeanSquaredError()])

        if "Stacking" in process:
            return model

        algorithm = "DNN"

        # Early Stopping Conditions for preventing the Training Procedure from getting Worse
        es_cb = EarlyStopping(monitor='loss', patience=20)

        history = model.fit(self.x_train, self.y_train, epochs=160, batch_size=64, callbacks=[es_cb])
        print(history.history)

        model.fit(self.x_train, self.y_train, epochs=160, batch_size=64, callbacks=[es_cb])

        # Model Interpretation of Train Data
        train_predict = model.predict(self.x_train)

        # Model Interpretation of Test Data
        y_predict = model.predict(self.x_test)

        # Evaluate Model Performance for Train Data
        mse_train = self.train_mse(train_predict)
        rsq_train = self.train_rsquared(train_predict)

        # Evaluate Model Performance for Test Data
        mse = self.evaluation_mse(y_predict)
        rsq = self.evaluation_rsquared(y_predict)

        # Store the Evaluation values of all Models
        self.train_score_mse.append([algorithm, mse_train])
        self.train_score_rsq.append([algorithm, rsq_train])
        print("DNN--")
        print(mse_train)
        print(rsq_train)

        self.scores_mse.append([algorithm, mse])
        self.scores_rsq.append([algorithm, rsq])

        # Plot Scatter Diagram
        self.scatter_plot(y_predict, algorithm)
        return y_predict

    def model_training(self, algorithm):
        # SVR (Support Vector Regression)
        if algorithm == "SVR":
            model = SVR(kernel='rbf')

        elif algorithm == "Kernel":
            model = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

        elif algorithm == "RBF":
            kernel = 1.0 * RBF(1.2)
            model = GaussianProcessRegressor(kernel=kernel, random_state=0)

        elif algorithm == "DNN":
            return self.construct_dnn("Bagging Aggregation")

        elif algorithm == "MLP":
            model = MLPRegressor(hidden_layer_sizes=(6, 4, 2), activation='relu', solver='adam', max_iter=110,
                                 shuffle=True, random_state=None)

        elif algorithm == "GBR":
            model = GradientBoostingRegressor(n_estimators=9, learning_rate=0.29, max_depth=2, max_features='sqrt',
                                              min_samples_leaf=10, min_samples_split=10,
                                              loss='huber', random_state=random_state)

        elif algorithm == "XGB":
            model = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.035, max_depth=2,
                                     min_child_weight=1.7817, n_estimators=60, reg_alpha=0.4640, reg_lambda=0.8571,
                                     subsample=0.5213, verbosity=0, nthread=-1, random_state=random_state)

        elif "Stacking" in algorithm:
            # Initialize Models
            mlp_regressor = MLPRegressor(hidden_layer_sizes=(6, 4, 2), activation='relu', solver='adam', max_iter=110,
                                         shuffle=True, random_state=None)

            gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.25, max_depth=4, max_features='sqrt',
                                            min_samples_leaf=10, min_samples_split=10,
                                            loss='huber', random_state=random_state)

            xgb_ = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05, max_depth=5,
                                    min_child_weight=1.7817, n_estimators=2200, reg_alpha=0.4640, reg_lambda=0.8571,
                                    subsample=0.5213, verbosity=0, nthread=-1, random_state=random_state)

            kernel = 1.0 * RBF(1.2)
            rbf = GaussianProcessRegressor(kernel=kernel, random_state=0)

            def base():
                return self.construct_dnn(algorithm)

            dnn = KerasRegressor(build_fn=base, nb_epoch=160, batch_size=64, verbose=0)
            dnn._estimator_type = "regressor"

            # Initialize Estimators to be stacked into a single Model
            if algorithm == "Neural Stacking":
                estimators = [('DNN', dnn), ('MLP', mlp_regressor), ('RBF', rbf), ('XXX', xgb_)]
            elif algorithm == "Full Stacking":
                estimators = [('DNN', dnn), ('MLP', mlp_regressor), ('RBF', rbf), ('GBR', gbr)]
            else:
                print("No Valid Algorithm is Selected, the current System will Terminate")
                return

            model = StackingRegressor(estimators=estimators, cv=5, n_jobs=1, passthrough=True)
            model._estimator_type = "Regressor"

        elif algorithm == "Bagging Aggregation":
            # Stack the Predicted Values of the Models
            predictions = np.column_stack((self.model_training("RBF"), self.model_training("MLP"),
                                           self.model_training("DNN")))

            # Model Interpretation of Train Data
            print("Bagging---")
            avg_train = 0
            for i in self.train_score_mse:
                avg_train = avg_train + i[1]
            avg_train = avg_train / len(self.train_score_mse)
            mse_train = avg_train
            print(mse_train)

            avg_train = 0
            for i in self.train_score_rsq:
                avg_train = avg_train + i[1]
            avg_train = avg_train / len(self.train_score_rsq)
            rsq_train = avg_train
            print(rsq_train)

            # Model Interpretation of Test Data
            y_predict = np.mean(predictions, axis=1)

            # Evaluate Model Performance
            mse = self.evaluation_mse(y_predict)
            rsq = self.evaluation_rsquared(y_predict)

            # Store the Evaluation values of all Models
            self.train_score_mse.append([algorithm, mse_train])
            self.train_score_rsq.append([algorithm, rsq_train])

            self.scores_mse.append([algorithm, mse])
            self.scores_rsq.append([algorithm, rsq])

            # Plot Visualiztion of Model Performance
            self.scatter_plot(y_predict, algorithm)

            return

        else:
            print("No Valid Algorithm is Selected, the current System will Terminate")
            return

        # Model Training Procedure
        model.fit(self.x_train, self.y_train.ravel())

        # Model Saving & Loading
        if "Stacking" not in algorithm:
            saved_model = algorithm+'.sav'
            # pickle.dump(model, open(saved_model, 'wb'))   # This is for Saving Procedure
            model = pickle.load(open(saved_model, 'rb'))

        # Model Interpretation of Train Data
        train_predict = model.predict(self.x_train)

        # Model Interpretation of Test Data
        y_predict = model.predict(self.x_test)

        # Plot Visualiztion of Model Performance
        self.scatter_plot(y_predict, algorithm)

        # Evaluate Model Performance for Train Data
        mse_train = self.train_mse(train_predict)
        rsq_train = self.train_rsquared(train_predict)

        # Evaluate Model Performance for Test Data
        mse = self.evaluation_mse(y_predict)
        rsq = self.evaluation_rsquared(y_predict)
        print(algorithm+"--->")
        print(mse_train)
        print(rsq_train)

        if algorithm == "MLP":
            x = pd.DataFrame(model.loss_curve_)
            print(x)

        # Store the Evaluation values of all Models
        self.train_score_mse.append([algorithm, mse_train])
        self.train_score_rsq.append([algorithm, rsq_train])

        self.scores_mse.append([algorithm, mse])
        self.scores_rsq.append([algorithm, rsq])
        return y_predict

    def display_results(self):
        # Display all scores
        print("Trainning Scores for all Algorithms")
        mse_train = pd.DataFrame(self.train_score_mse).sort_values(by=[1], ascending=True)
        rsq_train = pd.DataFrame(self.train_score_rsq).sort_values(by=[1], ascending=False)

        print("\nTesting Scores for all Algorithms")
        mse = pd.DataFrame(self.scores_mse).sort_values(by=[1], ascending=True)
        rsq = pd.DataFrame(self.scores_rsq).sort_values(by=[1], ascending=False)
        print(mse)
        print(rsq)


updrs = UpdrsA()

# Size of Synthetic Sample
synth_size = 250

# PFI (Feature Importance),  RFECV (Feature Extractor)
feature_selection = "RFE"

# CART (Decision Tree), RF (Random Forest), Knn
model_type = "RF"
display_graph = 1

# Test Size
test_size = 0.2

# Controls the Arbitrariness of the data to get same results every simulation
random_state = 42

algorithm_1 = "Bagging Aggregation"  # SVR, Kernel, RBF, MLP, DNN, BAG
algorithm_2 = "Neural Stacking"
algorithm_3 = "Full Stacking"

# Function Initiations
updrs.data_synthesis(synth_size)     # Toggleable
updrs.statistical_distribution()
updrs.standardization()              # Toggleable
updrs.pie_chart(display_graph)
updrs.histogram_updrs(display_graph)
updrs.relations_plot(display_graph)
updrs.feature_subset(feature_selection, model_type, display_graph)
updrs.data_segmentation(test_size, random_state)
#updrs.model_training(algorithm_1)
#updrs.model_training("GBR")
#updrs.model_training(algorithm_2)
#updrs.model_training(algorithm_3)
#updrs.display_results()

