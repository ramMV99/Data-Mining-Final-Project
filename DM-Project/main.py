#import required libraries
import pandas as pd
import numpy as np
from scipy.stats import iqr
from scipy.signal import periodogram
from sklearn.decomposition import PCA
import math
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import cluster
from scipy.fftpack import rfft
from scipy.integrate import simps
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# Read insulin data from csv file
insulin_data_ld = pd.read_csv("InsulinData.csv", usecols=["Date", "Time", "BWZ Carb Input (grams)"], index_col=False, )
cmg_df_Df_pat1 = pd.read_csv("CGMData.csv", usecols=["Date", "Time", "Sensor Glucose (mg/dL)"])

# Create date-time stamps from CGM data
t1, t2 = [], []
for index, rows in cmg_df_Df_pat1.iterrows():
    t1.append(rows["Date"] + " " + rows["Time"])
cmg_df_Df_pat1["date_time_stamp"] = pd.to_datetime(t1)

# Create date-time stamps from insulin data
for index, rows in insulin_data_ld.iterrows():
    t2.append(rows["Date"] + " " + rows["Time"])
insulin_data_ld["date_time_stamp"] = pd.to_datetime(t2)


def extract_Meal_Data(cgm_diff, insulin_diff):
    cgm_df_cp = cgm_diff.copy()
    cgm_df_cp = cgm_df_cp.set_index('date_time_stamp')
    # Sort the CGM dataframe by the index and reset it to the default index

    cgm_df_cp = cgm_df_cp.sort_index().reset_index()

    # Make a copy of the insulin dataframe and set its index to date_time_stamp
    insulin_df_cp = insulin_diff.copy()
    insulin_df_cp = insulin_df_cp.set_index('date_time_stamp')

    # Sort the insulin dataframe by the index and remove rows with missing values
    insulin_df_cp = insulin_df_cp.sort_index().dropna()

    insulin_df_cp = insulin_df_cp.replace(0.0, np.nan).reset_index().dropna()
    insulin_df_cp = insulin_df_cp.reset_index().drop(columns='index')

    # Create an empty list to store the valid meal timestamps
    meal_timestamp_legit_list = []
    # Create an empty list to store the valid insulin values
    insulin_val_legit = []
    for i in range(insulin_df_cp.shape[0] - 1):
        if (insulin_df_cp.iloc[i + 1]['date_time_stamp'] - insulin_df_cp.iloc[i]['date_time_stamp']).seconds >= 7200:
            meal_timestamp_legit_list.append(insulin_df_cp.iloc[i]['date_time_stamp'])
            insulin_val_legit.append(insulin_df_cp.iloc[i]['BWZ Carb Input (grams)'])

    # Add the timestamp and insulin value of the last row to the respective lists
    meal_timestamp_legit_list.append(insulin_df_cp.iloc[insulin_df_cp.shape[0] - 1]['date_time_stamp'])
    insulin_val_legit.append(insulin_df_cp.iloc[insulin_df_cp.shape[0] - 1]['BWZ Carb Input (grams)'])
    # Create an empty list to store the meal data
    meal_data_list = []

    for i in range(len(meal_timestamp_legit_list)):
        cgm_meal_values = cgm_df_cp[
            (cgm_df_cp['date_time_stamp'] >= meal_timestamp_legit_list[i] - pd.Timedelta(minutes=30)) & (
                    cgm_df_cp['date_time_stamp'] <= meal_timestamp_legit_list[i] + pd.Timedelta(minutes=120))][
            'Sensor Glucose (mg/dL)'].values.tolist()
        if len(cgm_meal_values) <= 30:
            cgm_meal_values = [np.nan] * (30 - len(cgm_meal_values)) + cgm_meal_values
        else:
            cgm_meal_values = cgm_meal_values[-30:]

        # add the cgm meal values to the meal_data_list
        meal_data_list.append(cgm_meal_values)

    # create a dataframe from the meal_data_list and remove any rows with more than 6 values
    meal_df = pd.DataFrame(meal_data_list)
    meal_df = meal_df[meal_df.isna().sum(axis=1) <= 6].reset_index()

    legit_non_nan_insulin_val = []
    # create a list of valid insulin values for each row in the meal_df dataframe
    for i in range(len(meal_df)):
        legit_non_nan_insulin_val.append(insulin_val_legit[int(meal_df.iloc[i]['index'])])
    ground_truth_val = []
    min_val = min(legit_non_nan_insulin_val)

    for i in range(len(legit_non_nan_insulin_val)):
        ground_truth_val.append(int((legit_non_nan_insulin_val[i] - min_val) / 20))

    meal_df = meal_df.reset_index().drop(columns='index')
    meal_df = meal_df.interpolate(method='linear', axis=1, limit_direction='both')
    print("Bins_Value:", math.ceil((max(legit_non_nan_insulin_val) - (min_val)) / 20))
    print("Ground_Truth_Matrix:", ground_truth_val)

    return meal_df, np.asarray(ground_truth_val), meal_timestamp_legit_list

# Extract meal data from the CGM dataframe and insulin data list
meal_df, ground_truth_matrix, valid_times = extract_Meal_Data(cmg_df_Df_pat1, insulin_data_ld)
meal_df = meal_df.drop('level_0', axis=1)

# function to calculate entropy for a given row of data
def calculate_entropy(data_row):
    unique_values, value_counts = np.unique(data_row, return_counts=True)
    probabilities = value_counts / len(data_row)
    entropy = np.sum(-probabilities * np.log2(probabilities))
    return entropy

def create_Meal_Feat_Matrix(input_Meal_data):
    index = input_Meal_data.isna().sum(axis=1).replace(0, np.nan).dropna().where(lambda x: x > 6).dropna().index
    meal_data_clean = input_Meal_data.drop(input_Meal_data.index[index]).reset_index().drop(columns="index")
    meal_data_clean = meal_data_clean.interpolate(method="linear", axis=1)
    drop_index = meal_data_clean.isna().sum(axis=1).replace(0, np.nan).dropna().index
    meal_data_clean = meal_data_clean.drop(input_Meal_data.index[drop_index]).reset_index().drop(columns="index")
    meal_data_clean = meal_data_clean.dropna().reset_index().drop(columns="index")
    (
        pow_first_max,
        pow_sec_max,
        pow_third_max,
        index_first_max,
        index_sec_max,
        rms_val,
        auc_val,
    ) = ([], [], [], [], [], [], [])
    # Calculate FFT features for each row of the cleaned meal data
    for i in range(len(meal_data_clean)):
        arr = abs(rfft(meal_data_clean.iloc[:, 0:30].iloc[i].values.tolist())).tolist()
        sndOrdArr = abs(rfft(meal_data_clean.iloc[:, 0:30].iloc[i].values.tolist())).tolist()
        sndOrdArr.sort()
        pow_first_max.append(sndOrdArr[-2])
        pow_sec_max.append(sndOrdArr[-3])
        pow_third_max.append(sndOrdArr[-4])
        index_first_max.append(arr.index(sndOrdArr[-2]))
        index_sec_max.append(arr.index(sndOrdArr[-3]))
        rms_row = np.sqrt(np.mean(meal_data_clean.iloc[i, 0:30] ** 2))
        rms_val.append(rms_row)
        auc_row = abs(simps(meal_data_clean.iloc[i, 0:30], dx=1))
        auc_val.append(auc_row)
    featured_meal_mat = pd.DataFrame()

    # Calculating velocity, acceleration, row entropies, and IQR values for each row of the cleaned meal data
    velocity = np.diff(meal_data_clean, axis=1)
    velocity_min = np.min(velocity, axis=1)
    velocity_max = np.max(velocity, axis=1)
    velocity_mean = np.mean(velocity, axis=1)

    acceleration = np.diff(velocity, axis=1)
    acceleration_min = np.min(acceleration, axis=1)
    acceleration_max = np.max(acceleration, axis=1)
    acceleration_mean = np.mean(acceleration, axis=1)

    # Add calculated features to the new DataFrame
    featured_meal_mat['velocity_min'] = velocity_min
    featured_meal_mat['velocity_max'] = velocity_max
    featured_meal_mat['velocity_mean'] = velocity_mean
    featured_meal_mat['acceleration_min'] = acceleration_min
    featured_meal_mat['acceleration_max'] = acceleration_max
    featured_meal_mat['acceleration_mean'] = acceleration_mean
    row_entropies = meal_data_clean.apply(calculate_entropy, axis=1)
    featured_meal_mat['row_entropies'] = row_entropies
    iqr_values = meal_data_clean.apply(iqr, axis=1)
    featured_meal_mat['iqr_values'] = iqr_values

    power_first_max = []
    power_second_max = []
    power_third_max = []
    power_fourth_max = []
    power_fifth_max = []
    power_sixth_max = []
    for it, rwdt in enumerate(meal_data_clean.iloc[:, 0:30].values.tolist()):
        sort_ara = abs(rfft(rwdt)).tolist()
        sort_ara.sort()
        power_first_max.append(sort_ara[-2])
        power_second_max.append(sort_ara[-3])
        power_third_max.append(sort_ara[-4])
        power_fourth_max.append(sort_ara[-5])
        power_fifth_max.append(sort_ara[-6])
        power_sixth_max.append(sort_ara[-7])

    featured_meal_mat['fft col1'] = power_first_max
    featured_meal_mat['fft col2'] = power_second_max
    featured_meal_mat['fft col3'] = power_third_max
    featured_meal_mat['fft col4'] = power_fourth_max
    featured_meal_mat['fft col5'] = power_fifth_max
    featured_meal_mat['fft col6'] = power_sixth_max
    frequencies, psd_values = periodogram(meal_data_clean, axis=1)

    psd1_values = np.mean(psd_values[:, 0:6], axis=1)
    psd2_values = np.mean(psd_values[:, 5:11], axis=1)
    psd3_values = np.mean(psd_values[:, 10:16], axis=1)
    featured_meal_mat['psd1_values'] = psd1_values
    featured_meal_mat['psd2_values'] = psd2_values
    featured_meal_mat['psd3_values'] = psd3_values
    return featured_meal_mat

feat = create_Meal_Feat_Matrix(meal_df)

def sse_calculation(bin):
    sse = 0
    if len(bin) != 0:
        avg = sum(bin) / len(bin)
        for i in bin:
            sse = sse + (i - avg) * (i - avg)
    return sse


def calculate_entropy(y_true, y_pred):
    # calculating the contingency matrix for the true and predicted labels
    contingency_matrix = cluster.contingency_matrix(y_true, y_pred)
    entropy_li = []
    for i in range(0, len(contingency_matrix)):
        p = contingency_matrix[i, :]
        p = pd.Series(p).value_counts(normalize=True, sort=False)
        entropy_li.append((-p / p.sum() * np.log(p / p.sum()) / np.log(2)).sum())

    total_val = sum(contingency_matrix, 1);
    completeEntropy = 0;

    # calculate the complete entropy of the predicted labels given the true labels
    for i in range(0, len(contingency_matrix)):
        p = contingency_matrix[i, :]
        completeEntropy = completeEntropy + ((sum(p)) / (sum(total_val))) * entropy_li[i]

    return completeEntropy

# Calculating purity score of predicted labels
def calculate_purity_score(y_true, y_pred):
    contingency_matrix = cluster.contingency_matrix(y_true, y_pred)
    purity = []
    for i in range(0, len(contingency_matrix)):
        p = contingency_matrix[i, :]
        purity.append(p.max() / p.sum())
    total_val = sum(contingency_matrix, 1);
    completePurity = 0;

    for i in range(0, len(contingency_matrix)):
        p = contingency_matrix[i, :]
        completePurity = completePurity + ((sum(p)) / (sum(total_val))) * purity[i]

    return completePurity

def train_Kmeans_model(X_principal):
    # K-means clustering using scikit-learn's KMeans implementation
    print("----------------- K-means---------------")
    clusterNum = 6
    k_means = KMeans(init="k-means++", n_clusters=clusterNum, n_init=12, random_state=42)
    k_means.fit(X_principal)
    kmeans_labels = k_means.labels_
    sse = k_means.inertia_
    print("SSE of K_means clustering is : ", sse)
    
    KMeans_Clusters = []
    for bin_val in range(0, 6):
        new = []
        for i in range(0, len(kmeans_labels)):
            if kmeans_labels[i] == bin_val:
                new.append(i)
        KMeans_Clusters.append(new)

    # finding the most frequent element in a list
    def most_freq(list):
        return max(set(list), key=list.count)

    # Updating the K-means assigning the majority true label for each cluster
    Updated_kmeans_labels = kmeans_labels.copy()
    for c in range(0, 6):
        kmeans_cluster = KMeans_Clusters[c]
        true_labels = []
        for i in range(0, len(kmeans_cluster)):
            val = kmeans_cluster[i]
            true_labels.append(ground_truth_matrix[val])
        updated_label = most_freq(true_labels)
        for i in range(0, len(kmeans_cluster)):
            val = kmeans_cluster[i]
            Updated_kmeans_labels[val] = updated_label
    # Obtaining the cluster labels and calculate the entropy and purity score of the K-means clustering
    y_pred = k_means.fit_predict(X_principal)
    kmean_entropy = calculate_entropy(ground_truth_matrix, y_pred)
    print("-----------------------------------------")
    print("Entropy of K_means clustering is : ", kmean_entropy)
    print("-----------------------------------------")
    kmean_purity_score = calculate_purity_score(ground_truth_matrix, y_pred)
    print("Purity of K_means clustering is : ", kmean_purity_score)
    print("-----------------------------------------")
    print("cluster_matrix:", confusion_matrix(ground_truth_matrix, y_pred))

    return sse, kmean_entropy, kmean_purity_score

feat_matrix_std = StandardScaler().fit_transform(feat)
feat_matrix_norm = normalize(feat_matrix_std)
# Convert the numpy array to a pandas dataframe
feat_matrix_norm = pd.DataFrame(feat_matrix_norm)
pca = PCA(n_components=2)
X_principal = pca.fit_transform(feat_matrix_norm)
X_principal = pd.DataFrame(X_principal)
# Rename the columns of the dataframe
X_principal.columns = ['PCA1', 'PCA2']

# Train the KMeans model and get the SSE, entropy, and purity
kMeansSSE, kMeansEntropy, kMeansPurity = train_Kmeans_model(X_principal)

def train_DBSCAN_model(X_principal):
    # Train the DBSCAN model with the given parameters
    db = DBSCAN(eps=0.4, min_samples=7)
    db.fit(X_principal)
    db_labels = db.labels_
    unique_labels = set(db_labels) - {-1}
    data = pd.DataFrame()
    sse = 0
    for label in unique_labels:
        cluster = X_principal[db_labels == label]
        centroid = np.mean(cluster)
        sse += np.sum((cluster - centroid) ** 2)
    DBSCAN_Clusters = []
    for bin in range(-1, 6):
        new = []
        for i in range(0, len(db_labels)):
            if db_labels[i] == bin:
                new.append(i)
        DBSCAN_Clusters.append(new)

    def most_freq(List):
        if not List:
            return None
        return max(set(List), key=List.count)

    dbscan_labels_updated = db_labels.copy()
    for c in range(0, 7):
        db_cluster = DBSCAN_Clusters[c]
        true_labels = []
        for i in range(0, len(db_cluster)):
            val = db_cluster[i]
            true_labels.append(ground_truth_matrix[val])
        updated_label = most_freq(true_labels)
        # Update the dbscan labels
        for i in range(0, len(db_cluster)):
            val = db_cluster[i]
            dbscan_labels_updated[val] = updated_label
    # Create a dataframe with the updated labels assigned by DBSCAN
    data['cluster'] = dbscan_labels_updated
    sse = data.groupby('cluster').apply(lambda x: ((x - data['cluster'].mean()) ** 2).sum()).sum()
    # Calculate the entropy and purity for the updated DBSCAN labels
    entropy_dbscan = calculate_entropy(ground_truth_matrix, dbscan_labels_updated)
    purity_score_dbscan = calculate_purity_score(ground_truth_matrix, dbscan_labels_updated)

    return sse, entropy_dbscan, purity_score_dbscan

# train a DBSCAN clustering model using normalized feature matrix
dbScanSSE, entropyDbScan, purityDbScan = train_DBSCAN_model(feat_matrix_norm)

# create a dictionary of the evaluation metrics for both clustering algorithms
req_data = {
    'kMeansSSE': kMeansSSE,
    'dbScanSSE': dbScanSSE,
    'kMeansEntropy': kMeansEntropy,
    'entropyDbScan': entropyDbScan,
    'kMeansPurity': kMeansPurity,
    'purityDbScan': purityDbScan
}
# convert the dictionary to a pandas DataFrame and write it to a CSV file
results = pd.DataFrame(req_data)
results.to_csv('Results.csv', header=False, index=False)