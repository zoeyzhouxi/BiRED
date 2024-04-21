# %% md
# Merging of the different MIMIC data sources

##### This file takes as inputs :

# -LAB_processed(
# from notebook LabEvents) with the pre-selected and cleaned lab measurements of the patients
#
# -INPUTS_processed(
# from notebook Admissions) with the pre-selected and cleaned inputs to the patients
#
# -Admissions_processed(
# from the notebook
#
# Admissions) with the death label of the patients
#
# -Diagnoses_ICD
# with the ICD9 codes of each patient.
#
# ##### This notebook outputs :
#
# -death_tags.csv.A
# dataframe
# with the patient id and the corresponding death label
#
# -complete_tensor_csv.A
# dataframe
# containing
# all
# the
# measurments in tensor
# version.
#
# -complete_tensor_train.csv.A
# dataframe
# containing
# all
# the
# training
# measurments in tensor
# version.
#
# -complete_tensor_val.csv.A
# dataframe
# containing
# all
# the
# validation
# measurments in tensor
# version.
#
# -complete_covariates.csv.A
# dataframe
# with the ICD9 covariates codes (binary) of each patient index.
# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import datetime
from datetime import timedelta
import numpy as np
import torch

file_path = "./Data/Full_MIMIC/"
outfile_path = "./Data/Clean_data/"
processed_path = file_path + "Processed/"
# %%
lab_df = pd.read_csv(processed_path + "LAB_processed.csv")[["SUBJECT_ID", "HADM_ID", "CHARTTIME", "VALUENUM", "LABEL"]]
inputs_df = pd.read_csv(processed_path + "INPUTS_processed.csv")[["SUBJECT_ID", "HADM_ID", "CHARTTIME", "AMOUNT", "LABEL"]]
outputs_df = pd.read_csv(processed_path + "OUTPUTS_processed.csv")[["SUBJECT_ID", "HADM_ID", "CHARTTIME", "VALUE", "LABEL"]]
presc_df = pd.read_csv(processed_path + "PRESCRIPTIONS_processed.csv")[
    ["SUBJECT_ID", "HADM_ID", "CHARTTIME", "DOSE_VAL_RX", "DRUG"]]

death_tags_df = pd.read_csv(processed_path+ "death_tags.csv")
# %%
# Process names of columns to have the same everywhere.

# Change the name of amount. Valuenum for every table
inputs_df["VALUENUM"] = inputs_df["AMOUNT"]
inputs_df.head()
inputs_df = inputs_df.drop(["AMOUNT"], axis=1).copy()

# Change the name of amount. Valuenum for every table
outputs_df["VALUENUM"] = outputs_df["VALUE"]
outputs_df = outputs_df.drop(["VALUE"], axis=1).copy()

# Change the name of amount. Valuenum for every table
presc_df["VALUENUM"] = presc_df["DOSE_VAL_RX"]
presc_df = presc_df.drop(["DOSE_VAL_RX"], axis=1).copy()
presc_df["LABEL"] = presc_df["DRUG"]
presc_df = presc_df.drop(["DRUG"], axis=1).copy()

# Tag to distinguish between lab and inputs events
inputs_df["Origin"] = "Inputs"
lab_df["Origin"] = "Lab"
outputs_df["Origin"] = "Outputs"
presc_df["Origin"] = "Prescriptions"

# merge both dfs.
merged_df1 = (inputs_df.append(lab_df)).reset_index()
merged_df2 = (merged_df1.append(outputs_df)).reset_index()
merged_df2.drop(["level_0"], axis=1, inplace=True)
merged_df = (merged_df2.append(presc_df)).reset_index()

# merged_df=lab_df.reset_index()

# Check that all labels have different names.
assert (merged_df["LABEL"].nunique() == (
            inputs_df["LABEL"].nunique() + lab_df["LABEL"].nunique() + outputs_df["LABEL"].nunique() + presc_df[
        "LABEL"].nunique()))

# %%
merged_df.head()
# %%
# Set the reference time as the lowest chart time for each admission.
merged_df['CHARTTIME'] = pd.to_datetime(merged_df["CHARTTIME"], format='%Y-%m-%d %H:%M:%S')
ref_time = merged_df.groupby("HADM_ID")["CHARTTIME"].min()

merged_df_1 = pd.merge(ref_time.to_frame(name="REF_TIME"), merged_df, left_index=True, right_on="HADM_ID")
merged_df_1["TIME_STAMP"] = merged_df_1["CHARTTIME"] - merged_df_1["REF_TIME"]
assert (len(merged_df_1.loc[merged_df_1["TIME_STAMP"] < timedelta(hours=0)].index) == 0)
# %%
# Create a label code (int) for the labels.
label_dict = dict(zip(list(merged_df_1["LABEL"].unique()), range(len(list(merged_df_1["LABEL"].unique())))))
merged_df_1["LABEL_CODE"] = merged_df_1["LABEL"].map(label_dict)

merged_df_short = merged_df_1[["HADM_ID", "VALUENUM", "TIME_STAMP", "LABEL_CODE", "Origin"]]

# To do : store the label dictionnary in a csv file.
# %%
label_dict_df = pd.Series(merged_df_1["LABEL"].unique()).reset_index()
label_dict_df.columns = ["index", "LABEL"]
label_dict_df["LABEL_CODE"] = label_dict_df["LABEL"].map(label_dict)
label_dict_df.drop(["index"], axis=1, inplace=True)
label_dict_df.to_csv(outfile_path + "label_dict.csv")
# %% md
#### Time binning of the data
# First
# we
# select
# the
# data
# up
# to
# a
# certain
# time
# limit(48
# hours)
# %%
# Now only select values within 48 hours.
merged_df_short = merged_df_short.loc[(merged_df_short["TIME_STAMP"] < timedelta(hours=48))]
print("Number of patients considered :" + str(merged_df_short["HADM_ID"].nunique()))
# %%
# Plot the number of "hits" based on the binning. That is, the number of measurements falling into the same bin in function of the number of bins
bins_num = range(1, 60)
merged_df_short_binned = merged_df_short.copy()
hits_vec = []
for bin_k in bins_num:
    time_stamp_str = "TIME_STAMP_Bin_" + str(bin_k)
    merged_df_short_binned[time_stamp_str] = round(
        merged_df_short_binned["TIME_STAMP"].dt.total_seconds() * bin_k / (100 * 36)).astype(int)
    hits_prop = merged_df_short_binned.duplicated(subset=["HADM_ID", "LABEL_CODE", time_stamp_str]).sum() / len(
        merged_df_short_binned.index)
    hits_vec += [hits_prop]
plt.plot(bins_num, hits_vec)
plt.title("Percentage of hits in function of the binning factor")
plt.xlabel("Number of bins/hour")
plt.ylabel("% of hits")
plt.show()
# %%
# We choose 60 bins per hour. We now need to aggregate the data in different ways.
bin_k = 60
merged_df_short["TIME"] = round(merged_df_short_binned["TIME_STAMP"].dt.total_seconds() * bin_k / (100 * 36)).astype(
    int)

# For lab, we have to average the duplicates.
lab_subset = merged_df_short.loc[merged_df_short["Origin"] == "Lab", ["HADM_ID", "TIME", "LABEL_CODE", "VALUENUM"]]
lab_subset["KEY_ID"] = lab_subset["HADM_ID"].astype(str) + "/" + lab_subset["TIME"].astype(str) + "/" + lab_subset[
    "LABEL_CODE"].astype(str)
lab_subset["VALUENUM"] = lab_subset["VALUENUM"].astype(float)

lab_subset_s = lab_subset.groupby("KEY_ID")["VALUENUM"].mean().to_frame().reset_index()

lab_subset.rename(inplace=True, columns={"VALUENUM": "ExVALUENUM"})
lab_s = pd.merge(lab_subset, lab_subset_s, on="KEY_ID")
assert (not lab_s.isnull().values.any())

# For inputs, we have to sum the duplicates.
input_subset = merged_df_short.loc[merged_df_short["Origin"] == "Inputs", ["HADM_ID", "TIME", "LABEL_CODE", "VALUENUM"]]
input_subset["KEY_ID"] = input_subset["HADM_ID"].astype(str) + "/" + input_subset["TIME"].astype(str) + "/" + \
                         input_subset["LABEL_CODE"].astype(str)
input_subset["VALUENUM"] = input_subset["VALUENUM"].astype(float)

input_subset_s = input_subset.groupby("KEY_ID")["VALUENUM"].sum().to_frame().reset_index()

input_subset.rename(inplace=True, columns={"VALUENUM": "ExVALUENUM"})
input_s = pd.merge(input_subset, input_subset_s, on="KEY_ID")
assert (not input_s.isnull().values.any())

# For outpus, we have to sum the duplicates as well.
output_subset = merged_df_short.loc[
    merged_df_short["Origin"] == "Outputs", ["HADM_ID", "TIME", "LABEL_CODE", "VALUENUM"]]
output_subset["KEY_ID"] = output_subset["HADM_ID"].astype(str) + "/" + output_subset["TIME"].astype(str) + "/" + \
                          output_subset["LABEL_CODE"].astype(str)
output_subset["VALUENUM"] = output_subset["VALUENUM"].astype(float)

output_subset_s = output_subset.groupby("KEY_ID")["VALUENUM"].sum().to_frame().reset_index()

output_subset.rename(inplace=True, columns={"VALUENUM": "ExVALUENUM"})
output_s = pd.merge(output_subset, output_subset_s, on="KEY_ID")
assert (not output_s.isnull().values.any())

# For prescriptions, we have to sum the duplicates as well.
presc_subset = merged_df_short.loc[
    merged_df_short["Origin"] == "Prescriptions", ["HADM_ID", "TIME", "LABEL_CODE", "VALUENUM"]]
presc_subset["KEY_ID"] = presc_subset["HADM_ID"].astype(str) + "/" + presc_subset["TIME"].astype(str) + "/" + \
                         presc_subset["LABEL_CODE"].astype(str)
presc_subset["VALUENUM"] = presc_subset["VALUENUM"].astype(float)

presc_subset_s = presc_subset.groupby("KEY_ID")["VALUENUM"].sum().to_frame().reset_index()

presc_subset.rename(inplace=True, columns={"VALUENUM": "ExVALUENUM"})
presc_s = pd.merge(presc_subset, presc_subset_s, on="KEY_ID")
assert (not presc_s.isnull().values.any())

# Now remove the duplicates/
lab_s = (lab_s.drop_duplicates(subset=["HADM_ID", "LABEL_CODE", "TIME"]))[
    ["HADM_ID", "TIME", "LABEL_CODE", "VALUENUM"]].copy()
input_s = (input_s.drop_duplicates(subset=["HADM_ID", "LABEL_CODE", "TIME"]))[
    ["HADM_ID", "TIME", "LABEL_CODE", "VALUENUM"]].copy()
output_s = (output_s.drop_duplicates(subset=["HADM_ID", "LABEL_CODE", "TIME"]))[
    ["HADM_ID", "TIME", "LABEL_CODE", "VALUENUM"]].copy()
presc_s = (presc_s.drop_duplicates(subset=["HADM_ID", "LABEL_CODE", "TIME"]))[
    ["HADM_ID", "TIME", "LABEL_CODE", "VALUENUM"]].copy()

# We append both subsets together to form the complete dataframe
complete_df1 = lab_s.append(input_s)
complete_df2 = complete_df1.append(output_s)
complete_df = complete_df2.append(presc_s)

assert (sum(
    complete_df.duplicated(subset=["HADM_ID", "LABEL_CODE", "TIME"]) == True) == 0)  # Check if no duplicates anymore.

# We remove patients with less than 50 observations.
id_counts = complete_df.groupby("HADM_ID").count()
id_list = list(id_counts.loc[id_counts["TIME"] < 50].index)
complete_df = complete_df.drop(complete_df.loc[complete_df["HADM_ID"].isin(id_list)].index).copy()
# %%
# We also choose 10 bins per hour. We now need to aggregate the data in different ways.
bin_k = 10
merged_df_short["TIME"] = round(merged_df_short_binned["TIME_STAMP"].dt.total_seconds() * bin_k / (100 * 36))

# For lab, we have to average the duplicates.
lab_subset = merged_df_short.loc[merged_df_short["Origin"] == "Lab", ["HADM_ID", "TIME", "LABEL_CODE", "VALUENUM"]]
lab_subset["KEY_ID"] = lab_subset["HADM_ID"].astype(str) + "/" + lab_subset["TIME"].astype(str) + "/" + lab_subset[
    "LABEL_CODE"].astype(str)
lab_subset["VALUENUM"] = lab_subset["VALUENUM"].astype(float)

lab_subset_s = lab_subset.groupby("KEY_ID")["VALUENUM"].mean().to_frame().reset_index()

lab_subset.rename(inplace=True, columns={"VALUENUM": "ExVALUENUM"})
lab_s = pd.merge(lab_subset, lab_subset_s, on="KEY_ID")
assert (not lab_s.isnull().values.any())

# For inputs, we have to sum the duplicates.
input_subset = merged_df_short.loc[merged_df_short["Origin"] == "Inputs", ["HADM_ID", "TIME", "LABEL_CODE", "VALUENUM"]]
input_subset["KEY_ID"] = input_subset["HADM_ID"].astype(str) + "/" + input_subset["TIME"].astype(str) + "/" + \
                         input_subset["LABEL_CODE"].astype(str)
input_subset["VALUENUM"] = input_subset["VALUENUM"].astype(float)

input_subset_s = input_subset.groupby("KEY_ID")["VALUENUM"].sum().to_frame().reset_index()

input_subset.rename(inplace=True, columns={"VALUENUM": "ExVALUENUM"})
input_s = pd.merge(input_subset, input_subset_s, on="KEY_ID")
assert (not input_s.isnull().values.any())

# For outpus, we have to sum the duplicates as well.
output_subset = merged_df_short.loc[
    merged_df_short["Origin"] == "Outputs", ["HADM_ID", "TIME", "LABEL_CODE", "VALUENUM"]]
output_subset["KEY_ID"] = output_subset["HADM_ID"].astype(str) + "/" + output_subset["TIME"].astype(str) + "/" + \
                          output_subset["LABEL_CODE"].astype(str)
output_subset["VALUENUM"] = output_subset["VALUENUM"].astype(float)

output_subset_s = output_subset.groupby("KEY_ID")["VALUENUM"].sum().to_frame().reset_index()

output_subset.rename(inplace=True, columns={"VALUENUM": "ExVALUENUM"})
output_s = pd.merge(output_subset, output_subset_s, on="KEY_ID")
assert (not output_s.isnull().values.any())

# For prescriptions, we have to sum the duplicates as well.
presc_subset = merged_df_short.loc[
    merged_df_short["Origin"] == "Prescriptions", ["HADM_ID", "TIME", "LABEL_CODE", "VALUENUM"]]
presc_subset["KEY_ID"] = presc_subset["HADM_ID"].astype(str) + "/" + presc_subset["TIME"].astype(str) + "/" + \
                         presc_subset["LABEL_CODE"].astype(str)
presc_subset["VALUENUM"] = presc_subset["VALUENUM"].astype(float)

presc_subset_s = presc_subset.groupby("KEY_ID")["VALUENUM"].sum().to_frame().reset_index()

presc_subset.rename(inplace=True, columns={"VALUENUM": "ExVALUENUM"})
presc_s = pd.merge(presc_subset, presc_subset_s, on="KEY_ID")
assert (not presc_s.isnull().values.any())

# Now remove the duplicates/
lab_s = (lab_s.drop_duplicates(subset=["HADM_ID", "LABEL_CODE", "TIME"]))[
    ["HADM_ID", "TIME", "LABEL_CODE", "VALUENUM"]].copy()
input_s = (input_s.drop_duplicates(subset=["HADM_ID", "LABEL_CODE", "TIME"]))[
    ["HADM_ID", "TIME", "LABEL_CODE", "VALUENUM"]].copy()
output_s = (output_s.drop_duplicates(subset=["HADM_ID", "LABEL_CODE", "TIME"]))[
    ["HADM_ID", "TIME", "LABEL_CODE", "VALUENUM"]].copy()
presc_s = (presc_s.drop_duplicates(subset=["HADM_ID", "LABEL_CODE", "TIME"]))[
    ["HADM_ID", "TIME", "LABEL_CODE", "VALUENUM"]].copy()

# We append both subsets together to form the complete dataframe
complete_df1 = lab_s.append(input_s)
complete_df2 = complete_df1.append(output_s)
complete_df10 = complete_df2.append(presc_s)

assert (sum(
    complete_df10.duplicated(subset=["HADM_ID", "LABEL_CODE", "TIME"]) == True) == 0)  # Check if no duplicates anymore.

# We remove patients with less than 50 observations.
id_counts = complete_df10.groupby("HADM_ID").count()
id_list = list(id_counts.loc[id_counts["TIME"] < 50].index)
complete_df10 = complete_df10.drop(complete_df10.loc[complete_df10["HADM_ID"].isin(id_list)].index).copy()
# %%
# # SAPSII data
# saps = pd.read_csv(processed_path + 'saps2.csv')
# valid_hadm_id = complete_df["HADM_ID"].unique()
# saps = saps.loc[saps["hadm_id"].isin(list(valid_hadm_id))].copy()
# saps["HADM_ID"] = saps["hadm_id"]
# saps.head()
# # %%
# saps["SUM_score"] = saps[
#     ['hr_score', 'sysbp_score', 'temp_score', 'pao2fio2_score', 'uo_score', 'bun_score', 'wbc_score', 'potassium_score',
#      'sodium_score', 'bicarbonate_score', 'bilirubin_score', 'gcs_score']].sum(axis=1)
# saps["X"] = -7.7631 + 0.0737 * saps["SUM_score"] + 0.9971 * (np.log(saps["SUM_score"] + 1))
# saps["PROB"] = np.exp(saps["X"]) / (1 + np.exp(saps["X"]))
#
# # %%
# saps.head()
# # %%
# saps.to_csv(processed_path + "sapsii_processed.csv")
# # %%
# saps_death = pd.merge(death_tags_df, saps, on="HADM_ID")
# y_pred = np.array(saps_death["PROB"])
# y_pred_full = np.array(saps_death["sapsii_prob"])
# y = np.array(saps_death["DEATHTAG"])
# # %%
# from sklearn.metrics import roc_auc_score
#
# print(roc_auc_score(y, y_pred))
# print(roc_auc_score(y, y_pred_full))
# # %%
# complete_df10["TIME"].max()
# # # %%


# %% md
# Dataframe creation for Tensor Decomposition

# Creation
# of
# a
# unique
# index
# for the admissions id.
# %%
# Creation of a unique index
unique_ids = np.arange(complete_df["HADM_ID"].nunique())
np.random.shuffle(unique_ids)
d = dict(zip(complete_df["HADM_ID"].unique(), unique_ids))
# d.to_csv(outfile_path + "UNIQUE_ID_dict.csv")
# # %%

# %%
Unique_id_dict=pd.Series(complete_df["HADM_ID"].unique()).reset_index().copy()
Unique_id_dict.columns=["index","HADM_ID"]
Unique_id_dict["UNIQUE_ID"]=Unique_id_dict["HADM_ID"].map(d)
Unique_id_dict.to_csv(outfile_path+"UNIQUE_ID_dict.csv")
# %% md
unique_id_df = pd.read_csv(outfile_path + "UNIQUE_ID_dict.csv")
d = dict(zip(unique_id_df["HADM_ID"].values, unique_id_df["UNIQUE_ID"].values))

### Death tags data set
# %%
admissions = pd.read_csv(processed_path + "Admissions_processed.csv")
death_tags_s = admissions.groupby("HADM_ID")["DEATHTAG"].unique().astype(int).to_frame().reset_index()
death_tags_df = death_tags_s.loc[death_tags_s["HADM_ID"].isin(complete_df["HADM_ID"])].copy()
death_tags_df["UNIQUE_ID"] = death_tags_df["HADM_ID"].map(d)
death_tags_df.sort_values(by="UNIQUE_ID", inplace=True)
death_tags_df.rename(columns={"DEATHTAG": "Value"}, inplace=True)
death_tags_df.to_csv(outfile_path + "complete_death_tags.csv")
# %% md
### Tensor Dataset
# %%
complete_df["UNIQUE_ID"] = complete_df["HADM_ID"].map(d)
# %%
# ICD9 codes
ICD_diag = pd.read_csv(file_path + "DIAGNOSES_ICD.csv")
# %%
main_diag = ICD_diag.loc[(ICD_diag["SEQ_NUM"] == 1)]
complete_tensor = pd.merge(complete_df, main_diag[["HADM_ID", "ICD9_CODE"]], on="HADM_ID")

# Only select the first 3 digits of each ICD9 code.
complete_tensor["ICD9_short"] = complete_tensor["ICD9_CODE"].astype(str).str[:3]
# Check that all codes are 3 digits long.
str_len = complete_tensor["ICD9_short"].str.len()
assert (str_len.loc[str_len != 3].count() == 0)

# Finer encoding (3 digits)
hot_encodings = pd.get_dummies(complete_tensor["ICD9_short"])
complete_tensor[hot_encodings.columns] = hot_encodings
complete_tensor_nocov = complete_tensor[["UNIQUE_ID", "LABEL_CODE", "TIME"] + ["VALUENUM"]].copy()

complete_tensor_nocov.rename(columns={"TIME": "TIME_STAMP"}, inplace=True)
# %% md
### Normalization of the data (N(0,1))
# %%
# Add a column with the mean and std of each different measurement type and then normalize them.
d_mean = dict(complete_tensor_nocov.groupby("LABEL_CODE")["VALUENUM"].mean())
complete_tensor_nocov["MEAN"] = complete_tensor_nocov["LABEL_CODE"].map(d_mean)
d_std = dict(complete_tensor_nocov.groupby("LABEL_CODE")["VALUENUM"].std())
complete_tensor_nocov["STD"] = complete_tensor_nocov["LABEL_CODE"].map(d_std)
complete_tensor_nocov["VALUENORM"] = (complete_tensor_nocov["VALUENUM"] - complete_tensor_nocov["MEAN"]) / \
                                     complete_tensor_nocov["STD"]

# %% md
### Train-Validation-Test split
# Random
# sampling
# %%
# Split training_validation_test sets RANDOM DIVISION.

df_train, df_test = train_test_split(complete_tensor_nocov, test_size=0.1)

# Make sure that patients of the test set have instances in the training set. (same with labels but this should be nearly certain)
assert (len(df_test.loc[~df_test["UNIQUE_ID"].isin(df_train["UNIQUE_ID"])].index) == 0)
assert (len(df_test.loc[~df_test["LABEL_CODE"].isin(df_train["LABEL_CODE"])].index) == 0)

# %%
# First train_val fold
df_train1, df_val1 = train_test_split(df_train, test_size=0.2)

# Make sure that patients of the test set have instances in the training set. (same with labels but this should be nearly certain)
assert (len(df_val1.loc[~df_val1["UNIQUE_ID"].isin(df_train1["UNIQUE_ID"])].index) == 0)
assert (len(df_val1.loc[~df_val1["LABEL_CODE"].isin(df_train1["LABEL_CODE"])].index) == 0)
# %%
# Second train_val fold
df_train2, df_val2 = train_test_split(df_train, test_size=0.2)

# Make sure that patients of the test set have instances in the training set. (same with labels but this should be nearly certain)
assert (len(df_val2.loc[~df_val2["UNIQUE_ID"].isin(df_train2["UNIQUE_ID"])].index) == 0)
assert (len(df_val2.loc[~df_val2["LABEL_CODE"].isin(df_train2["LABEL_CODE"])].index) == 0)
# %%
# Third train_val fold
df_train3, df_val3 = train_test_split(df_train, test_size=0.2)

# Make sure that patients of the test set have instances in the training set. (same with labels but this should be nearly certain)
assert (len(df_val3.loc[~df_val3["UNIQUE_ID"].isin(df_train3["UNIQUE_ID"])].index) == 0)
assert (len(df_val3.loc[~df_val3["LABEL_CODE"].isin(df_train3["LABEL_CODE"])].index) == 0)
# %% md
#### Venn diagram of the training sets. Visualization of the number of common samples.

# %%
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

temp12 = pd.merge(df_train1, df_train2, how="inner", on=["UNIQUE_ID", "LABEL_CODE", "TIME_STAMP"])
temp13 = pd.merge(df_train1, df_train3, how="inner", on=["UNIQUE_ID", "LABEL_CODE", "TIME_STAMP"])
temp23 = pd.merge(df_train2, df_train3, how="inner", on=["UNIQUE_ID", "LABEL_CODE", "TIME_STAMP"])
temp123 = pd.merge(temp12, temp23, how="inner", on=["UNIQUE_ID", "LABEL_CODE", "TIME_STAMP"])

# Make the diagram
venn3(subsets=(len(df_train1.index) - len(temp12.index) - len(temp13.index) + len(temp123.index),
               len(df_train2.index) - len(temp12.index) - len(temp23.index) + len(temp123.index),
               len(temp12.index) - len(temp123),
               len(df_train3.index) - len(temp13.index) - len(temp23.index) + len(temp123.index),
               len(temp13.index) - len(temp123.index), len(temp23.index) - len(temp123.index), len(temp123)))
plt.show()
# %%
# Save locally.
complete_tensor_nocov.to_csv(outfile_path + "complete_tensor.csv")  # Full data
df_train1.to_csv(outfile_path + "complete_tensor_train1.csv")  # Train data
df_val1.to_csv(outfile_path + "complete_tensor_val1.csv")  # Validation data
df_train2.to_csv(outfile_path + "complete_tensor_train2.csv")  # Train data
df_val2.to_csv(outfile_path + "complete_tensor_val2.csv")  # Validation data
df_train3.to_csv(outfile_path + "complete_tensor_train3.csv")  # Train data
df_val3.to_csv(outfile_path + "complete_tensor_val3.csv")  # Validation data
df_test.to_csv(outfile_path + "complete_tensor_test.csv")  # Test data

# %% md
#### Covariates dataset
# %%
# We create a data set with the covariates
covariates = complete_tensor.groupby("UNIQUE_ID").nth(0)[list(hot_encodings.columns)]
covariates.to_csv(outfile_path + "complete_covariates.csv")  # save locally
# %% md
## Creation of the dataset for LSTM operation

# We
# split
# the
# data
# patient - wise and provide
# imputation
# methods.
# %%
# Unique_ids of train and test
test_prop = 0.1
val_prop = 0.2
sorted_unique_ids = np.sort(unique_ids)
train_unique_ids = sorted_unique_ids[:int((1 - test_prop) * (1 - val_prop) * len(unique_ids))]
val_unique_ids = sorted_unique_ids[
                 int((1 - test_prop) * (1 - val_prop) * len(unique_ids)):int((1 - test_prop) * len(unique_ids))]
test_unique_ids = sorted_unique_ids[int((1 - test_prop) * len(unique_ids)):]
# %% md
#### Death tags
# %%
death_tags_train_df = death_tags_df.loc[death_tags_df["UNIQUE_ID"].isin(list(train_unique_ids))].sort_values(
    by="UNIQUE_ID")
death_tags_val_df = death_tags_df.loc[death_tags_df["UNIQUE_ID"].isin(list(val_unique_ids))].sort_values(by="UNIQUE_ID")
death_tags_test_df = death_tags_df.loc[death_tags_df["UNIQUE_ID"].isin(list(test_unique_ids))].sort_values(
    by="UNIQUE_ID")

death_tags_train_df.to_csv(outfile_path + "LSTM_death_tags_train.csv")
death_tags_val_df.to_csv(outfile_path + "LSTM_death_tags_val.csv")
death_tags_test_df.to_csv(outfile_path + "LSTM_death_tags_test.csv")
# %% md
#### Tensor split
# %%
# Create a segmented tensor (by patients)
complete_tensor_train = complete_tensor_nocov.loc[
    complete_tensor_nocov["UNIQUE_ID"].isin(list(train_unique_ids))].sort_values(by="UNIQUE_ID")
complete_tensor_val = complete_tensor_nocov.loc[
    complete_tensor_nocov["UNIQUE_ID"].isin(list(val_unique_ids))].sort_values(by="UNIQUE_ID")
complete_tensor_test = complete_tensor_nocov.loc[
    complete_tensor_nocov["UNIQUE_ID"].isin(list(test_unique_ids))].sort_values(by="UNIQUE_ID")

complete_tensor_train.to_csv(outfile_path + "LSTM_tensor_train.csv")
complete_tensor_val.to_csv(outfile_path + "LSTM_tensor_val.csv")
complete_tensor_test.to_csv(outfile_path + "LSTM_tensor_test.csv")
# %% md
#### Covariates split
# %%
covariates_train = covariates.loc[covariates.index.isin(train_unique_ids)].sort_index()
covariates_val = covariates.loc[covariates.index.isin(val_unique_ids)].sort_index()
covariates_test = covariates.loc[covariates.index.isin(test_unique_ids)].sort_index()

covariates_train.to_csv(outfile_path + "LSTM_covariates_train.csv")  # save locally
covariates_val.to_csv(outfile_path + "LSTM_covariates_val.csv")  # save locally
covariates_test.to_csv(outfile_path + "LSTM_covariates_test.csv")  # save locally
# %% md
#### Mean Imputation
# %%
# Vector containing the mean_values of each dimension.
mean_dims = complete_tensor_train.groupby("LABEL_CODE")["MEAN"].mean()
mean_dims.to_csv(outfile_path + "mean_features.csv")
# %% md
## Dataset for GRU_D (continuous time operation)
# %%
# map the admission id to the unique id
complete_df10["UNIQUE_ID"] = complete_df10["HADM_ID"].map(d)
complete_df10["TIME_CONTINUOUS"] = complete_df10["TIME"] / 10
# %%
# Add a column with the mean and std of each different measurement type and then normalize them.
d_mean = dict(complete_df10.groupby("LABEL_CODE")["VALUENUM"].mean())
complete_df10["MEAN"] = complete_df10["LABEL_CODE"].map(d_mean)
d_std = dict(complete_df10.groupby("LABEL_CODE")["VALUENUM"].std())
complete_df10["STD"] = complete_df10["LABEL_CODE"].map(d_std)
complete_df10["VALUENORM"] = (complete_df10["VALUENUM"] - complete_df10["MEAN"]) / complete_df10["STD"]

assert (len(complete_df10.loc[complete_df10[
                                  "VALUENORM"] == 0].index) == 0)  # Make sure that there are no zeros. (Zeros can be used to represent missing values then)
# %%
max_time_bins = complete_df10.groupby("UNIQUE_ID")[
    "TIME"].nunique().max()  # This is the maximal number of different time steps in a patient time series.

# %%
a = complete_df10.sort_values(by=["UNIQUE_ID", "TIME_CONTINUOUS"]).copy()
a.reset_index(inplace=True)
# %%
b = a.assign(Time_order=a.groupby('UNIQUE_ID').TIME.rank(method='dense') - 1)

# %%
# b
# %%
#### END OF FILE ####### (below is testing stuff.)
# %%
tags = pd.read_csv(outfile_path + "LSTM_death_tags_train.csv")
tags["UNIQUE_ID"].unique()
# %%
tags = pd.read_csv(outfile_path + "LSTM_death_tags_val.csv")
tags["UNIQUE_ID"].unique()
# %%
df = pd.read_csv(outfile_path + "LSTM_tensor_train.csv")
# %%
df["UNIQUE_ID"].unique()
# %%
df["UNIQUE_ID"].nunique()
# %%
df2 = pd.read_csv(outfile_path + "LSTM_tensor_val.csv")
# %%
df2["UNIQUE_ID"].unique()
# %%
df3 = pd.concat([df, df2])
# %%
df3["UNIQUE_ID"].unique()
# %%
means_df = pd.Series.from_csv("./Data/Clean_data/mean_features.csv")
means_vec = torch.tensor(means_df.as_matrix())
# %%
means_vec.size()
# %%
# mean_dims
# %%
cov = pd.read_csv(outfile_path + "complete_covariates.csv")
df_train = pd.read_csv(outfile_path + "complete_tensor_train1.csv")
df_val = pd.read_csv(outfile_path + "complete_tensor_val1.csv")
deaths = pd.read_csv(outfile_path + "complete_death_tags.csv")
df = pd.read_csv(outfile_path + "complete_tensor.csv")
# %% md
## Create a segmented dataset by patients for actual testing.
# %%
unique_ids = cov["UNIQUE_ID"]
train_unique_ids, test_unique_ids = train_test_split(unique_ids, test_size=0.1)
# %%
df_segment_train = df.loc[df["UNIQUE_ID"].isin(list(train_unique_ids))]
df_segment_test = df.loc[df["UNIQUE_ID"].isin(list(test_unique_ids))]
cov_segment_train = cov.loc[cov["UNIQUE_ID"].isin(list(train_unique_ids))]
cov_segment_test = cov.loc[cov["UNIQUE_ID"].isin(list(test_unique_ids))]
# %%
df_segment_train.to_csv(outfile_path + "segmented_tensor_train.csv")
df_segment_test.to_csv(outfile_path + "segmented_tensor_test.csv")
cov_segment_train.to_csv(outfile_path + "segmented_covariates_train.csv")
cov_segment_test.to_csv(outfile_path + "segmented_covariates_test.csv")
# %%
list(df["UNIQUE_ID"].unique())
# %%
df_train['UNIQUE_ID'].nunique()
# %%
df_val['UNIQUE_ID'].nunique()
# %%
deaths["UNIQUE_ID"].nunique()
# %%
df["UNIQUE_ID"].nunique()
# %%
import torch

# %%
a = torch.tensor([3, 2, 1])
# %%
b = a.repeat(10, 5, 1)
# %%
# b.shape
# %%
# torch.cumsum(b)
# %%
import torch
import numpy as np

# %%
a = torch.tensor([np.nan, 3, 4, np.nan])
# %%
observed_mask = a == a
# %%
# observed_mask
# %%
a = torch.randn((3, 4, 5))
b = torch.randn((3, 4, 5))
c = torch.randn((3, 4, 5))
# %%
d = torch.cat((a, b, c))
d.size()
# %%
print(d.dtype)
# %%
z = torch.zeros((4))
# %%
torch.zeros((4)).masked_scatter_(1 - observed_mask, a)
# %%
a = a.float()
# %%
# a.dtype
# %%
b = a.repeat(3)
# %%

# %%

# %%
device = torch.device("cpu")
b.to(device)
# %%
# b.dtype
# %%
train_test_split(np.arange(4), test_size=0.2)
# %%
a = 3
# a=None
# %%
# a is not None
# %%
df = pd.read_csv(outfile_path + "complete_tensor.csv")
# %%
len(df.index)
# %%
# df
# %%
lab_df = pd.read_csv(processed_path + "LAB_processed.csv")
# %%
lab_df["LABEL"].unique()
# %%
a = pd.read_csv(outfile_path + "LSTM_covariates_test.csv")
b = pd.read_csv(outfile_path + "LSTM_tensor_test.csv")
# %%
# len(a.index)
# %%
b["UNIQUE_ID"].nunique()
# %%
len(merged_df["LABEL"].unique())
# %%
len(merged_df.index) / (96 * merged_df["SUBJECT_ID"].nunique() * 48 * 2)
# %%
merged_df.head()
# %%
presc_df["LABEL"].unique()
# %%
merged_df["LABEL"].nunique()
# %%