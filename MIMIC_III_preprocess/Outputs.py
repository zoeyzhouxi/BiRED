#%% md
# Pre-processing of the outputevents dataset
#%%
import pandas as pd
import matplotlib.pyplot as plt

file_path="./Data/Full_MIMIC/"

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 300)
#%%
adm=pd.read_csv(file_path+"Processed/Admissions_processed.csv")
#%% md
# We now consider the outputevents dataset. We select only the patients with the same criteria as above.
#%%
outputs=pd.read_csv(file_path+"OUTPUTEVENTS.csv")
#%%
#Some checks
assert(len(outputs.loc[outputs["ISERROR"].notnull()].index)==0) #No entry with iserror==TRUE

#Restrict the dataset to the previously selected admission ids only.
adm_ids=list(adm["HADM_ID"])
outputs=outputs.loc[outputs["HADM_ID"].isin(adm_ids)]

print("Number of patients remaining in the database: ")
print(outputs["SUBJECT_ID"].nunique())
#%% md
# We load the D_ITEMS dataframe which contains the name of the ITEMID. And we merge both tables together.
#%%
#item_id
item_id=pd.read_csv(file_path+"D_ITEMS.csv")
item_id_1=item_id[["ITEMID","LABEL"]]
item_id_1.head()

#We merge the name of the item administrated.
outputs_2=pd.merge(outputs,item_id_1,on="ITEMID")
outputs_2.head()
print("Number of patients remaining in the database: ")
print(outputs_2["SUBJECT_ID"].nunique())
#%% md
# We compute the number of patients that have the specific outputs labels and we select only the features that are the most present over the whole data set. For this, we rank the features by number of patients and select the n_best.
#%%
n_best=15
#For each item, evaluate the number of patients who have been given this item.
pat_for_item=outputs_2.groupby("LABEL")["SUBJECT_ID"].nunique()
#Order by occurence and take the 20 best (the ones with the most patients)
frequent_labels=pat_for_item.sort_values(ascending=False)[:n_best]

#Select only the time series with high occurence.
outputs_3=outputs_2.loc[outputs_2["LABEL"].isin(list(frequent_labels.index))].copy()

print("Number of patients remaining in the database: ")
print(outputs_3["SUBJECT_ID"].nunique())
print("Number of datapoints remaining in the database: ")
print(len(outputs_3.index))

print(frequent_labels)
#%% md
#### Eventually, we select the same labels of the paper
#%%
outputs_label_list=['Gastric Gastric Tube','Stool Out Stool','Urine Out Incontinent','Ultrafiltrate Ultrafiltrate','Foley', 'Void','Condom Cath','Fecal Bag','Ostomy (output)','Chest Tube #1','Chest Tube #2','Jackson Pratt #1','OR EBL','Pre-Admission','TF Residual']
outputs_bis=outputs_2.loc[outputs_2["LABEL"].isin(outputs_label_list)].copy()

print("Number of patients remaining in the database: ")
print(outputs_bis["SUBJECT_ID"].nunique())
print("Number of datapoints remaining in the database: ")
print(len(outputs_bis.index))

outputs_3=outputs_bis.copy()
#%% md
# Cleaning of the output data

### Units Cleaning

#### 1) Amounts
#%%
#Verification that all input labels have the same amounts units.
outputs_3.groupby("LABEL")["VALUEUOM"].value_counts() #OK
#%% md
### Check for outliers

#### 1) In amounts
#%%
outputs_3.groupby("LABEL")["VALUE"].describe()
#%%
#Remove all entries whose rate is more than 4 std away from the mean.
out_desc=outputs_3.groupby("LABEL")["VALUE"].describe()
name_list=list(out_desc.loc[out_desc["count"]!=0].index)
for label in name_list:
    outputs_3=outputs_3.drop(outputs_3.loc[(outputs_3["LABEL"]==label)&(outputs_3["VALUE"]>(out_desc.loc[label,"mean"]+4*out_desc.loc[label,"std"]))].index).copy()

print("Number of patients remaining in the database: ")
print(outputs_3["SUBJECT_ID"].nunique())
print("Number of datapoints remaining in the database: ")
print(len(outputs_3.index))
#%%
#Clean Foley, remove too large values
outputs_3=outputs_3.drop(outputs_3.loc[(outputs_3["LABEL"]=="Foley") & (outputs_3["VALUE"]>5500)].index).copy()
#Clean Expected Blood Loss, remove too large values
outputs_3=outputs_3.drop(outputs_3.loc[(outputs_3["LABEL"]=="OR EBL") & (outputs_3["VALUE"]>5000)].index).copy()
#Clean Out Expected Blood Loss, remove too large values
outputs_3=outputs_3.drop(outputs_3.loc[(outputs_3["LABEL"]=="OR Out EBL") & (outputs_3["VALUE"]>5000)].index).copy()
#Clean OR Urine, remove too large values
outputs_3=outputs_3.drop(outputs_3.loc[(outputs_3["LABEL"]=="OR Urine") & (outputs_3["VALUE"]>5000)].index).copy()
#Clean Pre-Admission, remove too large and negative values
outputs_3=outputs_3.drop(outputs_3.loc[(outputs_3["LABEL"]=="Pre-Admission") & (outputs_3["VALUE"]<0)].index).copy()
outputs_3=outputs_3.drop(outputs_3.loc[(outputs_3["LABEL"]=="Pre-Admission") & (outputs_3["VALUE"]>5000)].index).copy()
#Clean Pre-Admission output, remove too large values
outputs_3=outputs_3.drop(outputs_3.loc[(outputs_3["LABEL"]=="Pre-Admission Output Pre-Admission Output") & (outputs_3["VALUE"]>5000)].index).copy()
#Clean Urine Out Foley output, remove too large values
outputs_3=outputs_3.drop(outputs_3.loc[(outputs_3["LABEL"]=="Urine Out Foley") & (outputs_3["VALUE"]>5000)].index).copy()
#Clean Void, remove negative values
outputs_3=outputs_3.drop(outputs_3.loc[(outputs_3["LABEL"]=="Void") & (outputs_3["VALUE"]<0)].index).copy()

outputs_3.dropna(subset=["VALUE"],inplace=True)

print("Number of patients remaining in the database: ")
print(outputs_3["SUBJECT_ID"].nunique())
print("Number of datapoints remaining in the database: ")
print(len(outputs_3.index))
#%% md
# As data is already in timestamp format, we don't neeed to consider rates
#%%
outputs_3.to_csv(file_path+"Processed/OUTPUTS_processed.csv")
#%%
