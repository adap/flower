import pandas as pd 
import numpy as np 
import pickle
import os 
from pathlib import Path
from imblearn.over_sampling import RandomOverSampler
from joblib import Parallel, delayed

def save_dataset(X, Y, out_dir="./data/output", fmt="pkl"):
    os.makedirs(out_dir, exist_ok=True)

    print("Saving dataset...")

    if fmt == "pkl":
        with open(os.path.join(out_dir, "X.pkl"), "wb") as fp:
            pickle.dump(X, fp)
        with open(os.path.join(out_dir, "Y.pkl"), "wb") as fp:
            pickle.dump(Y, fp)

    elif fmt == "npy":
        np.save(os.path.join(out_dir, "X.npy"), np.array(X))
        np.save(os.path.join(out_dir, "Y.npy"), np.array(Y))

    elif fmt == "csv":
        pd.DataFrame(X).to_csv(os.path.join(out_dir, "X.csv"), index=False)
        pd.DataFrame(Y).to_csv(os.path.join(out_dir, "Y.csv"), index=False)

    else:
        raise ValueError("Unsupported format. Choose from: pkl, npy, csv")

def build_dataset(use_ICU , label , disease_label , bucket , time  , oversampling , concat , fmt = "csv" ):

    cohort_output = (
        "cohort_"
        + use_ICU.lower()
        + "_"
        + label.lower().replace(" ", "_")
        + "_"
        + str(time)
        + "_"
        + str(bucket)
        + "_"
        + disease_label
    )
    output_dir = './data/output/'
    os.makedirs(output_dir, exist_ok=True)
    output_dir = output_dir + cohort_output
    os.makedirs(output_dir, exist_ok=True)

    hids = create_hids(oversampling)
    labels=pd.read_csv('./data/csv/labels.csv', header=0)
    concat_cols=[]
    if(concat):
        dyn=pd.read_csv('./data/csv/'+str(hids[0])+'/dynamic.csv',header=[0,1])
        dyn.columns=dyn.columns.droplevel(0)
        cols=dyn.columns
        time=dyn.shape[0]

        for t in range(time):
            cols_t = [x + "_"+str(t) for x in cols]

            concat_cols.extend(cols_t)
    X , Y = getXY_consolidated(hids, labels , concat_cols , concat , use_ICU)
    print(X.shape)
    print("Saving dataset...")
    save_dataset(X, Y, out_dir=output_dir, fmt=fmt) 

def create_hids(oversampling):

    labels = pd.read_csv('./data/csv/labels.csv', header=0)

    hids = labels.iloc[:, 0]
    y = labels.iloc[:, 1]

    print("Total Samples", len(hids))
    print("Positive Samples", y.sum())

    if oversampling:
        print("=============OVERSAMPLING===============")
        oversample = RandomOverSampler(sampling_strategy='minority')
        hids = np.asarray(hids).reshape(-1, 1)
        hids, y = oversample.fit_resample(hids, y)
        hids = hids[:, 0]

        print("Total Samples", len(hids))
        print("Positive Samples", y.sum())
    
    return hids

def process_single_sample(sample, label_val, concat, concat_cols, data_icu, mean_keys, base_path):
    sample_dir = base_path / str(sample)
    
    # Read files
    dyn = pd.read_csv(sample_dir / 'dynamic.csv', header=[0, 1])
    stat = pd.read_csv(sample_dir / 'static.csv', header=[0, 1])['COND']
    demo = pd.read_csv(sample_dir / 'demo.csv')

    if concat:
        dyn_values = dyn.to_numpy().flatten().reshape(1, -1)
        dyn_df = pd.DataFrame(dyn_values, columns=concat_cols)
    else:
        agg_map = {col: ("mean" if col in mean_keys else "max") 
                   for col in dyn.columns.levels[0]}
        dyn_df = dyn.groupby(level=0, axis=1).agg(agg_map)
        dyn_df.columns = dyn_df.columns.droplevel(0)

    # Return the combined row and the label
    return pd.concat([dyn_df, stat, demo], axis=1), label_val



def getXY_consolidated(ids, labels, concat_cols, concat, data_icu):
    print(f"Loading consolidated data for {len(ids)} samples...")

    # 1. Load Files
    dyn_all = pd.read_csv('./data/csv/all_dynamic.csv', header=[0, 1], low_memory=False)
    stat_all = pd.read_csv('./data/csv/all_static.csv', low_memory=False)
    demo_all = pd.read_csv('./data/csv/all_demo.csv', low_memory=False)

    # Automatically identify the stay_id column in the MultiIndex
    # It usually looks like ('stay_id', 'stay_id') or ('STAY_ID', 'STAY_ID')
    id_tuple = [col for col in dyn_all.columns if 'stay_id' in str(col[0]).lower()][0]
    id_name_simple = 'stay_id' if data_icu else 'hadm_id'

    # 2. Aggregation (Process Unique Patients)
    if concat:
        print("Flattening time steps (Concatenate=True)...")
        def flatten_patient(group):
            # Drop the ID column before flattening
            return pd.Series(group.drop(columns=id_tuple[0], level=0).to_numpy().flatten())
        
        X_dyn_unique = dyn_all.groupby(id_tuple).apply(flatten_patient)
        X_dyn_unique.columns = concat_cols
    else:
        print("Aggregating time steps (Concatenate=False)...")
        mean_keys = ["CHART", "MEDS"] if data_icu else ["LAB", "MEDS"]
        
        # 1. Extract only the unique Level 0 names available in the data
        level0_names = [n for n in dyn_all.columns.get_level_values(0).unique() if 'stay_id' not in n.lower()]
        
        # 2. Map aggregation functions
        agg_map = {name: ("mean" if name in mean_keys else "max") for name in level0_names}
        
        # 3. Aggregating specifically on Level 0
        # By passing 'level=0' to the dict-aggregator, Pandas knows to look at the top category
        X_dyn_unique = dyn_all.groupby(id_tuple).agg({
            col: agg_map[col[0]] for col in dyn_all.columns if col[0] in agg_map
        })
        
        # 4. Collapse MultiIndex columns: ('CHART', 'HeartRate') -> 'HeartRate'
        X_dyn_unique.columns = X_dyn_unique.columns.droplevel(0)

    # 3. Create Master Lookup Table
    # Ensure indices are strings/ints consistently for the join
    stat_all = stat_all.set_index('stay_id')
    demo_all = demo_all.set_index('stay_id')
    labels_idx = labels.set_index(id_name_simple)

    # Combine all unique data into one table
    master_unique = pd.concat([X_dyn_unique, stat_all, demo_all, labels_idx[['label']]], axis=1)

    # 4. Apply Oversampling by Re-indexing
    # This duplicates rows for the minority class based on your 'ids' list
    print("Expanding dataset based on oversampled IDs...")
    final_df = master_unique.loc[ids].reset_index(drop=True)

    # 5. Sanity Checks
    if final_df.isnull().values.any():
        print("Warning: NaNs detected! Some IDs in your list were missing from the data files.")
        final_df = final_df.fillna(0) # Or use your Missing_values_management strategy

    # 6. Split X and Y
    Y = final_df['label']
    X = final_df.drop(columns=['label'])

    print(f"--- Process Complete ---")
    print(f"Final X Shape: {X.shape}")
    print(f"Class Counts:\n{Y.value_counts()}")
    
    return X, Y
def getXY( ids, labels, concat_cols , concat , data_icu):

    X_rows = []

    y_rows = []


    if data_icu:

        label_map = labels.set_index('stay_id')['label']

    else:

        label_map = labels.set_index('hadm_id')['label']

    print(ids)
    for i, sample in enumerate(ids):

        if i % 100 == 0:
    
            print(i)


        y_rows.append(label_map.loc[sample])


        dyn = pd.read_csv(f'./data/csv/{sample}/dynamic.csv', header=[0,1])


        if concat:

            dyn.columns = dyn.columns.droplevel(0)

            dyn = dyn.to_numpy().reshape(1, -1)

            dyn_df = pd.DataFrame(dyn, columns=concat_cols)

        else:

            if data_icu:

                mean_keys = ["CHART", "MEDS"]

            else:

                mean_keys = ["LAB", "MEDS"]


            agg_map = {
                key: "mean" if key in mean_keys else "max"
                for key in dyn.columns.levels[0]

            }


            dyn_df = dyn.groupby(level=0, axis=1).agg(agg_map)

            dyn_df.columns = dyn_df.columns.droplevel(0)


        stat = pd.read_csv(f'./data/csv/{sample}/static.csv', header=[0,1])['COND']

        demo = pd.read_csv(f'./data/csv/{sample}/demo.csv')


        X_rows.append(pd.concat([dyn_df, stat, demo], axis=1))


    X_df = pd.concat(X_rows, axis=0, ignore_index=True)

    y_df = pd.Series(y_rows, name="label")


    return X_df, y_df 
