import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')
if not os.path.exists("./data/dict"):
    os.makedirs("./data/dict")
if not os.path.exists("./data/csv"):
    os.makedirs("./data/csv")
    
class Generator():
    def __init__(self,cohort_output,if_mort,if_admn,if_los,feat_cond,feat_proc,feat_out,feat_chart,feat_med,impute,include_time=24,bucket=1,predW=6):
        self.feat_cond,self.feat_proc,self.feat_out,self.feat_chart,self.feat_med = feat_cond,feat_proc,feat_out,feat_chart,feat_med
        self.cohort_output=cohort_output
        self.impute=impute
        self.data = self.generate_adm()
        print("[ READ COHORT ]")
        
        self.generate_feat()
        print("[ READ ALL FEATURES ]")
        
        if if_mort:
            self.mortality_length(include_time,predW)
            print("[ PROCESSED TIME SERIES TO EQUAL LENGTH  ]")
        elif if_admn:
            self.readmission_length(include_time)
            print("[ PROCESSED TIME SERIES TO EQUAL LENGTH  ]")
        elif if_los:
            self.los_length(include_time)
            print("[ PROCESSED TIME SERIES TO EQUAL LENGTH  ]")
        
        self.smooth_meds(bucket)
        print("[ SUCCESSFULLY SAVED DATA DICTIONARIES ]")
    
    def generate_feat(self):
        if(self.feat_cond):
            print("[ ======READING DIAGNOSIS ]")
            self.generate_cond()
        if(self.feat_proc):
            print("[ ======READING PROCEDURES ]")
            self.generate_proc()
        if(self.feat_out):
            print("[ ======READING OUT EVENTS ]")
            self.generate_out()
        if(self.feat_chart):
            print("[ ======READING CHART EVENTS ]")
            self.generate_chart()
        if(self.feat_med):
            print("[ ======READING MEDICATIONS ]")
            self.generate_meds()

    def generate_adm(self):
        data=pd.read_csv(f"./data/cohort/{self.cohort_output}.csv.gz", compression='gzip', header=0, index_col=None)
        data['intime'] = pd.to_datetime(data['intime'])
        data['outtime'] = pd.to_datetime(data['outtime'])
        data['los']=pd.to_timedelta(data['outtime']-data['intime'],unit='h')
        data['los']=data['los'].astype(str)
        data[['days', 'dummy', 'hours']] = data['los'].str.split(' ', n=-1, expand=True)
        data[['hours', 'min', 'sec']] = data['hours'].str.split(':', n=-1, expand=True)
        data['los']=pd.to_numeric(data['days'])*24+pd.to_numeric(data['hours'])
        data=data.drop(columns=['days', 'dummy','hours','min','sec'])
        data=data[data['los']>0]
        data['Age']=data['Age'].astype(int)
        #print(data.head())
        #print(data.shape)
        return data
    
    def generate_cond(self):
        cond=pd.read_csv("./data/features/preproc_diag_icu.csv.gz", compression='gzip', header=0, index_col=None)
        cond=cond[cond['stay_id'].isin(self.data['stay_id'])]
        cond_per_adm = cond.groupby('stay_id').size().max()
        self.cond, self.cond_per_adm = cond, cond_per_adm
    
    def generate_proc(self):
        proc=pd.read_csv("./data/features/preproc_proc_icu.csv.gz", compression='gzip', header=0, index_col=None)
        proc=proc[proc['stay_id'].isin(self.data['stay_id'])]
        proc[['start_days', 'dummy','start_hours']] = proc['event_time_from_admit'].str.split(' ', n=-1, expand=True)
        proc[['start_hours','min','sec']] = proc['start_hours'].str.split(':', n=-1, expand=True)
        proc['start_time']=pd.to_numeric(proc['start_days'])*24+pd.to_numeric(proc['start_hours'])
        proc=proc.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
        proc=proc[proc['start_time']>=0]
        
        ###Remove where event time is after discharge time
        proc=pd.merge(proc,self.data[['stay_id','los']],on='stay_id',how='left')
        proc['sanity']=proc['los']-proc['start_time']
        proc=proc[proc['sanity']>0]
        del proc['sanity']
        
        self.proc=proc
        
    def generate_out(self):
        out=pd.read_csv("./data/features/preproc_out_icu.csv.gz", compression='gzip', header=0, index_col=None)
        out=out[out['stay_id'].isin(self.data['stay_id'])]
        out[['start_days', 'dummy','start_hours']] = out['event_time_from_admit'].str.split(' ', n=-1, expand=True)
        out[['start_hours','min','sec']] = out['start_hours'].str.split(':', n=-1, expand=True)
        out['start_time']=pd.to_numeric(out['start_days'])*24+pd.to_numeric(out['start_hours'])
        out=out.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
        out=out[out['start_time']>=0]
        
        ###Remove where event time is after discharge time
        out=pd.merge(out,self.data[['stay_id','los']],on='stay_id',how='left')
        out['sanity']=out['los']-out['start_time']
        out=out[out['sanity']>0]
        del out['sanity']
        
        self.out=out
        
        
    def generate_chart(self):
        chunksize = 5000000
        final=pd.DataFrame()
        for chart in tqdm(pd.read_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip', header=0, index_col=None,chunksize=chunksize)):
            chart=chart[chart['stay_id'].isin(self.data['stay_id'])]
            chart[['start_days', 'dummy','start_hours']] = chart['event_time_from_admit'].str.split(' ', n=-1, expand=True)
            chart[['start_hours','min','sec']] = chart['start_hours'].str.split(':', n=-1, expand=True)
            chart['start_time']=pd.to_numeric(chart['start_days'])*24+pd.to_numeric(chart['start_hours'])
            chart=chart.drop(columns=['start_days', 'dummy','start_hours','min','sec','event_time_from_admit'])
            chart=chart[chart['start_time']>=0]

            ###Remove where event time is after discharge time
            chart=pd.merge(chart,self.data[['stay_id','los']],on='stay_id',how='left')
            chart['sanity']=chart['los']-chart['start_time']
            chart=chart[chart['sanity']>0]
            del chart['sanity']
            del chart['los']
            
            if final.empty:
                final=chart
            else:
                final=pd.concat([final, chart], ignore_index=True)
        
        self.chart=final
        
        
        
    def generate_meds(self):
        meds=pd.read_csv("./data/features/preproc_med_icu.csv.gz", compression='gzip', header=0, index_col=None)
        meds[['start_days', 'dummy','start_hours']] = meds['start_hours_from_admit'].str.split(' ', n=-1, expand=True)
        meds[['start_hours','min','sec']] = meds['start_hours'].str.split(':', n=-1, expand=True)
        meds['start_time']=pd.to_numeric(meds['start_days'])*24+pd.to_numeric(meds['start_hours'])
        meds[['start_days', 'dummy','start_hours']] = meds['stop_hours_from_admit'].str.split(' ', n=-1, expand=True)
        meds[['start_hours','min','sec']] = meds['start_hours'].str.split(':', n=-1, expand=True)
        meds['stop_time']=pd.to_numeric(meds['start_days'])*24+pd.to_numeric(meds['start_hours'])
        meds=meds.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
        #####Sanity check
        meds['sanity']=meds['stop_time']-meds['start_time']
        meds=meds[meds['sanity']>0]
        del meds['sanity']
        #####Select hadm_id as in main file
        meds=meds[meds['stay_id'].isin(self.data['stay_id'])]
        meds=pd.merge(meds,self.data[['stay_id','los']],on='stay_id',how='left')

        #####Remove where start time is after end of visit
        meds['sanity']=meds['los']-meds['start_time']
        meds=meds[meds['sanity']>0]
        del meds['sanity']
        ####Any stop_time after end of visit is set at end of visit
        meds.loc[meds['stop_time'] > meds['los'],'stop_time']=meds.loc[meds['stop_time'] > meds['los'],'los']
        del meds['los']
        
        meds['rate']=meds['rate'].apply(pd.to_numeric, errors='coerce')
        meds['amount']=meds['amount'].apply(pd.to_numeric, errors='coerce')
        
        self.meds=meds
    
    def mortality_length(self,include_time,predW):
        print("include_time",include_time)
        self.los=include_time
        self.data=self.data[(self.data['los']>=include_time+predW)]
        self.hids=self.data['stay_id'].unique()
        
        if(self.feat_cond):
            self.cond=self.cond[self.cond['stay_id'].isin(self.data['stay_id'])]
        
        self.data['los']=include_time

        ####Make equal length input time series and remove data for pred window if needed
        
        ###MEDS
        if(self.feat_med):
            self.meds=self.meds[self.meds['stay_id'].isin(self.data['stay_id'])]
            self.meds=self.meds[self.meds['start_time']<=include_time]
            self.meds.loc[self.meds.stop_time >include_time, 'stop_time']=include_time
                    
        
        ###PROCS
        if(self.feat_proc):
            self.proc=self.proc[self.proc['stay_id'].isin(self.data['stay_id'])]
            self.proc=self.proc[self.proc['start_time']<=include_time]
            
        ###OUT
        if(self.feat_out):
            self.out=self.out[self.out['stay_id'].isin(self.data['stay_id'])]
            self.out=self.out[self.out['start_time']<=include_time]
            
       ###CHART
        if(self.feat_chart):
            self.chart=self.chart[self.chart['stay_id'].isin(self.data['stay_id'])]
            self.chart=self.chart[self.chart['start_time']<=include_time]
        
        #self.los=include_time
    def los_length(self,include_time):
        print("include_time",include_time)
        self.los=include_time
        self.data=self.data[(self.data['los']>=include_time)]
        self.hids=self.data['stay_id'].unique()
        
        if(self.feat_cond):
            self.cond=self.cond[self.cond['stay_id'].isin(self.data['stay_id'])]
        
        self.data['los']=include_time

        ####Make equal length input time series and remove data for pred window if needed
        
        ###MEDS
        if(self.feat_med):
            self.meds=self.meds[self.meds['stay_id'].isin(self.data['stay_id'])]
            self.meds=self.meds[self.meds['start_time']<=include_time]
            self.meds.loc[self.meds.stop_time >include_time, 'stop_time']=include_time
                    
        
        ###PROCS
        if(self.feat_proc):
            self.proc=self.proc[self.proc['stay_id'].isin(self.data['stay_id'])]
            self.proc=self.proc[self.proc['start_time']<=include_time]
            
        ###OUT
        if(self.feat_out):
            self.out=self.out[self.out['stay_id'].isin(self.data['stay_id'])]
            self.out=self.out[self.out['start_time']<=include_time]
            
       ###CHART
        if(self.feat_chart):
            self.chart=self.chart[self.chart['stay_id'].isin(self.data['stay_id'])]
            self.chart=self.chart[self.chart['start_time']<=include_time]
            
    def readmission_length(self,include_time):
        self.los=include_time
        self.data=self.data[(self.data['los']>=include_time)]
        self.hids=self.data['stay_id'].unique()
        
        if(self.feat_cond):
            self.cond=self.cond[self.cond['stay_id'].isin(self.data['stay_id'])]
        self.data['select_time']=self.data['los']-include_time
        self.data['los']=include_time

        ####Make equal length input time series and remove data for pred window if needed
        
        ###MEDS
        if(self.feat_med):
            self.meds=self.meds[self.meds['stay_id'].isin(self.data['stay_id'])]
            self.meds=pd.merge(self.meds,self.data[['stay_id','select_time']],on='stay_id',how='left')
            self.meds['stop_time']=self.meds['stop_time']-self.meds['select_time']
            self.meds['start_time']=self.meds['start_time']-self.meds['select_time']
            self.meds=self.meds[self.meds['stop_time']>=0]
            self.meds.loc[self.meds.start_time <0, 'start_time']=0
        
        ###PROCS
        if(self.feat_proc):
            self.proc=self.proc[self.proc['stay_id'].isin(self.data['stay_id'])]
            self.proc=pd.merge(self.proc,self.data[['stay_id','select_time']],on='stay_id',how='left')
            self.proc['start_time']=self.proc['start_time']-self.proc['select_time']
            self.proc=self.proc[self.proc['start_time']>=0]
            
        ###OUT
        if(self.feat_out):
            self.out=self.out[self.out['stay_id'].isin(self.data['stay_id'])]
            self.out=pd.merge(self.out,self.data[['stay_id','select_time']],on='stay_id',how='left')
            self.out['start_time']=self.out['start_time']-self.out['select_time']
            self.out=self.out[self.out['start_time']>=0]
            
       ###CHART
        if(self.feat_chart):
            self.chart=self.chart[self.chart['stay_id'].isin(self.data['stay_id'])]
            self.chart=pd.merge(self.chart,self.data[['stay_id','select_time']],on='stay_id',how='left')
            self.chart['start_time']=self.chart['start_time']-self.chart['select_time']
            self.chart=self.chart[self.chart['start_time']>=0]
        
            
    def smooth_meds(self,bucket):
        final_meds=pd.DataFrame()
        final_proc=pd.DataFrame()
        final_out=pd.DataFrame()
        final_chart=pd.DataFrame()
        
        if(self.feat_med):
            self.meds=self.meds.sort_values(by=['start_time'])
        if(self.feat_proc):
            self.proc=self.proc.sort_values(by=['start_time'])
        if(self.feat_out):
            self.out=self.out.sort_values(by=['start_time'])
        if(self.feat_chart):
            self.chart=self.chart.sort_values(by=['start_time'])
        
        t=0
        for i in tqdm(range(0,self.los,bucket)): 
            ###MEDS
             if(self.feat_med):
                sub_meds=self.meds[(self.meds['start_time']>=i) & (self.meds['start_time']<i+bucket)].groupby(['stay_id','itemid','orderid']).agg({'stop_time':'max','subject_id':'max','rate':'mean','amount':'mean'})
                sub_meds=sub_meds.reset_index()
                sub_meds['start_time']=t
                sub_meds['stop_time']=sub_meds['stop_time']/bucket
                if final_meds.empty:
                    final_meds=sub_meds
                else:
                    final_meds=pd.concat([final_meds, sub_meds], ignore_index=True)
                    
            
            ###PROC
             if(self.feat_proc):
                sub_proc=self.proc[(self.proc['start_time']>=i) & (self.proc['start_time']<i+bucket)].groupby(['stay_id','itemid']).agg({'subject_id':'max'})
                sub_proc=sub_proc.reset_index()
                sub_proc['start_time']=t
                if final_proc.empty:
                    final_proc=sub_proc
                else:    
                    final_proc=pd.concat([final_proc, sub_proc], ignore_index=True)
                    
              ###OUT
             if(self.feat_out):
                sub_out=self.out[(self.out['start_time']>=i) & (self.out['start_time']<i+bucket)].groupby(['stay_id','itemid']).agg({'subject_id':'max'})
                sub_out=sub_out.reset_index()
                sub_out['start_time']=t
                if final_out.empty:
                    final_out=sub_out
                else:    
                    final_out=pd.concat([final_out, sub_out], ignore_index=True)
                    
                    
              ###CHART
             if(self.feat_chart):
                sub_chart=self.chart[(self.chart['start_time']>=i) & (self.chart['start_time']<i+bucket)].groupby(['stay_id','itemid']).agg({'valuenum':np.nanmean})
                sub_chart=sub_chart.reset_index()
                sub_chart['start_time']=t
                if final_chart.empty:
                    final_chart=sub_chart
                else:    
                    final_chart=pd.concat([final_chart, sub_chart], ignore_index=True)
            
             t=t+1
        print("bucket",bucket)
        los=int(self.los/bucket)
        
        
        ###MEDS
        if(self.feat_med):
            f2_meds=final_meds.groupby(['stay_id','itemid','orderid']).size()
            self.med_per_adm=f2_meds.groupby('stay_id').sum().reset_index()[0].max()                 
            self.medlength_per_adm=final_meds.groupby('stay_id').size().max()
        
        ###PROC
        if(self.feat_proc):
            f2_proc=final_proc.groupby(['stay_id','itemid']).size()
            self.proc_per_adm=f2_proc.groupby('stay_id').sum().reset_index()[0].max()       
            self.proclength_per_adm=final_proc.groupby('stay_id').size().max()
            
        ###OUT
        if(self.feat_out):
            f2_out=final_out.groupby(['stay_id','itemid']).size()
            self.out_per_adm=f2_out.groupby('stay_id').sum().reset_index()[0].max() 
            self.outlength_per_adm=final_out.groupby('stay_id').size().max()
            
            
        ###chart
        if(self.feat_chart):
            f2_chart=final_chart.groupby(['stay_id','itemid']).size()
            self.chart_per_adm=f2_chart.groupby('stay_id').sum().reset_index()[0].max()             
            self.chartlength_per_adm=final_chart.groupby('stay_id').size().max()
        
        print("[ PROCESSED TIME SERIES TO EQUAL TIME INTERVAL ]")
        ###CREATE DICT
#         if(self.feat_chart):
#             self.create_chartDict(final_chart,los)
#         else:
        self.create_Dict(final_meds,final_proc,final_out,final_chart,los)
        
    
    def create_chartDict(self,chart,los):
        dataDic={}
        for hid in self.hids:
            grp=self.data[self.data['stay_id']==hid]
            dataDic[hid]={'Chart':{},'label':int(grp['label'])}
        for hid in tqdm(self.hids):
            ###CHART
            if(self.feat_chart):
                df2=chart[chart['stay_id']==hid]
                val=df2.pivot_table(index='start_time',columns='itemid',values='valuenum')
                df2['val']=1
                df2=df2.pivot_table(index='start_time',columns='itemid',values='val')
                #print(df2.shape)
                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                df2=pd.concat([df2, add_df])
                df2=df2.sort_index()
                df2=df2.fillna(0)
                
                val=pd.concat([val, add_df])
                val=val.sort_index()
                if self.impute=='Mean':
                    val=val.ffill()
                    val=val.bfill()
                    val=val.fillna(val.mean())
                elif self.impute=='Median':
                    val=val.ffill()
                    val=val.bfill()
                    val=val.fillna(val.median())
                val=val.fillna(0)
                
                
                df2[df2>0]=1
                df2[df2<0]=0
                #print(df2.head())
                dataDic[hid]['Chart']['signal']=df2.iloc[:,0:].to_dict(orient="list")
                dataDic[hid]['Chart']['val']=val.iloc[:,0:].to_dict(orient="list")
            
            
                
        ######SAVE DICTIONARIES##############
        with open("./data/dict/metaDic", 'rb') as fp:
            metaDic=pickle.load(fp)
        
        with open("./data/dict/dataChartDic", 'wb') as fp:
            pickle.dump(dataDic, fp)

      
        with open("./data/dict/chartVocab", 'wb') as fp:
            pickle.dump(list(chart['itemid'].unique()), fp)
        self.chart_vocab = chart['itemid'].nunique()
        metaDic['Chart']=self.chart_per_adm
        
            
        with open("./data/dict/metaDic", 'wb') as fp:
            pickle.dump(metaDic, fp)
            

    def create_Dict(self, meds, proc, out, chart, los, chunk_size=5000):
        dataDic = {}
        time_index = pd.Index(range(los))
        
        # File Paths
        csv_dir = "./data/csv"
        os.makedirs(csv_dir, exist_ok=True)
        dyn_path = os.path.join(csv_dir, "all_dynamic.csv")
        demo_path = os.path.join(csv_dir, "all_demo.csv")
        static_path = os.path.join(csv_dir, "all_static.csv")
        label_path = os.path.join(csv_dir, "labels.csv") # Path for labels

        # Clear existing files to prevent appending to old data
        for p in [dyn_path, demo_path, static_path, label_path]:
            if os.path.exists(p): os.remove(p)

        # Temporary storage for the current chunk
        chunk_demo, chunk_dyn, chunk_static = [], [], []
        
        print("Saving labels.csv...")
        # Extract stay_id and label from the main data cohort
        labels_df = self.data[['stay_id', 'label']].drop_duplicates()
        labels_df.to_csv(label_path, index=False)
        
        # 1. Pre-grouping (The key for speed)
        print("Pre-grouping large dataframes...")
        med_groups = dict(tuple(meds.groupby("stay_id"))) if self.feat_med else {}
        proc_groups = dict(tuple(proc.groupby("stay_id"))) if self.feat_proc else {}
        out_groups = dict(tuple(out.groupby("stay_id"))) if self.feat_out else {}
        chart_groups = dict(tuple(chart.groupby("stay_id"))) if self.feat_chart else {}
        cond_groups = dict(tuple(self.cond.groupby("stay_id"))) if self.feat_cond else {}
        data_groups = dict(tuple(self.data.groupby("stay_id")))

        med_feat = meds['itemid'].unique() if self.feat_med else []
        proc_feat = proc['itemid'].unique() if self.feat_proc else []
        out_feat = out['itemid'].unique() if self.feat_out else []
        chart_feat = chart['itemid'].unique() if self.feat_chart else []
        cond_feat = self.cond['new_icd_code'].unique() if self.feat_cond else []

        def flush_to_disk(data_list, path):
            if not data_list: return
            df = pd.concat(data_list)
            # Write header only if file doesn't exist
            is_new = not os.path.exists(path)
            df.to_csv(path, mode='a', index=False, header=is_new)
            data_list.clear()

        print(f"Processing {len(self.hids)} patients...")
        for i, hid in enumerate(tqdm(self.hids)):
            grp = data_groups.get(hid)
            if grp is None: continue

            # Populate dataDic (keep in memory for pickle)
            dataDic[hid] = {
                'Cond': {}, 'Proc': {}, 'Med': {}, 'Out': {}, 'Chart': {},
                'ethnicity': grp['ethnicity'].iloc[0],
                'age': int(grp['Age'].iloc[0]),
                'gender': grp['gender'].iloc[0],
                'label': int(grp['label'].iloc[0])
            }

            # --- DEMO ---
            d_df = grp[['Age', 'gender', 'ethnicity', 'insurance']].copy()
            d_df['stay_id'] = hid
            chunk_demo.append(d_df)

            # --- DYNAMIC ---
            p_dyn = []
            # MEDS Logic
            if self.feat_med:
                m = med_groups.get(hid)
                if m is None:
                    amt = pd.DataFrame(np.zeros((los, len(med_feat))), columns=med_feat)
                else:
                    rt = m.pivot_table(index='start_time', columns='itemid', values='rate').reindex(time_index).ffill().fillna(-1)
                    am = m.pivot_table(index='start_time', columns='itemid', values='amount').reindex(time_index).ffill().fillna(-1)
                    st = m.pivot_table(index='start_time', columns='itemid', values='stop_time').reindex(time_index).ffill().fillna(0)
                    
                    sig = (st.sub(st.index, axis=0) > 0).astype(int)
                    dataDic[hid]['Med'] = {'signal': sig.to_dict("list"), 'rate': (sig*rt).to_dict("list"), 'amount': (sig*am).to_dict("list")}
                    amt = (sig*am).reindex(columns=med_feat, fill_value=0)
                amt.columns = pd.MultiIndex.from_product([["MEDS"], amt.columns])
                p_dyn.append(amt)

            # PROC & OUT Logic
            for feat_flag, g_dict, lbl, f_list in [(self.feat_proc, proc_groups, "PROC", proc_feat), (self.feat_out, out_groups, "OUT", out_feat)]:
                if feat_flag:
                    d2 = g_dict.get(hid)
                    if d2 is None:
                        res = pd.DataFrame(np.zeros((los, len(f_list))), columns=f_list)
                    else:
                        d2 = d2.copy(); d2['val'] = 1
                        res = d2.pivot_table(index='start_time', columns='itemid', values='val').reindex(time_index).fillna(0)
                        res = (res > 0).astype(int)
                        dataDic[hid][lbl.capitalize()] = res.to_dict("list")
                        res = res.reindex(columns=f_list, fill_value=0)
                    res.columns = pd.MultiIndex.from_product([[lbl], res.columns])
                    p_dyn.append(res)

            # --- CHART Logic (Updated Fix) ---
            if self.feat_chart:
                c = chart_groups.get(hid)
                if c is None:
                    v = pd.DataFrame(np.zeros((los, len(chart_feat))), columns=chart_feat)
                else:
                    # 1. Ensure the dataframe columns are flat
                    if isinstance(c.columns, pd.MultiIndex):
                        c.columns = c.columns.get_level_values(-1)

                    # 2. Perform the pivots with explicit column selection
                    # Using values='valuenum' and index/columns ensures a 2D result
                    v = c.pivot_table(index='start_time', columns='itemid', values='valuenum').reindex(time_index)
                    
                    # This was the line crashing:
                    # We create a 'present' column to make the signal pivot simpler
                    temp_c = c.copy()
                    temp_c['present'] = 1
                    s = temp_c.pivot_table(index='start_time', columns='itemid', values='present').reindex(time_index).notnull().astype(int)

                    if self.impute == "Mean": 
                        v = v.ffill().bfill().fillna(v.mean())
                    elif self.impute == "Median": 
                        v = v.ffill().bfill().fillna(v.median())
                    
                    v = v.fillna(0)
                    dataDic[hid]['Chart'] = {'signal': s.to_dict("list"), 'val': v.to_dict("list")}
                    v = v.reindex(columns=chart_feat, fill_value=0)
                
                v.columns = pd.MultiIndex.from_product([["CHART"], v.columns])
                p_dyn.append(v)
            if p_dyn:
                # Join Meds, Procs, Outs, and Chart horizontally
                patient_dynamic_df = pd.concat(p_dyn, axis=1)
                # Add stay_id so we can filter/load it later
                patient_dynamic_df[('stay_id', 'stay_id')] = hid
                chunk_dyn.append(patient_dynamic_df)

            # --- STATIC (COND) ---
            if self.feat_cond:
                cg = cond_groups.get(hid)
                if cg is None:
                    dataDic[hid]['Cond'] = {'fids': ['<PAD>']}
                    stc = pd.DataFrame(np.zeros((1, len(cond_feat))), columns=cond_feat)
                else:
                    dataDic[hid]['Cond'] = {'fids': list(cg['new_icd_code'])}
                    cg = cg.copy(); cg['val'] = 1
                    stc = cg.drop_duplicates().pivot(index='stay_id', columns='new_icd_code', values='val').reindex(columns=cond_feat, fill_value=0)
                stc.columns = pd.MultiIndex.from_product([["COND"], stc.columns])
                stc['stay_id'] = hid
                chunk_static.append(stc)

            # --- PERIODIC FLUSH (Memory Safety) ---
            if (i + 1) % chunk_size == 0:
                flush_to_disk(chunk_demo, demo_path)
                flush_to_disk(chunk_dyn, dyn_path)
                flush_to_disk(chunk_static, static_path)

        # Final flush for remaining data
        flush_to_disk(chunk_demo, demo_path)
        flush_to_disk(chunk_dyn, dyn_path)
        flush_to_disk(chunk_static, static_path)

        # Final Pickles
        print("Saving metadata dictionaries...")
        os.makedirs("./data/dict", exist_ok=True)
        ######SAVE DICTIONARIES##############
        metaDic={'Cond':{},'Proc':{},'Med':{},'Out':{},'Chart':{},'LOS':{}}
        metaDic['LOS']=los
        with open("./data/dict/dataDic", 'wb') as fp:
            pickle.dump(dataDic, fp)

        with open("./data/dict/hadmDic", 'wb') as fp:
            pickle.dump(self.hids, fp)
        
        with open("./data/dict/ethVocab", 'wb') as fp:
            pickle.dump(list(self.data['ethnicity'].unique()), fp)
            self.eth_vocab = self.data['ethnicity'].nunique()
            
        with open("./data/dict/ageVocab", 'wb') as fp:
            pickle.dump(list(self.data['Age'].unique()), fp)
            self.age_vocab = self.data['Age'].nunique()
            
        with open("./data/dict/insVocab", 'wb') as fp:
            pickle.dump(list(self.data['insurance'].unique()), fp)
            self.ins_vocab = self.data['insurance'].nunique()
            
        if(self.feat_med):
            with open("./data/dict/medVocab", 'wb') as fp:
                pickle.dump(list(meds['itemid'].unique()), fp)
            self.med_vocab = meds['itemid'].nunique()
            metaDic['Med']=self.med_per_adm
            
        if(self.feat_out):
            with open("./data/dict/outVocab", 'wb') as fp:
                pickle.dump(list(out['itemid'].unique()), fp)
            self.out_vocab = out['itemid'].nunique()
            metaDic['Out']=self.out_per_adm
            
        if(self.feat_chart):
            with open("./data/dict/chartVocab", 'wb') as fp:
                pickle.dump(list(chart['itemid'].unique()), fp)
            self.chart_vocab = chart['itemid'].nunique()
            metaDic['Chart']=self.chart_per_adm
        
        if(self.feat_cond):
            with open("./data/dict/condVocab", 'wb') as fp:
                pickle.dump(list(self.cond['new_icd_code'].unique()), fp)
            self.cond_vocab = self.cond['new_icd_code'].nunique()
            metaDic['Cond']=self.cond_per_adm
        
        if(self.feat_proc):    
            with open("./data/dict/procVocab", 'wb') as fp:
                pickle.dump(list(proc['itemid'].unique()), fp)
            self.proc_vocab = proc['itemid'].nunique()
            metaDic['Proc']=self.proc_per_adm
            
        with open("./data/dict/metaDic", 'wb') as fp:
            pickle.dump(metaDic, fp)
        print("Done! Data consolidated into 3 CSV files and dictionaries.")
