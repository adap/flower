def drop_wrong_uom(data, cut_off):
#     count=0
    grouped = data.groupby(['itemid'])['valueuom']
    for id_number, uom in grouped:
        value_counts = uom.value_counts()
        num_observations = len(uom)
        if(value_counts.size >1):
#             count+=1
            most_frequent_measurement = value_counts.index[0]
            print(value_counts)
            frequency = value_counts.iloc[0]
#             print(id_number,value_counts.size,frequency/num_observations)
            if(frequency/num_observations > cut_off):
                values = uom
                index_to_drop = values[values != most_frequent_measurement].index
                data.drop(index_to_drop, axis=0, inplace=True)
    data = data.reset_index(drop=True)
#     print(count)
    return data

