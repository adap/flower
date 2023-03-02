import math
import pathlib
from typing import Union
from constants import RANDOM_SEED
import pandas as pd


def _create_samples_division_list(n_samples, n_groups, keep_remainder=True):
    group_size = n_samples // n_groups
    n_samples_in_full_groups = n_groups * group_size
    samples_division_list = []
    for i in range(n_groups):
        samples_division_list.extend([i] * group_size)
    if n_samples_in_full_groups != n_samples:
        # add remainder only if it is needed == remainder is not equal zero
        remainder = n_samples - n_samples_in_full_groups
        samples_division_list.extend([n_groups] * remainder)
    return samples_division_list


class NistSampler:
    def __init__(self, data_info_df: pd.DataFrame):
        self._data_info_df = data_info_df

    def sample(self, type: str, frac: float, n_clients: Union[int, None] = None) -> pd.DataFrame:
        # n_clients is not used when niid
        # The question is if that hold in memory
        if type == "iid":
            if n_clients is None:
                raise ValueError("n_clients can not be None for idd training")
            idd_data_info_df = self._data_info_df.sample(frac=frac, random_state=RANDOM_SEED)
            # add client ids (todo: maybe better in the index)
            idd_data_info_df["client_id"] = _create_samples_division_list(idd_data_info_df.shape[0], n_clients, True)
            return idd_data_info_df
        elif type == "niid":
            if n_clients is not None:
                print(
                    "Warning: the n_clinets is ignored in case of niid training. "
                    "The number of clients is equal to the number of unique writers")
            # It uses the following sampling logic:
            # Take N users with their full data till it doesn't exceed the total number of data that can be used
            # Then take remaining M samples (from 1 or more users, probably only one) till the  total number of samples
            # is reached
            frac_samples = math.ceil(frac * self._data_info_df.shape[0])  # make it consistent with pd.DatFrame.sample()
            niid_data_info_full = self._data_info_df.copy()
            writer_ids_to_cumsum = niid_data_info_full.groupby("writer_id").size().sample(frac=1.,
                                                                                          random_state=RANDOM_SEED).cumsum()
            writer_ids_to_consider_mask = writer_ids_to_cumsum < frac_samples
            partial_writer_id = writer_ids_to_consider_mask.idxmin()
            niid_data_info_full = niid_data_info_full.set_index("writer_id", append=True)
            writer_ids_to_consider = writer_ids_to_consider_mask[writer_ids_to_consider_mask].index.values
            niid_data_info = niid_data_info_full.loc[niid_data_info_full.index.get_level_values("writer_id").isin(
                writer_ids_to_consider)]
            # fill in remainder
            current_n_samples = niid_data_info.shape[0]
            missing_samples = frac_samples - current_n_samples
            partial_writer_samples = niid_data_info_full.loc[niid_data_info_full.index.get_level_values(
                "writer_id") == partial_writer_id].iloc[:missing_samples]

            niid_data_info = pd.concat([niid_data_info, partial_writer_samples], axis=0)
            return niid_data_info  # todo: think if you should change the writer_id here to client_id and if it should be a number
        else:
            raise ValueError(f"The given type of smapling is not supported. Given: {type}")


if __name__ == "__main__":
    df_info_path = pathlib.Path("data/processed/resized_images_to_labels.csv")
    df_info = pd.read_csv(df_info_path, index_col=0)
    sampler = NistSampler(df_info)
    sampled_data_info = sampler.sample("niid", 0.05, 100)
    sampled_data_info.to_csv("data/processed/niid_sampled_images_to_labels.csv")

    import matplotlib.pyplot as plt

    plt.figure()
    sampled_data_info.reset_index(drop=False)["writer_id"].value_counts().plot.hist()
    print(sampled_data_info.shape)
    # plt.show()
