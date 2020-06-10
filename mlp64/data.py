import json
import pickle
from pathlib import Path

import librosa
from librosa.core import fft
import numpy as np
import pandas as pd
import scipy.fft as scipyfft
import torch
import torchaudio as ta
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# Do this to maintain 32bit data with librosa 0.72
fft.set_fftlib(scipyfft)

N_SAMPLES_NSYNTH = 64000
SAMPLE_RATE_NSYNTH = 16000

# Used for train_test_split to ensure reproducibility
SPLIT_SEED = 42


class CachedNSynth(Dataset):
    """PyTorch Dataset for the NSynth dataset."""
    
    def __init__(self, root, df: pd.DataFrame, target_field: str, transform=None, normalise=True, cache="cache",
                 overwrite=False, n=1, noclass=False):
        """
        Creates cached NSynth dataset for PyTorch. Cache is essentially pickling the transformed files onto disk and
        loading them from there. Concatenates n notes to create small tunes. Set seed before loading to ensure
        reproducibility.
        :param root: Path or str to an nsynth-.../audio folder
        :param df: output of create_dataset_df(..)
        :param target_field: one from ["instrument", "instrument_source", "instrument_family", "instrument_class"] to
                            make class labels based on instrument (out of the 1006 instrument), source (acoustic,
                            electronic or synthetic), family (out of the 10: guitar, reed, ...) or the 33 possibilites
                            when using both source and family (instrument_class). As there are e.g. no acoustic synth-
                            lead instruments, there will be a total of 28 classes (in the nsynth-train dataset).
        :param transform: function used to transform input data (e.g. torchaudio.transforms.MelSpectrogram())
        :param cache: location (under root) to write the .pkl files
        :param overwrite: flag to determine if existing files should be overwritten (use if you want to write files with a
                          different transformation to a path already containing other cached files.
        """
        if isinstance(root, (str, Path)):
            root = Path(root)
            assert root.parts[-1] == "audio", "root needs to point to the nsynth-***/audio folder."
        assert target_field in ["instrument", "instrument_source", "instrument_family", "instrument_class"]
        
        self.root = root
        self.filenames = dftostr(df)
        
        if target_field == "instrument_class":
            # Encode source and family together to make 33 distinct classes
            df["instrument_class"] = df.apply(lambda x: x["instrument_source"] * 11 + x["instrument_family"], axis=1)
        self.df = df
        
        self.targets = df[target_field].to_list()
        self.transform = transform
        self.normalise = normalise
        self.n = n
        
        self.cache = self.root / cache
        self.cache.mkdir(exist_ok=True)
        self.cached_files = [self.cache / (x.split('.')[0] + '.pkl') for x in self.filenames]

        self.noclass = noclass

        if overwrite:
            for file in self.cached_files:
                if file.is_file():
                    file.unlink()
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        if self.cached_files[index].is_file():
            with open(self.cached_files[index], 'rb') as f:
                x = pickle.load(f)
        else:
            path = self.root / self.filenames[index]
            x = torch.empty(N_SAMPLES_NSYNTH * self.n)
            x[:N_SAMPLES_NSYNTH], _ = ta.load(path, normalization=self.normalise)
            # Choose self.n - 1 other notes to concatenate
            j = 1
            for i in np.random.choice(range(len(self.filenames)), self.n - 1):
                x[N_SAMPLES_NSYNTH * j:N_SAMPLES_NSYNTH * (j + 1)], _ = ta.load(self.root / self.filenames[i],
                                                                                normalization=self.normalise)
                j += 1
            if self.transform is not None:
                x = self.transform(x)
            with open(self.cached_files[index], 'wb') as f:
                pickle.dump(x, f)
        if self.noclass:
            return x
        else:
            return x, self.targets[index]


class NSynth(Dataset):
    """PyTorch Dataset for the NSynth dataset."""
    
    def __init__(self, root, df: pd.DataFrame, target_field: str, transform=None, normalise=True, n=1):
        """
        Creates NSynth dataset for PyTorch. Concatenates n notes to create small tunes. Set seed before loading to
        ensure reproducibility.
        :param root: Path or str to an nsynth-.../audio folder
        :param df: output of create_dataset_df(..)
        :param target_field: one from ["instrument", "instrument_source", "instrument_family", "instrument_class"] to
                            make class labels based on instrument (out of the 1006 instrument), source (acoustic,
                            electronic or synthetic), family (out of the 10: guitar, reed, ...) or the 33 possibilites
                            when using both source and family (instrument_class). As there are e.g. no acoustic synth-
                            lead instruments, there will be a total of 28 classes (in the nsynth-train dataset).
        :param transform: function used to transform input data (e.g. torchaudio.transforms.MelSpectrogram())
        """
        if isinstance(root, (str, Path)):
            root = Path(root)
            assert root.parts[-1] == "audio", "root needs to point to the nsynth-***/audio folder."
        assert target_field in ["instrument", "instrument_source", "instrument_family", "instrument_class"]
        
        self.root = root
        self.filenames = dftostr(df)
        
        if target_field == "instrument_class":
            # Encode source and family together to make 33 distinct classes
            df["instrument_class"] = df.apply(lambda x: x["instrument_source"] * 11 + x["instrument_family"], axis=1)
        self.df = df
        
        self.targets = df[target_field].to_list()
        self.transform = transform
        self.normalise = normalise
        self.n = n
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        path = self.root / self.filenames[index]
        x = torch.empty(N_SAMPLES_NSYNTH * self.n)
        x[:N_SAMPLES_NSYNTH], _ = ta.load(path, normalization=self.normalise)
        # Choose self.n - 1 other notes to concatenate
        j = 1
        for i in np.random.choice(range(len(self.filenames)), self.n - 1):
            x[N_SAMPLES_NSYNTH * j:N_SAMPLES_NSYNTH * (j + 1)], _ = ta.load(self.root / self.filenames[i],
                                                                            normalization=self.normalise)
            j += 1
        if self.transform is not None:
            x = self.transform(x)
        return x, self.targets[index]


def create_dataset_df(json_file, instrument_family_str="all", instrument_source_str="all", instrument="all",
                      pitch=[0, 127], velocity=[0, 127], include_qualities=[], exclude_qualities=["distortion"]):
    """
    Creates a DataFrame corresponding to a (possibly filtered) NSynth dataset. By default excludes distorted notes.
    https://magenta.tensorflow.org/datasets/nsynth
    
    :param json_file: Path/str to the examples.json file, or a previously loaded pd.DataFrame
    :param instrument_family_str: filter according to instrument_family (e.g. "guitar") - default uses all.
    :param instrument_source_str: filter according to instrument_source (e.g. "electronic") - default uses all.
    :param pitch: list of [min_pitch, max_pitch] to filter according to pitch. Limits are inclusive
    :param velocity: list of [min_velocity, max_velocity] to filter according to velocity. Limits are inclusive
    :param include_qualities: list of qualities (according to qualities_str) that need to be present in a note
    :param exclude_qualities: list of qualities according to which notes will be filtered. default excludes "distortion"
    :return: examples.json as pd.DataFrame, filtered according to parameters
    """
    if isinstance(json_file, (str, Path)):
        path = Path(json_file)
        assert path.is_file(), "json_file needs to point to an examples.json file."
        with open(path) as f:
            df = pd.DataFrame.from_records(json.load(f)).T
    else:
        assert isinstance(json_file,
                          pd.DataFrame), "json_file needs to be a pandas DataFrame or a string containing a path to the examples.json file"
        df = json_file
    
    print("JSON loaded into DataFrame!")
    
    try:
        if instrument_family_str != "all":
            df = df[df["instrument_family_str"] == instrument_family_str]
        if instrument_source_str != "all":
            df = df[df["instrument_source_str"] == instrument_source_str]
        if instrument != "all":
            df = df[df["instrument"] == instrument]
        
        if pitch != [0, 127]:
            df = df[(df["pitch"] >= pitch[0]) & (df["pitch"] <= pitch[1])]
        if velocity != [0, 127]:
            df = df[(df["velocity"] >= velocity[0]) & (df["velocity"] <= velocity[1])]
        
        # Map quality strings to 1/0 lists and filter based on those
        qualities_str = ["bright", "dark", "distortion", "fast_decay", "long_release", "multiphonic", "nonlinear_env",
                         "percussive", "reverb", "tempo-synced"]
        assert all([x in qualities_str for x in
                    (include_qualities + exclude_qualities)]), "Some referred qualities are not legal quality strings."
        
        if len(include_qualities) != 0:
            qualities = [1 if x in include_qualities else 0 for x in qualities_str]
            # sum(a and b) == sum(a) - all those in included need to be in there
            df = df[df.qualities.apply(lambda x: sum(qualities) == sum(np.logical_and(x, qualities)))]
        if len(exclude_qualities) > 0:
            qualities = [1 if x in exclude_qualities else 0 for x in qualities_str]
            # not any(df_qualities and qualities) - there can't be any of the excluded ones in there
            df = df[df.qualities.apply(lambda x: not any(np.logical_and(x, qualities)))]
    except AttributeError as a:
        print("Most likely the selection you want doesn't exist. Try broadening the options or find the bug!")
        raise a
    except Exception as e:
        raise e
    
    return df


def get_train_test(df: pd.DataFrame, target_field, test_size=0.2):
    """
    Splits json dataframe into train/test split, stratified according to target_field.
    :param df: output of create_dataset_df(..)
    :param target_field: field to use as the target for the classifier.
                        one from ["instrument", "instrument_source", "instrument_family", "instrument_class"] to
                        make class labels based on instrument (out of the 1006 instrument), source (acoustic,
                        electronic or synthetic), family (out of the 10: guitar, reed, ...) or the 33 possibilites
                        when using both source and family (instrument_class). As there are e.g. no acoustic synth-
                        lead instruments, there will be a total of 28 classes (in the nsynth-train dataset).
    :param test_size:
    :return:
    """
    if target_field == "instrument_class":
        # Encode source and family together to make 33 (actually 28) distinct classes
        df["instrument_class"] = df.apply(lambda x: x["instrument_source"] * 11 + x["instrument_family"], axis=1)
    targets = df[target_field]
    trdf, tedf = train_test_split(df, test_size=test_size, stratify=df[target_field], random_state=SPLIT_SEED)
    return trdf, tedf


def get_source_target_tt(df: pd.DataFrame, source_label, target_label, test_size=0.2,
                         distinguisher="instrument_family_str"):
    """
    Splits json dataframe into train/test splits for both the source and target data.
    :param df: output of create_dataset_df(..)
    :param source_label: value of the source dataset in the distinguisher field
                        (e.g. "bass_electronic" if distinguisher = "instrument_family")
    :param target_label: value of the target dataset in the distinguisher field
    :param distinguisher: field by which the dataset is split into source and target sets.
                        one from ["instrument", "instrument_source", "instrument_family", "instrument_class"] to
                        make class labels based on instrument (out of the 1006 instrument), source (acoustic,
                        electronic or synthetic), family (out of the 10: guitar, reed, ...) or the 33 possibilites
                        when using both source and family (instrument_class). As there are e.g. no acoustic synth-
                        lead instruments, there will be a total of 28 classes (in the nsynth-train dataset).
    :param test_size: size of the test set as a proportion of the original dataset
    :return:
    """
    sdf = df[df[distinguisher] == source_label]
    tdf = df[df[distinguisher] == target_label]
    strdf, stedf = train_test_split(sdf, test_size=test_size, random_state=SPLIT_SEED)
    ttrdf, ttedf = train_test_split(tdf, test_size=test_size, random_state=SPLIT_SEED)
    return strdf, stedf, ttrdf, ttedf


def dftostr(df: pd.DataFrame):
    """
    Returns the notes present in a DataFrame as list of NSynth filenames (e.g. ["bass_synthetic_033-022-050.wav", ...]
    :param df: output of create_dataset_df(..)
    :return:
    """
    return df["note_str"].apply(lambda x: x + ".wav").to_list()


def load_files(path, files: list, format="numpy"):
    """
    Loads NSynth files into numpy array.
    :param path: Path to NSynth audio folder
    :param files: list of files (e.g. from dftostr(..)) to load
    :param format: currently only numpy supported
    :return:
    """
    assert format in ["pytorch", "numpy"], "Format {} not yet supported.".format(format)
    
    data = np.empty((N_SAMPLES_NSYNTH, len(files)), dtype=np.float32, order="F")
    for i, f in enumerate(files):
        data[:, i], _ = librosa.load(path / f, sr=SAMPLE_RATE_NSYNTH)
    
    if format == "numpy":
        return data
    if format == "pytorch":
        Warning("If you want to train PyTorch models, you should use the NSynth dataset class.")
        return torch.as_tensor(data)
