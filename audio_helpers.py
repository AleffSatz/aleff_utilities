import os
import pandas as pd
import numpy as np
import glob
import seaborn as sns
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
import pickle
from tempfile import mkdtemp
import librosa as lr
import librosa.display as display
import numba
import warnings
from tqdm import tqdm

def plot_left_right(y_left, y_right, sr):
    '''
    plot left and right channel on top of each other
    '''
    plt.subplot(2,1,1)
    display.waveplot(y_right, sr)
    plt.title('Right channel ')

    plt.subplot(2,1,2)
    display.waveplot(y_left, sr)
    plt.title('Left channel ')
    plt.tight_layout()
    plt.xlabel('')
    
    plt.show()

def plot_active_segments(y_right, change_times
                         , sr=22050
                        ):
    '''
    plot active segments of an audio file after splitting by silence
    '''
    try:
        display.waveplot(y_right, sr)
    except Exception as error:
        # there might be an error here where waveplot expects a fotran array
        # this behaviour, if persistent, is new
        display.waveplot(np.asfortranarray(y_right), sr)
    plt.title('Right channel')

    #plt.subplot(2,1,2, sharex=ax1)
    #plt.plot(change_times, label='states')
    plt.vlines(change_times, y_right.max() * (-1), y_right.max()+0.1, color='r', alpha=0.9,
                linestyle='--', label='active segments')
    plt.axis('tight')
    plt.legend(bbox_to_anchor=(1.,1.))
    
    plt.xlabel('')
    plt.show()
    
def visualize_energy_distribution(wave, return_energy=True):
    '''
    visualize energy distribution with option to return the values for further analysis
    '''
    audio_energy = lr.feature.rms(wave)**2
    audio_energy_db = lr.power_to_db(audio_energy, ref=np.median)
    sns.distplot(audio_energy_db)
    return audio_energy_db

###########################################################################
def split_by_median(wave
                    , sr
                    , temp_top_db=None
                    , visualize=False
                   ):
    '''
    split wave by silence using median_db as reference
    
    good for long audio files with plenty of dialogue
    '''
    if not temp_top_db:
        audio_energy = lr.feature.rms(wave)**2
        audio_energy_db = lr.power_to_db(audio_energy, ref=np.median)
        wave_min = np.min(audio_energy_db)
        wave_median = np.median(audio_energy_db)
        temp_top_db = (wave_median - wave_min)/2 + 1e-2
    
    intervals = lr.effects.split(wave, top_db=temp_top_db + 1, ref=np.median)
    intervals_length = (intervals[:,1] - intervals[:,0])/sr
    active_segments = intervals[intervals_length > 1.,:]/sr
    
    if visualize:
        plot_active_segments(np.asfortranarray(wave), active_segments, sr=sr)
    
    return active_segments


def split_by_max(wave
                 , sr
                 , temp_top_db=None
                 , visualize=False
               ):
    '''
    split wave by silence using max_db as reference
    
    good for short audio files with plenty of silence
    '''
    
    if not temp_top_db:
        audio_energy = lr.feature.rms(wave)**2
        audio_energy_db = lr.power_to_db(audio_energy, ref=np.median)
        wave_max = np.max(audio_energy_db)
        wave_median = np.percentile(audio_energy_db, 25)
        temp_top_db = -(wave_median - wave_max)/2 + 1e-2
    
    intervals = lr.effects.split(wave, top_db=temp_top_db + 1, ref=np.max)
    intervals_length = (intervals[:,1] - intervals[:,0])/sr
    active_segments = intervals[intervals_length > 1.,:]/sr
    
    if visualize:
        plot_active_segments(np.asfortranarray(wave), active_segments, sr=sr)
    
    return active_segments


def split_by_silence(test_file, sr=None, visualize=False
                     , length_threshold=30
                    ):
    '''
    input: path of audio file
    output: array of audio segments split by silence
    
    flow: 
    + first split file by median_db to get active_segments
    + get lengths of active segments
    + for segment in active_segments:
        if length > say 30 seconds: 
            get wave and starting point of that segment
            split by max_db on segment -> mini-active-segments
            add starting point to mini-segments
            add mini-segments to list of final segments
    + concat mini-segments
    '''
    if not sr:
        y, sr = lr.load(test_file, sr=sr, mono=False)
    else:
        y, sr = lr.load(test_file, mono=False)
        
    active_segments = split_by_median(y[1,:], sr, visualize=False)
    segments_length = active_segments[:,1] - active_segments[:,0]
    final_segments_list = []
    for i, temp_length in enumerate(segments_length):
        if (temp_length) > length_threshold:
            #print('mini')
            start_point, end_point = active_segments[i]
            #print(start_point, end_point)
            mini_y = y[1, int(start_point * sr):int(end_point * sr)]
            mini_segments = split_by_max(mini_y, sr, visualize=False) + start_point
            final_segments_list.append(mini_segments)
        else:
            final_segments_list.append(active_segments[i].reshape(1,-1))
            
    return np.concatenate(final_segments_list)


def export_audio_segments(samples_dir
                          , sr=None
                          , length_threshold=18
                          , visualize=False
                          , export=True
                          , progress_bar=True
                         ):
    '''
    segment and export a corresponding csv files for all audio files in samples_dir
    '''
    warnings.filterwarnings('ignore', category=UserWarning)
    if not sr:
        sr = 22050 # default librosa value
    
    segments_df_list = []
    files_list = tqdm(os.listdir(samples_dir)) if progress_bar else os.listdir(samples_dir)
    for file in files_list:
        path = samples_dir + file
        try:

            active_segments = split_by_silence(path, sr=sr, length_threshold=length_threshold)
            if visualize:
                print(file)
                y, sampling_rate = lr.load(path, sr=sr, mono=False)
                plot_active_segments(np.asfortranarray(y[1,:]), active_segments, sampling_rate)
            temp_df = pd.DataFrame({'File': [file] * len(active_segments)
                                    , 'Path': [path] * len(active_segments)
                                    , 'Segment': np.arange(len(active_segments))
                                    , 'SegmentStart': active_segments[:,0]
                                    , 'SegmentEnd': active_segments[:,1]
                                    , 'SegmentLength': active_segments[:,1] - active_segments[:,0]
                                   })
            segments_df_list.append(temp_df)
        except Exception as error:
            print('Error loading', file)
        
    if export:
        temp_file_name = 'audio_segments.csv'
        pd.concat(segments_df_list).to_csv(samples_dir + temp_file_name, index=False, sep='\t')
    return pd.concat(segments_df_list)
