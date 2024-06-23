import librosa
import numpy as np
from fastdtw import fastdtw
import noisereduce as nr
import os
from scipy.spatial.distance import cosine


def remove_silence(y):
    parts = librosa.effects.split(y, top_db=25, frame_length=1024, hop_length=512)
    y_non_silent = []
    for start, end in parts:
        y_non_silent.extend(y[start:end])

    y = np.array(y_non_silent)
    return y

def preprocessing(audio):
    y , sr = librosa.load(audio, sr=44100)
    y = librosa.effects.preemphasis(y)
    y = y/np.max(np.abs(y))
    y = remove_silence(y)
    y = nr.reduce_noise(y, sr, prop_decrease=0.8)
    return y,sr

def extraction(audio):
    y , sr = preprocessing(audio)
    mfccs = librosa.feature.mfcc(y=y,
                                 sr=sr,
                                 n_mfcc=13,
                                 window='hamming',
                                 win_length=1024,
                                 htk=True,
                                 hop_length=512,
                                 n_fft=1024,
                                 norm=None,
                                 dct_type=2,
                                 n_mels=20)
    return mfccs.T

def calculate_dtw(mfccs1, mfccs2):
    distance, _ = fastdtw(mfccs1, mfccs2, dist=cosine)
    return distance

def scoring(file_input, folder_template, template_score, max_threshold_score):
    audio_input = file_input
    # print(f"Checking folder: {folder_template}")  # Debug statement
    templates = os.listdir(folder_template)
    total_distances = 0

    for template in templates:
        audio_template = os.path.join(folder_template, template)

        mfccs1 = extraction(audio_input)
        mfccs2 = np.loadtxt(audio_template, delimiter=',')

        dtw_distance = calculate_dtw(mfccs1, mfccs2)

        total_distances += dtw_distance

    # Calculate average distance
    average_distance = total_distances / len(templates)
    # Calculate the difference between the average distance and the template score
    dif_distance = average_distance - template_score

    # Calculate the final score based on the difference
    if dif_distance <= 0:
        final_score = 100
    elif dif_distance >= max_threshold_score:
        final_score = 0
    else:
        final_score = int(100 - ((dif_distance / max_threshold_score) * 100))

    return final_score

root = "./Dataset_MFCC_prem_norm_non_ortho/"

list_reference_folders = [
    ("01.Ha\'/", 50),
    ("02.Kha\'/", 50),
    ("03.Shad/", 50),
    ("04.Dhad/", 50),
    ("05.Tha\'/", 50),
    ("06.Dhza\'/", 50),
    ("07.\'AIn/", 50),
    ("08.Ghain/", 50),
    ("09.Qaf/", 50),
    ("10.Ha^\'/", 50)
]

file_input, index_input = "./Dataset/Testing_Rafi/rafi_shad.wav", 2

dirPath_reference, score_reference = list_reference_folders[index_input]
max_threshold = 5

folder_reference = os.path.join(root, dirPath_reference)

skor = scoring(file_input, folder_reference, score_reference, max_threshold)
print(skor)

# print(f"Folder reference: {folder_reference}")  # Debug statement

# # Ensure the folder exists before proceeding
# if not os.path.exists(folder_reference):
#     print(f"Error: The folder {folder_reference} does not exist.")
# else:
#     skor = scoring(file_input, folder_reference, score_reference, max_threshold)
#     print(skor)


