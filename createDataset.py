import os, gc
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import pandas as pd, numpy as np
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
import pickle
from pydub import AudioSegment
import re

sound_file = AudioSegment.from_mp3("assets/audio/Hades_voice_line_Athena.mp3")
train = pd.read_csv('Hades_voice_line_Athena.tsv', names=['start',	'end',	'text'], skiprows=1, sep='\t')
train["text"] = train['text'].str.replace('[^\w\s]','')
print(train.head())
index = 0
all_data = []
for _, row in tqdm(train.iterrows()):
    start = int(row.start)
    end = int(row.end)
    audio = sound_file[start:end]
    audio.export(
        "/mnt/d/Data/Athena_voice/audio/chunk{0}.wav".format(index),
        bitrate = "63k",
        format = "wav"
    )
    res = re.sub(r'[^\w\s]', '', row.text)
    all_data.append({
        "audio": "chunk{0}.wav".format(index),
        "text": res,
    })
    index += 1
    # print(start, )

output_df = pd.DataFrame(all_data)
output_df.to_csv("/mnt/d/Data/Athena_voice/train_audio.txt", sep='|', header=False, index=False)