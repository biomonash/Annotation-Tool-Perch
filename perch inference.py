# this script requires onnxruntime and scipy, all other libraries are base python. you can just install both of them with `pip install onnxruntime scipy`.
# you also need to download the model from https://huggingface.co/justinchuby/Perch-onnx/blob/main/perch_v2_no_dft.onnx and redirect line 14 to point at it.
# input filepath is currently on line 69, replace that with the input file you want to process.
# this script has the audio clipping fix built-in through `get_reconstruction`. you can take it out on line 65 if you don't want it.

import datetime
import scipy
import os
from pathlib import Path
import glob
import json

import onnxruntime as ort
import numpy as np
ort_sess = ort.InferenceSession("perch_v2_no_dft.onnx")

OUTPUT_NAMES = ["embeddings", "spatiotemporal_embeddings", "spectrogram", "class_logits"]
SAVE_OUTPUT_NAMES = ["embeddings", "class_logits"]


def prepare_data(sample):
    resampled = scipy.signal.resample(sample, 5 * 32000)
    resampled = np.interp(resampled, [np.iinfo(np.int16).min, np.iinfo(np.int16).max], [0,1])
    inputs = resampled.astype(np.float32)[None, ]

    return inputs

def get_reconstruction(data):
    # boolean array where starts[i] is True if data[i] > 10000 and data[i+1] < -30000
    starts = (data[:-1]>10000) & (data[1:]<-30000)

    # convert to index representation
    starts = np.where(starts)[0]

    # for each detected overflow start, figure out where the overflow region ends
    ends = np.zeros_like(starts)
    for i, s in enumerate(starts):
        for j in range(s+1, len(data)):
            if data[j] > -15000:
                break

        ends[i] = j

    # overwrite the overflow values with the large positive clipped value it should've been recorded as
    reconstruction = data.copy()
    for s, e in zip(starts, ends):
        reconstruction[s+1:e] = np.iinfo(np.int16).max

    return reconstruction

def process_waveform(rate, waveform):
    outputs_list = []
    for i in range(int(np.floor(waveform.shape[0]/(5*24000)))):
        sample = waveform[i*5*rate:(i+1)*5*rate]
        inputs = prepare_data(sample)
        outputs = ort_sess.run(None, {'inputs': inputs})
        outputs_list.append(outputs)

    outputs_fused = [np.concatenate(A, axis=0) for A in zip(*outputs_list)]

    return outputs_fused

def process_file(input_filepath):
    rate, waveform = scipy.io.wavfile.read(input_filepath)
    waveform = get_reconstruction(waveform)
    outputs_fused = process_waveform(rate, waveform)
    return outputs_fused


input_filepath = "LISA_20260605_055402.wav"
outputs_fused = process_file(input_filepath)


outputs_dict = dict(zip(OUTPUT_NAMES, outputs_fused))
is_penguin_arr = outputs_dict["class_logits"].argmax(axis=-1) == 4878
is_penguin_arr = np.concatenate([is_penguin_arr, np.zeros(1, dtype=bool)]) # pad one more non-penguin period to the end so that penguins in the last period are detected correctly


datetime_arr = np.arange(len(is_penguin_arr))*5


runs = []
is_run = False
run_start = None
run_end = None
for i, is_penguin in enumerate(is_penguin_arr):
    if is_penguin:
        if is_run:
            run_end += 1
        else:
            run_start = i
            run_end = i+1
            is_run = True
    elif is_penguin_arr[i-1]:
        is_run = False
        runs.append((run_start, run_end))

times = [(int(datetime_arr[i]), int(datetime_arr[j])) for i, j in runs]


with open(Path(input_filepath).stem + ".json", "w") as f:
    json.dump(times, f)





