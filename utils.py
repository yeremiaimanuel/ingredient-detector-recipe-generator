import numpy as np

def preprocess(input_list):
    # Jika input_list adalah list angka panjang n, ubah ke bentuk (1, n)
    arr = np.array(input_list, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

def postprocess(pred):
    # Default: kembalikan prediksi as float atau array
    try:
        val = pred.tolist()
        return val
    except:
        return float(pred)