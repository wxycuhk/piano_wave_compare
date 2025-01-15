import pandas as pd
from scipy.signal import find_peaks
import os

def find_extrema(series, h, d):
    peaks, _ = find_peaks(series, height = h,distance = d)
    valleys, _ = find_peaks(-series, height = h,distance = d)
    return {
        "peaks": [series.iloc[idx] for idx in peaks],
        "valleys": [series.iloc[idx] for idx in valleys]
    }

def compare_and_store_extrema(file1, file2, output_file):
    data1 = pd.read_csv(file1).iloc[:, 1]
    data2 = pd.read_csv(file2).iloc[:, 1]
    extrema1 = find_extrema(data1, 8000, 10)
    extrema2 = find_extrema(data2, 0, 1)

    valleys = extrema1["valleys"] 
    peaks = extrema2["peaks"]
    print(f"valley number: {len(valleys)}, peak number: {len(peaks)}")
    print(f"valleys: {valleys}")
    print(f"peaks: {peaks}")
    if len(valleys) == len(peaks):
        combined_data = [{"valley_value": valley, "peak_value": peak} for valley, peak in zip(valleys, peaks)]
        df = pd.DataFrame(combined_data)

        if os.path.exists(output_file):
            df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            df.to_csv(output_file, mode='w', header=False, index=False)

        print(f"Data saved to {output_file}")
    else:
        print("Number of valleys and peaks do not match")

file1 = "piano_velocity.csv"
file2 = "RMS_filtered_data.csv"
output_file = "RMS_vel_match.csv"

compare_and_store_extrema(file1, file2, output_file)
