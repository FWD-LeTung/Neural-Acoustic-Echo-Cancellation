import os
import io
import math
import soundfile as sf
import pandas as pd
from datasets.features import Features, Value, Audio
from datasets import DatasetInfo
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

root = "D:/Synthetic_Neural_AEC_dataset/NoisyAndDistortion"
csv_path = os.path.join(root, "dataset_config.csv")

meta = pd.read_csv(csv_path)
meta["mic_path"] = meta["mic_path"].str.replace("\\", "/")
meta["ref_path"] = meta["ref_path"].str.replace("\\", "/")
meta["clean_path"] = meta["clean_path"].str.replace("\\", "/")
NUM_SHARDS = 6
total = len(meta)
shard_size = math.ceil(total / NUM_SHARDS)

output_dir = "D:/Synthetic_Neural_AEC_dataset/NoiseAndDistortion_arrow"
os.makedirs(output_dir, exist_ok=True)

# ===== HF FEATURES =====
features = Features({
    "id": Value("string"),
    "mic": Audio(sampling_rate=16000),
    "ref": Audio(sampling_rate=16000),
    "clean": Audio(sampling_rate=16000),
    "endpoint_bool": Value("bool"),
    "ser_db": Value("float32"),
    "snr_db": Value("float32"),
    "distortion_type": Value("string"),
    "source_near_end": Value("string"),
    "source_echo": Value("string"),
})

row_i = 0

for shard_id in range(NUM_SHARDS):
    shard_path = os.path.join(output_dir, f"data-{shard_id:05d}.arrow")
    print(f"SAVE AT: {shard_path}")

    with ArrowWriter(path=shard_path, features=features) as writer:
        for _ in tqdm(range(shard_size)):
            if row_i >= total:
                break

            row = meta.iloc[row_i]
            row_i += 1

            mic_file = os.path.join(root, row["mic_path"])
            ref_file = os.path.join(root, row["ref_path"])
            clean_file = os.path.join(root, row["clean_path"])
            # ==== Load audio ==== 
            mic_audio, sr_mic = sf.read(mic_file)
            ref_audio, sr_ref = sf.read(ref_file)
            clean_audio, sr_clean = sf.read(clean_file)
            # encode to bytes
            mic_bytes = io.BytesIO()
            sf.write(mic_bytes, mic_audio, sr_mic, format="WAV")
            mic_bytes = mic_bytes.getvalue()

            ref_bytes = io.BytesIO()
            sf.write(ref_bytes, ref_audio, sr_ref, format="WAV")
            ref_bytes = ref_bytes.getvalue()

            clean_bytes = io.BytesIO()
            sf.write(clean_bytes, clean_audio, sr_clean, format="WAV")
            clean_bytes = clean_bytes.getvalue()  

            writer.write({
                "id": row["id"],
                "mic": {"bytes": mic_bytes, "path": row["mic_path"]},
                "ref": {"bytes": ref_bytes, "path": row["ref_path"]},
                "clean": {"bytes": clean_bytes, "path": row["clean_path"]},
                "endpoint_bool": bool(row["endpoint_bool"]),
                "ser_db": float(row["ser_db"]),
                "snr_db": float(row["snr_db"]),
                "distortion_type": row["distortion_type"],
                "source_near_end": row["source_near_end"],
                "source_echo": row["source_echo"],
            })

    print("---Shard saved---")

# dataset_info.json
info = DatasetInfo(features=features)
info.write_to_directory(output_dir)

print("🎉 DONE! Arrow files generated successfully.")
