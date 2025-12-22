import os
import io
import math
import soundfile as sf
import numpy as np
import pandas as pd
from datasets.features import Features, Value, Audio
from datasets import DatasetInfo
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

# --- HÀM QUAN TRỌNG: Sửa lỗi Tuple index out of range ---
def ensure_2d(audio):
    """Chuyển đổi mảng 1D (Mono) thành 2D (N, 1) để tránh lỗi soundfile."""
    if len(audio.shape) == 1:
        return audio[:, np.newaxis]
    return audio

root = "D:/Synthetic_Neural_AEC_dataset/TestSet_FixedSER"
csv_path = os.path.join(root, "dataset_test_config.csv")

# 1. Đọc và Kiểm tra số lượng dòng trong CSV
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Không tìm thấy file CSV tại: {csv_path}")

meta = pd.read_csv(csv_path)
print(f"📊 TỔNG SỐ DÒNG TRONG CSV: {len(meta)}") # Kiểm tra xem CSV có đủ 10004 dòng không

# Chuẩn hóa đường dẫn
path_cols = ["ref_path", "clean_path", "mic_path_ser_-10dB", 
             "mic_path_ser_-5dB", "mic_path_ser_0dB", 
             "mic_path_ser_5dB", "mic_path_ser_10dB"]

for col in path_cols:
    if col in meta.columns:
        meta[col] = meta[col].str.replace("\\", "/")

NUM_SHARDS = 1
total = len(meta)
shard_size = math.ceil(total / NUM_SHARDS)

output_dir = "D:/Synthetic_Neural_AEC_dataset/Test/FixdB2"
os.makedirs(output_dir, exist_ok=True)

# ===== HF FEATURES =====
features = Features({
    "id": Value("string"),
    "clean": Audio(sampling_rate=16000),
    "ref": Audio(sampling_rate=16000),
    "mic_neg_10dB": Audio(sampling_rate=16000),
    "mic_neg_5dB": Audio(sampling_rate=16000),
    "mic_0dB": Audio(sampling_rate=16000),
    "mic_5dB": Audio(sampling_rate=16000),
    "mic_10dB": Audio(sampling_rate=16000),
    "distortion_type": Value("string"),
    "endpoint_bool": Value("bool"),
    "source_near_end": Value("string"),
    "source_echo": Value("string"),
})

row_i = 0

for shard_id in range(NUM_SHARDS):
    shard_path = os.path.join(output_dir, f"data-{shard_id:05d}.arrow")
    print(f"🚀 SAVE AT: {shard_path}")

    with ArrowWriter(path=shard_path, features=features) as writer:
        for _ in tqdm(range(shard_size)):
            if row_i >= total:
                break

            try:
                row = meta.iloc[row_i]
                row_i += 1

                # Đường dẫn file
                clean_file = os.path.join(root, row["clean_path"])
                ref_file = os.path.join(root, row["ref_path"])
                mic_neg_10dB_file = os.path.join(root, row["mic_path_ser_-10dB"])
                mic_neg_5dB_file = os.path.join(root, row["mic_path_ser_-5dB"])
                mic_0dB_file = os.path.join(root, row["mic_path_ser_0dB"])
                mic_5dB_file = os.path.join(root, row["mic_path_ser_5dB"])
                mic_10dB_file = os.path.join(root, row["mic_path_ser_10dB"])

                # ==== Load audio ==== 
                # (Thêm logic kiểm tra file tồn tại nếu cần thiết)
                mic_neg_10dB_audio, sr_mic_neg_10dB_audio = sf.read(mic_neg_10dB_file)
                mic_neg_5dB_audio, sr_mic_neg_5dB_audio = sf.read(mic_neg_5dB_file)
                mic_0dB_audio, sr_mic_0dB_audio = sf.read(mic_0dB_file)
                mic_5dB_audio, sr_mic_5dB_audio = sf.read(mic_5dB_file)
                mic_10dB_audio, sr_mic_10dB_audio = sf.read(mic_10dB_file)
                ref_audio, sr_ref = sf.read(ref_file)
                clean_audio, sr_clean = sf.read(clean_file)

                # ==== Encode to bytes (SỬA LỖI: Bọc bằng ensure_2d) ====
                
                mic_neg_10dB_bytes_io = io.BytesIO()
                sf.write(mic_neg_10dB_bytes_io, ensure_2d(mic_neg_10dB_audio), sr_mic_neg_10dB_audio, format="WAV")
                mic_neg_10dB_val = mic_neg_10dB_bytes_io.getvalue()

                mic_neg_5dB_bytes_io = io.BytesIO()
                sf.write(mic_neg_5dB_bytes_io, ensure_2d(mic_neg_5dB_audio), sr_mic_neg_5dB_audio, format="WAV")
                mic_neg_5dB_val = mic_neg_5dB_bytes_io.getvalue()

                mic_0dB_bytes_io = io.BytesIO()
                sf.write(mic_0dB_bytes_io, ensure_2d(mic_0dB_audio), sr_mic_0dB_audio, format="WAV")
                mic_0dB_val = mic_0dB_bytes_io.getvalue()

                mic_5dB_bytes_io = io.BytesIO()
                sf.write(mic_5dB_bytes_io, ensure_2d(mic_5dB_audio), sr_mic_5dB_audio, format="WAV")
                mic_5dB_val = mic_5dB_bytes_io.getvalue()

                mic_10dB_bytes_io = io.BytesIO()
                sf.write(mic_10dB_bytes_io, ensure_2d(mic_10dB_audio), sr_mic_10dB_audio, format="WAV")
                mic_10dB_val = mic_10dB_bytes_io.getvalue()

                ref_bytes_io = io.BytesIO()
                sf.write(ref_bytes_io, ensure_2d(ref_audio), sr_ref, format="WAV")
                ref_val = ref_bytes_io.getvalue()

                clean_bytes_io = io.BytesIO()
                sf.write(clean_bytes_io, ensure_2d(clean_audio), sr_clean, format="WAV")
                clean_val = clean_bytes_io.getvalue()  

                # Xử lý endpoint_bool an toàn
                endpoint_val = bool(row["endpoint_bool"]) if "endpoint_bool" in row else False

                writer.write({
                    "id": str(row["id"]),
                    "clean": {"bytes": clean_val, "path": row["clean_path"]},
                    "ref": {"bytes": ref_val, "path": row["ref_path"]},
                    "mic_neg_10dB": {"bytes": mic_neg_10dB_val, "path": row["mic_path_ser_-10dB"]},
                    "mic_neg_5dB": {"bytes": mic_neg_5dB_val, "path": row["mic_path_ser_-5dB"]},
                    "mic_0dB": {"bytes": mic_0dB_val, "path": row["mic_path_ser_0dB"]},
                    "mic_5dB": {"bytes": mic_5dB_val, "path": row["mic_path_ser_5dB"]},
                    "mic_10dB": {"bytes": mic_10dB_val, "path": row["mic_path_ser_10dB"]},
                    "distortion_type": str(row.get("distortion_type", "unknown")),
                    "endpoint_bool": endpoint_val,
                    "source_near_end": str(row["source_near_end"]),
                    "source_echo": str(row["source_echo"]),
                })

            except Exception as e:
                print(f"❌ ERROR tại dòng {row_i} (ID: {row.get('id', 'unknown')}): {e}")
                # Không dừng chương trình, tiếp tục qua file tiếp theo

    print("---Shard saved---")

# dataset_info.json
info = DatasetInfo(features=features)
info.write_to_directory(output_dir)

print("🎉 DONE! Arrow files generated successfully.")