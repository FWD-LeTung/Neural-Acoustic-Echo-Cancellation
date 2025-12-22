import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from huggingface_hub import login
from datasets import Dataset, Audio, Features, Value, Sequence

# 1. Đăng nhập vào Hugging Face
login()

def create_manifest_with_vad(mic_dir, ref_dir, clean_dir, vad_dir):
    # Lấy danh sách file và sắp xếp để đảm bảo thứ tự khớp nhau
    mic_files = sorted(glob.glob(os.path.join(mic_dir, "*.wav")))
    ref_files = sorted(glob.glob(os.path.join(ref_dir, "*.wav")))
    clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
    vad_files = sorted(glob.glob(os.path.join(vad_dir, "*.npy")))

    # Kiểm tra số lượng file có khớp nhau không
    if not (len(mic_files) == len(ref_files) == len(clean_files) == len(vad_files)):
        print(f"CẢNH BÁO: Số lượng file không khớp!")
        print(f"Mic: {len(mic_files)}, Ref: {len(ref_files)}, Clean: {len(clean_files)}, VAD: {len(vad_files)}")

    # Đọc nội dung file VAD (.npy) để đưa vào DataFrame
    print("Đang đọc nhãn VAD từ các file .npy...")
    vad_labels_data = []
    for f in tqdm(vad_files):
        label = np.load(f).astype(np.int8).tolist() # Chuyển sang list để HF Dataset xử lý
        vad_labels_data.append(label)

    data = {
        "mic_path": mic_files,
        "ref_path": ref_files,
        "clean_path": clean_files,
        "vad_label": vad_labels_data # Chứa trực tiếp mảng 0 và 1
    }
    return pd.DataFrame(data)

# --- THỰC THI ---

# 2. Tạo DataFrame
df = create_manifest_with_vad(
    mic_dir="D:/AEC-Challenge/datasets/synthetic/nearend_mic_signal",
    ref_dir="D:/AEC-Challenge/datasets/synthetic/farend_speech",
    clean_dir="D:/AEC-Challenge/datasets/synthetic/nearend_speech",
    vad_dir="D:/AEC-Challenge/datasets/synthetic/vad" # Thư mục vad bạn vừa tạo
)

# 3. Tạo Dataset từ Pandas
dataset = Dataset.from_pandas(df)

# 4. Định nghĩa cấu trúc (Features)
# Sử dụng Sequence(Value("int8")) cho nhãn VAD để tối ưu dung lượng
features = dataset.features.copy()
dataset = dataset.cast_column("mic_path", Audio(sampling_rate=16000))
dataset = dataset.cast_column("ref_path", Audio(sampling_rate=16000))
dataset = dataset.cast_column("clean_path", Audio(sampling_rate=16000))

# 5. Đổi tên cột cho chuyên nghiệp
dataset = dataset.rename_column("mic_path", "mic")
dataset = dataset.rename_column("ref_path", "ref")
dataset = dataset.rename_column("clean_path", "clean")
# Giữ nguyên tên vad_label

# 6. Đẩy lên Hugging Face
repo_id = "PandaLT/microsoft-AEC-vad-dataset" 
print(f"Đang đẩy dataset lên: {repo_id}...")
dataset.push_to_hub(repo_id, private=False)

print("Hoàn thành! Giờ bạn có thể load dataset kèm nhãn VAD chỉ bằng 1 dòng lệnh load_dataset().")