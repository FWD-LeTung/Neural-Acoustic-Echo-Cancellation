import os
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm
import json

def extract_audio_from_hf(dataset_name, output_dir, split='train', max_samples=None):
    """
    Tải dataset (dạng parquet) từ HuggingFace và tách thành file audio riêng lẻ.
    """
    # 1. Tạo thư mục output
    audio_dir = os.path.join(output_dir, "audio_files")
    os.makedirs(audio_dir, exist_ok=True)
    
    print(f"🚀 Đang kết nối tới dataset: {dataset_name}...")
    
    # 2. Load dataset ở chế độ Streaming (Quan trọng với dataset 30GB)
    # Streaming giúp không cần tải hết 30GB Parquet về máy rồi mới xử lý
    try:
        ds = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as e:
        print(f"Lỗi khi load dataset: {e}")
        return

    print("✅ Đã kết nối. Bắt đầu tải và tách file...")
    
    metadata_path = os.path.join(output_dir, "dataset_info.jsonl")
    
    count = 0
    with open(metadata_path, "w", encoding="utf-8") as f_meta:
        # Duyệt qua từng dòng trong dataset
        for sample in tqdm(ds):
            try:
                # 3. Lấy dữ liệu Audio
                # Hugging Face tự động decode cột 'audio' từ Parquet thành dictionary:
                # {'array': numpy.ndarray, 'sampling_rate': int}
                audio_data = sample['audio']['array']
                sample_rate = sample['audio']['sampling_rate']
                
                # 4. Tạo tên file
                # Dựa trên dataset_info.json của bạn, có cột 'id'. Ta dùng nó làm tên file.
                # Nếu không có 'id', ta dùng biến đếm count.
                file_id = sample.get('id', f"audio_{count:06d}")
                
                # Làm sạch file_id để tránh các ký tự lạ gây lỗi đường dẫn
                safe_filename = "".join([c for c in str(file_id) if c.isalnum() or c in ('-','_')])
                filename = f"{safe_filename}.wav"
                file_path = os.path.join(audio_dir, filename)
                
                # 5. Lưu file Audio (.wav)
                sf.write(file_path, audio_data, sample_rate)
                
                # 6. Lưu Metadata (Rất quan trọng để training sau này)
                # Loại bỏ mảng audio nặng nề khỏi metadata trước khi lưu
                meta_record = {k: v for k, v in sample.items() if k != 'audio'}
                meta_record['file_path'] = filename # Link metadata với file audio
                
                # Ghi vào file jsonl
                f_meta.write(json.dumps(meta_record) + "\n")
                
                count += 1
                if max_samples and count >= max_samples:
                    print(f"🛑 Đã đạt giới hạn {max_samples} mẫu.")
                    break
                    
            except Exception as e:
                print(f"\n[WARN] Lỗi xử lý mẫu {count}: {e}")
                continue

    print(f"\n🎉 Hoàn tất! Đã trích xuất {count} file vào thư mục: {audio_dir}")
    print(f"📋 Metadata được lưu tại: {metadata_path}")


DATASET_NAME = "PandaLT/vie_train" 
OUTPUT_FOLDER = "/media/disk_360GB/00_datasets/vie_train/near_end_signal"

if __name__ == "__main__":
    extract_audio_from_hf(DATASET_NAME, OUTPUT_FOLDER, split='train', max_samples=None)