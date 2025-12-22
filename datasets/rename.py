input_folder = "D:/nearend_speech/audio_files"  # Thư mục chứa file audio gốc
output_folder = "D:/nearend_speech/rename"     # Thư mục lưu file đã đổi tên


import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

def rename_and_update_metadata(audio_folder, output_folder=None):
    """
    Đổi tên file audio và cập nhật lại file metadata JSON/JSONL.
    
    Args:
        audio_folder: Thư mục chứa file audio tên cũ.
        metadata_file: Đường dẫn đến file metadata (thường là .json hoặc .jsonl).
        output_folder: (Tuỳ chọn) Thư mục lưu file mới. Nếu None, sẽ ghi đè tại chỗ.
    """
    
    # 1. Thiết lập thư mục đầu ra
    if output_folder:
        # Tạo folder mới để an toàn, không làm hỏng dữ liệu gốc
        target_audio_dir = os.path.join(output_folder, "audio_files")
        target_meta_path = os.path.join(output_folder, "metadata.jsonl")
        Path(target_audio_dir).mkdir(parents=True, exist_ok=True)
    else:
        # Ghi đè trực tiếp (Cẩn thận!)
        target_audio_dir = audio_folder
        

    # Lấy danh sách file wav trong thư mục input
    audio_files = [f for f in os.listdir(audio_folder) if f.lower().endswith('.wav')]
    print(f"🔍 Tìm thấy {len(audio_files)} file audio.")

    for idx, filename in enumerate(tqdm(audio_files, desc="Renaming")):
        old_path = os.path.join(audio_folder, filename)
        new_filename = f"clean_{idx}.wav"
        new_path = os.path.join(target_audio_dir, new_filename)
        if output_folder:
            shutil.copy2(old_path, new_path)
        else:
            os.rename(old_path, new_path)

    print(f"\n✅ Hoàn tất!")
    print(f"- Đã xử lý thành công: {len(audio_files)} files")
    print(f"- Audio mới lưu tại: {target_audio_dir}")


if __name__ == "__main__":
    rename_and_update_metadata(input_folder,  output_folder)