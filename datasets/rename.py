import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

def rename_and_update_metadata(audio_folder, metadata_file, output_folder=None):
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
        target_meta_path = metadata_file

    print(f"📂 Đang quét metadata từ: {metadata_file}")
    
    # 2. Đọc Metadata cũ
    # Xử lý cả trường hợp file là JSON list hoặc JSONL (Line-delimited JSON)
    original_records = []
    is_jsonl = str(metadata_file).endswith('.jsonl')
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        if is_jsonl:
            for line in f:
                if line.strip():
                    original_records.append(json.loads(line))
        else:
            # Giả sử là file JSON chuẩn chứa 1 list
            try:
                data = json.load(f)
                if isinstance(data, list):
                    original_records = data
                elif isinstance(data, dict) and 'samples' in data: # Trường hợp cấu trúc khác
                     original_records = data['samples']
                else:
                    print("⚠️ Cấu trúc JSON không phải là List. Vui lòng kiểm tra lại.")
                    return
            except json.JSONDecodeError:
                print("❌ File JSON bị lỗi format.")
                return

    print(f"🔍 Tìm thấy {len(original_records)} bản ghi.")
    
    # 3. Thực hiện Đổi tên và Cập nhật
    new_records = []
    missing_count = 0
    
    # Sắp xếp lại records nếu cần thứ tự nhất quán (tuỳ chọn)
    # original_records.sort(key=lambda x: x.get('id', '')) 

    for idx, record in enumerate(tqdm(original_records, desc="Renaming")):
        # Lấy tên file cũ từ metadata
        # Giả sử trường chứa tên file là 'file_path' hoặc 'file_name' hoặc 'id'
        # Dựa trên dataset của bạn, có thể là 'file_path' tạo từ bước download trước
        old_filename = record.get('file_path') 
        
        # Nếu trong metadata không có đuôi .wav, hãy tự thêm vào
        if not old_filename.endswith('.wav'):
             old_filename += '.wav'
             
        old_path = os.path.join(audio_folder, old_filename)
        
        # Kiểm tra file có tồn tại thực tế không
        if not os.path.exists(old_path):
            # Thử tìm bằng ID nếu file_path sai
            possible_id_name = os.path.join(audio_folder, f"{record.get('id')}.wav")
            if os.path.exists(possible_id_name):
                old_path = possible_id_name
            else:
                missing_count += 1
                continue # Bỏ qua nếu không tìm thấy file audio

        # Tạo tên mới: audio_00000.wav
        new_filename = f"audio_{idx:05d}.wav"
        new_path = os.path.join(target_audio_dir, new_filename)
        
        # Copy (hoặc Move) file sang tên mới
        if output_folder:
            shutil.copy2(old_path, new_path)
        else:
            os.rename(old_path, new_path)
            
        # Cập nhật thông tin trong record
        record['file_path'] = new_filename
        record['original_filename'] = old_filename # Lưu lại tên cũ để trace nếu cần
        record['id'] = f"audio_{idx:05d}" # Cập nhật luôn ID cho đồng bộ
        
        new_records.append(record)

    # 4. Ghi file Metadata mới
    with open(target_meta_path, 'w', encoding='utf-8') as f_out:
        if is_jsonl:
            for rec in new_records:
                f_out.write(json.dumps(rec) + '\n')
        else:
            json.dump(new_records, f_out, indent=2)

    print(f"\n✅ Hoàn tất!")
    print(f"- Đã xử lý thành công: {len(new_records)}/{len(original_records)} files")
    if missing_count > 0:
        print(f"- ⚠️ Không tìm thấy: {missing_count} file audio (Đã bỏ qua trong metadata mới)")
    print(f"- Metadata mới lưu tại: {target_meta_path}")
    print(f"- Audio mới lưu tại: {target_audio_dir}")

# --- CẤU HÌNH ---
# Folder chứa file audio tên cũ (ví dụ: 5dec87f7....wav)
OLD_AUDIO_DIR = "D:/near_end_signal/test_audio_files"

# File metadata hiện tại (được tạo ra từ bước download trước)
METADATA_FILE = "D:/near_end_signal/dataset_info.jsonl"
# Nơi lưu dataset mới (Nên tạo folder mới để an toàn)
OUTPUT_DIR = "D:/near_end_signal/test_audio_files_v2"

if __name__ == "__main__":
    rename_and_update_metadata(OLD_AUDIO_DIR, METADATA_FILE, OUTPUT_DIR)