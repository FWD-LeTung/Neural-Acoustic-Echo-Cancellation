from huggingface_hub import HfApi, login

login()
api = HfApi()

api.upload_large_folder(
    folder_path="D:/Synthetic_Neural_AEC_dataset/Test/FixdB",
    repo_id="PandaLT/Test-Fixed-SER",
    repo_type="dataset",
)