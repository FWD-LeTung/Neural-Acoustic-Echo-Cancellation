from huggingface_hub import HfApi, login

login()
api = HfApi()

api.upload_large_folder(
    folder_path="D:/Synthetic_Neural_AEC_dataset/Test/NoNoiseArrow",
    repo_id="PandaLT/Neural-AEC-Test-No-Noise",
    repo_type="dataset",
)