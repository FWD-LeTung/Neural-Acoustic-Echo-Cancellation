from huggingface_hub import HfApi, login

login()
api = HfApi()

api.upload_large_folder(
    folder_path="D:/Synthetic_Neural_AEC_dataset/Test/NoisyArrow",
    repo_id="PandaLT/Neural-AEC-Test-Noisy",
    repo_type="dataset",
)