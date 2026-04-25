import openxlab
from openxlab.dataset import download

# Log in with your AK/SK from OpenDataLab user center
openxlab.login(ak="evmlyombrqj8b1olgb2v", sk="lbjvz0wnqrrpdkemdbdljgq7g4obxgepd9jx52ok")

# Dataset repository and target path
dataset_repo = "OpenDataLab/ScanNet_v2"
target_path = "/mnt/disk4/chenyt/LLaVA-3D/playground/data/scanet_v2"

# List of specific files to download from /raw
files_to_download = [
    "scans.tar.part-00",
    "scans_test.tar",
    "tasks.tar",
    "scannetv2-labels.combined.tsv",
]

print(f"Downloading {len(files_to_download)} files from {dataset_repo}/raw...")

for filename in files_to_download:
    source_path = f"/raw/{filename}"
    print(f"\nDownloading: {source_path}")
    try:
        download(
            dataset_repo=dataset_repo,
            source_path=source_path,
            target_path=target_path,
        )
        print(f"✓ Successfully downloaded: {filename}")
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")

print("\nDownload process completed!")
