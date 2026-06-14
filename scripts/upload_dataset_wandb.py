import wandb
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Upload QM9 raw files to WandB Artifacts")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Path to the directory containing the 3 raw files (dsgdb9nsd.xyz.tar.bz2, uncharacterized.txt, atomref.txt)")
    args = parser.parse_args()

    files_to_upload = [
        "dsgdb9nsd.xyz.tar.bz2",
        "uncharacterized.txt",
        "atomref.txt"
    ]

    for f in files_to_upload:
        if not os.path.exists(os.path.join(args.dataset_dir, f)):
            raise FileNotFoundError(f"Missing {f} in {args.dataset_dir}. Please download all 3 files from FigShare first.")

    print("Initializing WandB...")
    run = wandb.init(project="e3_diffusion", job_type="dataset_upload")

    print("Creating Dataset Artifact...")
    artifact = wandb.Artifact("qm9_raw", type="dataset", description="Raw QM9 dataset files from FigShare")

    for f in files_to_upload:
        file_path = os.path.join(args.dataset_dir, f)
        print(f"Adding {f} to artifact...")
        artifact.add_file(file_path)

    print("Uploading to WandB (this may take a few minutes for the 260MB file)...")
    run.log_artifact(artifact)
    run.finish()
    print("Upload complete! You can now use the train_cfg_colab.ipynb notebook to automatically download it to Colab.")

if __name__ == "__main__":
    main()
