import subprocess
import shutil
import sys

# -----------------------------
# USER CONFIGURATION
# -----------------------------
PROJECT_ID = "ibeat-fatwater"
ZONE = "europe-west2-b"
VM_NAME = "nnunet-gpu-vm"
MACHINE_TYPE = "g2-standard-8"
GPU_TYPE = "nvidia-l4"
GPU_COUNT = 1
BUCKET_NAME = "nnunet_fatwater"
NNUNET_MOUNT = "/mnt/nnunet_data"
NNUNET_RAW = f"{NNUNET_MOUNT}/nnUNet_raw"
NNUNET_PREPROC = f"{NNUNET_MOUNT}/nnUNet_preprocessed"
NNUNET_RESULTS = f"{NNUNET_MOUNT}/nnUNet_results"
NNUNET_N_PROC_DA = 2
FOLDS = [0, 1, 2, 3, 4]
LOG_FILE = f"{NNUNET_RESULTS}/training.log"
# -----------------------------

# 0. Detect gcloud on Windows
gcloud_cmd = r"C:\Users\md1spsx\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"

# 1. Create GPU VM
print("Creating GPU VM...")
subprocess.run([
    gcloud_cmd, "compute", "instances", "create", VM_NAME,
    "--project", PROJECT_ID,
    "--zone", ZONE,
    "--machine-type", MACHINE_TYPE,
    "--accelerator", f"type={GPU_TYPE},count={GPU_COUNT}",
    "--image-family", "ubuntu-2204-lts",
    "--image-project", "ubuntu-os-cloud",
    "--maintenance-policy", "TERMINATE",
    "--boot-disk-size", "200GB",
    "--metadata", "install-nvidia-driver=True",
    "--scopes", "https://www.googleapis.com/auth/cloud-platform",
    "--quiet"
], check=True)

# 2. Prepare SSH command for VM setup and training
ssh_command = f"""
sudo apt-get update && sudo apt-get install -y python3-pip git fuse gcsfuse tmux &&
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 &&
pip install nnunetv2 gcsfs &&
sudo mkdir -p {NNUNET_MOUNT} && sudo chown $USER:$USER {NNUNET_MOUNT} &&
gcsfuse {BUCKET_NAME} {NNUNET_MOUNT} &&
if mountpoint -q {NNUNET_MOUNT}; then echo 'Bucket mounted'; else echo 'Mount failed'; exit 1; fi &&
mkdir -p {NNUNET_RESULTS} &&
export nnUNet_raw_data_base={NNUNET_RAW} &&
export nnUNet_preprocessed={NNUNET_PREPROC} &&
export RESULTS_FOLDER={NNUNET_RESULTS} &&
export nnUNet_n_proc_DA={NNUNET_N_PROC_DA} &&
tmux new-session -d -s nnunet_training \"
"""

# 3. Add training commands for all folds
for fold in FOLDS:
    ssh_command += f"""
echo 'Starting training for fold {fold}';
nnUNetv2_train 011 3d_fullres {fold} --npz | tee -a {LOG_FILE};
"""

# 4. Run best configuration
ssh_command += f"""
echo 'Training complete. Finding best configuration...';
nnUNetv2_find_best_configuration 011 3d_fullres | tee -a {LOG_FILE};
\" 
"""

# 5. SSH into VM and run the setup/training
print("Starting training on VM...")
subprocess.run([
    gcloud_cmd, "compute", "ssh", VM_NAME,
    "--zone", ZONE,
    "--project", PROJECT_ID,
    "--command", ssh_command
], check=True)

# 6. Stream logs in real-time
print("Streaming training logs. Press Ctrl+C to stop (training continues in tmux)...\n")
try:
    subprocess.run([
        gcloud_cmd, "compute", "ssh", VM_NAME,
        "--zone", ZONE,
        "--project", PROJECT_ID,
        "--command", f"tail -f {LOG_FILE}"
    ], check=True)
except KeyboardInterrupt:
    print("\nStopped streaming logs. Training continues in detached tmux session 'nnunet_training'.")
