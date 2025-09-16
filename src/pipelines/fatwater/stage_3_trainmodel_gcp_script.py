
import os
import subprocess

def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True, executable="/bin/bash")
        print(f"Command executed successfully: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        raise

# This code runs on the remote VM
def main():
    # Install nnUNet on the remote VM
    run_command("pip install git+https://github.com/MIC-DKFZ/nnUNet.git")

    # Define local working directories on the VM's disk
    local_preprocessed_dir = "/mnt/nnunet_preprocessed/"
    local_results_dir = "/mnt/nnunet_results/"

    # Create local directories
    run_command(f"mkdir -p {local_preprocessed_dir}")
    run_command(f"mkdir -p {local_results_dir}")

    # Download preprocessed data from the bucket to the local VM
    run_command(f"gsutil -m cp -r gs://nnunet_fatwater/nnunet_preprocessed/* {local_preprocessed_dir}")

    # Set nnUNet environment variables
    os.environ["nnUNet_preprocessed"] = local_preprocessed_dir
    os.environ["nnUNet_results"] = local_results_dir
    os.environ['nnUNet_n_proc_DA'] = '2'

    # Perform training over 5 folds
    for fold in range(5):
        run_command(f"nnUNetv2_train 011 3d_fullres {fold} --npz")

    # Find the best configuration
    run_command(f"nnUNetv2_find_best_configuration 011 3d_fullres")

    # Upload all training results back to the Cloud Storage bucket
    run_command(f"gsutil -m cp -r {local_results_dir}/* gs://nnunet_fatwater/nnunet_results/")

if __name__ == "__main__":
    main()
