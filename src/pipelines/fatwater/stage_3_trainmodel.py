import os
import subprocess


path = os.path.join(os.getcwd(), 'build', 'fatwater', 'stage_2_trainingdata')
trainingdatapath = os.path.join(path, 'nnUNet_raw')
path = os.path.join(os.getcwd(), 'build', 'fatwater', 'stage_3_trainmodel')
preprocpath = os.path.join(path, 'nnUNet_preprocessed')
resultspath = os.path.join(path, 'nnUNet_results')


def train(cont=False):

    # Ensure folders exist
    os.makedirs(preprocpath, exist_ok=True)
    os.makedirs(resultspath, exist_ok=True) 

    # Define environment variables
    # https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md
    os.environ["nnUNet_raw"] = trainingdatapath
    os.environ["nnUNet_preprocessed"] = preprocpath
    os.environ["nnUNet_results"] = resultspath
    os.environ['nnUNet_n_proc_DA'] = '2' # depends on hardware

    # https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md
    for FOLD in [0, 1, 2, 3, 4]:
        cmd = [
            "nnUNetv2_train",
            "011",
            "3d_fullres",
            str(FOLD),
            "--npz",
        ]
        if cont:
            cmd.append("--c")

        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            encoding="utf-8",   # <-- force UTF-8 decoding
            errors="replace"    # <-- avoids crash if weird bytes appear
        )

        # Stream logs in real-time
        for line in process.stdout:
            print(line, end="")

        process.wait()  # wait for completion

train()



def preprocess():

    # Ensure folders exist
    os.makedirs(preprocpath, exist_ok=True)
    os.makedirs(resultspath, exist_ok=True) 

    # Define environment variables
    # https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md
    os.environ["nnUNet_raw"] = trainingdatapath
    os.environ["nnUNet_preprocessed"] = preprocpath
    os.environ["nnUNet_results"] = resultspath
    os.environ['nnUNet_n_proc_DA'] = '10' # depends on hardware

    # https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md

    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d",
        "011",
        "-c", 
        "3d_fullres",
        "--verify_dataset_integrity",
    ]

    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        encoding="utf-8",   # <-- force UTF-8 decoding
        errors="replace"    # <-- avoids crash if weird bytes appear
    )

    # Stream logs in real-time
    for line in process.stdout:
        print(line, end="")

    process.wait()  # wait for completion




def find_config():

    # Ensure folders exist
    os.makedirs(preprocpath, exist_ok=True)
    os.makedirs(resultspath, exist_ok=True)

    # Find best configuration    
    cmd = [
        "nnUNetv2_find_best_configuration",
        "011",
        "3d_fullres"
    ]

    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        encoding="utf-8",   # <-- force UTF-8 decoding
        errors="replace"    # <-- avoids crash if weird bytes appear
    )

    # Stream logs in real-time
    for line in process.stdout:
        print(line, end="")

    process.wait()  # wait for completion
