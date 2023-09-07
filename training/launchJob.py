import json
from azure.ai.ml import MLClient, command, Input, Output, PyTorchDistribution
from azure.identity import DefaultAzureCredential


datasets = {
    "laion-coco": "azureml:laion_coco:2",
    "laion-coco-images": "azureml:laion_coco_images:1",
    "unzip-laion": "azureml:laion-coco-unzip-2:1",
}

computes = {
    "A100MultiNode": {
        "name": "A100Cluster",
        "num_machine": 2,
        "num_process": 8,
    },
    "A100MultiNodeNorth": {
        "name": "A100ClusterNorth",
        "num_machine": 2,
        "num_process": 8,
    },
    "A100SingleNode": {
        "name": "A100Cluster",
        "num_machine": 1,
        "num_process": 4,
    },
    "A100SingleGPU": {
        "name": "A100Single",
        "num_machine": 1,
        "num_process": 1,
    },
    "CPU": {
        "name": "CPUCompute",
        "num_machine": 1,
        "num_process": 1,
    },
}


# Preset Dataset =======================================
# compute_target = "HighMemoryCPU"
# environment = "datasetEnv"

# exp_name = "clip"
# jobName = "datasetGeneration"

# dataset = datasets["laion-coco"]

# command_to_run = "./generateDataset.sh ${{inputs.data_path}} ${{outputs.output}}"


# Preset CLIP test =======================================
compute_target = "A100SingleGPU"
compute_target = "CPU"

environment = "clipTraining"

exp_name = "clip"
jobName = "clip_datasetGeneration"

dataset = datasets["laion-coco-images"]

command_to_run = (
    f"accelerate launch --mixed_precision fp16 --num_machines {computes[compute_target]['num_machine']} --num_processes {computes[compute_target]['num_process']}"
    + (
        " --machine_rank $NODE_RANK --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT"
        if computes[compute_target]["num_machine"] > 1
        else ""
    )
    + " training.py --data-path ${{inputs.data_path}} --image-path ${{inputs.image_path}} --epochs 500"
)

# Preset CLIP full training =======================================
compute_target = "A100MultiNodeNorth"

environment = "clipTraining"

exp_name = "clip"
jobName = "clip_transformer"

dataset = datasets["laion-coco-images"]

command_to_run = (
    f"accelerate launch --mixed_precision fp16 --num_machines {computes[compute_target]['num_machine']} --num_processes {computes[compute_target]['num_process']}"
    + (
        " --machine_rank $NODE_RANK --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT"
        if computes[compute_target]["num_machine"] > 1
        else ""
    )
    + " training.py --data-path ${{inputs.data_path}}  --image-path ${{inputs.image_path}}"
)

# =========================================================================================== #

with open("azureCredentials.json", "r") as f:
    azureCredentials = json.load(f)

ml_client = MLClient(
    DefaultAzureCredential(
        exclude_environment_credential=True,
        exclude_managed_identity_credential=True,
        exclude_powershell_credential=True,
        exclude_shared_token_cache_credential=True,
        exclude_visual_studio_code_credential=True,
    ),
    azureCredentials["subscription_id"],
    azureCredentials["resource_group_name"],
    azureCredentials["workspace_name"],
)

try:
    ml_client.compute.get(computes[compute_target]["name"])
except Exception:
    raise ValueError("Impossible to get compute target.")

# define the command
command_job = command(
    code="./training",
    command=command_to_run,
    environment=f"{environment}@latest",
    inputs={
        "data_path": Input(
            type="uri_folder",
            path=dataset,
        ),
        "image_path": Input(type="uri_folder", path=datasets["unzip-laion"]),
    },
    compute=computes[compute_target]["name"],
    experiment_name=exp_name,
    docker_args="--shm-size=500g",
    display_name=jobName,
    instance_count=computes[compute_target]["num_machine"],
    distribution=PyTorchDistribution(
        process_count_per_node=computes[compute_target]["num_process"] / computes[compute_target]["num_machine"]
    )
    if computes[compute_target]["num_machine"] > 1
    else None,
)

# submit the command
returned_job = ml_client.jobs.create_or_update(command_job)
# get a URL for the status of the job
print(returned_job.services["Studio"].endpoint)
