from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml import command
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment, BuildContext
import os
from azure.core.exceptions import ResourceNotFoundError
from typing import Optional

def create_instruction_dataset_job(
    workspace_name: str,
    subscription_id: str,
    resource_group: str,
    experiment_name: str = "tunisian-insurance-qa-generation",
    compute_name: str = "maxcalculatorcpu",
    input_data_path: str = "data/trainingtext__.jsonl",
    vm_size: str = "Standard_D4_v3"  # Added VM size parameter with a default
) -> Optional[any]:
    try:
        # Connect to Azure ML workspace
        ml_client = MLClient(
            DefaultAzureCredential(),
            subscription_id,
            resource_group,
            workspace_name
        )
        
        # Verify compute cluster exists and check its configuration
        try:
            compute_target = ml_client.compute.get(compute_name)
            print(f"Found compute target: {compute_name}")
            print(f"Current VM size: {compute_target.size}")
            
            # If you need to create or update the compute cluster with a larger size:
            """
            from azure.ai.ml.entities import AmlCompute
            
            compute_config = AmlCompute(
                name=compute_name,
                size=vm_size,
                min_instances=0,
                max_instances=1,
                idle_time_before_scale_down=120
            )
            ml_client.compute.begin_create_or_update(compute_config).result()
            """
            
        except ResourceNotFoundError:
            print(f"ERROR: Compute cluster '{compute_name}' not found!")
            print("Available compute targets:")
            for compute in ml_client.compute.list():
                print(f"- {compute.name}")
            return None

        # Define environment with conda dependencies and minimal base image
        custom_env = Environment(
            name="qa-generation-env",
            description="Environment for Tunisian insurance QA generation",
            image="mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference:latest",  # Changed to minimal image
            conda_file={
                "name": "qa-generation",
                "channels": ["conda-forge", "defaults"],
                "dependencies": [
                    "python=3.8",
                    "pip=21.2.4",
                    {
                        "pip": [
                            "transformers>=4.37.0",
                            "torch>=2.0.0",
                            "nltk>=3.8.1",
                            "bitsandbytes>=0.41.0",
                            "accelerate>=0.21.0",
                            "azure-ai-ml"
                        ]
                    }
                ]
            }
        )

        # Define the command job
        job = command(
            code="./src",
            command="python instruction_dataset_generator.py",
            environment=custom_env,
            compute=compute_name,
            display_name=experiment_name,
            experiment_name=experiment_name,
            inputs={
                "input_data": Input(type="uri_file", path=input_data_path)
            },
            outputs={
                "instruction_dataset": Output(
                    type="uri_file",
                    path="azureml://datastores/workspaceblobstore/paths/outputs/instruction_qwen.jsonl",
                    mode="upload"
                )
            },
            instance_count=1
        )

        # Submit the job
        returned_job = ml_client.jobs.create_or_update(job)
        print(f"Successfully submitted job: {returned_job.name}")
        print(f"Job status: {returned_job.status}")
        print(f"Job tracking URI: {returned_job.studio_url}")
        
        return returned_job

    except Exception as e:
        print(f"Error creating/submitting job: {str(e)}")
        print("\nDetailed error information:")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Get workspace details from environment variables or configure directly
    workspace_name = os.getenv("WORKSPACE_NAME", "mltalel")
    subscription_id = os.getenv("SUBSCRIPTION_ID", "877b760f-539d-43f5-8127-f2ab9891a58a")
    resource_group = os.getenv("RESOURCE_GROUP", "ai")
    compute_name = os.getenv("COMPUTE_NAME", "llama-compute")
    
    # Suggested VM sizes for different workloads
    VM_SIZES = {
        'small': 'Standard_DS3_v2',    # 4 cores, 14 GB RAM
        'medium': 'Standard_D4_v3',    # 4 cores, 16 GB RAM
        'large': 'Standard_D8_v3',     # 8 cores, 32 GB RAM
        'xlarge': 'Standard_D16_v3'    # 16 cores, 64 GB RAM
    }
    
    # Choose a larger VM size if needed
    selected_vm_size = VM_SIZES['medium']  # Adjust as needed
    
    print("Submitting Azure ML job...")
    job = create_instruction_dataset_job(
        workspace_name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        compute_name=compute_name,
        vm_size=selected_vm_size
    )
    
    if job is None:
        print("\nJob submission failed! Please check the error messages above.")
    else:
        print("\nJob submitted successfully!")
        print(f"Track your job at: {job.studio_url}")
