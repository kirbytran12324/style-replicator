import modal
import sys
import os
import shutil
from pathlib import Path

# Initialize App
app = modal.App("selective-cleanup")

# Connect to the existing backend resources
job_store = modal.Dict.from_name("user-job-store")
model_volume = modal.Volume.from_name("flux-lora-models")

# Constants matching your backend
MOUNT_DIR = "/root/modal_output"


@app.function(volumes={MOUNT_DIR: model_volume})
def delete_remote_paths(paths: list[str]):
    """
    Deletes a list of relative paths (files or folders) inside the volume.
    """
    root = Path(MOUNT_DIR)
    deleted_count = 0

    for p in paths:
        target = root / p
        if target.exists():
            try:
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
                print(f"Deleted: {p}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {p}: {e}")
        else:
            print(f"Not found (already deleted?): {p}")

    if deleted_count > 0:
        print("Committing changes to volume...")
        model_volume.commit()

    return deleted_count


@app.local_entrypoint()
def main():
    print("--- Fetching Job Records ---")

    # 1. Fetch all jobs
    jobs = []
    try:
        # modal.Dict keys() returns an iterator
        for key in job_store.keys():
            data = job_store.get(key)
            if data:
                jobs.append(data)
    except Exception as e:
        print(f"Error connecting to job store: {e}")
        return

    # Sort jobs by creation date (Newest first)
    jobs.sort(key=lambda x: x.get("created_at", "0"), reverse=True)

    if not jobs:
        print("No jobs found in the store.")
        return

    # 2. Display Jobs
    print(f"\n{'IDX':<5} {'STATUS':<10} {'CONFIG NAME':<25} {'JOB ID'}")
    print("-" * 80)
    for i, job in enumerate(jobs):
        status = job.get("status", "???")
        name = job.get("config_name", "???")
        jid = job.get("job_id", "???")
        print(f"{i:<5} {status:<10} {name[:24]:<25} {jid}")

    print("-" * 80)
    print("Enter the index numbers of the jobs to delete (separated by space).")
    print("Example: '0 2 5'. Type 'all' to delete everything. Press Enter to cancel.")

    choice = input("\nSelection: ").strip()

    if not choice:
        print("Cancelled.")
        return

    # 3. Process Selection
    selected_jobs = []
    if choice.lower() == 'all':
        selected_jobs = jobs
    else:
        try:
            indices = [int(x) for x in choice.split()]
            for idx in indices:
                if 0 <= idx < len(jobs):
                    selected_jobs.append(jobs[idx])
                else:
                    print(f"Warning: Index {idx} out of range, skipping.")
        except ValueError:
            print("Invalid input. Please enter numbers.")
            return

    if not selected_jobs:
        print("No jobs selected.")
        return

    print(f"\nYou selected {len(selected_jobs)} jobs.")
    confirm = input("This will permanently delete Metadata AND Files. Proceed? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return

    # 4. Perform Deletion
    paths_to_delete = []

    print("\n--- Cleaning Metadata ---")
    for job in selected_jobs:
        jid = job.get("job_id")
        user_id = job.get("user_id", "default_user")
        config_name = job.get("config_name")

        # Remove from Dict
        print(f"Removing record: {jid}")
        try:
            job_store.pop(jid)
        except Exception:
            print(f"  (Record {jid} was already missing)")

        # Prepare file paths to delete
        if config_name:
            # 1. The Output Folder
            paths_to_delete.append(f"trainings/{user_id}/{config_name}")
            # 2. The Config File
            paths_to_delete.append(f"trainings/{user_id}/_configs/{config_name}.yaml")

    # 5. Clean Remote Files
    if paths_to_delete:
        print(f"\n--- Cleaning {len(paths_to_delete)} Remote Paths ---")
        delete_remote_paths.remote(paths_to_delete)

    print("\nDone! Selective cleanup complete.")