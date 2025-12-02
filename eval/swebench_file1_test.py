#!/usr/bin/env python3

"""Run mini-SWE-agent on SWE-bench instances in batch mode."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

import sys
sys.path.append("./")

from file1agent.config import File1AgentConfig
from file1agent.file_manager import FileManager
import subprocess, os

def file1_main(task: str, local_repo_dir: str):
    # Copy repo files
    # os.makedirs(local_repo_dir, exist_ok=True)
    
    # out = subprocess.Popen(["docker", "cp", f"{container_id}:{repo_dir_in_container}", local_repo_dir], 
    #                        stdout=subprocess.PIPE,
    #                        stderr=subprocess.PIPE)
    # stdout, stderr = out.communicate()
    # print(stdout.decode())
    # print(stderr.decode())
    
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(current_dir, "..", "config.toml"))
    
    # Load configuration
    config = File1AgentConfig.from_toml(config_path)
    file_manager = FileManager(
        config=config,
        analyze_dir=local_repo_dir,
        log_level="INFO",
        # log_path=os.path.join("outputs", "file1agent.log"),
        worker_num=8
    )
    
    graph_summary = file_manager.search_workspace(
        max_token_cnt=1000, question=task, use_graph=True
    )

    direct_summary = file_manager.search_workspace(
        max_token_cnt=1000, question=task, use_graph=False
    )
    
    additional_prompt = f"""
    
    
Below are the file tree and file graph summaries for the repository, please understand them before coding:
{direct_summary}
Below are the most related files to the problem and their related files:
{graph_summary}
Avoid viewing full content of files if you only need to understand the overall functionality of the code.
    """
    
    with open(os.path.join(local_dir, "file1agent_prompt.txt"), "w") as f:
        f.write(additional_prompt)
    
    return additional_prompt

if __name__ == "__main__":
    task = "Add a new feature to the code"
    local_repo_dir = "outputs/swebench/astropy__astropy"
    additional_prompt = file1_main(task, local_repo_dir)
    print(additional_prompt)
