import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "infofusion_rag"

list_of_files = [
    ".github/workflows/main.yaml",
    f"data/sample_data.txt",
    "research_notebooks/01_EDA.ipynb",
    "research_notebooks/02_Model_Building.ipynb",

    f"src/{project_name}/__init__.py",
    f"src/{project_name}/ingestion/__init__.py",
    f"src/{project_name}/ingestion/pdf_loader.py",
    f"src/{project_name}/retrieval/__init__.py",
    f"src/{project_name}/retrieval/chromadb_connector.py",
    f"src/{project_name}/agents/__init__.py",
    f"src/{project_name}/agents/orchestrator.py",
    f"src/{project_name}/api/__init__.py",
    f"src/{project_name}/api/main.py",
    f"src/{project_name}/ui/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/logger.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",

    "dags/pdf_ingestion_dag.py",
    "tests/__init__.py",

    "main.py",
    "app.py",                     
    "Dockerfile",
    ".gitignore",
    ".dockerignore",
    "setup.py",
    "requirements.txt",
    "README.md",
    ".env",
]

def create_structure():
    for filepath in list_of_files:
        filepath = Path(filepath)
        filedir, filename = os.path.split(filepath)
        if filedir != "":
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Created directory: {filedir} for file: {filename}")

        if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
            with open(filepath, "w") as f:
                if filename.endswith(".py"):
                    f.write(f'"""Module: {filename}"""' + "\n\n")
            logging.info(f"Created empty file: {filepath}")
        else:
            logging.info(f"{filename} already exists")

if __name__ == "__main__":
    create_structure()
