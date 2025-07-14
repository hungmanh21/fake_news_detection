import wandb
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def get_wandb_api_key():
    """
    Retrieve the WandB API key from environment variables.
    If not found, prompt the user to input it or issue a warning.

    Returns:
        str: The WandB API key.
    """
    api_key = os.getenv('WANDB_API_KEY')
    if not api_key:
        print("Warning: WANDB_API_KEY not found in environment variables.")
        api_key = input("Please enter your WandB API key: ")
    return api_key


def init_wandb(run_name: str, project: str, config: dict):
    """
    Initialize wandb with the given run name, project, and configuration.

    Args:
        run_name (str): Name of the wandb run.
        project (str): Name of the wandb project.
        config (dict): Configuration dictionary to log.

    Returns:
        wandb.run: The initialized wandb run object.
    """
    try:
        api_key = get_wandb_api_key()

        wandb.login(key=api_key)

        wandb.init(
            name=run_name,
            project=project,
            config=config
        )
        return wandb.run
    except Exception as e:
        print(f"Error initializing WandB: {e}")
        return None
