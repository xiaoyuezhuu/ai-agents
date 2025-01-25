#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from xz_demo.crew import XzDemo

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information


IDEA = "A grocery delivery service like InstaCart"
MVP_OUTPUT_DIR = "outputs/mvp"


def run():
    """
    Run the crew.
    """
    inputs = {
        "idea": IDEA,
        "mvp_output_dir": MVP_OUTPUT_DIR,
        # 'current_year': str(datetime.now().year)
    }

    try:
        # run the crew
        # XzDemo().crew().kickoff(inputs=inputs)

        # test the agents performance with LLM-as-a-judge
        # XzDemo().crew().test(inputs=inputs, n_iterations=1, openai_model_name="gpt-4o")

        # train the agents with human feedback
        XzDemo().crew().train(inputs=inputs, n_iterations=1, filename="training.pkl")

    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {"idea": IDEA}
    try:
        XzDemo().crew().train(
            n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        XzDemo().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {"idea": IDEA}
    try:
        XzDemo().crew().test(
            n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
