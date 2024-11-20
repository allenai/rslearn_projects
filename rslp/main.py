"""Main entrypoint for rslp."""

import argparse
import importlib

import multiprocessing

import sys

import dotenv
import jsonargparse


from rslp.log_utils import get_logger

logger = get_logger(__name__)
from rslp.utils.mp import init_mp


def main() -> None:
    """Main entrypoint function for rslp."""
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(description="rslearn")
    parser.add_argument("project", help="The project to execute a workflow for.")
    parser.add_argument("workflow", help="The name of the workflow.")
    args = parser.parse_args(args=sys.argv[1:3])

    module = importlib.import_module(f"rslp.{args.project}")
    workflow_fn = module.workflows[args.workflow]
    logger.info(f"running {args.workflow} for {args.project}")
    logger.info(f"args: {sys.argv[3:]}")
    jsonargparse.CLI(workflow_fn, args=sys.argv[3:])


if __name__ == "__main__":
    init_mp()
    main()
