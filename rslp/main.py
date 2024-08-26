"""Main entrypoint for rslp."""

import argparse
import importlib
import sys

import dotenv
import jsonargparse


def main():
    """Main entrypoint function for rslp."""
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(description="rslearn")
    parser.add_argument("project", help="The project to execute a workflow for.")
    parser.add_argument("workflow", help="The name of the workflow.")
    args = parser.parse_args(args=sys.argv[1:3])

    module = importlib.import_module(f"rslp.{args.project}")
    workflow_fn = module.workflows[args.workflow][1]
    jsonargparse.CLI(workflow_fn, args=sys.argv[3:])


if __name__ == "__main__":
    main()
