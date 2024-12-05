"""Main entrypoint for rslp."""

import argparse
import importlib
import logging
import sys
from datetime import datetime

import dotenv
import jsonargparse
import jsonargparse.typing

from rslp.utils.mp import init_mp

logging.basicConfig()


def datetime_serializer(v: datetime) -> str:
    """Serialize datetime for jsonargparse.

    Args:
        v: the datetime object.

    Returns:
        the datetime encoded to string
    """
    return v.isoformat()


def datetime_deserializer(v: str) -> datetime:
    """Deserialize datetime for jsonargparse.

    Args:
        v: the encoded datetime.

    Returns:
        the decoded datetime object
    """
    return datetime.fromisoformat(v)


def run_workflow(project: str, workflow: str, args: list[str]) -> None:
    """Run the specified workflow.

    Args:
        project: the project that the workflow is in. This is the name of the module.
        workflow: the workflow name.
        args: arguments to pass to jsonargparse for running the workflow function.
    """
    module = importlib.import_module(f"rslp.{project}")
    workflow_fn = module.workflows[workflow]
    jsonargparse.CLI(workflow_fn, args=args)


def main() -> None:
    """Main entrypoint function for rslp."""
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(description="rslearn")
    parser.add_argument("project", help="The project to execute a workflow for.")
    parser.add_argument("workflow", help="The name of the workflow.")
    args = parser.parse_args(args=sys.argv[1:3])
    run_workflow(args.project, args.workflow, sys.argv[3:])


if __name__ == "__main__":
    init_mp()

    # Setup jsonargparse.
    jsonargparse.typing.register_type(
        datetime, datetime_serializer, datetime_deserializer
    )

    main()
