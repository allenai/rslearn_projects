"""Main entrypoint for rslp."""

import argparse
import importlib
import logging
import multiprocessing
import sys
from datetime import datetime

import dotenv
import jsonargparse
import jsonargparse.typing

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


def main() -> None:
    """Main entrypoint function for rslp."""
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(description="rslearn")
    parser.add_argument("project", help="The project to execute a workflow for.")
    parser.add_argument("workflow", help="The name of the workflow.")
    args = parser.parse_args(args=sys.argv[1:3])

    module = importlib.import_module(f"rslp.{args.project}")
    workflow_fn = module.workflows[args.workflow]

    # Setup jsonargparse.
    jsonargparse.typing.register_type(
        datetime, datetime_serializer, datetime_deserializer
    )

    # Parse arguments and run function.
    jsonargparse.CLI(workflow_fn, args=sys.argv[3:])


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    main()
