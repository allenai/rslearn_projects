"""Main entrypoint for rslp."""

import argparse
import importlib
import sys
from datetime import datetime
from pathlib import Path

import dotenv
import jsonargparse
import jsonargparse.typing
from jsonargparse import ActionConfigFile

from rslp.log_utils import get_logger
from rslp.utils.mp import init_mp

logger = get_logger(__name__)


class RelativePathActionConfigFile(ActionConfigFile):
    """Custom action to handle relative paths to config files."""

    def __call__(
        self,
        parser: jsonargparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str,
        option_string: str | None = None,
    ) -> None:
        """Convert relative paths to absolute before loading config."""
        if not str(values).startswith(("/", "gs://")):
            repo_root = (
                Path(__file__).resolve().parents[1]
            )  # Go up to rslearn_projects root
            values = str(repo_root / values)
        super().__call__(parser, namespace, values, option_string)


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
    logger.info(f"running {workflow} for {project}")
    logger.info(f"args: {args}")

    # Enable relative path support for config files
    jsonargparse.set_config_read_mode("default")
    jsonargparse.CLI(workflow_fn, args=args)


def main() -> None:
    """Main entrypoint function for rslp."""
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(description="rslearn")
    parser.register("action", "config_file", RelativePathActionConfigFile)
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
