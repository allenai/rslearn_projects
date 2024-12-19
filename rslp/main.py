"""Main entrypoint for rslp."""

import argparse
import importlib
import sys
from pathlib import Path

import dotenv
import jsonargparse
from jsonargparse import ActionConfigFile

from rslp.log_utils import get_logger
from rslp.utils.mp import init_mp

logger = get_logger(__name__)


class RelativePathActionConfigFile(ActionConfigFile):
    """Custom action to handle relative paths in config files."""

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


def main() -> None:
    """Main entrypoint function for rslp."""
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(description="rslearn")
    parser.register("action", "config_file", RelativePathActionConfigFile)
    parser.add_argument("project", help="The project to execute a workflow for.")
    parser.add_argument("workflow", help="The name of the workflow.")
    args = parser.parse_args(args=sys.argv[1:3])

    module = importlib.import_module(f"rslp.{args.project}")
    workflow_fn = module.workflows[args.workflow]
    logger.info(f"running {args.workflow} for {args.project}")
    logger.info(f"args: {sys.argv[3:]}")

    # Enable relative path support for config files
    jsonargparse.set_config_read_mode("default")
    jsonargparse.CLI(workflow_fn, args=sys.argv[3:])


if __name__ == "__main__":
    init_mp()
    main()
