"""Entrypoint when using rslp directly."""

import logging

from rslp.utils.mp import init_mp

logging.basicConfig()


def main() -> None:
    """Main function when using rslp directly (outside of an automatic job).

    Includes rslp tooling except the code upload/download steps. Credentials must also
    be set either in environmental variables or in .env.
    """
    from dotenv import load_dotenv

    load_dotenv()
    import rslearn.main

    from rslp.lightning_cli import CustomLightningCLI, custom_model_handler

    rslearn.main.RslearnLightningCLI = CustomLightningCLI
    rslearn.main.model_handler = custom_model_handler
    rslearn.main.main()


if __name__ == "__main__":
    init_mp()
    main()
