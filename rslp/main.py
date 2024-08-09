"""Entrypoint when using rslp directly."""


def main():
    """Main function when using rslp directly (outside of an automatic job).

    Includes rslp tooling except the code upload/download steps. Credentials must also
    be set either in environmental variables or in .env.
    """
    from dotenv import load_dotenv

    load_dotenv()
    import rslearn.main

    from rslp.lightning_cli import CustomLightningCLI

    rslearn.main.RslearnLightningCLI = CustomLightningCLI
    rslearn.main.main()


if __name__ == "__main__":
    main()
