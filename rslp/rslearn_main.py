"""Entrypoint when using rslp directly."""

import multiprocessing


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
    multiprocessing.set_start_method("forkserver")
    multiprocessing.set_forkserver_preload(
        [
            "pickle",
            "fiona",
            "gcsfs",
            "jsonargparse",
            "numpy",
            "PIL",
            "torch",
            "torch.multiprocessing",
            "torchvision",
            "upath",
            "wandb",
            "rslearn.main",
            "rslearn.train.dataset",
            "rslearn.train.data_module",
        ]
    )
    main()
