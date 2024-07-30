if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    import rslearn.main
    from rslp.lightning_cli import CustomLightningCLI
    rslearn.main.RslearnLightningCLI = CustomLightningCLI
    rslearn.main.main()
