import os

from dotenv import load_dotenv


load_dotenv()


APP_NAME = os.getenv(
    "APP_NAME",
    "VFI Investment Analyzer"
)

ENVIRONMENT = os.getenv(
    "ENVIRONMENT",
    "development"
)