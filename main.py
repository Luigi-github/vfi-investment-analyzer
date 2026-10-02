from config.settings import APP_NAME, ENVIRONMENT


def main():
    print("=" * 50)
    print(APP_NAME)
    print("=" * 50)

    print(f"Environment: {ENVIRONMENT}")
    print("Project started successfully!")


if __name__ == "__main__":
    main()