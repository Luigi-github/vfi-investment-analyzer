from app.data.market_data import MarketDataService


def main():

    service = MarketDataService()

    daily_data = service.get_daily_data(
        symbol="ETH-USD",
        period="1y"
    )

    weekly_data = service.convert_to_weekly(
        daily_data
    )

    print("=== DATOS DIARIOS ===")
    print(daily_data.tail())

    print()
    print("Cantidad de registros diarios:")
    print(len(daily_data))

    print()
    print("=== DATOS SEMANALES ===")
    print(weekly_data.tail())

    print()
    print("Cantidad de registros semanales:")
    print(len(weekly_data))


if __name__ == "__main__":
    main()