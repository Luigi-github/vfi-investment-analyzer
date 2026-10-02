from app.data.market_data import MarketDataService
from app.indicators.vfi import VFIIndicator


def main():

    symbol = "ETH-USD"

    # ==========================================================
    # 1. MARKET DATA
    # ==========================================================

    market_data = MarketDataService()

    daily_data = market_data.get_daily_data(
        symbol=symbol,
        period="5y"
    )

    weekly_data = market_data.convert_to_weekly(
        daily_data
    )

    # ==========================================================
    # 2. VFI
    # ==========================================================

    indicator = VFIIndicator()

    data = indicator.calculate(
        weekly_data
    )

    # ==========================================================
    # 3. DIAGNOSTIC
    # ==========================================================

    print()
    print("=" * 100)
    print("DIAGNÓSTICO DE DATOS VFI")
    print("=" * 100)

    print()

    print(
        f"Total semanas: {len(data)}"
    )

    print(
        f"VFI válidos: "
        f"{data['VFI'].notna().sum()}"
    )

    print(
        f"VFI NaN: "
        f"{data['VFI'].isna().sum()}"
    )

    print(
        f"Slope válidos: "
        f"{data['VFI_Slope'].notna().sum()}"
    )

    print(
        f"Slope 4W válidos: "
        f"{data['VFI_Slope_4W'].notna().sum()}"
    )

    print(
        f"Slope 8W válidos: "
        f"{data['VFI_Slope_8W'].notna().sum()}"
    )

    print()

    # ==========================================================
    # 4. LAST 15 WEEKS
    # ==========================================================

    columns = [
        "Close",
        "Volume",
        "VFI",
        "VFI_Slope",
        "VFI_Slope_4W",
        "VFI_Slope_8W"
    ]

    print("=" * 100)
    print("ÚLTIMAS 15 SEMANAS")
    print("=" * 100)

    print()

    print(
        data[columns]
        .tail(15)
        .to_string()
    )

    # ==========================================================
    # 5. FIRST VALID VFI
    # ==========================================================

    valid_vfi = data[
        data["VFI"].notna()
    ]

    if not valid_vfi.empty:

        first_valid_date = valid_vfi.index[0]

        print()
        print("=" * 100)
        print("PRIMER VFI VÁLIDO")
        print("=" * 100)

        print()

        print(
            f"Fecha: {first_valid_date}"
        )

        print(
            f"VFI: "
            f"{valid_vfi.iloc[0]['VFI']:.4f}"
        )

    # ==========================================================
    # 6. CURRENT VALUES
    # ==========================================================

    current = data.iloc[-1]

    print()
    print("=" * 100)
    print("VALORES ACTUALES")
    print("=" * 100)

    print()

    print(
        f"Fecha:        {data.index[-1]}"
    )

    print(
        f"VFI:          {current['VFI']}"
    )

    print(
        f"Slope:        {current['VFI_Slope']}"
    )

    print(
        f"Slope 4W:     {current['VFI_Slope_4W']}"
    )

    print(
        f"Slope 8W:     {current['VFI_Slope_8W']}"
    )


if __name__ == "__main__":
    main()