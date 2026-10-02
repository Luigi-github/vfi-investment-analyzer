import yfinance as yf
import pandas as pd


class MarketDataService:

    def get_daily_data(
        self,
        symbol: str,
        period: str = "5y"
    ) -> pd.DataFrame:

        data = yf.Ticker(symbol).history(
            period=period,
            interval="1d",
            auto_adjust=False
        )

        if data.empty:
            raise ValueError(
                f"No data found for symbol: {symbol}"
            )

        columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        data = data[columns].copy()

        # Convertir el índice a datetime
        data.index = pd.to_datetime(data.index)

        # Eliminar la información de zona horaria.
        # Trabajaremos solamente con fechas.
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        # Fecha actual del sistema
        today = pd.Timestamp.now().normalize()

        # Eliminar cualquier fecha posterior a hoy
        data = data[
            data.index.normalize() <= today
        ]

        return data

    def convert_to_weekly(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:

        weekly = data.resample(
            "W-FRI"
        ).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        })

        weekly = weekly.dropna()

        # Fecha actual
        today = pd.Timestamp.now().normalize()

        # Encontrar el viernes de la semana actual
        current_friday = (
            today
            + pd.Timedelta(
                days=(4 - today.weekday()) % 7
            )
        )

        # Solo semanas cuyo viernes sea anterior
        # al viernes de la semana actual.
        weekly = weekly[
            weekly.index < current_friday
        ]

        print()
        print("DEBUG")
        print(f"Today: {today}")
        print(f"Current Friday: {current_friday}")
        print(f"Last weekly date: {weekly.index[-1]}")

        return weekly