from app.paper_trading import PaperAccount, PaperTradingEngine


def main() -> None:
    account = PaperAccount(equity=10_000, cash=10_000)
    engine = PaperTradingEngine(account)
    engine.open_position(
        symbol="BTCUSDT", direction="long", entry_price=100_000,
        quantity=0.01, stop_price=92_000, target_price=116_000,
    )
    event = engine.process_bar("BTCUSDT", high=101_500, low=98_500, close=100_800)
    print({"event": event, **account.snapshot()})


if __name__ == "__main__":
    main()
