import asyncio

from app.models import Side, Signal
from app.storage import Storage


def test_storage_roundtrip(tmp_path) -> None:
    async def scenario() -> None:
        storage = Storage(tmp_path / "state.db")
        await storage.initialise()
        signal = Signal(
            symbol="BTC_USDT",
            strategy="order_flow",
            setup="TREND_PULLBACK",
            side=Side.SHORT,
            score=81,
            stop_pct=0.01,
            target_r=2.8,
            ts=1234,
            reasons=["test"],
        )
        await storage.save_signal(signal)
        await storage.save_account_states(1234, {"DEMO": {"balance": 1000}})

        signals = await storage.recent_signals()
        accounts = await storage.load_account_states()

        assert signals[0]["side"] == "SHORT"
        assert accounts["DEMO"]["balance"] == 1000
        await storage.close()

    asyncio.run(scenario())

