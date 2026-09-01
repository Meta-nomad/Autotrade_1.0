import asyncio
import logging

from app import __version__

from app.service import PaperTradingService

from conftest import make_settings


def test_synthetic_service_lifecycle(tmp_path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.service")
    async def scenario() -> None:
        settings = make_settings(
            tmp_path / "service.db",
            evaluation_interval_seconds=0.05,
            feature_persist_seconds=1,
            account_persist_seconds=1,
        )
        service = PaperTradingService(settings)
        await service.start()
        try:
            await asyncio.sleep(0.8)
            status = await service.status()
            feature = status["market"]["symbols"]["BTC_USDT"]["feature"]
            assert status["mode"] == "PAPER_ONLY"
            assert status["live_trading_enabled"] is False
            assert __version__ == "0.3.0"
            assert list(service.broker.accounts) == ["COMPOSITE_FLOW"]
            assert "RELEASE v0.3.0-20260901 accounts=COMPOSITE_FLOW execution=PAPER_ONLY" in caplog.text
            assert status["last_engine_error"] == ""
            assert feature is not None
            assert feature["data_ready"] is True
        finally:
            await service.stop()

    asyncio.run(scenario())
