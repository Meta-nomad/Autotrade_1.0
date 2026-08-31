from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse

from .config import Settings
from .service import PaperTradingService


settings = Settings.from_env()
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
service = PaperTradingService(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await service.start()
    try:
        yield
    finally:
        await service.stop()


app = FastAPI(
    title="Order Flow Paper Lab",
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)


def authorise(
    token: str = Query(default=""),
    x_dashboard_token: str = Header(default=""),
) -> None:
    if settings.dashboard_token and token != settings.dashboard_token and x_dashboard_token != settings.dashboard_token:
        raise HTTPException(status_code=401, detail="Invalid dashboard token")


@app.get("/health")
async def health() -> dict[str, object]:
    lag = time.time() - service.last_engine_tick if service.last_engine_tick else None
    healthy = bool(service.last_engine_tick and lag is not None and lag < 30 and not service.last_engine_error)
    return {
        "status": "ok" if healthy else "starting",
        "paper_only": True,
        "live_trading_enabled": False,
        "engine_lag_seconds": lag,
        "error": service.last_engine_error,
    }


@app.get("/api/status", dependencies=[Depends(authorise)])
async def api_status() -> dict[str, object]:
    return await service.status()


@app.get("/api/trades", dependencies=[Depends(authorise)])
async def api_trades(limit: int = Query(default=100, ge=1, le=1_000)) -> list[dict[str, object]]:
    return await service.storage.recent_trades(limit)


@app.get("/api/signals", dependencies=[Depends(authorise)])
async def api_signals(limit: int = Query(default=100, ge=1, le=1_000)) -> list[dict[str, object]]:
    return await service.storage.recent_signals(limit)


@app.get("/api/diagnostics", dependencies=[Depends(authorise)])
async def api_diagnostics() -> dict[str, object]:
    return await service.diagnostics()


@app.get("/api/equity/{account}", dependencies=[Depends(authorise)])
async def api_equity(account: str, limit: int = Query(default=1_000, ge=1, le=10_000)) -> list[dict[str, object]]:
    if account not in service.broker.accounts:
        raise HTTPException(status_code=404, detail="Unknown account")
    return await service.storage.equity_history(account, limit)


@app.get("/", response_class=HTMLResponse, dependencies=[Depends(authorise)])
async def dashboard() -> str:
    return DASHBOARD_HTML


DASHBOARD_HTML = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Order Flow Paper Lab</title>
<style>
:root{color-scheme:dark;--bg:#0a0d12;--card:#111722;--line:#253044;--muted:#8e9bb0;--green:#30d17b;--red:#ff5c6c;--amber:#f6c453;--blue:#5ca9ff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 20% 0,#142038 0,var(--bg) 35%);font:14px Inter,system-ui,sans-serif;color:#eef3fb}
main{max-width:1440px;margin:auto;padding:28px}.top{display:flex;justify-content:space-between;align-items:flex-end;gap:20px;margin-bottom:22px}
h1{font-size:26px;margin:0 0 5px}.muted{color:var(--muted)}.badge{padding:7px 11px;border:1px solid #245b3d;border-radius:99px;color:var(--green);background:#0d261b;font-weight:700}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.card{background:linear-gradient(145deg,#151d2a,var(--card));border:1px solid var(--line);border-radius:15px;padding:17px;box-shadow:0 15px 40px #0004}
.account h2{font-size:16px;margin:0 0 14px}.metric{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #ffffff0b}.metric:last-child{border:0}.value{font-variant-numeric:tabular-nums;font-weight:650}
.positive{color:var(--green)}.negative{color:var(--red)}.section{margin-top:20px}.section h2{font-size:17px;margin:0 0 10px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}th{font-size:12px;color:var(--muted);background:#151d29}tr:last-child td{border:0}
.feeds{display:flex;gap:8px;flex-wrap:wrap}.feed{padding:7px 10px;border-radius:8px;background:#17202d;border:1px solid var(--line)}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;background:var(--red)}.dot.on{background:var(--green)}
@media(max-width:900px){.grid{grid-template-columns:1fr}.top{align-items:flex-start;flex-direction:column}main{padding:18px}.tablewrap{overflow:auto}}
</style></head><body><main>
<div class="top"><div><h1>Order Flow Paper Lab</h1><div class="muted">Реальные рыночные данные · виртуальные деньги · реальные ордера отключены</div></div><div class="badge">PAPER ONLY</div></div>
<div id="runtime" class="muted">Подключение к рыночным потокам…</div>
<div id="accounts" class="grid"></div>
<section class="section"><h2>Потоки данных</h2><div id="feeds" class="feeds"></div></section>
<section class="section"><h2>Открытые позиции</h2><div class="tablewrap"><table><thead><tr><th>Счёт</th><th>Пара</th><th>Сторона</th><th>Сетап</th><th>Вход</th><th>Mark</th><th>PnL</th><th>R</th></tr></thead><tbody id="positions"></tbody></table></div></section>
<section class="section"><h2>Последние закрытые сделки</h2><div class="tablewrap"><table><thead><tr><th>Время</th><th>Счёт</th><th>Пара</th><th>Сторона</th><th>Причина</th><th>PnL</th><th>R</th></tr></thead><tbody id="trades"></tbody></table></div></section>
</main><script>
const token=new URLSearchParams(location.search).get('token')||'';const q=token?'?token='+encodeURIComponent(token):'';
const n=(v,d=2)=>Number(v||0).toLocaleString('ru-RU',{minimumFractionDigits:d,maximumFractionDigits:d});
const cls=v=>Number(v)>=0?'positive':'negative';
async function load(){try{const [s,t]=await Promise.all([fetch('/api/status'+q).then(r=>r.json()),fetch('/api/trades'+q).then(r=>r.json())]);
document.getElementById('runtime').textContent=`Движок: ${s.last_engine_error?'ошибка: '+s.last_engine_error:'работает'} · задержка ${n(s.engine_lag_seconds,1)} с · uptime ${n(s.uptime_seconds/60,1)} мин`;
document.getElementById('accounts').innerHTML=Object.values(s.accounts).map(a=>`<div class="card account"><h2>${a.name}</h2><div class="metric"><span class="muted">Equity</span><span class="value">${n(a.equity)} USDT</span></div><div class="metric"><span class="muted">Доходность</span><span class="value ${cls(a.return_pct)}">${n(a.return_pct)}%</span></div><div class="metric"><span class="muted">Просадка</span><span class="value ${cls(a.drawdown_pct)}">${n(a.drawdown_pct)}%</span></div><div class="metric"><span class="muted">W / L</span><span class="value">${a.wins} / ${a.losses}</span></div><div class="metric"><span class="muted">Комиссии</span><span class="value">${n(a.total_fees)} USDT</span></div><div class="metric"><span class="muted">Статус</span><span class="value">${a.halted_reason||'активен'}</span></div></div>`).join('');
document.getElementById('feeds').innerHTML=Object.values(s.market.feeds).map(f=>`<div class="feed"><span class="dot ${f.connected?'on':''}"></span>${f.name}: ${f.connected?'online':'offline'} · ${f.messages}</div>`).join('');
const pos=[];Object.values(s.accounts).forEach(a=>a.positions.forEach(p=>pos.push({...p,account:a.name})));document.getElementById('positions').innerHTML=pos.length?pos.map(p=>`<tr><td>${p.account}</td><td>${p.symbol}</td><td>${p.side}</td><td>${p.setup}</td><td>${n(p.entry_price,5)}</td><td>${n(p.mark_price,5)}</td><td class="${cls(p.unrealized_pnl)}">${n(p.unrealized_pnl)}</td><td>${n(p.current_r)}</td></tr>`).join(''):'<tr><td colspan="8" class="muted">Пока нет открытых позиций</td></tr>';
document.getElementById('trades').innerHTML=t.length?t.slice(0,50).map(x=>`<tr><td>${new Date(x.closed_at*1000).toLocaleString()}</td><td>${x.account}</td><td>${x.symbol}</td><td>${x.side}</td><td>${x.reason}</td><td class="${cls(x.net_pnl)}">${n(x.net_pnl)}</td><td>${n(x.r_multiple)}</td></tr>`).join(''):'<tr><td colspan="7" class="muted">Закрытых сделок пока нет</td></tr>';
}catch(e){document.getElementById('runtime').textContent='Dashboard API недоступен: '+e;console.error(e)}}load();setInterval(load,10000);
</script></body></html>"""
