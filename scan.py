"""
BUZZZR — scanner
================
Pulls Reddit ticker mentions (ApeWisdom, free, no key) + prices (yfinance, no key),
and BUY-side sentiment from Tradestie's free WSB feed (no key). Computes a heat
score that favours positive/bullish attention and writes the top 25 to Supabase.

Run offline to test:   MOCK=1 python scan.py

Env (GitHub Actions secrets for live):
  SUPABASE_URL, SUPABASE_SERVICE_KEY     # service key bypasses RLS to write
  RESEND_API_KEY                         # (optional) subscriber alert emails
  APEWISDOM_FILTER                       # (optional) default 'all-stocks'
  TRADESTIE_URL                          # (optional) override sentiment endpoint
"""
import os, json, time, datetime as dt
import numpy as np, requests

MOCK              = os.environ.get("MOCK", "") == "1"
SUPABASE_URL      = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY      = os.environ.get("SUPABASE_SERVICE_KEY", "")
AW_FILTER         = os.environ.get("APEWISDOM_FILTER", "all-stocks")
RESEND_KEY        = os.environ.get("RESEND_API_KEY", "")
ALERT_FROM        = os.environ.get("ALERT_FROM", "Buzzzr <alerts@bzzzr.co>")
SITE_URL          = os.environ.get("SITE_URL", "https://bzzzr.co")
ALERT_TIER        = os.environ.get("ALERT_TIER", "🔥 Hot")
ALERT_COOLDOWN_H  = float(os.environ.get("ALERT_COOLDOWN_HRS", "18"))
TRADESTIE_URL     = os.environ.get("TRADESTIE_URL", "https://api.tradestie.com/v1/apps/reddit")

MIN_MENTIONS      = 8
TOP_CANDIDATES    = 25          # board size
FREE_ROWS         = 5           # rows visible without a subscription
WEIGHTS           = dict(V=0.32, A=0.15, M=0.20, R=0.10, E=0.04, O=0.14)
TIERS             = [("🔥 Hot", 1.7), ("🌡️ Heating", 1.1), ("👀 Watch", 0.6)]
TIER_ORDER        = {"🔥 Hot": 3, "🌡️ Heating": 2, "👀 Watch": 1, None: 0}

HERE = os.path.dirname(os.path.abspath(__file__))
HIST = os.path.join(HERE, "docs", "data", "mention_history.json")
ALERT_STATE = os.path.join(HERE, "docs", "data", "alert_state.json")
os.makedirs(os.path.dirname(HIST), exist_ok=True)

# ----------------------------- MENTIONS (ApeWisdom) -------------------------
def fetch_apewisdom():
    if MOCK: return _mock_aw()
    rows = []
    for page in (1, 2):
        url = f"https://apewisdom.io/api/v1.0/filter/{AW_FILTER}/page/{page}"
        try:
            r = requests.get(url, timeout=25); r.raise_for_status()
            results = r.json().get("results", [])
        except Exception as e:
            print(f"WARNING: ApeWisdom page {page} failed: {e}"); continue
        for d in results:
            try:
                rows.append(dict(
                    ticker=d["ticker"], name=d.get("name", d["ticker"]),
                    mentions=int(d.get("mentions", 0)),
                    mentions_24h=int(d.get("mentions_24h_ago", 0) or 0),
                    rank=int(d.get("rank", 9999)),
                    rank_24h=int(d.get("rank_24h_ago", 9999) or 9999),
                    upvotes=int(d.get("upvotes", 0) or 0)))
            except Exception:
                continue
    if not rows:
        raise SystemExit("ERROR: ApeWisdom returned no usable data — aborting")
    print(f"ApeWisdom: fetched {len(rows)} tickers")
    return rows

# ----------------------------- SENTIMENT (Tradestie) ------------------------
def fetch_sentiment():
    """{ticker: {'sentiment': float(-1..1), 'comments': int}} from Tradestie's free WSB feed."""
    if MOCK: return _mock_sent()
    try:
        r = requests.get(TRADESTIE_URL, timeout=25,
                         headers={"User-Agent": "buzzzr/1.0 (+https://bzzzr.co)"})
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"WARNING: Tradestie fetch failed ({e}) — sentiment will be neutral")
        return {}
    out = {}
    for d in (data if isinstance(data, list) else []):
        tk = str(d.get("ticker", "")).upper()
        if not tk:
            continue
        try:
            out[tk] = {"sentiment": float(d.get("sentiment_score", 0) or 0),
                       "comments": int(d.get("no_of_comments", 0) or 0)}
        except Exception:
            continue
    print(f"Tradestie: got sentiment for {len(out)} tickers")
    return out

# ----------------------------- PRICES + MOVES (yfinance) --------------------
_EXCH = {"NMS":"NASDAQ","NGM":"NASDAQ","NCM":"NASDAQ","NASDAQ":"NASDAQ",
         "NYQ":"NYSE","NYSE":"NYSE","NYE":"NYSE","PCX":"NYSEARCA",
         "ASE":"NYSEAMERICAN","AMEX":"NYSEAMERICAN","BATS":"BATS","PNK":"OTCMKTS"}

def _pct(a, b):
    try:
        if b in (None, 0) or a is None: return None
        return round((float(a)/float(b) - 1.0) * 100.0, 1)
    except Exception:
        return None

def fetch_prices(tickers):
    """Returns {ticker: dict(volz, price, vol_ratio, chg_1d/5d/1m/ytd, exchange)}."""
    if MOCK: return _mock_prices(tickers)
    out = {}
    try:
        import yfinance as yf
        data = yf.download(sorted(set(tickers)), period="1y", interval="1d",
                           auto_adjust=False, progress=False, group_by="ticker", threads=True)
    except Exception as e:
        print(f"WARNING: yfinance download failed ({e}) — continuing without price data")
        return {}
    for tk in tickers:
        rec = {}
        try:
            v = data[tk]["Volume"].dropna(); c = data[tk]["Close"].dropna()
            if len(c) >= 2:
                rec["price"] = round(float(c.iloc[-1]), 2)
                rec["chg_1d"] = _pct(c.iloc[-1], c.iloc[-2])
                if len(c) >= 6:  rec["chg_5d"] = _pct(c.iloc[-1], c.iloc[-6])
                if len(c) >= 22: rec["chg_1m"] = _pct(c.iloc[-1], c.iloc[-22])
                yr = c.index[-1].year
                prior = c[c.index.year < yr]
                base = prior.iloc[-1] if len(prior) else c.iloc[0]
                rec["chg_ytd"] = _pct(c.iloc[-1], base)
            if len(v) >= 20:
                base = v.iloc[-31:-1]
                rec["volz"] = float((v.iloc[-1] - base.mean()) / (base.std() + 1e-9))
                rec["vol_ratio"] = round(float(v.iloc[-1] / (base.mean() + 1e-9)), 2)
        except Exception:
            pass
        try:
            fi = yf.Ticker(tk).fast_info
            code = None
            try:    code = fi["exchange"]
            except Exception: code = getattr(fi, "exchange", None)
            if code: rec["exchange"] = _EXCH.get(str(code).upper())
        except Exception:
            pass
        if rec: out[tk] = rec
    print(f"yfinance: got price data for {len(out)}/{len(tickers)} tickers")
    return out

# ----------------------------- SCORING --------------------------------------
def zscore(hist, today):
    if len(hist) < 8: return None
    a = np.array(hist, float); sd = a.std()
    return 0.0 if sd == 0 else float((today - a.mean()) / sd)

def make_why(ratio, vratio, sent, has_sent):
    bits = []
    if ratio and ratio >= 1.5:    bits.append(f"mentions up {ratio:.1f}x in 24h")
    elif ratio and ratio >= 1.15: bits.append("mentions climbing")
    if has_sent:
        if sent >= 0.35:    bits.append("strongly bullish, buy-side tone")
        elif sent >= 0.12:  bits.append("buy-side lean")
        elif sent <= -0.35: bits.append("bearish, short-side tone")
        elif sent <= -0.12: bits.append("cautious, mixed tone")
    if vratio and vratio >= 1.5: bits.append(f"trading {vratio:.1f}x normal volume")
    if not bits: bits.append("early, low-conviction interest")
    s = "; ".join(bits[:3])
    return s[0].upper() + s[1:]

def score(rows, hist, px, sent_map):
    out = []
    for r in rows:
        if r["mentions"] < MIN_MENTIONS: continue
        tk = r["ticker"]
        h = hist.get(tk, [])
        z = zscore(h[-30:], r["mentions"])
        if z is None:
            surge = (r["mentions"] - r["mentions_24h"]) / (r["mentions_24h"] + 10)
            V = float(np.clip(surge * 2.5, -1, 5)); A = 0.0
        else:
            V = z
            zp = zscore(h[-30:-1], h[-1]) if len(h) >= 9 else V
            A = V - (zp if zp is not None else V)
        R = float(np.clip((r["rank_24h"] - r["rank"]) / 50, -1, 1))
        E = float(np.clip(r["upvotes"] / (r["mentions"] + 1) / 5, 0, 1))
        p = px.get(tk, {})
        vz = p.get("volz", 0.0); M = max(vz, 0.0)
        srec = sent_map.get(tk)
        has_s = srec is not None
        sent = float(srec["sentiment"]) if has_s else 0.0
        O = 0.0   # options/gamma needs raw post text; not available from Tradestie
        raw = (WEIGHTS["V"]*V + WEIGHTS["A"]*A + WEIGHTS["M"]*M +
               WEIGHTS["R"]*R + WEIGHTS["E"]*E + WEIGHTS["O"]*O)
        # positive-attention gate: bullish talk amplifies, bearish talk dampens
        mult = float(np.clip(1.0 + 0.45 * sent, 0.6, 1.5)) if has_s else 1.0
        sc = raw * mult
        tier = None
        for nm, thr in TIERS:
            if sc >= thr:
                if nm == "🔥 Hot" and (vz <= 0 or (has_s and sent < -0.15)): continue
                tier = nm; break
        ratio = r["mentions"] / max(r["mentions_24h"], 1)
        out.append(dict(
            ticker=tk, name=r["name"], score=round(sc, 3), tier=tier,
            mentions=r["mentions"], mentions_24h=r["mentions_24h"], rank=r["rank"],
            price=p.get("price"), vol_ratio=p.get("vol_ratio"),
            chg_1d=p.get("chg_1d"), chg_5d=p.get("chg_5d"),
            chg_1m=p.get("chg_1m"), chg_ytd=p.get("chg_ytd"),
            exchange=p.get("exchange"),
            sentiment=(round(sent, 3) if has_s else None),
            options_score=None,
            why=make_why(ratio, p.get("vol_ratio"), sent, has_s),
            components=dict(V=round(V,2), A=round(A,2), M=round(M,2), R=round(R,2),
                            E=round(E,2), O=round(O,2), sent=round(sent,3), mult=round(mult,2))))
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:TOP_CANDIDATES]

# ----------------------------- SINK -----------------------------------------
def write_supabase(results):
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0, tzinfo=None).isoformat() + "Z"
    payload = [{**r, "board_rank": i+1, "is_premium": i+1 > FREE_ROWS, "updated_at": now}
               for i, r in enumerate(results)]
    print(f"ENV CHECK: SUPABASE_URL set={bool(SUPABASE_URL)}  "
          f"SUPABASE_SERVICE_KEY set={bool(SUPABASE_KEY)} (len {len(SUPABASE_KEY)})")
    if not (SUPABASE_URL and SUPABASE_KEY):
        print("[no supabase creds — would write %d rows]" % len(payload))
        for r in payload[:8]:
            lock = "🔒" if r["is_premium"] else "🆓"
            print(f"  {lock} #{r['board_rank']:>2} {r['tier'] or '  '} {r['ticker']:<6} "
                  f"score {r['score']:>5}  sent {r['sentiment']}  why: {r['why']}")
        return
    h = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
         "Content-Type": "application/json", "Prefer": "return=minimal"}
    d = requests.delete(f"{SUPABASE_URL}/rest/v1/signals?board_rank=gte.0", headers=h, timeout=20)
    print(f"DELETE old rows -> status {d.status_code}")
    p = requests.post(f"{SUPABASE_URL}/rest/v1/signals", headers=h, json=payload, timeout=30)
    print(f"INSERT new rows -> status {p.status_code}")
    if p.status_code >= 300 or d.status_code >= 300:
        print("WRITE FAILED. Supabase said:", (p.text or d.text)[:600])
        raise SystemExit(f"Supabase write rejected (delete {d.status_code}, insert {p.status_code}). "
                         "This almost always means SUPABASE_SERVICE_KEY is not the service_role key.")
    print(f"SUCCESS: wrote {len(payload)} signals to Supabase")

# ----------------------------- ALERTS ---------------------------------------
def fetch_subscribers():
    if not (SUPABASE_URL and SUPABASE_KEY):
        return ["test.subscriber@example.com"] if MOCK else []
    h = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    url = f"{SUPABASE_URL}/rest/v1/profiles?subscription_status=eq.active&select=email"
    try:
        r = requests.get(url, headers=h, timeout=20); r.raise_for_status()
        return [p["email"] for p in r.json() if p.get("email")]
    except Exception as e:
        print("subscriber fetch error:", e); return []

def _digest_html(rows):
    items = "".join(
        f'<tr><td style="padding:8px 0;font:700 16px monospace;color:#b8860b">${r["ticker"]}</td>'
        f'<td style="padding:8px 12px;color:#444">{r["name"] or ""}<br>'
        f'<span style="color:#888;font-size:12px">{r.get("why","")}</span></td>'
        f'<td style="padding:8px 0;font:600 15px monospace;text-align:right">{r["score"]:.2f}</td></tr>'
        for r in rows)
    link = f'<p><a href="{SITE_URL}" style="color:#b8860b">Open the board →</a></p>' if SITE_URL else ""
    return (f'<div style="font-family:system-ui,sans-serif;max-width:520px">'
            f'<h2 style="font-family:Archivo,sans-serif">🔥 New on the buzz board</h2>'
            f'<p style="color:#555">These names just crossed into <b>Hot</b>:</p>'
            f'<table style="width:100%;border-collapse:collapse">{items}</table>{link}'
            f'<hr style="border:none;border-top:1px solid #eee;margin:20px 0">'
            f'<p style="color:#999;font-size:11px;line-height:1.6">Buzzzr reports public '
            f'social-media activity. Not financial advice or a recommendation to trade; '
            f'trending names often underperform. Manage or cancel your subscription'
            f'{" at " + SITE_URL if SITE_URL else " from your account"}.</p></div>')

def _send_resend(emails, subject, html):
    if not RESEND_KEY:
        print(f"[no RESEND_API_KEY — would email {len(emails)} subscriber(s)]: {subject}"); return
    h = {"Authorization": f"Bearer {RESEND_KEY}", "Content-Type": "application/json"}
    for i in range(0, len(emails), 100):
        batch = [dict(**{"from": ALERT_FROM, "to": [e], "subject": subject, "html": html})
                 for e in emails[i:i+100]]
        try:
            requests.post("https://api.resend.com/emails/batch", headers=h, json=batch, timeout=30)
        except Exception as e:
            print("resend error:", e)

def notify(results):
    now = time.time()
    threshold = TIER_ORDER.get(ALERT_TIER, 3)
    hot = [r for r in results if TIER_ORDER.get(r["tier"], 0) >= threshold]
    try:    state = json.load(open(ALERT_STATE))
    except Exception: state = {}
    newly = [r for r in hot
             if r["ticker"] not in state or (now - state[r["ticker"]]) > ALERT_COOLDOWN_H * 3600]
    if newly:
        emails = fetch_subscribers()
        tickers = " ".join("$" + r["ticker"] for r in newly[:5])
        subject = f"🔥 Buzzzr: {len(newly)} name{'s' if len(newly)>1 else ''} just went Hot: {tickers}"
        if emails:
            _send_resend(emails, subject, _digest_html(newly))
            print(f"alerted {len(emails)} subscriber(s) on: {tickers}")
        else:
            print(f"newly hot ({tickers}) but no active subscribers to email")
    for r in hot:
        state[r["ticker"]] = now
    state = {k: v for k, v in state.items() if now - v < 7 * 86400}
    try:    json.dump(state, open(ALERT_STATE, "w"))
    except Exception: pass

def main():
    rows = fetch_apewisdom()
    try:    hist = json.load(open(HIST))
    except Exception: hist = {}
    cands = sorted(rows, key=lambda r: r["mentions"], reverse=True)[:TOP_CANDIDATES + 15]
    tickers = [r["ticker"] for r in cands]
    px       = fetch_prices(tickers)
    sent_map = fetch_sentiment()
    results  = score(cands, hist, px, sent_map)
    print(f"scored {len(results)} tickers (sentiment: {'on' if sent_map else 'off'})")
    write_supabase(results)
    try:
        notify(results)
    except Exception as e:
        print(f"WARNING: notify step failed ({e}) — board was still saved")
    for r in cands:
        hist.setdefault(r["ticker"], []).append(r["mentions"])
        hist[r["ticker"]] = hist[r["ticker"]][-60:]
    try:    json.dump(hist, open(HIST, "w"))
    except Exception: pass
    hot  = sum(1 for r in results if r["tier"] == "🔥 Hot")
    heat = sum(1 for r in results if r["tier"] == "🌡️ Heating")
    print(f"DONE: {len(results)} tickers | hot {hot} heating {heat}")

# ----------------------------- MOCK -----------------------------------------
def _mock_aw():
    rng = np.random.default_rng(4)
    tks = [("GME","GameStop"),("NVDA","Nvidia"),("PLTR","Palantir"),("SOFI","SoFi"),
           ("RKLB","Rocket Lab"),("ASTS","AST SpaceMobile"),("HOOD","Robinhood"),
           ("SMCI","Super Micro"),("MARA","Marathon"),("RIVN","Rivian"),
           ("LUNR","Intuitive Machines"),("CVNA","Carvana"),("OKLO","Oklo"),
           ("TSLA","Tesla"),("AMD","AMD"),("COIN","Coinbase"),("DNUT","Krispy Kreme"),
           ("BBAI","BigBear.ai"),("ACHR","Archer"),("UPST","Upstart"),
           ("SOUN","SoundHound"),("IONQ","IonQ"),("CHPT","ChargePoint"),
           ("AFRM","Affirm"),("PLUG","Plug Power"),("NIO","NIO"),("F","Ford")]
    out = []
    for i, (tk, nm) in enumerate(tks):
        m24 = int(rng.integers(10, 400)); surge = rng.random() < 0.35
        m = m24 + (int(rng.integers(50, 400)) if surge else int(rng.integers(-30, 40)))
        out.append(dict(ticker=tk, name=nm, mentions=max(0, m), mentions_24h=m24,
                        rank=i+1, rank_24h=int(np.clip(i+1+rng.integers(-6,6),1,40)),
                        upvotes=int(m*rng.uniform(1,6))))
    return out

def _mock_prices(tickers):
    rng = np.random.default_rng(6); out = {}; exs = ["NASDAQ","NYSE","NYSEARCA"]
    for tk in tickers:
        out[tk] = dict(volz=float(rng.normal(0.7,1.2)), price=round(float(rng.uniform(3,400)),2),
                       vol_ratio=round(float(rng.uniform(0.7,4.5)),2),
                       chg_1d=round(float(rng.normal(0,3)),1), chg_5d=round(float(rng.normal(1,7)),1),
                       chg_1m=round(float(rng.normal(2,14)),1), chg_ytd=round(float(rng.normal(8,40)),1),
                       exchange=exs[rng.integers(0,3)])
    return out

def _mock_sent():
    rng = np.random.default_rng(9)
    tks = ["GME","NVDA","PLTR","TSLA","AMD","SOFI","HOOD","MARA","COIN","ASTS","CVNA","SOUN"]
    return {tk: {"sentiment": round(float(rng.uniform(-0.5, 0.6)), 3),
                 "comments": int(rng.integers(5, 300))} for tk in tks}

if __name__ == "__main__":
    main()
