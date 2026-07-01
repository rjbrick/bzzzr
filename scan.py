"""
BUZZZR — scanner (free data sources)
======================================
Pulls Reddit ticker mentions from ApeWisdom's free public API (no key) + prices
from yfinance (no key), computes the heat score, and writes the ranked board to
Supabase, which the website reads. Builds its OWN mention baseline over time
(stored in docs/data/mention_history.json, committed by the Action), so velocity
and acceleration sharpen after ~a week of runs.

Run offline to test:   MOCK=1 python scan.py

Env (set as GitHub Actions secrets for live):
  SUPABASE_URL, SUPABASE_SERVICE_KEY   # service key bypasses RLS to write
  (optional) APEWISDOM_FILTER          # default 'all-stocks'
"""
import os, json, time, datetime as dt
import numpy as np, requests

MOCK              = os.environ.get("MOCK", "") == "1"
SUPABASE_URL      = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY      = os.environ.get("SUPABASE_SERVICE_KEY", "")
AW_FILTER         = os.environ.get("APEWISDOM_FILTER", "all-stocks")
# email alerts to subscribers (Resend)
RESEND_KEY        = os.environ.get("RESEND_API_KEY", "")
ALERT_FROM        = os.environ.get("ALERT_FROM", "Buzzzr <alerts@bzzzr.co>")
SITE_URL          = os.environ.get("SITE_URL", "https://bzzzr.co")
ALERT_TIER        = os.environ.get("ALERT_TIER", "🔥 Hot")    # min tier that triggers an email
ALERT_COOLDOWN_H  = float(os.environ.get("ALERT_COOLDOWN_HRS", "18"))

MIN_MENTIONS      = 8
TOP_CANDIDATES    = 50
FREE_ROWS         = 5            # rows visible without a subscription
WEIGHTS           = dict(V=0.40, A=0.20, M=0.22, R=0.13, E=0.05)
TIERS             = [("🔥 Hot", 1.7), ("🌡️ Heating", 1.1), ("👀 Watch", 0.6)]
TIER_ORDER        = {"🔥 Hot": 3, "🌡️ Heating": 2, "👀 Watch": 1, None: 0}

HERE = os.path.dirname(os.path.abspath(__file__))
HIST = os.path.join(HERE, "docs", "data", "mention_history.json")
ALERT_STATE = os.path.join(HERE, "docs", "data", "alert_state.json")
os.makedirs(os.path.dirname(HIST), exist_ok=True)

# ----------------------------- DATA -----------------------------------------
def fetch_apewisdom():
    if MOCK: return _mock_aw()
    rows = []
    for page in (1, 2):
        url = f"https://apewisdom.io/api/v1.0/filter/{AW_FILTER}/page/{page}"
        try:
            r = requests.get(url, timeout=25); r.raise_for_status()
            results = r.json().get("results", [])
        except Exception as e:
            print(f"WARNING: ApeWisdom page {page} failed: {e}")
            continue
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

def fetch_volume(tickers):
    # Volume is an OPTIONAL signal. yfinance breaks often in CI, so never let it
    # crash the whole scan — on any failure, just return {} and carry on.
    if MOCK: return _mock_vol(tickers)
    out = {}
    try:
        import yfinance as yf
        data = yf.download(sorted(set(tickers)), period="3mo", interval="1d",
                           auto_adjust=False, progress=False, group_by="ticker", threads=True)
    except Exception as e:
        print(f"WARNING: yfinance download failed ({e}) — continuing without volume signal")
        return {}
    for tk in tickers:
        try:
            v = data[tk]["Volume"].dropna(); c = data[tk]["Close"].dropna()
            if len(v) < 20: continue
            base = v.iloc[-31:-1]
            z = (v.iloc[-1] - base.mean()) / (base.std() + 1e-9)
            out[tk] = (float(z), float(c.iloc[-1]), float(v.iloc[-1] / (base.mean() + 1e-9)))
        except Exception:
            pass
    print(f"yfinance: got volume for {len(out)}/{len(tickers)} tickers")
    return out

# ----------------------------- SCORING --------------------------------------
def zscore(hist, today):
    if len(hist) < 8: return None
    a = np.array(hist, float); sd = a.std()
    return 0.0 if sd == 0 else float((today - a.mean()) / sd)

def score(rows, hist, volz):
    out = []
    for r in rows:
        if r["mentions"] < MIN_MENTIONS: continue
        h = hist.get(r["ticker"], [])
        z = zscore(h[-30:], r["mentions"])
        if z is None:                                   # warm-up: use 24h surge
            surge = (r["mentions"] - r["mentions_24h"]) / (r["mentions_24h"] + 10)
            V = float(np.clip(surge * 2.5, -1, 5)); A = 0.0
        else:
            V = z
            zp = zscore(h[-30:-1], h[-1]) if len(h) >= 9 else V
            A = V - (zp if zp is not None else V)
        R = float(np.clip((r["rank_24h"] - r["rank"]) / 50, -1, 1))
        E = float(np.clip(r["upvotes"] / (r["mentions"] + 1) / 5, 0, 1))
        vz, price, vratio = volz.get(r["ticker"], (0.0, None, None))
        M = max(vz, 0.0)
        sc = (WEIGHTS["V"]*V + WEIGHTS["A"]*A + WEIGHTS["M"]*M +
              WEIGHTS["R"]*R + WEIGHTS["E"]*E)
        tier = None
        for nm, thr in TIERS:
            if sc >= thr:
                if nm == "🔥 Hot" and vz <= 0: continue   # Hot needs volume confirm
                tier = nm; break
        out.append(dict(ticker=r["ticker"], name=r["name"], score=round(sc, 3),
                        tier=tier, mentions=r["mentions"], mentions_24h=r["mentions_24h"],
                        rank=r["rank"], price=price,
                        vol_ratio=None if vratio is None else round(vratio, 2),
                        components=dict(V=round(V,2), A=round(A,2), M=round(M,2),
                                        R=round(R,2), E=round(E,2))))
    out.sort(key=lambda x: x["score"], reverse=True)
    return out

# ----------------------------- SINK -----------------------------------------
def write_supabase(results):
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0, tzinfo=None).isoformat() + "Z"
    payload = []
    for i, r in enumerate(results):
        payload.append({**r, "board_rank": i + 1,
                        "is_premium": i + 1 > FREE_ROWS, "updated_at": now})
    print(f"ENV CHECK: SUPABASE_URL set={bool(SUPABASE_URL)}  "
          f"SUPABASE_SERVICE_KEY set={bool(SUPABASE_KEY)} (len {len(SUPABASE_KEY)})")
    if not (SUPABASE_URL and SUPABASE_KEY):
        print("[no supabase creds — would write %d rows]" % len(payload))
        for r in payload[:8]:
            lock = "🔒" if r["is_premium"] else "🆓"
            print(f"  {lock} #{r['board_rank']:>2} {r['tier'] or '  '} {r['ticker']:<6} "
                  f"score {r['score']:>5}  {r['mentions']} mentions  {r['components']}")
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

# ----------------------------- ALERTS (email subscribers) -------------------
def fetch_subscribers():
    """Emails of users with an active subscription."""
    if not (SUPABASE_URL and SUPABASE_KEY):
        return ["test.subscriber@example.com"] if MOCK else []
    h = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    url = (f"{SUPABASE_URL}/rest/v1/profiles"
           "?subscription_status=eq.active&select=email")
    try:
        r = requests.get(url, headers=h, timeout=20); r.raise_for_status()
        return [p["email"] for p in r.json() if p.get("email")]
    except Exception as e:
        print("subscriber fetch error:", e); return []

def _digest_html(rows):
    items = "".join(
        f'<tr><td style="padding:8px 0;font:700 16px monospace;color:#b8860b">${r["ticker"]}</td>'
        f'<td style="padding:8px 12px;color:#444">{r["name"] or ""}</td>'
        f'<td style="padding:8px 0;font:600 15px monospace;text-align:right">{r["score"]:.2f}</td>'
        f'<td style="padding:8px 0 8px 12px;color:#888;font:12px monospace;text-align:right">'
        f'{r["mentions"]} mentions</td></tr>' for r in rows)
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
        print(f"[no RESEND_API_KEY — would email {len(emails)} subscriber(s)]: {subject}")
        return
    h = {"Authorization": f"Bearer {RESEND_KEY}", "Content-Type": "application/json"}
    for i in range(0, len(emails), 100):       # Resend batch = 100 max
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
        subject = f"🔥 Buzzzr: {len(newly)} name{'s' if len(newly)>1 else ''} just went Hot — {tickers}"
        if emails:
            _send_resend(emails, subject, _digest_html(newly))
            print(f"alerted {len(emails)} subscriber(s) on: {tickers}")
        else:
            print(f"newly hot ({tickers}) but no active subscribers to email")
    # refresh state: stamp current hot names; prune anything stale (>7d)
    for r in hot:
        if r["ticker"] in newly_set(newly) or r["ticker"] not in state:
            state[r["ticker"]] = now
    state = {k: v for k, v in state.items() if now - v < 7 * 86400}
    json.dump(state, open(ALERT_STATE, "w"))

def newly_set(newly):
    return {r["ticker"] for r in newly}


def main():
    rows = fetch_apewisdom()
    try:    hist = json.load(open(HIST))
    except Exception: hist = {}
    cands = sorted(rows, key=lambda r: r["mentions"], reverse=True)[:TOP_CANDIDATES]
    volz = fetch_volume([r["ticker"] for r in cands])
    results = score(cands, hist, volz)
    print(f"scored {len(results)} tickers")
    write_supabase(results)                 # the important bit — do this first
    try:
        notify(results)                     # alerts must never undo a good write
    except Exception as e:
        print(f"WARNING: notify step failed ({e}) — board was still saved")
    # update self-history baseline
    for r in cands:
        hist.setdefault(r["ticker"], []).append(r["mentions"])
        hist[r["ticker"]] = hist[r["ticker"]][-60:]
    try:    json.dump(hist, open(HIST, "w"))
    except Exception: pass
    hot = sum(1 for r in results if r["tier"] == "🔥 Hot")
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
           ("BBAI","BigBear.ai"),("ACHR","Archer"),("UPST","Upstart")]
    out = []
    for i, (tk, nm) in enumerate(tks):
        m24 = int(rng.integers(10, 400))
        surge = rng.random() < 0.35
        m = m24 + (int(rng.integers(50, 400)) if surge else int(rng.integers(-30, 40)))
        out.append(dict(ticker=tk, name=nm, mentions=max(0, m), mentions_24h=m24,
                        rank=i+1, rank_24h=int(np.clip(i+1+rng.integers(-6,6),1,40)),
                        upvotes=int(m*rng.uniform(1,6))))
    return out

def _mock_vol(tickers):
    rng = np.random.default_rng(6)
    return {tk: (float(rng.normal(0.7, 1.2)), round(float(rng.uniform(3,400)),2),
                 round(float(rng.uniform(0.7,4.5)),2)) for tk in tickers}

if __name__ == "__main__":
    main()
