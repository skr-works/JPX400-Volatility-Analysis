import os
import sys
import json
import base64
import datetime
import requests
import pandas as pd
import yfinance as yf
import jpholiday
import warnings
from time import sleep

# Google Generative AIの警告を抑制
warnings.filterwarnings("ignore")

import google.generativeai as genai

# ==========================================
# 1. Config & Constants
# ==========================================
class Config:
    """設定情報を管理するクラス"""
    
    # 閾値設定
    CHANGE_THRESHOLD = 0.06  # 騰落率 ±6%
    VOL_RATIO_THRESHOLD = 1.5  # 出来高倍率 1.5倍
    
    def __init__(self):
        # 環境変数からSecretsを取得
        self.secret_ai = self._load_json_secret("APP_SECRET_AI")
        self.secret_wp = self._load_json_secret("APP_SECRET_WORDPRESS")
        
        # Gemini設定 (エラーが出ても止まらないようにする)
        self.ai_enabled = False
        if self.secret_ai and self.secret_ai.get("api_key"):
            try:
                genai.configure(api_key=self.secret_ai.get("api_key"))
                self.ai_model_name = self.secret_ai.get("model", "gemini-1.5-flash")
                self.ai_enabled = True
            except Exception as e:
                print(f"[WARNING] Gemini config failed: {e}")

    def _load_json_secret(self, key):
        """JSON形式の環境変数をパースする"""
        raw = os.getenv(key)
        if not raw:
            print(f"[WARNING] Secret '{key}' is not set.")
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse secret '{key}': {e}")
            return {}

# ==========================================
# 2. Market Data Fetcher
# ==========================================
class MarketData:
    """株価・指数データを取得するクラス"""
    JPX400_URL = "https://site1.sbisec.co.jp/ETGate/WPLETmgR001Control?OutSide=on&getFlg=on&burl=search_market&cat1=market&cat2=info&dir=info&file=market_meigara_400.html"

    @staticmethod
    def get_indices():
        """日経平均とS&P500の直近データを取得"""
        print("[INFO] Fetching Market Indices...")
        tickers = {"^N225": "日経平均", "^GSPC": "S&P500"}
        results = {}
        try:
            for code, name in tickers.items():
                try:
                    data = yf.download(code, period="5d", interval="1d", progress=False, threads=False)
                    if isinstance(data.columns, pd.MultiIndex):
                        if 'Close' in data.columns.levels[0]:
                            series = data['Close'][code].dropna()
                        else:
                            series = data.xs('Close', axis=1, level=0)[code].dropna()
                    else:
                        series = data['Close'].dropna()

                    if len(series) >= 2:
                        today_val = float(series.iloc[-1])
                        prev_val = float(series.iloc[-2])
                        change_pct = (today_val - prev_val) / prev_val * 100
                        results[name] = {"price": today_val, "change_pct": change_pct}
                except Exception:
                    pass # 指数取得失敗は無視して進める
        except Exception as e:
            print(f"[ERROR] Indices fetch failed: {e}")
        return results

    @staticmethod
    def get_jpx400_tickers():
        """SBI証券からJPX400銘柄リストを取得"""
        print("[INFO] Fetching JPX400 List...")
        try:
            res = requests.get(MarketData.JPX400_URL, timeout=10)
            res.encoding = "cp932"
            dfs = pd.read_html(res.text)
            target_df = None
            for df in dfs:
                if df.shape[1] > 0 and df.iloc[:, 0].astype(str).str.match(r'\d{4}').any():
                    target_df = df
                    break
            if target_df is None: target_df = dfs[1]
            codes = target_df.iloc[:, 0].astype(str).str.zfill(4) + ".T"
            return codes.tolist()
        except Exception as e:
            print(f"[ERROR] JPX400 list fetch failed: {e}")
            return []

    @staticmethod
    def get_stock_data(tickers):
        """対象銘柄の株価データ取得"""
        print(f"[INFO] Fetching stock data for {len(tickers)} tickers...")
        if not tickers: return pd.DataFrame()
        try:
            # yfinanceのエラーでも止まらないようにする
            data = yf.download(tickers, period="120d", interval="1d", group_by="ticker", threads=True, progress=True)
            return data
        except Exception as e:
            print(f"[ERROR] Bulk download failed: {e}")
            return pd.DataFrame()

# ==========================================
# 3. Logic Analyzer
# ==========================================
class Analyzer:
    """分析ロジックを担当するクラス"""

    @staticmethod
    def analyze_stocks(tickers, data, config):
        results = []
        if data.empty: return []

        for t in tickers:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if t not in data.columns.levels[0]: continue
                    df = data[t].copy()
                else:
                    df = data.copy()

                if 'Close' not in df.columns or 'Volume' not in df.columns: continue
                df = df.dropna(subset=["Close", "Volume"])
                if len(df) < 25: continue
                
                last_close = float(df["Close"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2])
                if prev_close == 0: continue
                
                ret = (last_close - prev_close) / prev_close
                ret_pct = ret * 100
                
                vol_today = float(df["Volume"].iloc[-1])
                vol_avg_20 = df["Volume"].iloc[-22:-2].mean()
                vol_ratio = vol_today / vol_avg_20 if (vol_avg_20 and vol_avg_20 > 0) else 0

                is_spike = ret >= config.CHANGE_THRESHOLD
                is_drop = ret <= -config.CHANGE_THRESHOLD
                is_active = vol_ratio >= config.VOL_RATIO_THRESHOLD

                if (is_spike or is_drop) and is_active:
                    ticker_info = Analyzer._get_ticker_info(t)
                    scores, comments = Analyzer._evaluate_fundamentals(ticker_info, ret_pct, vol_ratio)
                    results.append({
                        "ticker": t,
                        "name": ticker_info.get("longName", t),
                        "price": last_close,
                        "change_pct": ret_pct,
                        "volume_ratio": vol_ratio,
                        "scores": scores,
                        "comments": comments,
                        "summary": ticker_info.get("longBusinessSummary", "事業概要データなし")
                    })
                    sleep(0.5)
            except Exception:
                continue
        return results

    @staticmethod
    def _get_ticker_info(ticker):
        try: return yf.Ticker(ticker).info
        except: return {}

    @staticmethod
    def _evaluate_fundamentals(info, ret_pct, vol_ratio):
        scores = {}
        comments = []
        def get_val(key, default=None):
            val = info.get(key, default)
            return val if val is not None else default

        per = get_val("trailingPE", 100)
        pbr = get_val("priceToBook", 10)
        score_val = 0
        if per < 10: score_val += 5
        elif per < 15: score_val += 3
        if pbr < 1.0: score_val += 5
        elif pbr < 1.5: score_val += 3
        scores["Valuation"] = min(score_val, 10)
        if score_val >= 8: comments.append("PER/PBR水準から見て、割安感が強い状態です。")

        roe = get_val("returnOnEquity", 0)
        margin = get_val("operatingMargins", 0)
        score_prof = 0
        if roe > 0.15: score_prof += 5
        elif roe > 0.08: score_prof += 3
        if margin > 0.10: score_prof += 5
        elif margin > 0.05: score_prof += 3
        scores["Profitability"] = min(score_prof, 10)

        de_ratio = get_val("debtToEquity", 1000)
        score_fin = 5
        if de_ratio < 50: score_fin = 10
        elif de_ratio < 100: score_fin = 7
        scores["Financial"] = score_fin

        rev_growth = get_val("revenueGrowth", 0)
        score_grow = 0
        if rev_growth > 0.20: score_grow = 10
        elif rev_growth > 0.10: score_grow = 7
        elif rev_growth > 0.05: score_grow = 5
        scores["Growth"] = score_grow
        if score_grow >= 7: comments.append("直近の売上成長率が高く、事業拡大が続いています。")

        yield_val = get_val("dividendYield", 0)
        score_div = 0
        if yield_val is not None:
            if yield_val > 0.04: score_div = 10
            elif yield_val > 0.03: score_div = 7
            elif yield_val > 0.02: score_div = 5
        scores["Dividend"] = score_div
        if score_div >= 7: comments.append(f"配当利回りが{round(yield_val*100, 2)}%と高く、インカムゲインの魅力があります。")

        score_mom = 5
        if abs(ret_pct) > 10: score_mom = 10
        elif abs(ret_pct) > 6: score_mom = 8
        scores["Momentum"] = score_mom
        
        if ret_pct > 0: comments.append(f"本日は+{round(ret_pct, 2)}%の大幅上昇となりました。")
        else: comments.append(f"本日は{round(ret_pct, 2)}%の大幅下落となりました。")
        if vol_ratio > 3.0: comments.append("普段の3倍以上の出来高を伴っており、投資家の注目が集まっています。")

        return scores, comments

# ==========================================
# 4. AI Writer (Gemini)
# ==========================================
class AIWriter:
    """Gemini APIを使用して概況テキストを生成"""
    
    @staticmethod
    def write_market_summary(indices, config):
        # AIが無効、またはキーがない場合は定型文
        if not config.ai_enabled:
            return "（※本日の市場概況データは取得できませんでした）"

        n225 = indices.get("日経平均", {})
        sp500 = indices.get("S&P500", {})
        
        # 値がない場合のガード
        n225_p = n225.get('price', 0)
        n225_c = n225.get('change_pct', 0)
        sp500_p = sp500.get('price', 0)
        sp500_c = sp500.get('change_pct', 0)

        prompt = f"""
        あなたはベテランの経済記者です。以下の市場データを基に、本日の市場概況を日本語200文字程度で簡潔にまとめてください。
        客観的な事実を中心に記述してください。
        [データ]
        日経平均株価: {round(n225_p)}円 (前日比 {round(n225_c, 2)}%)
        S&P500: {round(sp500_p)} (前日比 {round(sp500_c, 2)}%)
        """
        
        try:
            model = genai.GenerativeModel(config.ai_model_name)
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"[WARNING] AI Generation failed: {e}")
            return f"本日の日経平均は{round(n225_c, 2)}%、S&P500は{round(sp500_c, 2)}%の動きとなりました。"

# ==========================================
# 5. WordPress Publisher
# ==========================================
class WordPressPublisher:
    """HTMLを組み立ててWPへ投稿"""
    
    @staticmethod
    def publish(config, indices, stocks, summary_text):
        if not config.secret_wp:
            print("[INFO] WP secret not set. Skip publishing.")
            return

        # --- 重要: URLの自動補正ロジック ---
        base_url = config.secret_wp['url'].rstrip("/")
        # URLに wp-json が含まれていない場合、自動で付与する
        if "wp-json" not in base_url:
            base_url = f"{base_url}/wp-json/wp/v2"
        
        # エンドポイントの構築
        wp_api_url = f"{base_url}/pages/{config.secret_wp['page_id']}"
        # --------------------------------

        today_str = datetime.date.today().strftime("%Y/%m/%d")
        
        html_parts = []
        html_parts.append(f"<h2>{today_str} 市場概況</h2>")
        html_parts.append(f"<p>{summary_text}</p>")
        html_parts.append("<hr>")
        
        if not stocks:
            html_parts.append(f"<p><strong>本日は、変動率±{int(config.CHANGE_THRESHOLD*100)}%の基準を超える急騰・急落銘柄はありませんでした。</strong></p>")
            html_parts.append("<p>市場は比較的落ち着いた動き、または特定の材料難による模様眺めムードの可能性があります。</p>")
        else:
            html_parts.append(f"<h2>本日の注目銘柄（{len(stocks)}件）</h2>")
            for s in stocks:
                color = "#d32f2f" if s['change_pct'] > 0 else "#1976d2" # 赤/青
                sign = "+" if s['change_pct'] > 0 else ""
                
                radar_text = " / ".join([f"{k}:{v}" for k, v in s['scores'].items()])
                full_comment = "".join(s['comments'])
                biz_summary = s['summary'][:100] + "..." if len(s['summary']) > 100 else s['summary']

                section = f"""
                <div style="border:1px solid #ddd; padding:15px; margin-bottom:20px; border-radius:5px; background-color:#fff;">
                    <h3 style="margin-top:0; border-bottom:1px solid #eee; padding-bottom:5px;">{s['name']} ({s['ticker']})</h3>
                    <p style="font-size:1.4em; font-weight:bold; color:{color}; margin:10px 0;">
                        前日比: {sign}{round(s['change_pct'], 2)}% <span style="font-size:0.7em; color:#333;">(出来高倍率: {round(s['volume_ratio'], 2)}倍)</span>
                    </p>
                    <div style="background:#f5f5f5; padding:10px; font-size:0.9em; margin-bottom:10px; border-radius:4px;">
                        <strong>【AIスコア】</strong> {radar_text}
                    </div>
                    <p style="line-height:1.6;"><strong>【分析】</strong> {full_comment}</p>
                    <p style="font-size:0.8em; color:#666; margin-top:10px;">{biz_summary}</p>
                </div>
                """
                html_parts.append(section)
        
        html_parts.append("<hr>")
        html_parts.append("<p style='font-size:0.8em; color:#888;'>※本情報はAIおよびプログラムによる自動生成です。投資判断は自己責任でお願いします。</p>")
        
        final_html = "\n".join(html_parts)
        
        credentials = f"{config.secret_wp['username']}:{config.secret_wp['password']}"
        token = base64.b64encode(credentials.encode()).decode()
        
        headers = {
            "Authorization": f"Basic {token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "content": final_html,
            "title": f"【JPX400】本日の急騰・急落銘柄分析 ({today_str})"
        }
        
        print(f"[INFO] Posting to WordPress API: {wp_api_url}")
        
        try:
            res = requests.post(wp_api_url, headers=headers, json=payload)
            res.raise_for_status()
            print("[INFO] Successfully updated WordPress page.")
        except Exception as e:
            print(f"[ERROR] WordPress update failed: {e}")
            if 'res' in locals():
                print(f"Response Status: {res.status_code}")
                # HTMLが返ってきていないか確認（API URLミスの場合はHTMLが返る）
                if "<html" in res.text[:100]:
                     print("Received HTML instead of JSON. Check your URL.")
                else:
                     print(res.text[:200])

# ==========================================
# Main Execution Flow
# ==========================================
def main():
    try:
        today = datetime.date.today()
        if jpholiday.is_holiday(today):
            print(f"[INFO] {today} is a holiday in Japan. Exiting.")
            sys.exit(0)
    except Exception as e:
        print(f"[WARNING] Holiday check failed: {e}. Proceeding.")
    
    print(f"[INFO] Starting analysis for {today}...")
    
    config = Config()
    indices = MarketData.get_indices()
    jpx_codes = MarketData.get_jpx400_tickers()
    
    if not jpx_codes:
        print("[ERROR] No tickers found. Exiting.")
        return

    stock_data = MarketData.get_stock_data(jpx_codes)
    target_stocks = Analyzer.analyze_stocks(jpx_codes, stock_data, config)
    print(f"[INFO] Found {len(target_stocks)} target stocks.")
    
    summary_text = AIWriter.write_market_summary(indices, config)
    print("[INFO] Market summary generated (or skipped).")
    
    WordPressPublisher.publish(config, indices, target_stocks, summary_text)
    print("[INFO] All done.")

if __name__ == "__main__":
    main()
