import os
import sys
import json
import base64
import datetime
import requests
import pandas as pd
import yfinance as yf
import jpholiday
import google.generativeai as genai
from time import sleep

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
        
        # Gemini設定
        if self.secret_ai:
            genai.configure(api_key=self.secret_ai.get("api_key"))
            self.ai_model_name = self.secret_ai.get("model", "gemini-1.5-flash")

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
            data = yf.download(list(tickers.keys()), period="5d", interval="1d", progress=False)
            # yfinanceのレスポンス形式対応（MultiIndexの場合）
            adj_close = data["Close"] if "Close" in data else data
            
            for code, name in tickers.items():
                if code in adj_close.columns:
                    series = adj_close[code].dropna()
                    if len(series) >= 2:
                        today_val = series.iloc[-1]
                        prev_val = series.iloc[-2]
                        change_pct = (today_val - prev_val) / prev_val * 100
                        results[name] = {"price": today_val, "change_pct": change_pct}
        except Exception as e:
            print(f"[ERROR] Indices fetch failed: {e}")
            
        return results

    @staticmethod
    def get_jpx400_tickers():
        """SBI証券からJPX400銘柄リストを取得"""
        print("[INFO] Fetching JPX400 List...")
        try:
            dfs = pd.read_html(MarketData.JPX400_URL)
            # 通常、銘柄テーブルはインデックス1にあることが多いが、列名で判定推奨
            # ここでは単純化のためテーブル[1]を使用
            df = dfs[1]
            # 銘柄コード列（0列目）を取得し、文字列化 + ".T" 付与
            codes = df.iloc[:, 0].astype(str).str.zfill(4) + ".T"
            return codes.tolist()
        except Exception as e:
            print(f"[ERROR] JPX400 list fetch failed: {e}")
            return []

    @staticmethod
    def get_stock_data(tickers):
        """対象銘柄の株価データ（過去120日）を取得"""
        print(f"[INFO] Fetching stock data for {len(tickers)} tickers...")
        if not tickers:
            return pd.DataFrame()
            
        # yfinanceの一括ダウンロード
        data = yf.download(tickers, period="120d", interval="1d", group_by="ticker", threads=True, progress=True)
        return data

# ==========================================
# 3. Logic Analyzer
# ==========================================
class Analyzer:
    """分析ロジックを担当するクラス"""

    @staticmethod
    def analyze_stocks(tickers, data, config):
        """急騰・急落銘柄を抽出し、詳細分析を行う"""
        results = []
        
        for t in tickers:
            # データフレームから対象銘柄の切り出し
            try:
                df = data[t].dropna(subset=["Close", "Volume"]) if isinstance(data.columns, pd.MultiIndex) else data
                if len(df) < 25:
                    continue
                
                # テクニカル指標計算
                # 1. リターン
                last_close = df["Close"].iloc[-1]
                prev_close = df["Close"].iloc[-2]
                ret = (last_close - prev_close) / prev_close
                ret_pct = ret * 100
                
                # 2. 出来高比率
                vol_today = df["Volume"].iloc[-1]
                vol_avg_20 = df["Volume"].iloc[-22:-2].mean() # 当日を含めない過去平均
                vol_ratio = vol_today / (vol_avg_20 + 1) # ゼロ除算回避

                # スクリーニング判定
                is_spike = ret >= config.CHANGE_THRESHOLD
                is_drop = ret <= -config.CHANGE_THRESHOLD
                is_active = vol_ratio >= config.VOL_RATIO_THRESHOLD

                if (is_spike or is_drop) and is_active:
                    # 対象銘柄！ -> 詳細情報の取得（APIコールが発生するため、ここで行う）
                    ticker_info = Analyzer._get_ticker_info(t)
                    
                    # 6項目スコア計算 & コメント生成
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
                    sleep(0.5) # API負荷軽減
                    
            except Exception as e:
                # 個別銘柄のエラーはスキップして続行
                continue
                
        return results

    @staticmethod
    def _get_ticker_info(ticker):
        """詳細情報(info)を取得"""
        try:
            return yf.Ticker(ticker).info
        except:
            return {}

    @staticmethod
    def _evaluate_fundamentals(info, ret_pct, vol_ratio):
        """6項目のスコアリングと定型コメント生成"""
        scores = {}
        comments = []
        
        # 安全に値を取得するヘルパー
        def get_val(key, default=None):
            return info.get(key, default)

        # 1. Valuation (割安性)
        per = get_val("trailingPE", 100)
        pbr = get_val("priceToBook", 10)
        score_val = 0
        if per < 10: score_val += 5
        elif per < 15: score_val += 3
        if pbr < 1.0: score_val += 5
        elif pbr < 1.5: score_val += 3
        scores["Valuation"] = min(score_val, 10)
        
        if score_val >= 8: comments.append("PER/PBR水準から見て、割安感が強い状態です。")

        # 2. Profitability (収益性)
        roe = get_val("returnOnEquity", 0)
        margin = get_val("operatingMargins", 0)
        score_prof = 0
        if roe > 0.15: score_prof += 5
        elif roe > 0.08: score_prof += 3
        if margin > 0.10: score_prof += 5
        elif margin > 0.05: score_prof += 3
        scores["Profitability"] = min(score_prof, 10)

        # 3. Financial (財務健全性)
        # infoから自己資本比率を直接取れない場合が多いが、簡易的にdebtToEquityなどを利用
        de_ratio = get_val("debtToEquity", 1000) # 低いほうがいい
        score_fin = 5 # デフォルト
        if de_ratio < 50: score_fin = 10
        elif de_ratio < 100: score_fin = 7
        scores["Financial"] = score_fin

        # 4. Growth (成長性)
        rev_growth = get_val("revenueGrowth", 0)
        score_grow = 0
        if rev_growth > 0.20: score_grow = 10
        elif rev_growth > 0.10: score_grow = 7
        elif rev_growth > 0.05: score_grow = 5
        scores["Growth"] = score_grow
        
        if score_grow >= 7: comments.append("直近の売上成長率が高く、事業拡大が続いています。")

        # 5. Dividend (配当)
        yield_val = get_val("dividendYield", 0)
        score_div = 0
        if yield_val is not None:
            if yield_val > 0.04: score_div = 10
            elif yield_val > 0.03: score_div = 7
            elif yield_val > 0.02: score_div = 5
        scores["Dividend"] = score_div
        
        if score_div >= 7: comments.append(f"配当利回りが{round(yield_val*100, 2)}%と高く、インカムゲインの魅力があります。")

        # 6. Momentum (モメンタム)
        # 今回は単純に当日の動きで判定
        score_mom = 5
        if abs(ret_pct) > 10: score_mom = 10
        elif abs(ret_pct) > 6: score_mom = 8
        scores["Momentum"] = score_mom
        
        # 騰落コメント
        if ret_pct > 0:
            comments.append(f"本日は+{round(ret_pct, 2)}%の大幅上昇となりました。")
        else:
            comments.append(f"本日は{round(ret_pct, 2)}%の大幅下落となりました。")
            
        if vol_ratio > 3.0:
            comments.append("普段の3倍以上の出来高を伴っており、投資家の注目が集まっています。")

        return scores, comments

# ==========================================
# 4. AI Writer (Gemini)
# ==========================================
class AIWriter:
    """Gemini APIを使用して概況テキストを生成"""
    
    @staticmethod
    def write_market_summary(indices, config):
        if not config.secret_ai or not indices:
            return "（API設定不足またはデータ不足のため概況生成スキップ）"
            
        n225 = indices.get("日経平均", {})
        sp500 = indices.get("S&P500", {})
        
        prompt = f"""
        あなたはベテランの経済記者です。以下の市場データを基に、本日の市場概況を日本語200文字程度で簡潔にまとめてください。
        客観的な事実を中心に記述してください。

        [データ]
        日経平均株価: {round(n225.get('price', 0))}円 (前日比 {round(n225.get('change_pct', 0), 2)}%)
        S&P500 (前日終値): {round(sp500.get('price', 0))} (前日比 {round(sp500.get('change_pct', 0), 2)}%)
        """
        
        try:
            model = genai.GenerativeModel(config.ai_model_name)
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"[ERROR] AI Generation failed: {e}")
            return "（AI概況生成中にエラーが発生しました）"

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

        # 1. HTML生成
        today_str = datetime.date.today().strftime("%Y/%m/%d")
        
        html_parts = []
        html_parts.append(f"<h2>{today_str} 市場概況</h2>")
        html_parts.append(f"<p>{summary_text}</p>")
        html_parts.append("<hr>")
        
        if not stocks:
            html_parts.append("<p>本日は、変動率±6%の基準を超える急騰・急落銘柄はありませんでした。</p>")
        else:
            html_parts.append(f"<h2>本日の注目銘柄（{len(stocks)}件）</h2>")
            for s in stocks:
                color = "red" if s['change_pct'] > 0 else "blue"
                sign = "+" if s['change_pct'] > 0 else ""
                
                # レーダーチャート（テキスト簡易版）
                radar_text = " / ".join([f"{k}:{v}" for k, v in s['scores'].items()])
                
                # コメント結合
                full_comment = "".join(s['comments'])
                
                # 事業概要（最初の100文字のみ）
                biz_summary = s['summary'][:100] + "..." if len(s['summary']) > 100 else s['summary']

                section = f"""
                <div style="border:1px solid #ddd; padding:15px; margin-bottom:20px; border-radius:5px;">
                    <h3 style="margin-top:0;">{s['name']} ({s['ticker']})</h3>
                    <p style="font-size:1.2em; font-weight:bold; color:{color};">
                        前日比: {sign}{round(s['change_pct'], 2)}% / 出来高倍率: {round(s['volume_ratio'], 2)}倍
                    </p>
                    <div style="background:#f9f9f9; padding:10px; font-size:0.9em;">
                        <strong>【AIスコア】</strong><br>{radar_text}
                    </div>
                    <p><strong>【分析】</strong>{full_comment}</p>
                    <p style="font-size:0.8em; color:#666;">{biz_summary}</p>
                </div>
                """
                html_parts.append(section)
        
        html_parts.append("<hr>")
        html_parts.append("<p><small>※本情報はAIおよびプログラムによる自動生成です。投資判断は自己責任でお願いします。</small></p>")
        
        final_html = "\n".join(html_parts)
        
        # 2. WP APIへPOST
        wp_url = f"{config.secret_wp['url']}/pages/{config.secret_wp['page_id']}"
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
        
        try:
            res = requests.post(wp_url, headers=headers, json=payload)
            res.raise_for_status()
            print("[INFO] Successfully updated WordPress page.")
        except Exception as e:
            print(f"[ERROR] WordPress update failed: {e}")
            if 'res' in locals():
                print(res.text)

# ==========================================
# Main Execution Flow
# ==========================================
def main():
    # 0. 祝日判定
    today = datetime.date.today()
    if jpholiday.is_holiday(today):
        print(f"[INFO] {today} is a holiday in Japan. Exiting.")
        sys.exit(0)
    
    print(f"[INFO] Starting analysis for {today}...")
    
    # 1. 設定読み込み
    config = Config()
    
    # 2. データ収集
    indices = MarketData.get_indices()
    jpx_codes = MarketData.get_jpx400_tickers()
    
    if not jpx_codes:
        print("[ERROR] No tickers found. Exiting.")
        return

    # 株価データ一括取得
    stock_data = MarketData.get_stock_data(jpx_codes)
    
    # 3. 分析実行
    target_stocks = Analyzer.analyze_stocks(jpx_codes, stock_data, config)
    print(f"[INFO] Found {len(target_stocks)} target stocks.")
    
    # 4. 市場概況生成 (AI)
    summary_text = AIWriter.write_market_summary(indices, config)
    print("[INFO] Market summary generated.")
    
    # 5. WordPress投稿
    WordPressPublisher.publish(config, indices, target_stocks, summary_text)
    
    print("[INFO] All done.")

if __name__ == "__main__":
    main()
