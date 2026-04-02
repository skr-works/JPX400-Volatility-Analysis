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
    
    # API制御設定
    BATCH_SIZE = 5            # 1リクエストでまとめる銘柄数
    MAX_ANALYZE_LIMIT = 5     # AI分析・掲載する最大銘柄数

    def __init__(self):
        # 環境変数からSecretsを取得
        self.secret_ai = self._load_json_secret("APP_SECRET_AI")
        self.secret_wp = self._load_json_secret("APP_SECRET_WORDPRESS")
        
        # Gemini設定
        self.ai_enabled = False
        if self.secret_ai and self.secret_ai.get("api_key"):
            try:
                genai.configure(api_key=self.secret_ai.get("api_key"))
                self.ai_model_name = self.secret_ai.get("model", "gemini-2.5-flash")
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
                        if "Close" in data.columns.levels[0]:
                            series = data["Close"][code].dropna()
                        else:
                            series = data.xs("Close", axis=1, level=0)[code].dropna()
                    else:
                        series = data["Close"].dropna()

                    if len(series) >= 2:
                        today_val = float(series.iloc[-1])
                        prev_val = float(series.iloc[-2])
                        change_pct = (today_val - prev_val) / prev_val * 100
                        results[name] = {"price": today_val, "change_pct": change_pct}
                except Exception:
                    pass
        except Exception as e:
            print(f"[ERROR] Indices fetch failed: {e}")
        return results

    @staticmethod
    def get_jpx400_mapping():
        """data/jpx400_constituents.csv からJPX400銘柄リストを取得"""
        print("[INFO] Loading JPX400 List & Names from CSV...")
        ticker_map = {}

        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(base_dir, "data", "jpx400_constituents.csv")

            df = pd.read_csv(csv_path, dtype=str).fillna("")

            if "code" not in df.columns or "name" not in df.columns:
                print(f"[ERROR] Required columns 'code' and 'name' not found in: {csv_path}")
                return {}

            if "sector" not in df.columns:
                df["sector"] = ""

            codes = df["code"].astype(str).str.strip()
            names = df["name"].astype(str).str.strip()
            sectors = df["sector"].astype(str).str.strip()

            ticker_map = {
                f"{code}.T": {
                    "name": name,
                    "sector": sector
                }
                for code, name, sector in zip(codes, names, sectors)
                if code and name
            }

        except Exception as e:
            print(f"[ERROR] JPX400 CSV load failed: {e}")
            return {}

        return ticker_map

    @staticmethod
    def get_stock_data(tickers):
        """対象銘柄の株価データ取得"""
        print(f"[INFO] Fetching stock data for {len(tickers)} tickers...")
        if not tickers:
            return pd.DataFrame()
        try:
            data = yf.download(
                tickers,
                period="120d",
                interval="1d",
                group_by="ticker",
                threads=True,
                progress=True
            )
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
    def analyze_stocks(ticker_map, data, config):
        """
        ticker_map: {code: {"name": str, "sector": str}} の辞書
        処理フロー:
        1. 全銘柄スキャンして候補リストを作成
        2. priority_score順にソートし、上位N件に絞る
        3. 上位N件をバッチ(最大5件)でAI処理
        """
        candidates = []
        if data.empty:
            return []

        tickers = list(ticker_map.keys())

        print("[INFO] Screening stocks...")
        for t in tickers:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if t not in data.columns.levels[0]:
                        continue
                    df = data[t].copy()
                else:
                    df = data.copy()

                if "Close" not in df.columns or "Volume" not in df.columns:
                    continue

                df = df.dropna(subset=["Close", "Volume"])
                if len(df) < 25:
                    continue

                last_close = float(df["Close"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2])
                if prev_close == 0:
                    continue

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

                    ticker_meta = ticker_map.get(t, {})
                    jp_name = ticker_meta.get("name", ticker_info.get("longName", t))
                    sector = ticker_meta.get("sector", "")

                    close_5d_ago = float(df["Close"].iloc[-6])
                    five_day_change_pct = ((last_close - close_5d_ago) / close_5d_ago * 100) if close_5d_ago else 0

                    close_prev_20 = df["Close"].iloc[-21:-1]
                    ma20 = close_prev_20.mean()
                    ma20_gap_pct = ((last_close - ma20) / ma20 * 100) if ma20 else 0

                    is_20d_high = last_close > close_prev_20.max()
                    is_20d_low = last_close < close_prev_20.min()

                    scores_dict, total_score = Analyzer._calc_scores(ticker_info, ret_pct)
                    judgment = Analyzer._judge_status(total_score, ret_pct)

                    priority_score = abs(ret_pct) * min(vol_ratio, 3.0)

                    candidates.append({
                        "ticker": t,
                        "name": jp_name,
                        "sector": sector,
                        "price": last_close,
                        "change_pct": ret_pct,
                        "volume_ratio": vol_ratio,
                        "five_day_change_pct": five_day_change_pct,
                        "ma20_gap_pct": ma20_gap_pct,
                        "is_20d_high": is_20d_high,
                        "is_20d_low": is_20d_low,
                        "scores": scores_dict,
                        "total_score": total_score,
                        "judgment": judgment,
                        "priority_score": priority_score,
                        "analysis": ""
                    })

            except Exception:
                continue

        if not candidates:
            return []

        candidates.sort(key=lambda x: x["priority_score"], reverse=True)
        target_list = candidates[:config.MAX_ANALYZE_LIMIT]

        print(f"[INFO] Candidates found: {len(candidates)}. Published: {len(target_list)}")

        final_results = []
        batches = [target_list[i:i + config.BATCH_SIZE] for i in range(0, len(target_list), config.BATCH_SIZE)]

        for batch in batches:
            if config.ai_enabled:
                try:
                    print(f"[INFO] Processing batch of {len(batch)} stocks via Gemini...")
                    comments_map = Analyzer._generate_batch_comments(config, batch)

                    for item in batch:
                        if item["ticker"] in comments_map:
                            item["analysis"] = comments_map[item["ticker"]]
                        else:
                            item["analysis"] = Analyzer._generate_logic_comment(item)

                    sleep(4.0)

                except Exception as e:
                    print(f"[ERROR] Batch processing failed: {e}. Fallback to logic.")
                    for item in batch:
                        item["analysis"] = Analyzer._generate_logic_comment(item)
            else:
                for item in batch:
                    item["analysis"] = Analyzer._generate_logic_comment(item)

            final_results.extend(batch)

        return final_results

    @staticmethod
    def _get_ticker_info(ticker):
        try:
            return yf.Ticker(ticker).info
        except Exception:
            return {}

    @staticmethod
    def _calc_scores(info, ret_pct):
        """スコア計算"""
        def get_val(key, default=None):
            val = info.get(key, default)
            return val if val is not None else default

        # 1. 割安性
        per = get_val("trailingPE", 100)
        pbr = get_val("priceToBook", 10)
        s_val = 0
        if per < 10:
            s_val += 5
        elif per < 15:
            s_val += 3
        if pbr < 1.0:
            s_val += 5
        elif pbr < 1.5:
            s_val += 3
        s_val = min(s_val, 10)

        # 2. 収益性
        roe = get_val("returnOnEquity", 0)
        margin = get_val("operatingMargins", 0)
        s_prof = 0
        if roe > 0.15:
            s_prof += 5
        elif roe > 0.08:
            s_prof += 3
        if margin > 0.10:
            s_prof += 5
        elif margin > 0.05:
            s_prof += 3
        s_prof = min(s_prof, 10)

        # 3. 財務健全性
        de_ratio = get_val("debtToEquity", 1000)
        s_fin = 5
        if de_ratio < 50:
            s_fin = 10
        elif de_ratio < 100:
            s_fin = 7

        # 4. 成長性
        rev_growth = get_val("revenueGrowth", 0)
        s_grow = 0
        if rev_growth > 0.20:
            s_grow = 10
        elif rev_growth > 0.10:
            s_grow = 7
        elif rev_growth > 0.05:
            s_grow = 5

        # 5. 配当
        yield_val = get_val("dividendYield", 0)
        s_div = 0
        if yield_val and yield_val > 0.04:
            s_div = 10
        elif yield_val and yield_val > 0.03:
            s_div = 7
        elif yield_val and yield_val > 0.02:
            s_div = 5

        # 6. モメンタム
        s_mom = 5
        if abs(ret_pct) > 10:
            s_mom = 10
        elif abs(ret_pct) > 6:
            s_mom = 8

        scores = {
            "割安性": s_val,
            "収益性": s_prof,
            "財務": s_fin,
            "成長性": s_grow,
            "配当": s_div,
            "モメンタム": s_mom
        }
        total = sum(scores.values())
        return scores, total

    @staticmethod
    def _judge_status(total_score, ret_pct):
        if total_score >= 40:
            return "買い検討"
        elif total_score <= 20:
            return "売り検討"
        else:
            return "様子見"

    @staticmethod
    def _generate_batch_comments(config, batch_items):
        """
        複数銘柄(batch_items)の情報をまとめてGeminiに投げ、JSON形式でコメントを受け取る。
        """
        data_text = ""
        for item in batch_items:
            sign_day = "+" if item["change_pct"] > 0 else ""
            sign_5d = "+" if item["five_day_change_pct"] > 0 else ""
            sign_ma20 = "+" if item["ma20_gap_pct"] > 0 else ""

            data_text += f"""
            - 銘柄コード: {item['ticker']}
              銘柄名: {item['name']}
              業種: {item['sector']}
              前日比: {sign_day}{round(item['change_pct'], 2)}%
              出来高倍率: {round(item['volume_ratio'], 2)}倍
              5日騰落率: {sign_5d}{round(item['five_day_change_pct'], 2)}%
              20日線乖離率: {sign_ma20}{round(item['ma20_gap_pct'], 2)}%
              20日高値更新: {"はい" if item["is_20d_high"] else "いいえ"}
              20日安値更新: {"はい" if item["is_20d_low"] else "いいえ"}
            """

        prompt = f"""
あなたは日本株の市況コメンテーターです。
以下の各銘柄について、当日の値動きを読む価値のある短評を日本語で作成してください。

【目的】
投資判断ではなく、「今日何が起きたか」「どう読むか」「次に何を確認すべきか」を簡潔に伝えること。

【入力データ】
{data_text}

【出力ルール】
- 必ず正しいJSON形式のみを出力すること。Markdownのコードブロックは不要。
- キーは銘柄コード、値はコメント。
- 各コメントは130〜200字。
- 2文まで。
- 前日比、出来高倍率、5日騰落率、20日線乖離率のうち、最低2つの具体的な数値を入れること。
- 「何が起きたか」→「その読みまたは次の確認点」の順で書くこと。
- 業績断定やニュース断定はしないこと。入力データから読める範囲だけで書くこと。
- 同じ表現を繰り返さないこと。

【禁止表現】
- 将来性が期待できます
- 見直し買いの余地があります
- 高配当の魅力があります
- 特段の材料は見当たりません
- 動意付いています
- 買い検討です
- 売り検討です

【出力例】
{{
  "7203.T": "前日比+7.2%、出来高2.1倍で資金流入が目立つ上昇です。5日騰落率も+10%を超えるなら短期過熱の可能性もあり、翌日以降に出来高を保ったまま高値を維持できるかを確認したい局面です。"
}}
"""

        model = genai.GenerativeModel(config.ai_model_name)
        response = model.generate_content(prompt)
        text = response.text.strip()

        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()

        return json.loads(text)

    @staticmethod
    def _generate_logic_comment(item):
        """AIを使わないロジック生成（フォールバック用）"""
        sign_day = "+" if item["change_pct"] > 0 else ""
        sign_5d = "+" if item["five_day_change_pct"] > 0 else ""
        sign_ma20 = "+" if item["ma20_gap_pct"] > 0 else ""

        if item["change_pct"] > 0:
            first_sentence = (
                f"前日比{sign_day}{round(item['change_pct'], 1)}%、出来高{round(item['volume_ratio'], 1)}倍で急騰しました。"
            )

            if item["is_20d_high"]:
                second_sentence = (
                    f"5日騰落率は{sign_5d}{round(item['five_day_change_pct'], 1)}%で、20日高値更新も伴う強い上昇ですが、"
                    f"20日線乖離率{sign_ma20}{round(item['ma20_gap_pct'], 1)}%が大きい場合は過熱感も出やすく、"
                    f"次も出来高を保てるかを確認したい局面です。"
                )
            else:
                second_sentence = (
                    f"5日騰落率は{sign_5d}{round(item['five_day_change_pct'], 1)}%で短期資金の流入は見えますが、"
                    f"20日線乖離率は{sign_ma20}{round(item['ma20_gap_pct'], 1)}%のため、"
                    f"一段高よりも上昇の持続性を見極めたい場面です。"
                )
        else:
            first_sentence = (
                f"前日比{round(item['change_pct'], 1)}%、出来高{round(item['volume_ratio'], 1)}倍で急落しました。"
            )

            if item["is_20d_low"]:
                second_sentence = (
                    f"5日騰落率は{sign_5d}{round(item['five_day_change_pct'], 1)}%で、20日安値更新も伴う弱い動きです。"
                    f"20日線乖離率は{sign_ma20}{round(item['ma20_gap_pct'], 1)}%で、"
                    f"まずは下げ止まりよりも売り圧力の継続有無を確認したい局面です。"
                )
            else:
                second_sentence = (
                    f"5日騰落率は{sign_5d}{round(item['five_day_change_pct'], 1)}%で短期的な需給悪化が意識されます。"
                    f"20日線乖離率は{sign_ma20}{round(item['ma20_gap_pct'], 1)}%で、"
                    f"自律反発の有無よりも安値更新を回避できるかを先に見たい場面です。"
                )

        return first_sentence + second_sentence

# ==========================================
# 4. AI Writer (Market Summary)
# ==========================================
class AIWriter:
    """市場概況用"""
    @staticmethod
    def write_market_summary(indices, config):
        if not config.ai_enabled:
            return "（※本日の市場概況データは取得できませんでした）"

        n225 = indices.get("日経平均", {})
        sp500 = indices.get("S&P500", {})
        
        n225_c = n225.get("change_pct", 0)
        sp500_c = sp500.get("change_pct", 0)

        if config.ai_enabled:
            sleep(2.0)

        prompt = f"""
        あなたは経済記者です。以下のデータから本日の市場概況を日本語200文字程度で要約してください。
        日経平均: 前日比 {round(n225_c, 2)}%
        S&P500: 前日比 {round(sp500_c, 2)}%
        """
        try:
            model = genai.GenerativeModel(config.ai_model_name)
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception:
            return f"日経平均は{round(n225_c, 2)}%、S&P500は{round(sp500_c, 2)}%の動きとなりました。"

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

        base_url = config.secret_wp["url"].rstrip("/")
        if "wp-json" not in base_url:
            base_url = f"{base_url}/wp-json/wp/v2"
        wp_api_url = f"{base_url}/pages/{config.secret_wp['page_id']}"

        today_str = datetime.date.today().strftime("%Y/%m/%d")
        
        html_parts = []
        html_parts.append(f"<h2>{today_str} 市場概況</h2>")
        html_parts.append(f"<p>{summary_text}</p>")
        
        if not stocks:
            html_parts.append(
                f"<p><strong>本日は、出来高推移や変動率±{int(config.CHANGE_THRESHOLD*100)}%が当サイト基準を超える急騰・急落銘柄はありませんでした。</strong></p>"
            )
        else:
            html_parts.append(f"<h2>本日の注目銘柄（{len(stocks)}件）</h2>")
            for s in stocks:
                color = "#d32f2f" if s["change_pct"] > 0 else "#1976d2"
                sign = "+" if s["change_pct"] > 0 else ""
                score_display = " ".join([f"[{k}:{v}]" for k, v in s["scores"].items()])

                judge_color = "#e65100"
                if s["judgment"] == "買い検討":
                    judge_color = "#d32f2f"
                elif s["judgment"] == "売り検討":
                    judge_color = "#1976d2"

                section = f"""
                <div style="margin-bottom:30px; padding:15px; border:1px solid #eee;">
                    <h3>{s['name']} ({s['ticker']})</h3>
                    <p style="font-size:1.2em;">
                        <strong style="color:{color};">前日比: {sign}{round(s['change_pct'], 2)}%</strong> 
                        <span style="font-size:0.8em; color:#555;">(出来高: {round(s['volume_ratio'], 1)}倍)</span>
                    </p>
                    <div style="background:#f9f9f9; padding:10px; margin:10px 0;">
                        <p style="margin:0; font-weight:bold; color:{judge_color};">判定: {s['judgment']}</p>
                        <p style="margin:5px 0 0 0; font-size:0.9em; color:#444;">{score_display}</p>
                    </div>
                    <p style="line-height:1.6;">{s['analysis']}</p>
                </div>
                """
                html_parts.append(section)
        
        html_parts.append("<p style='font-size:0.8em; color:#888; margin-top:20px;'>※本情報による投資判断は自己責任でお願いします。</p>")
        final_html = "\n".join(html_parts)
        
        credentials = f"{config.secret_wp['username']}:{config.secret_wp['password']}"
        token = base64.b64encode(credentials.encode()).decode()
        
        headers = {
            "Authorization": f"Basic {token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "content": final_html,
            "title": f"本日の急騰・急落銘柄分析 ({today_str})"
        }
        
        print(f"[INFO] Posting to WordPress API: {wp_api_url}")
        
        try:
            res = requests.post(wp_api_url, headers=headers, json=payload)
            res.raise_for_status()
            print("[INFO] Successfully updated WordPress page.")
        except Exception as e:
            print(f"[ERROR] WordPress update failed: {e}")

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
    
    jpx_map = MarketData.get_jpx400_mapping()
    
    if not jpx_map:
        print("[ERROR] No tickers found. Exiting.")
        return

    codes = list(jpx_map.keys())
    stock_data = MarketData.get_stock_data(codes)
    
    target_stocks = Analyzer.analyze_stocks(jpx_map, stock_data, config)
    print(f"[INFO] Analyzed {len(target_stocks)} stocks.")
    
    summary_text = AIWriter.write_market_summary(indices, config)
    print("[INFO] Market summary generated.")
    
    WordPressPublisher.publish(config, indices, target_stocks, summary_text)
    print("[INFO] All done.")

if __name__ == "__main__":
    main()
