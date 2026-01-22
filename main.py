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
import random  # 追加: ランダム選択用
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
    MAX_ANALYZE_LIMIT = 20    # AI分析する最大銘柄数（足切りライン）

    def __init__(self):
        # 環境変数からSecretsを取得
        self.secret_ai = self._load_json_secret("APP_SECRET_AI")
        self.secret_wp = self._load_json_secret("APP_SECRET_WORDPRESS")
        
        # Gemini設定
        self.ai_enabled = False
        if self.secret_ai and self.secret_ai.get("api_key"):
            try:
                genai.configure(api_key=self.secret_ai.get("api_key"))
                self.ai_model_name = self.secret_ai.get("model", "gemini-2.5-flash") # 最新推奨モデル
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
                    pass
        except Exception as e:
            print(f"[ERROR] Indices fetch failed: {e}")
        return results

    @staticmethod
    def get_jpx400_mapping():
        """SBI証券からJPX400銘柄リストを取得（失敗時は主要銘柄をフォールバックとして使用）"""
        print("[INFO] Fetching JPX400 List & Names...")
        ticker_map = {}
        
        # 10種類のUser-Agentリスト
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
        ]

        # ランダムにUser-Agentを選択
        selected_ua = random.choice(user_agents)

        headers = {
            "User-Agent": selected_ua,
            "Referer": "https://www.sbisec.co.jp/",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8"
        }

        try:
            res = requests.get(MarketData.JPX400_URL, headers=headers, timeout=15)
            res.encoding = "cp932"
            
            # HTMLが含まれているかチェック
            try:
                dfs = pd.read_html(res.text)
                target_df = None
                for df in dfs:
                    # 銘柄コード(数字4桁)が含まれる列を探す
                    if df.shape[1] >= 2 and df.iloc[:, 0].astype(str).str.match(r'\d{4}').any():
                        target_df = df
                        break
                
                if target_df is not None:
                    codes = target_df.iloc[:, 0].astype(str).str.zfill(4) + ".T"
                    names = target_df.iloc[:, 1].astype(str)
                    ticker_map = dict(zip(codes, names))
            except ValueError:
                print("[WARNING] Could not parse table from response.")

        except Exception as e:
            print(f"[WARNING] JPX400 list fetch failed: {e}")

        # 取得失敗、または0件だった場合のフォールバック（エラーで止まらないように主要銘柄を入れる）
        if not ticker_map:
            print("[INFO] Using fallback ticker list (Major JP Stocks).")
            # 代表的な銘柄（TOPIX Core30等の一部）をハードコードで定義
            fallback_tickers = {
                "7203.T": "トヨタ自動車", "6758.T": "ソニーG", "8306.T": "三菱UFJ", 
                "9984.T": "ソフトバンクG", "9983.T": "ファーストリテイリング", "8035.T": "東京エレクトロン",
                "6861.T": "キーエンス", "9432.T": "NTT", "4063.T": "信越化学", 
                "6098.T": "リクルート", "4502.T": "武田薬品", "7974.T": "任天堂",
                "8316.T": "三井住友FG", "8058.T": "三菱商事", "8001.T": "伊藤忠",
                "6501.T": "日立製作所", "6902.T": "デンソー", "4568.T": "第一三共"
            }
            return fallback_tickers

        return ticker_map

    @staticmethod
    def get_stock_data(tickers):
        """対象銘柄の株価データ取得"""
        print(f"[INFO] Fetching stock data for {len(tickers)} tickers...")
        if not tickers: return pd.DataFrame()
        try:
            # yfinanceエラー回避
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
    def analyze_stocks(ticker_map, data, config):
        """
        ticker_map: {code: name} の辞書
        処理フロー:
        1. 全銘柄スキャンして候補リストを作成
        2. 候補リストをソートし、上位N件(ハードキャップ)に絞る
        3. 上位N件をバッチ(5件ずつ)でAI処理
        4. それ以外はロジック生成
        """
        candidates = []
        if data.empty: return []

        tickers = list(ticker_map.keys())

        # --- 1. スクリーニングとデータ収集 ---
        print("[INFO] Screening stocks...")
        for t in tickers:
            try:
                # データ抽出
                if isinstance(data.columns, pd.MultiIndex):
                    if t not in data.columns.levels[0]: continue
                    df = data[t].copy()
                else:
                    df = data.copy()

                if 'Close' not in df.columns or 'Volume' not in df.columns: continue
                df = df.dropna(subset=["Close", "Volume"])
                if len(df) < 25: continue
                
                # テクニカル指標
                last_close = float(df["Close"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2])
                if prev_close == 0: continue
                
                ret = (last_close - prev_close) / prev_close
                ret_pct = ret * 100
                
                vol_today = float(df["Volume"].iloc[-1])
                vol_avg_20 = df["Volume"].iloc[-22:-2].mean()
                vol_ratio = vol_today / vol_avg_20 if (vol_avg_20 and vol_avg_20 > 0) else 0

                # 判定ロジック
                is_spike = ret >= config.CHANGE_THRESHOLD
                is_drop = ret <= -config.CHANGE_THRESHOLD
                is_active = vol_ratio >= config.VOL_RATIO_THRESHOLD

                if (is_spike or is_drop) and is_active:
                    # 候補として登録（この時点ではAIを呼ばない）
                    ticker_info = Analyzer._get_ticker_info(t)
                    jp_name = ticker_map.get(t, ticker_info.get("longName", t))
                    scores_dict, total_score = Analyzer._calc_scores(ticker_info, ret_pct)
                    judgment = Analyzer._judge_status(total_score, ret_pct)

                    candidates.append({
                        "ticker": t,
                        "name": jp_name,
                        "price": last_close,
                        "change_pct": ret_pct,
                        "volume_ratio": vol_ratio,
                        "scores": scores_dict,
                        "total_score": total_score,
                        "judgment": judgment,
                        "analysis": "" # 後で埋める
                    })

            except Exception as e:
                continue

        if not candidates:
            return []

        # --- 2. 優先順位付けとハードキャップ ---
        # 騰落率の絶対値が大きい順にソート
        candidates.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
        
        # 上位N件をAI対象(target_list)、それ以外を非対象(overflow_list)とする
        target_list = candidates[:config.MAX_ANALYZE_LIMIT]
        overflow_list = candidates[config.MAX_ANALYZE_LIMIT:]

        print(f"[INFO] Candidates found: {len(candidates)}. AI Target: {len(target_list)}, Overflow: {len(overflow_list)}")

        # --- 3. バッチ処理でAI分析 ---
        final_results = []

        # バッチ分割 (例: 5件ずつ)
        batches = [target_list[i:i + config.BATCH_SIZE] for i in range(0, len(target_list), config.BATCH_SIZE)]

        for batch in batches:
            if config.ai_enabled:
                try:
                    print(f"[INFO] Processing batch of {len(batch)} stocks via Gemini...")
                    # バッチ生成実行
                    comments_map = Analyzer._generate_batch_comments(config, batch)
                    
                    # 結果を割り当て
                    for item in batch:
                        # AI生成があればそれを使う、なければロジック生成
                        if item["ticker"] in comments_map:
                            item["analysis"] = comments_map[item["ticker"]]
                        else:
                            item["analysis"] = Analyzer._generate_logic_comment(item["scores"], item["change_pct"])
                    
                    # レート制限対策の待機 (バッチ処理しているので数秒で十分)
                    sleep(4.0)

                except Exception as e:
                    print(f"[ERROR] Batch processing failed: {e}. Fallback to logic.")
                    # バッチ失敗時はロジック生成へフォールバック
                    for item in batch:
                        item["analysis"] = Analyzer._generate_logic_comment(item["scores"], item["change_pct"])
            else:
                # AI無効時
                for item in batch:
                    item["analysis"] = Analyzer._generate_logic_comment(item["scores"], item["change_pct"])
            
            final_results.extend(batch)

        # --- 4. 溢れた分はロジック生成のみで処理 ---
        for item in overflow_list:
            item["analysis"] = Analyzer._generate_logic_comment(item["scores"], item["change_pct"])
            final_results.extend([item])

        return final_results

    @staticmethod
    def _get_ticker_info(ticker):
        try: return yf.Ticker(ticker).info
        except: return {}

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
        if per < 10: s_val += 5
        elif per < 15: s_val += 3
        if pbr < 1.0: s_val += 5
        elif pbr < 1.5: s_val += 3
        s_val = min(s_val, 10)

        # 2. 収益性
        roe = get_val("returnOnEquity", 0)
        margin = get_val("operatingMargins", 0)
        s_prof = 0
        if roe > 0.15: s_prof += 5
        elif roe > 0.08: s_prof += 3
        if margin > 0.10: s_prof += 5
        elif margin > 0.05: s_prof += 3
        s_prof = min(s_prof, 10)

        # 3. 財務健全性
        de_ratio = get_val("debtToEquity", 1000)
        s_fin = 5
        if de_ratio < 50: s_fin = 10
        elif de_ratio < 100: s_fin = 7

        # 4. 成長性
        rev_growth = get_val("revenueGrowth", 0)
        s_grow = 0
        if rev_growth > 0.20: s_grow = 10
        elif rev_growth > 0.10: s_grow = 7
        elif rev_growth > 0.05: s_grow = 5

        # 5. 配当
        yield_val = get_val("dividendYield", 0)
        s_div = 0
        if yield_val and yield_val > 0.04: s_div = 10
        elif yield_val and yield_val > 0.03: s_div = 7
        elif yield_val and yield_val > 0.02: s_div = 5

        # 6. モメンタム
        s_mom = 5
        if abs(ret_pct) > 10: s_mom = 10
        elif abs(ret_pct) > 6: s_mom = 8

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
        if total_score >= 40: return "買い検討"
        elif total_score <= 20: return "売り検討"
        else: return "様子見"

    @staticmethod
    def _generate_batch_comments(config, batch_items):
        """
        複数銘柄(batch_items)の情報をまとめてGeminiに投げ、JSON形式でコメントを受け取る。
        """
        # プロンプト用のデータを構築
        data_text = ""
        for item in batch_items:
            sign = "+" if item['change_pct'] > 0 else ""
            score_str = ", ".join([f"{k}:{v}" for k,v in item['scores'].items()])
            data_text += f"""
            - 銘柄コード: {item['ticker']}
              銘柄名: {item['name']}
              本日の変動: {sign}{round(item['change_pct'], 2)}% (出来高: {round(item['volume_ratio'], 1)}倍)
              スコア: {score_str}
              判定: {item['judgment']}
            """

        prompt = f"""
        あなたはプロの株式アナリストです。
        以下の複数の銘柄について、それぞれのスコアと変動状況に基づき、100文字以内で投資判断コメントを作成してください。
        
        【入力データ】
        {data_text}

        【出力形式】
        **必ず** 正しいJSON形式のみを出力してください。Markdownのコードブロックは不要です。
        キーは「銘柄コード」、値は「コメント」にしてください。
        
        例:
        {{
            "7203.T": "指標面での割安感があり...",
            "9984.T": "ボラティリティが高く..."
        }}
        """

        model = genai.GenerativeModel(config.ai_model_name)
        # JSONモードを明示（モデルが対応していればより確実だが、ここではPrompt指示で対応）
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Markdownの ```json 等を削除してパース
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()
            
        return json.loads(text)

    @staticmethod
    def _generate_logic_comment(scores, ret_pct):
        """AIを使わないロジック生成（フォールバック用）"""
        comments = []
        if scores["割安性"] >= 8: comments.append("指標面での割安感が強く、見直し買いの余地があります。")
        if scores["成長性"] >= 7: comments.append("高い成長性が評価されており、将来性が期待できます。")
        if scores["配当"] >= 7: comments.append("高配当によるインカムゲインの魅力があります。")
        
        if ret_pct > 0:
            comments.append(f"本日は+{round(ret_pct, 1)}%と大きく上昇しました。")
        else:
            comments.append(f"本日は{round(ret_pct, 1)}%と大きく下落しました。")
            
        if not comments:
            comments.append("特段の材料は見当たりませんが、出来高を伴って動意付いています。")
            
        return "".join(comments)

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
        
        n225_c = n225.get('change_pct', 0)
        sp500_c = sp500.get('change_pct', 0)

        # 連続リクエスト回避のため少し待機
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

        base_url = config.secret_wp['url'].rstrip("/")
        if "wp-json" not in base_url:
            base_url = f"{base_url}/wp-json/wp/v2"
        wp_api_url = f"{base_url}/pages/{config.secret_wp['page_id']}"

        today_str = datetime.date.today().strftime("%Y/%m/%d")
        
        # --- HTML構築 ---
        html_parts = []
        html_parts.append(f"<h2>{today_str} 市場概況</h2>")
        html_parts.append(f"<p>{summary_text}</p>")
        
        if not stocks:
            html_parts.append(f"<p><strong>本日は、出来高推移や変動率±{int(config.CHANGE_THRESHOLD*100)}%が当サイト基準を超える急騰・急落銘柄はありませんでした。</strong></p>")
        else:
            html_parts.append(f"<h2>本日の注目銘柄（{len(stocks)}件）</h2>")
            # 優先度順（既にソート済み）で表示
            for s in stocks:
                color = "#d32f2f" if s['change_pct'] > 0 else "#1976d2"
                sign = "+" if s['change_pct'] > 0 else ""
                score_display = " ".join([f"[{k}:{v}]" for k, v in s['scores'].items()])

                judge_color = "#e65100" 
                if s['judgment'] == "買い検討": judge_color = "#d32f2f"
                elif s['judgment'] == "売り検討": judge_color = "#1976d2"

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
    
    # 銘柄リスト取得
    jpx_map = MarketData.get_jpx400_mapping()
    
    if not jpx_map:
        print("[ERROR] No tickers found. Exiting.")
        return

    # データ取得
    codes = list(jpx_map.keys())
    stock_data = MarketData.get_stock_data(codes)
    
    # 分析実行
    target_stocks = Analyzer.analyze_stocks(jpx_map, stock_data, config)
    print(f"[INFO] Analyzed {len(target_stocks)} stocks.")
    
    # AI概況
    summary_text = AIWriter.write_market_summary(indices, config)
    print("[INFO] Market summary generated.")
    
    WordPressPublisher.publish(config, indices, target_stocks, summary_text)
    print("[INFO] All done.")

if __name__ == "__main__":
    main()
