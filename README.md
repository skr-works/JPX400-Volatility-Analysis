# JPX400 AI Market Analyzer & Publisher

JPX400採用銘柄の急騰・急落銘柄を抽出、分析コメントスクリプトです。

## 機能概要

* **自動スクリーニング**: JPX400銘柄の株価データを分析します。
* **急騰・急落検知**:
    * 前日比 ±6% 以上の変動
    * 出来高倍率 1.5倍 以上
* **休場日対応**: `jpholiday` ライブラリを使用し、日本の祝日は実行をスキップします。

## インストール

必要なライブラリをインストールします。

```bash
pip install requests pandas yfinance jpholiday google-generativeai lxml html5lib

```

## 実行方法

環境変数を設定した状態でスクリプトを実行します。

```bash
python main.py

```
