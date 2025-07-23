# PDF2Anki - 数学書PDF自動カード化システム

数学書のPDFから定義・定理・命題を自動抽出し、Obsidian + Anki対応のフラッシュカードを生成するシステム

## 概要

特定のPDF（数学書・教材）をインプットとして、以下を自動化：

1. **PDF解析**: 画像化されたPDFから定義・定理・命題を検出
2. **OCR & LaTeX変換**: 数式を含むテキストをLaTeX形式で抽出
3. **カード生成**: Obsidian to Ankiプラグイン対応形式で出力
4. **依存関係管理**: 概念間の前提関係を自動解析・タグ付け

## 特徴

- 📚 **数学特化**: 数式・定理・証明に特化したOCR処理
- 🔗 **依存関係可視化**: ObsidianのグラフビューでKnowledge Graph構築
- 🎯 **高精度**: PaddleOCR + GPT-4oによる最新技術スタック
- ⚡ **バッチ処理**: 大容量PDF対応の文脈保持チャンク分割
- 🎨 **カスタマイズ可能**: プロンプト・テンプレート調整対応

## 技術スタック

### OCR・数式認識
- **PaddleOCR 3.x + PP-FormulaNet-L**: 高精度数式認識
- **Pix2Text v1.1**: 軽量版OCR（CPU対応）
- **pdf2image + PyMuPDF**: PDF前処理

### LLM・変換層
- **OpenAI GPT-4o**: 最高精度のテキスト→LaTeX変換
- **Claude 3.5 Sonnet**: 長文コンテキスト対応
- **Self-refine**: エラー修正ループ

### 出力・連携
- **Obsidian to Anki**: フラッシュカード形式出力
- **YAML Frontmatter**: メタデータ管理
- **依存関係グラフ**: 自動リンク生成

## 出力カード例

```markdown
---
title: "ボルツァーノ・ワイエルシュトラス定理"
type: "定理"
source: "解析入門"
chapter: "第3章 数列と級数"
page: 42
requires:
  - "[[コンパクト集合の定義]]"
  - "[[有界閉集合の性質]]"
tags: [解析学, 実数論, 収束]
---

START
解析学_基礎
Front: 有界な数列に関するボルツァーノ・ワイエルシュトラス定理を述べよ
Back: $\mathbb{R}$における有界な数列は収束する部分列を持つ
Tags: 解析学, 定理, 収束
END

**定理の詳細:**
$\{a_n\}$を有界な実数列とする。このとき、$\{a_n\}$の部分列$\{a_{n_k}\}$で収束するものが存在する。

**前提条件:**
- [[コンパクト集合の定義]]
- [[有界閉集合の性質]]
```

## システム構成

```
pdf2anki/
├── src/
│   ├── ocr/                    # OCR・数式認識層
│   │   ├── paddle_processor.py
│   │   └── pix2text_processor.py
│   ├── llm/                    # LLM変換層
│   │   ├── gpt_client.py
│   │   ├── claude_client.py
│   │   └── self_refine.py
│   ├── core/                   # コア機能
│   │   ├── chunker.py
│   │   ├── card_generator.py
│   │   └── dependency_analyzer.py
│   └── utils/                  # ユーティリティ
│       ├── pdf_utils.py
│       └── latex_validator.py
├── config/                     # 設定ファイル
│   ├── models.yaml
│   └── prompts/
├── templates/                  # 出力テンプレート
│   └── obsidian_card.md
└── tests/                      # テスト
```

## クイックスタート

```bash
# 1. インストール
git clone https://github.com/[username]/pdf2anki.git
cd pdf2anki
pip install -r requirements.txt

# 2. 設定
cp config/settings.example.yaml config/settings.yaml
# API keys等を設定

# 3. 実行
python -m pdf2anki process sample.pdf --output cards/
```

## 使用例

```python
from pdf2anki import PDF2AnkiProcessor

processor = PDF2AnkiProcessor(
    ocr_model="paddle",      # "paddle" or "pix2text"
    llm_model="gpt-4o",      # "gpt-4o" or "claude-3.5"
    output_format="obsidian"  # 出力形式
)

# PDF処理
cards = processor.process("math_textbook.pdf")

# Obsidian vault保存
processor.save_to_obsidian(cards, "path/to/obsidian/vault")
```

## 開発状況

- [ ] 基本プロジェクト構造
- [ ] PDF前処理機能
- [ ] OCR実装（PaddleOCR）
- [ ] LLM連携（GPT-4o）
- [ ] カード生成機能
- [ ] 依存関係解析
- [ ] Obsidian出力
- [ ] テスト・ドキュメント

## ライセンス

MIT License

## 貢献

Issue・PRお待ちしています！

## 参考文献

- [PaddleOCR Documentation](https://paddlepaddle.github.io/PaddleOCR/)
- [Obsidian to Anki Plugin](https://github.com/ObsidianToAnki/Obsidian_to_Anki)
- [Pix2Text](https://github.com/breezedeus/Pix2Text)