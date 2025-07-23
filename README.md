# PDF2Anki - 数学書PDF自動カード化システム

数学書のPDFから定義・定理・命題を自動抽出し、Obsidian + Anki対応のフラッシュカードを生成するシステム

## 概要

数学のPDFファイルを入力として、以下の処理を自動化します：

1. **PDF解析**: PDFからテキストブロックを抽出
2. **LaTeX変換**: GPT-4oを使用してOCRテキストを高品質なLaTeX形式に変換
3. **カード生成**: 定義・定理・命題を自動検出してフラッシュカード化
4. **Obsidian出力**: Obsidian-to-Ankiプラグイン対応形式で保存

## 特徴

- 📚 **数学特化**: 数式・定理・証明に特化したLaTeX変換
- 🤖 **AI高精度**: OpenAI GPT-4oによる最新技術での変換精度
- ⚡ **高速処理**: キャッシュ機能による効率的な処理
- 🎯 **実用的**: 実際のObsidianとAnkiで使用可能な形式で出力
- 🛠 **カスタマイズ可能**: プロンプトテンプレートの調整が可能

## 技術スタック

### 主要技術
- **OpenAI GPT-4o**: 高精度LaTeX変換とカード抽出
- **PyMuPDF**: PDF処理とテキスト抽出
- **PaddleOCR**: OCR処理（数式認識対応）
- **Self-refine**: LaTeX構文自動検証・修正

### 出力形式
- **Obsidian-to-Anki形式**: START/BACK/END区切りでの出力
- **Markdown + YAML**: メタデータ付きの構造化出力

## インストール

```bash
# リポジトリのクローン
git clone https://github.com/[username]/pdf2anki.git
cd pdf2anki

# 依存関係のインストール
pip install -r requirements.txt

# 設定ファイルのセットアップ
cp config/settings.example.yaml config/settings.yaml
```

## 設定

`config/settings.yaml` でOpenAI APIキーを設定：

```yaml
api_keys:
  openai: "your-openai-api-key-here"

llm:
  openai:
    model: "gpt-4o"
    max_tokens: 4000
    temperature: 0.1
```

## 使用方法

### 基本的な使用方法

```bash
# メインデモの実行（サンプルPDFを使用）
python final_demo.py

# カスタムPDFでテスト
python custom_test.py "your_math_book.pdf"
```

### プログラムでの使用

```python
from src.utils.config_loader import ConfigLoader
from src.llm.gpt_client import GPTClient, ConversionRequest
from src.utils.pdf_utils import PDFProcessor

# 設定読み込み
config = ConfigLoader().config

# PDF処理
pdf_processor = PDFProcessor(config)
pages_info = pdf_processor.extract_text_blocks("math_book.pdf")

# GPTクライアント初期化
gpt_client = GPTClient(config)

# LaTeX変換
request = ConversionRequest(
    text="サンプルテキスト",
    page_number=1,
    source_info={'chapter': '第1章', 'page': 1}
)
result = gpt_client.convert_to_latex(request)

# カード抽出
cards = gpt_client.extract_cards(result.latex_content, {'page': 1})
```

## 出力例

システムが生成するObsidian形式のカードファイル例：

```markdown
# 開集合

Type: 定義
Confidence: 0.95
Source: Page 1

---

START
距離空間における開集合とは何ですか？
BACK
距離空間 $(X, d)$ において、集合 $U \subseteq X$ が開集合であるとは、任意の点 $x \in U$ に対して、ある $\epsilon > 0$ が存在し、$B(x, \epsilon) \subseteq U$ となることをいう。
END

---

## Full Content

距離空間 $(X, d)$ において、集合 $U \subseteq X$ が開集合であるとは、任意の点 $x \in U$ に対して、ある $\epsilon > 0$ が存在し、$B(x, \epsilon) \subseteq U$ となることをいう。
```

## プロジェクト構成

```
pdf2anki/
├── src/
│   ├── core/
│   │   └── card_generator.py      # メインのカード生成機能
│   ├── llm/
│   │   └── gpt_client.py          # OpenAI GPT-4o クライアント
│   ├── ocr/
│   │   └── paddle_processor.py    # PaddleOCR処理
│   └── utils/
│       ├── config_loader.py       # 設定管理
│       └── pdf_utils.py           # PDF処理ユーティリティ
├── config/
│   ├── settings.yaml              # メイン設定ファイル
│   ├── settings.example.yaml      # 設定ファイルのテンプレート
│   └── prompts/                   # GPTプロンプトテンプレート
│       ├── latex_conversion.txt   # LaTeX変換プロンプト
│       └── card_extraction.txt    # カード抽出プロンプト
├── tests/                         # ユニットテスト
├── final_demo.py                  # メインデモスクリプト
├── custom_test.py                 # カスタムテスト用
└── requirements.txt               # 依存関係
```

## テスト方法

### 1. 基本テスト
```bash
# 全体の動作確認
python final_demo.py
```

### 2. カスタムPDFテスト
```bash
# 自分のPDFファイルでテスト
python custom_test.py "path/to/your/math_book.pdf"
```

### 3. コンポーネント別テスト
```bash
# ユニットテストの実行
python -m pytest tests/

# 特定のテストのみ実行
python -m pytest tests/test_gpt_client.py
```

## 動作確認済み環境

- Python 3.8+
- macOS (Darwin 24.5.0)
- OpenAI API (GPT-4o)

## 制限事項

- OpenAI APIキーが必要
- 数式を含む日本語テキストに特化
- PaddleOCRの初期化に時間がかかる場合がある

## トラブルシューティング

### よくある問題

1. **OpenAI API エラー**
   - API キーが正しく設定されているか確認
   - API利用制限に達していないか確認

2. **PaddleOCR初期化エラー**
   - モデルのダウンロードに時間がかかる場合があります
   - ネットワーク接続を確認してください

3. **依存関係エラー**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## ライセンス

MIT License

## 開発状況

- ✅ 基本プロジェクト構造
- ✅ PDF前処理機能
- ✅ OCR実装（PaddleOCR）
- ✅ LLM連携（GPT-4o）
- ✅ カード生成機能
- ✅ Obsidian形式出力
- ✅ テスト・デモスクリプト
- 🔄 依存関係解析（部分実装）
- 🔄 CLI インターフェース（今後実装予定）

## 貢献

Issue・PRお待ちしています！

## 参考文献

- [OpenAI GPT-4o Documentation](https://platform.openai.com/docs)
- [PaddleOCR Documentation](https://paddlepaddle.github.io/PaddleOCR/)
- [Obsidian to Anki Plugin](https://github.com/ObsidianToAnki/Obsidian_to_Anki)
- [PyMuPDF Documentation](https://pymupdf.readthedocs.io/)