#!/usr/bin/env python3
"""
カスタムPDFテスト用スクリプト

自分のPDFファイルでPDF2Ankiシステムをテストできます
"""

import os
import sys
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def test_custom_pdf(pdf_path: str):
    """カスタムPDFでテスト
    
    Args:
        pdf_path: PDFファイルのパス
    """
    logger = logging.getLogger(__name__)
    
    # ファイル存在確認
    if not Path(pdf_path).exists():
        logger.error(f"❌ PDFファイルが見つかりません: {pdf_path}")
        return False
    
    try:
        from src.utils.config_loader import ConfigLoader
        from src.utils.pdf_utils import PDFProcessor
        from src.llm.gpt_client import GPTClient, ConversionRequest
        from src.core.card_generator import CardGenerator
        
        logger.info(f"🔍 テスト対象PDF: {pdf_path}")
        
        # 設定読み込み
        logger.info("⚙️  設定を読み込んでいます...")
        config_loader = ConfigLoader()
        config = config_loader.config
        
        # PDF処理
        logger.info("📄 PDFを処理しています...")
        pdf_processor = PDFProcessor(config)
        pages_info = pdf_processor.extract_text_blocks(pdf_path)
        
        logger.info(f"✅ {len(pages_info)}ページからテキストを抽出しました")
        
        # 最初のページの内容をサンプル表示
        if pages_info:
            sample_text = ""
            for block in pages_info[0].text_blocks[:3]:  # 最初の3ブロック
                if hasattr(block, 'text'):
                    sample_text += block.text + " "
                elif isinstance(block, dict):
                    sample_text += block.get('text', '') + " "
            
            logger.info(f"📝 サンプルテキスト: {sample_text[:200]}...")
        
        # GPTでLaTeX変換
        logger.info("🤖 LaTeX変換を実行しています...")
        gpt_client = GPTClient(config)
        
        # 実際の抽出テキストまたはサンプルテキストを使用
        if sample_text.strip():
            conversion_text = sample_text
        else:
            conversion_text = """
            定義 1.1 (実数)
            実数とは、有理数と無理数を合わせた数の集合である。
            実数全体の集合を R で表す。
            """
        
        request = ConversionRequest(
            text=conversion_text,
            page_number=1,
            source_info={'file': Path(pdf_path).name, 'page': 1}
        )
        
        result = gpt_client.convert_to_latex(request)
        
        if result.error_message:
            logger.error(f"❌ LaTeX変換に失敗: {result.error_message}")
            return False
        
        logger.info("✅ LaTeX変換が成功しました！")
        logger.info(f"📋 LaTeX内容: {result.latex_content[:300]}...")
        
        # カード抽出
        logger.info("🃏 フラッシュカードを抽出しています...")
        cards = gpt_client.extract_cards(result.latex_content, {'file': Path(pdf_path).name})
        
        logger.info(f"✅ {len(cards)}枚のカードが抽出されました！")
        
        # カード詳細表示
        for i, card in enumerate(cards, 1):
            logger.info(f"📇 カード {i}:")
            logger.info(f"   タイプ: {card.get('type', 'unknown')}")
            logger.info(f"   タイトル: {card.get('title', 'unknown')}")
            logger.info(f"   信頼度: {card.get('confidence', 0.0)}")
            logger.info(f"   表: {card.get('front', '')[:60]}...")
            logger.info(f"   裏: {card.get('back', '')[:60]}...")
            logger.info("")
        
        # Obsidian形式で保存
        output_dir = project_root / "custom_test_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        for i, card in enumerate(cards, 1):
            filename = f"card_{i}_{card.get('title', 'untitled').replace(' ', '_')}.md"
            filepath = output_dir / filename
            
            content = f"""# {card.get('title', 'Untitled')}

Type: {card.get('type', 'unknown')}
Confidence: {card.get('confidence', 0.0):.2f}
Source: {card.get('file', 'unknown')}

---

START
{card.get('front', '')}
BACK
{card.get('back', '')}
END

---

## Full Content

{card.get('content', '')}
"""
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            saved_files.append(str(filepath))
        
        logger.info(f"💾 {len(saved_files)}枚のカードを保存しました: {output_dir}")
        
        logger.info("🎉 カスタムPDFテストが完了しました！")
        logger.info(f"📁 出力ディレクトリ: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ テストに失敗しました: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """メイン処理"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # コマンドライン引数からPDFパスを取得
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # デフォルトのPDFを使用
        pdf_path = str(project_root / "20250424_位相1-2.pdf")
        logger.info(f"PDFパスが指定されていません。デフォルトを使用: {pdf_path}")
    
    logger.info("🚀 カスタムPDF2Ankiテスト開始")
    logger.info("=" * 50)
    
    success = test_custom_pdf(pdf_path)
    
    if success:
        logger.info("=" * 50)
        logger.info("✅ テスト完了！")
        logger.info("📁 'custom_test_output' ディレクトリで結果を確認してください")
    else:
        logger.error("❌ テストに失敗しました")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)