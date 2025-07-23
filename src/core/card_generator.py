"""
カード生成モジュール

PDF→OCR→LaTeX→カード生成の統合処理
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

# プロジェクト内モジュール
from ..utils.config_loader import ConfigLoader
from ..utils.pdf_utils import PDFProcessor
from ..ocr.paddle_processor import PaddleOCRProcessor, PageOCRResult
from ..llm.gpt_client import GPTClient, ConversionRequest, ConversionResult

logger = logging.getLogger(__name__)


@dataclass
class CardGenerationRequest:
    """カード生成リクエスト"""
    pdf_path: str
    output_dir: str
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    chapter_filter: Optional[str] = None


@dataclass
class Card:
    """生成されたカード"""
    type: str          # 定義、定理、命題、補題、系、例、注意
    title: str         # カードのタイトル
    content: str       # LaTeX形式の内容
    front: str         # カードの表面（質問）
    back: str          # カードの裏面（答え）
    confidence: float  # 信頼度 (0.0-1.0)
    source: Dict[str, Any]  # ソース情報（ページ、章など）
    dependencies: List[str] = None  # 依存関係（他のカードへの参照）


@dataclass
class GenerationResult:
    """カード生成結果"""
    cards: List[Card]
    statistics: Dict[str, Any]
    processing_time: float
    errors: List[str]


class CardGenerator:
    """PDFからフラッシュカードを生成するメインクラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: 設定ファイルのパス
        """
        # 設定読み込み
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.config
        
        # 各プロセッサーの初期化
        self.pdf_processor = PDFProcessor(self.config)
        
        try:
            self.ocr_processor = PaddleOCRProcessor(self.config)
        except ImportError as e:
            logger.warning(f"PaddleOCR not available: {e}")
            self.ocr_processor = None
        
        try:
            self.llm_client = GPTClient(self.config)
        except (ImportError, ValueError) as e:
            logger.warning(f"GPT client not available: {e}")
            self.llm_client = None
        
        # 出力設定
        self.output_config = self.config.get('output', {})
        self.obsidian_config = self.output_config.get('obsidian', {})
        
        logger.info("CardGenerator initialized successfully")
    
    def generate_cards(self, request: CardGenerationRequest) -> GenerationResult:
        """PDFからカードを生成
        
        Args:
            request: 生成リクエスト
            
        Returns:
            生成結果
        """
        start_time = time.time()
        errors = []
        
        logger.info(f"Starting card generation for: {request.pdf_path}")
        
        try:
            # 1. PDF処理
            logger.info("Step 1: Processing PDF...")
            pdf_info = self._process_pdf(request)
            if not pdf_info:
                errors.append("PDF processing failed")
                return self._create_error_result(errors, start_time)
            
            # 2. OCR処理
            logger.info("Step 2: Performing OCR...")
            ocr_results = self._perform_ocr(pdf_info['image_paths'])
            if not ocr_results:
                errors.append("OCR processing failed")
                return self._create_error_result(errors, start_time)
            
            # 3. LaTeX変換
            logger.info("Step 3: Converting to LaTeX...")
            latex_results = self._convert_to_latex(ocr_results, pdf_info)
            if not latex_results:
                errors.append("LaTeX conversion failed")
                return self._create_error_result(errors, start_time)
            
            # 4. カード抽出
            logger.info("Step 4: Extracting cards...")
            cards = self._extract_cards(latex_results)
            
            # 5. 後処理
            logger.info("Step 5: Post-processing...")
            processed_cards = self._post_process_cards(cards)
            
            processing_time = time.time() - start_time
            
            # 統計情報作成
            statistics = self._create_statistics(processed_cards, pdf_info, ocr_results, processing_time)
            
            logger.info(f"Card generation completed: {len(processed_cards)} cards in {processing_time:.2f}s")
            
            return GenerationResult(
                cards=processed_cards,
                statistics=statistics,
                processing_time=processing_time,
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Card generation failed: {e}")
            errors.append(str(e))
            return self._create_error_result(errors, start_time)
    
    def _process_pdf(self, request: CardGenerationRequest) -> Optional[Dict[str, Any]]:
        """PDF処理"""
        try:
            # 出力ディレクトリ準備
            output_dir = Path(request.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 画像変換
            image_dir = output_dir / "images"
            image_paths = self.pdf_processor.convert_to_images(
                request.pdf_path, 
                str(image_dir)
            )
            
            # テキスト情報抽出
            pages_info = self.pdf_processor.extract_text_blocks(request.pdf_path)
            
            # 章検出
            chapters = self.pdf_processor.detect_chapters(pages_info)
            
            return {
                'image_paths': image_paths,
                'pages_info': pages_info,
                'chapters': chapters,
                'pdf_path': request.pdf_path
            }
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return None
    
    def _perform_ocr(self, image_paths: List[str]) -> Optional[List[PageOCRResult]]:
        """OCR処理"""
        if not self.ocr_processor:
            logger.error("OCR processor not available")
            return None
            
        try:
            results = self.ocr_processor.process_batch(image_paths)
            return results
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return None
    
    def _convert_to_latex(self, ocr_results: List[PageOCRResult], pdf_info: Dict[str, Any]) -> Optional[List[ConversionResult]]:
        """LaTeX変換"""
        if not self.llm_client:
            logger.error("LLM client not available")
            return None
            
        try:
            requests = []
            
            for page_result in ocr_results:
                # ページ全体のテキストを結合
                full_text = " ".join([result.text for result in page_result.results])
                
                # 章情報の取得
                chapter_info = self._get_chapter_info(page_result.page_number, pdf_info['chapters'])
                
                # 変換リクエスト作成
                request = ConversionRequest(
                    text=full_text,
                    page_number=page_result.page_number,
                    source_info={
                        'page': page_result.page_number,
                        'chapter': chapter_info,
                        'pdf_path': pdf_info['pdf_path']
                    }
                )
                requests.append(request)
            
            # バッチ変換
            results = self.llm_client.process_batch(requests)
            return results
            
        except Exception as e:
            logger.error(f"LaTeX conversion failed: {e}")
            return None
    
    def _extract_cards(self, latex_results: List[ConversionResult]) -> List[Card]:
        """カード抽出"""
        cards = []
        
        for latex_result in latex_results:
            if latex_result.error_message:
                logger.warning(f"Skipping page {latex_result.original_text[:50]}... due to error: {latex_result.error_message}")
                continue
            
            try:
                # GPTクライアントでカード抽出
                source_info = {
                    'page': getattr(latex_result, 'page_number', 0),
                    'pdf_path': getattr(latex_result, 'pdf_path', ''),
                }
                
                card_data_list = self.llm_client.extract_cards(
                    latex_result.latex_content, 
                    source_info
                )
                
                # Card オブジェクトに変換
                for card_data in card_data_list:
                    card = Card(
                        type=card_data.get('type', 'unknown'),
                        title=card_data.get('title', ''),
                        content=card_data.get('content', ''),
                        front=card_data.get('front', ''),
                        back=card_data.get('back', ''),
                        confidence=card_data.get('confidence', 0.0),
                        source=card_data,
                        dependencies=[]  # 後で依存関係解析で設定
                    )
                    cards.append(card)
                    
            except Exception as e:
                logger.warning(f"Failed to extract cards from page: {e}")
                continue
        
        return cards
    
    def _post_process_cards(self, cards: List[Card]) -> List[Card]:
        """カード後処理"""
        # 信頼度によるフィルタリング
        min_confidence = self.config.get('card_generation', {}).get('min_confidence', 0.5)
        filtered_cards = [card for card in cards if card.confidence >= min_confidence]
        
        logger.info(f"Filtered {len(cards) - len(filtered_cards)} low-confidence cards")
        
        # 重複除去（タイトルベース）
        seen_titles = set()
        unique_cards = []
        
        for card in filtered_cards:
            if card.title not in seen_titles:
                unique_cards.append(card)
                seen_titles.add(card.title)
            else:
                logger.debug(f"Removed duplicate card: {card.title}")
        
        logger.info(f"Removed {len(filtered_cards) - len(unique_cards)} duplicate cards")
        
        return unique_cards
    
    def _get_chapter_info(self, page_number: int, chapters: Dict[str, List[int]]) -> str:
        """ページの章情報を取得"""
        for chapter_title, page_list in chapters.items():
            if page_number in page_list:
                return chapter_title
        return "unknown"
    
    def _create_statistics(self, cards: List[Card], pdf_info: Dict[str, Any], 
                          ocr_results: List[PageOCRResult], processing_time: float) -> Dict[str, Any]:
        """統計情報作成"""
        # カードタイプ別集計
        card_types = {}
        total_confidence = 0.0
        
        for card in cards:
            card_types[card.type] = card_types.get(card.type, 0) + 1
            total_confidence += card.confidence
        
        # OCR統計
        total_ocr_items = sum(len(result.results) for result in ocr_results)
        
        return {
            'total_cards': len(cards),
            'card_types': card_types,
            'average_confidence': total_confidence / len(cards) if cards else 0.0,
            'total_pages_processed': len(ocr_results),
            'total_ocr_items': total_ocr_items,
            'processing_time': processing_time,
            'cards_per_page': len(cards) / len(ocr_results) if ocr_results else 0.0
        }
    
    def _create_error_result(self, errors: List[str], start_time: float) -> GenerationResult:
        """エラー結果作成"""
        return GenerationResult(
            cards=[],
            statistics={},
            processing_time=time.time() - start_time,
            errors=errors
        )
    
    def save_cards_to_obsidian(self, cards: List[Card], output_dir: str) -> List[str]:
        """カードをObsidian形式で保存
        
        Args:
            cards: 保存するカード
            output_dir: 出力ディレクトリ
            
        Returns:
            作成されたファイルパスのリスト
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        created_files = []
        
        for card in cards:
            try:
                # ファイル名作成（不正な文字を除去）
                safe_title = "".join(c for c in card.title if c.isalnum() or c in (' ', '-', '_')).strip()
                if not safe_title:
                    safe_title = f"card_{hash(card.content) % 10000}"
                
                filename = f"{card.type}_{safe_title}.md"
                filepath = output_path / filename
                
                # Obsidian-to-Anki形式のコンテンツ作成
                content = self._create_obsidian_content(card)
                
                # ファイル保存
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                created_files.append(str(filepath))
                logger.debug(f"Created card file: {filename}")
                
            except Exception as e:
                logger.warning(f"Failed to save card {card.title}: {e}")
                continue
        
        logger.info(f"Saved {len(created_files)} cards to Obsidian format")
        return created_files
    
    def _create_obsidian_content(self, card: Card) -> str:
        """Obsidian-to-Anki形式のコンテンツ作成"""
        # メタデータ
        metadata = [
            f"# {card.title}",
            "",
            f"Type: {card.type}",
            f"Confidence: {card.confidence:.2f}",
            f"Source: Page {card.source.get('page', 'unknown')}",
            "",
            "---",
            ""
        ]
        
        # フラッシュカード部分
        flashcard = [
            "START",
            card.front,
            "BACK",
            card.back,
            "END",
            "",
            "---",
            "",
            "## Full Content",
            "",
            card.content
        ]
        
        return "\n".join(metadata + flashcard)