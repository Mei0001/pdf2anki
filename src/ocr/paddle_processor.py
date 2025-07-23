"""
PaddleOCR処理モジュール

PaddleOCRを使用したテキスト認識と数式認識機能
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import json
from dataclasses import dataclass

import numpy as np
import cv2
from PIL import Image

# PaddleOCRのインポート（遅延インポートでエラーハンドリング）
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    logging.warning("PaddleOCR not available. Please install with: pip install paddleocr")

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """OCR認識結果"""
    bbox: List[float]  # [x1, y1, x2, y2, x3, y3, x4, y4] 
    text: str
    confidence: float
    type: str = "text"  # "text" or "formula"


@dataclass
class PageOCRResult:
    """ページ単位のOCR結果"""
    page_number: int
    image_path: str
    results: List[OCRResult]
    processing_time: float
    total_confidence: float


class PaddleOCRProcessor:
    """PaddleOCR処理クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: OCR設定
        """
        if not PADDLE_AVAILABLE:
            raise ImportError("PaddleOCR is not installed. Please install with: pip install paddleocr")
        
        self.config = config
        self.ocr_config = config.get('ocr', {})
        self.paddle_config = self.ocr_config.get('paddle', {})
        self.formula_config = self.ocr_config.get('formula', {})
        
        # PaddleOCRインスタンスの初期化
        self._init_ocr_engines()
        
        # キャッシュ設定
        self.cache_enabled = config.get('cache', {}).get('ocr_cache', True)
        self.cache_dir = Path(config.get('cache', {}).get('directory', 'cache'))
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_ocr_engines(self):
        """OCRエンジンの初期化"""
        try:
            # テキストOCR初期化
            self.text_ocr = PaddleOCR(
                use_angle_cls=self.paddle_config.get('use_angle_cls', True),
                lang=self.paddle_config.get('lang', 'ja+en'),
                use_gpu=self.paddle_config.get('use_gpu', False),
                det_model_dir=None,  # デフォルトモデル使用
                rec_model_dir=None,
                cls_model_dir=None,
                show_log=False
            )
            
            # 数式OCRの初期化（PP-FormulaNetが利用可能な場合）
            self.formula_ocr = None
            if self.formula_config.get('enabled', True):
                try:
                    # 数式認識モジュールの初期化を試行
                    from paddleocr import PPFormula
                    model_name = self.formula_config.get('model', 'PP-FormulaNet-L')
                    self.formula_ocr = PPFormula(model_name=model_name)
                    logger.info(f"Formula OCR initialized with model: {model_name}")
                except Exception as e:
                    logger.warning(f"Formula OCR initialization failed: {e}")
                    logger.info("Falling back to text OCR for formula regions")
            
            logger.info("PaddleOCR engines initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OCR engines: {e}")
            raise
    
    def process_image(self, image_path: str, page_number: int = 1) -> PageOCRResult:
        """画像をOCR処理
        
        Args:
            image_path: 画像ファイルパス
            page_number: ページ番号
            
        Returns:
            ページOCR結果
        """
        start_time = time.time()
        
        # キャッシュチェック
        if self.cache_enabled:
            cached_result = self._load_cache(image_path)
            if cached_result:
                logger.info(f"Loaded OCR result from cache for {image_path}")
                return cached_result
        
        try:
            # 画像読み込み
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # テキストOCR実行
            text_results = self._run_text_ocr(image)
            
            # 数式OCR実行（利用可能な場合）
            formula_results = []
            if self.formula_ocr:
                formula_results = self._run_formula_ocr(image)
            
            # 結果の統合とソート
            all_results = text_results + formula_results
            all_results = self._sort_results_by_position(all_results)
            
            # 総合信頼度の計算
            total_confidence = self._calculate_total_confidence(all_results)
            
            processing_time = time.time() - start_time
            
            result = PageOCRResult(
                page_number=page_number,
                image_path=image_path,
                results=all_results,
                processing_time=processing_time,
                total_confidence=total_confidence
            )
            
            # キャッシュ保存
            if self.cache_enabled:
                self._save_cache(image_path, result)
            
            logger.info(f"OCR completed for page {page_number}: {len(all_results)} items, "
                       f"confidence: {total_confidence:.3f}, time: {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"OCR processing failed for {image_path}: {e}")
            raise
    
    def _run_text_ocr(self, image: np.ndarray) -> List[OCRResult]:
        """テキストOCR実行"""
        try:
            # PaddleOCRでテキスト認識
            ocr_results = self.text_ocr.ocr(image, cls=True)
            
            results = []
            if ocr_results and ocr_results[0]:
                for detection in ocr_results[0]:
                    bbox = detection[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text_info = detection[1]  # (text, confidence)
                    
                    if text_info and len(text_info) == 2:
                        text, confidence = text_info
                        
                        # bboxを平坦化 [x1, y1, x2, y2, x3, y3, x4, y4]
                        flat_bbox = [coord for point in bbox for coord in point]
                        
                        # 数式パターンを検出
                        is_formula = self._detect_formula_pattern(text)
                        
                        result = OCRResult(
                            bbox=flat_bbox,
                            text=text.strip(),
                            confidence=confidence,
                            type="formula" if is_formula else "text"
                        )
                        
                        results.append(result)
            
            logger.debug(f"Text OCR found {len(results)} items")
            return results
            
        except Exception as e:
            logger.error(f"Text OCR execution failed: {e}")
            return []
    
    def _run_formula_ocr(self, image: np.ndarray) -> List[OCRResult]:
        """数式OCR実行"""
        if not self.formula_ocr:
            return []
        
        try:
            # 数式領域の検出と認識
            formula_results = self.formula_ocr(image)
            
            results = []
            if formula_results:
                for result in formula_results:
                    # PP-FormulaNetの結果形式に応じて処理
                    if isinstance(result, dict):
                        bbox = result.get('bbox', [])
                        latex_text = result.get('text', '')
                        confidence = result.get('confidence', 0.0)
                        
                        if bbox and latex_text:
                            ocr_result = OCRResult(
                                bbox=bbox,
                                text=latex_text,
                                confidence=confidence,
                                type="formula"
                            )
                            results.append(ocr_result)
            
            logger.debug(f"Formula OCR found {len(results)} formulas")
            return results
            
        except Exception as e:
            logger.error(f"Formula OCR execution failed: {e}")
            return []
    
    def _detect_formula_pattern(self, text: str) -> bool:
        """テキストが数式かどうかを判定
        
        Args:
            text: 認識されたテキスト
            
        Returns:
            数式判定結果
        """
        import re
        
        # 数式パターンの定義
        formula_patterns = [
            r'\$.*?\$',              # LaTeX インライン数式
            r'\\[a-zA-Z]+',          # LaTeX コマンド
            r'[∫∑∏∆∇∂∞α-ωΑ-Ω]',    # 数学記号
            r'[{}^_]',               # 上下付き文字、ブレース
            r'\b\d+[+\-*/=]\d+',     # 簡単な数式
            r'[xy]\^[0-9]+',         # 指数表記
            r'√\d+',                 # 平方根
            r'[≤≥≠≈≡]',             # 関係演算子
        ]
        
        # パターンマッチング
        for pattern in formula_patterns:
            if re.search(pattern, text):
                return True
        
        # 数学的文脈の単語
        math_keywords = [
            '定理', '補題', '命題', '証明', '定義', '系',
            'lim', 'sin', 'cos', 'tan', 'log', 'ln', 'exp',
            'max', 'min', 'sup', 'inf'
        ]
        
        for keyword in math_keywords:
            if keyword in text:
                return True
        
        return False
    
    def _sort_results_by_position(self, results: List[OCRResult]) -> List[OCRResult]:
        """結果を画像上の位置でソート（上から下、左から右）"""
        def get_center_y(result: OCRResult) -> float:
            bbox = result.bbox
            # bbox: [x1, y1, x2, y2, x3, y3, x4, y4]
            y_coords = [bbox[1], bbox[3], bbox[5], bbox[7]]
            return sum(y_coords) / 4
        
        def get_center_x(result: OCRResult) -> float:
            bbox = result.bbox
            x_coords = [bbox[0], bbox[2], bbox[4], bbox[6]]
            return sum(x_coords) / 4
        
        # Y座標でソート、同じ行は左から右へ
        return sorted(results, key=lambda r: (get_center_y(r), get_center_x(r)))
    
    def _calculate_total_confidence(self, results: List[OCRResult]) -> float:
        """総合信頼度を計算"""
        if not results:
            return 0.0
        
        # 文字数で重み付けした信頼度の計算
        total_weighted_conf = 0.0
        total_weight = 0.0
        
        for result in results:
            weight = len(result.text) if result.text else 1
            total_weighted_conf += result.confidence * weight
            total_weight += weight
        
        return total_weighted_conf / total_weight if total_weight > 0 else 0.0
    
    def process_batch(self, image_paths: List[str]) -> List[PageOCRResult]:
        """複数画像のバッチ処理
        
        Args:
            image_paths: 画像ファイルパスのリスト
            
        Returns:
            ページOCR結果のリスト
        """
        results = []
        
        logger.info(f"Starting batch OCR processing for {len(image_paths)} images")
        
        for i, image_path in enumerate(image_paths, 1):
            try:
                result = self.process_image(image_path, page_number=i)
                results.append(result)
                
                # 進捗ログ
                if i % 10 == 0 or i == len(image_paths):
                    logger.info(f"Processed {i}/{len(image_paths)} images")
                    
            except Exception as e:
                logger.error(f"Failed to process image {i}: {image_path} - {e}")
                # エラーがあっても続行
                continue
        
        logger.info(f"Batch processing completed: {len(results)}/{len(image_paths)} successful")
        return results
    
    def _get_cache_path(self, image_path: str) -> Path:
        """キャッシュファイルパスを生成"""
        import hashlib
        
        # 画像パスとタイムスタンプでハッシュ生成
        image_stat = os.stat(image_path)
        cache_key = f"{image_path}_{image_stat.st_mtime}_{image_stat.st_size}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        
        return self.cache_dir / f"ocr_{cache_hash}.json"
    
    def _save_cache(self, image_path: str, result: PageOCRResult):
        """OCR結果をキャッシュに保存"""
        try:
            cache_path = self._get_cache_path(image_path)
            
            # データクラスを辞書に変換
            cache_data = {
                'page_number': result.page_number,
                'image_path': result.image_path,
                'processing_time': result.processing_time,
                'total_confidence': result.total_confidence,
                'results': [
                    {
                        'bbox': r.bbox,
                        'text': r.text,
                        'confidence': r.confidence,
                        'type': r.type
                    }
                    for r in result.results
                ]
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
            logger.debug(f"Saved OCR cache: {cache_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save OCR cache: {e}")
    
    def _load_cache(self, image_path: str) -> Optional[PageOCRResult]:
        """キャッシュからOCR結果を読み込み"""
        try:
            cache_path = self._get_cache_path(image_path)
            
            if not cache_path.exists():
                return None
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # 辞書からデータクラスに変換
            results = [
                OCRResult(
                    bbox=r['bbox'],
                    text=r['text'], 
                    confidence=r['confidence'],
                    type=r['type']
                )
                for r in cache_data['results']
            ]
            
            return PageOCRResult(
                page_number=cache_data['page_number'],
                image_path=cache_data['image_path'],
                results=results,
                processing_time=cache_data['processing_time'],
                total_confidence=cache_data['total_confidence']
            )
            
        except Exception as e:
            logger.warning(f"Failed to load OCR cache: {e}")
            return None
    
    def get_statistics(self, results: List[PageOCRResult]) -> Dict[str, Any]:
        """OCR結果の統計情報を取得
        
        Args:
            results: ページOCR結果のリスト
            
        Returns:
            統計情報
        """
        if not results:
            return {}
        
        total_items = sum(len(r.results) for r in results)
        total_text_items = sum(len([item for item in r.results if item.type == "text"]) for r in results)
        total_formula_items = sum(len([item for item in r.results if item.type == "formula"]) for r in results)
        
        avg_confidence = sum(r.total_confidence for r in results) / len(results)
        total_processing_time = sum(r.processing_time for r in results)
        
        return {
            'total_pages': len(results),
            'total_items': total_items,
            'text_items': total_text_items,
            'formula_items': total_formula_items,
            'average_confidence': avg_confidence,
            'total_processing_time': total_processing_time,
            'average_time_per_page': total_processing_time / len(results),
            'items_per_page': total_items / len(results)
        }