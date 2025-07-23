"""
PaddleOCR処理モジュールのテスト
"""

import os
import pytest
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# プロジェクトルートをsys.pathに追加
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr.paddle_processor import PaddleOCRProcessor, OCRResult, PageOCRResult
from src.utils.config_loader import ConfigLoader


class TestPaddleOCRProcessor:
    """PaddleOCRProcessor クラスのテスト"""
    
    def setup_method(self):
        """各テストメソッド実行前の準備"""
        self.test_dir = tempfile.mkdtemp()
        
        # テスト用設定
        self.test_config = {
            'ocr': {
                'engine': 'paddle',
                'paddle': {
                    'use_angle_cls': True,
                    'lang': 'ja+en',
                    'use_gpu': False,
                    'det_model': 'PP-OCRv4',
                    'rec_model': 'PP-OCRv4'
                },
                'formula': {
                    'enabled': True,
                    'model': 'PP-FormulaNet-L'
                }
            },
            'cache': {
                'ocr_cache': True,
                'directory': os.path.join(self.test_dir, 'cache')
            }
        }
        
    def teardown_method(self):
        """各テストメソッド実行後のクリーンアップ"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('src.ocr.paddle_processor.PADDLE_AVAILABLE', True)
    @patch('paddleocr.PaddleOCR')
    def test_processor_initialization(self, mock_paddle_ocr):
        """プロセッサー初期化のテスト"""
        mock_ocr_instance = MagicMock()
        mock_paddle_ocr.return_value = mock_ocr_instance
        
        processor = PaddleOCRProcessor(self.test_config)
        
        # 初期化パラメータの確認
        assert processor.cache_enabled == True
        assert mock_paddle_ocr.called
        
        # 初期化引数の確認
        call_args = mock_paddle_ocr.call_args
        assert call_args[1]['use_angle_cls'] == True
        assert call_args[1]['lang'] == 'ja+en'
        assert call_args[1]['use_gpu'] == False
    
    @patch('src.ocr.paddle_processor.PADDLE_AVAILABLE', False)
    def test_processor_initialization_without_paddle(self):
        """PaddleOCRが利用できない場合の初期化テスト"""
        with pytest.raises(ImportError, match="PaddleOCR is not installed"):
            PaddleOCRProcessor(self.test_config)
    
    @patch('src.ocr.paddle_processor.PADDLE_AVAILABLE', True)
    @patch('paddleocr.PaddleOCR')
    @patch('cv2.imread')
    @patch('os.path.exists')
    def test_process_image_success(self, mock_exists, mock_imread, mock_paddle_ocr):
        """画像処理成功ケースのテスト"""
        # モックの設定
        mock_exists.return_value = True
        mock_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        mock_imread.return_value = mock_image
        
        # OCR結果のモック
        mock_ocr_instance = MagicMock()
        mock_ocr_results = [
            [
                [
                    [[10, 20], [100, 20], [100, 50], [10, 50]],  # bbox
                    ['定理 1.1', 0.95]  # text, confidence
                ],
                [
                    [[10, 60], [200, 60], [200, 90], [10, 90]],
                    ['実数の連続性について', 0.88]
                ]
            ]
        ]
        mock_ocr_instance.ocr.return_value = mock_ocr_results
        mock_paddle_ocr.return_value = mock_ocr_instance
        
        processor = PaddleOCRProcessor(self.test_config)
        
        # テスト用画像ファイル
        test_image = os.path.join(self.test_dir, "test.png")
        
        # 画像処理実行
        result = processor.process_image(test_image, page_number=1)
        
        # 結果検証
        assert isinstance(result, PageOCRResult)
        assert result.page_number == 1
        assert result.image_path == test_image
        assert len(result.results) == 2
        
        # 最初の結果を詳細確認
        first_result = result.results[0]
        assert isinstance(first_result, OCRResult)
        assert first_result.text == '定理 1.1'
        assert first_result.confidence == 0.95
        assert first_result.bbox == [10, 20, 100, 20, 100, 50, 10, 50]
    
    @patch('src.ocr.paddle_processor.PADDLE_AVAILABLE', True)
    @patch('paddleocr.PaddleOCR')
    def test_detect_formula_pattern(self, mock_paddle_ocr):
        """数式パターン検出のテスト"""
        mock_paddle_ocr.return_value = MagicMock()
        processor = PaddleOCRProcessor(self.test_config)
        
        # 数式パターンのテストケース
        formula_texts = [
            '$x + y = z$',           # LaTeX インライン数式
            '\\int_0^1 x dx',        # LaTeX コマンド
            'sin(x) + cos(y)',       # 三角関数
            'α + β = γ',             # ギリシャ文字
            '2^3 = 8',               # 指数表記
            '√16 = 4',               # 平方根
            'x ≤ y',                 # 不等式
            '定理 1.1',              # 数学文脈
        ]
        
        non_formula_texts = [
            '通常のテキスト',
            'Hello World',
            '今日は良い天気です',
            'Python programming'
        ]
        
        # 数式パターンの検証
        for text in formula_texts:
            assert processor._detect_formula_pattern(text), f"Should detect as formula: {text}"
        
        # 非数式パターンの検証
        for text in non_formula_texts:
            assert not processor._detect_formula_pattern(text), f"Should not detect as formula: {text}"
    
    @patch('src.ocr.paddle_processor.PADDLE_AVAILABLE', True)
    @patch('paddleocr.PaddleOCR')
    def test_sort_results_by_position(self, mock_paddle_ocr):
        """結果位置ソートのテスト"""
        mock_paddle_ocr.return_value = MagicMock()
        processor = PaddleOCRProcessor(self.test_config)
        
        # テスト用OCR結果（位置がバラバラ）
        results = [
            OCRResult(bbox=[100, 100, 200, 100, 200, 130, 100, 130], text="下段左", confidence=0.9),
            OCRResult(bbox=[10, 50, 90, 50, 90, 80, 10, 80], text="上段左", confidence=0.9),
            OCRResult(bbox=[300, 100, 400, 100, 400, 130, 300, 130], text="下段右", confidence=0.9),
            OCRResult(bbox=[200, 50, 290, 50, 290, 80, 200, 80], text="上段右", confidence=0.9),
        ]
        
        # ソート実行
        sorted_results = processor._sort_results_by_position(results)
        
        # ソート結果の検証（上から下、左から右の順序）
        expected_order = ["上段左", "上段右", "下段左", "下段右"]
        actual_order = [r.text for r in sorted_results]
        
        assert actual_order == expected_order
    
    @patch('src.ocr.paddle_processor.PADDLE_AVAILABLE', True)
    @patch('paddleocr.PaddleOCR')
    def test_calculate_total_confidence(self, mock_paddle_ocr):
        """総合信頼度計算のテスト"""
        mock_paddle_ocr.return_value = MagicMock()
        processor = PaddleOCRProcessor(self.test_config)
        
        # テスト用結果
        results = [
            OCRResult(bbox=[0, 0, 10, 10, 10, 20, 0, 20], text="短い", confidence=0.9),  # 2文字
            OCRResult(bbox=[0, 0, 10, 10, 10, 20, 0, 20], text="長いテキスト", confidence=0.8),  # 6文字
            OCRResult(bbox=[0, 0, 10, 10, 10, 20, 0, 20], text="中程度", confidence=0.7),  # 3文字
        ]
        
        # 計算実行
        total_conf = processor._calculate_total_confidence(results)
        
        # 期待値計算（文字数重み付け）
        # (2*0.9 + 6*0.8 + 3*0.7) / (2+6+3) = (1.8 + 4.8 + 2.1) / 11 = 8.7/11 ≈ 0.791
        expected = (2*0.9 + 6*0.8 + 3*0.7) / (2+6+3)
        
        assert abs(total_conf - expected) < 0.001
        
        # 空リストのテスト
        assert processor._calculate_total_confidence([]) == 0.0
    
    @patch('src.ocr.paddle_processor.PADDLE_AVAILABLE', True)
    @patch('paddleocr.PaddleOCR')
    @patch('cv2.imread')
    @patch('os.path.exists')
    def test_process_batch(self, mock_exists, mock_imread, mock_paddle_ocr):
        """バッチ処理のテスト"""
        # モックの設定
        mock_exists.return_value = True
        mock_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        mock_imread.return_value = mock_image
        
        mock_ocr_instance = MagicMock()
        mock_ocr_instance.ocr.return_value = [
            [
                [
                    [[10, 20], [100, 20], [100, 50], [10, 50]],
                    ['テスト', 0.9]
                ]
            ]
        ]
        mock_paddle_ocr.return_value = mock_ocr_instance
        
        processor = PaddleOCRProcessor(self.test_config)
        
        # テスト用画像パス
        image_paths = [
            os.path.join(self.test_dir, "page1.png"),
            os.path.join(self.test_dir, "page2.png"),
            os.path.join(self.test_dir, "page3.png")
        ]
        
        # バッチ処理実行
        results = processor.process_batch(image_paths)
        
        # 結果検証
        assert len(results) == 3
        for i, result in enumerate(results, 1):
            assert result.page_number == i
            assert len(result.results) >= 1
    
    @patch('src.ocr.paddle_processor.PADDLE_AVAILABLE', True)
    @patch('paddleocr.PaddleOCR')
    def test_cache_functionality(self, mock_paddle_ocr):
        """キャッシュ機能のテスト"""
        mock_paddle_ocr.return_value = MagicMock()
        processor = PaddleOCRProcessor(self.test_config)
        
        # テスト用OCR結果
        test_result = PageOCRResult(
            page_number=1,
            image_path="test.png",
            results=[
                OCRResult(bbox=[0, 0, 10, 10, 10, 20, 0, 20], text="テスト", confidence=0.9)
            ],
            processing_time=1.5,
            total_confidence=0.9
        )
        
        # テスト用画像ファイル作成
        test_image = os.path.join(self.test_dir, "test.png")
        with open(test_image, 'w') as f:
            f.write("dummy image")
        
        # キャッシュ保存
        processor._save_cache(test_image, test_result)
        
        # キャッシュ読み込み
        loaded_result = processor._load_cache(test_image)
        
        # 結果検証
        assert loaded_result is not None
        assert loaded_result.page_number == test_result.page_number
        assert loaded_result.total_confidence == test_result.total_confidence
        assert len(loaded_result.results) == 1
        assert loaded_result.results[0].text == "テスト"
    
    @patch('src.ocr.paddle_processor.PADDLE_AVAILABLE', True)
    @patch('paddleocr.PaddleOCR')
    def test_get_statistics(self, mock_paddle_ocr):
        """統計情報取得のテスト"""
        mock_paddle_ocr.return_value = MagicMock()
        processor = PaddleOCRProcessor(self.test_config)
        
        # テスト用結果
        results = [
            PageOCRResult(
                page_number=1,
                image_path="page1.png",
                results=[
                    OCRResult(bbox=[0, 0, 10, 10, 10, 20, 0, 20], text="テキスト", confidence=0.9, type="text"),
                    OCRResult(bbox=[0, 0, 10, 10, 10, 20, 0, 20], text="$x=1$", confidence=0.8, type="formula")
                ],
                processing_time=2.0,
                total_confidence=0.85
            ),
            PageOCRResult(
                page_number=2,
                image_path="page2.png",
                results=[
                    OCRResult(bbox=[0, 0, 10, 10, 10, 20, 0, 20], text="定理", confidence=0.95, type="text")
                ],
                processing_time=1.5,
                total_confidence=0.95
            )
        ]
        
        # 統計情報取得
        stats = processor.get_statistics(results)
        
        # 結果検証
        assert stats['total_pages'] == 2
        assert stats['total_items'] == 3
        assert stats['text_items'] == 2
        assert stats['formula_items'] == 1
        assert stats['average_confidence'] == (0.85 + 0.95) / 2
        assert stats['total_processing_time'] == 3.5
        assert stats['items_per_page'] == 1.5
        
        # 空リストのテスト
        empty_stats = processor.get_statistics([])
        assert empty_stats == {}


if __name__ == "__main__":
    pytest.main([__file__])