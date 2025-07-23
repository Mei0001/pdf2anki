"""
PDF処理ユーティリティのテスト
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# プロジェクトルートをsys.pathに追加
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.pdf_utils import PDFProcessor, PDFMetadata, PageInfo, validate_pdf
from src.utils.config_loader import get_config, reset_config


class TestPDFProcessor:
    """PDFProcessor クラスのテスト"""
    
    def setup_method(self):
        """各テストメソッド実行前の準備"""
        reset_config()
        self.config = get_config()
        self.processor = PDFProcessor(self.config.config)
        self.test_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """各テストメソッド実行後のクリーンアップ"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_processor_initialization(self):
        """PDFProcessorの初期化テスト"""
        assert self.processor.dpi == 300
        assert self.processor.image_format == "PNG"
        assert isinstance(self.processor.preprocessing, dict)
    
    @patch('fitz.open')
    def test_extract_metadata_success(self, mock_fitz_open):
        """メタデータ抽出の成功ケース"""
        # モックPDFドキュメントを設定
        mock_doc = MagicMock()
        mock_doc.metadata = {
            'title': 'Test Mathematics Book',
            'author': 'Test Author',
            'subject': 'Mathematics',  
            'creator': 'Test Creator',
            'producer': 'Test Producer',
            'creationDate': '2024-01-01',
            'modDate': '2024-01-02'
        }
        mock_doc.__len__.return_value = 100
        mock_doc.needs_pass = False
        mock_fitz_open.return_value = mock_doc
        
        # テスト用PDFファイルのパス
        test_pdf = os.path.join(self.test_dir, "test.pdf")
        with open(test_pdf, 'w') as f:
            f.write("dummy pdf content")
        
        # メタデータ抽出を実行
        metadata = self.processor.extract_metadata(test_pdf)
        
        # 結果検証
        assert isinstance(metadata, PDFMetadata)
        assert metadata.title == 'Test Mathematics Book'
        assert metadata.author == 'Test Author'
        assert metadata.page_count == 100
        assert metadata.encryption == False
    
    @patch('fitz.open')
    def test_extract_metadata_encrypted_pdf(self, mock_fitz_open):
        """暗号化PDFのメタデータ抽出テスト"""
        mock_doc = MagicMock()
        mock_doc.metadata = {'title': 'Encrypted PDF'}
        mock_doc.__len__.return_value = 50
        mock_doc.needs_pass = True
        mock_fitz_open.return_value = mock_doc
        
        test_pdf = os.path.join(self.test_dir, "encrypted.pdf")
        with open(test_pdf, 'w') as f:
            f.write("dummy encrypted pdf")
        
        metadata = self.processor.extract_metadata(test_pdf)
        
        assert metadata.encryption == True
        assert metadata.page_count == 50
    
    @patch('pdf2image.convert_from_path')
    @patch('PIL.Image.Image.save')
    def test_convert_to_images_success(self, mock_save, mock_convert):
        """PDF→画像変換の成功ケース"""
        # モック画像オブジェクトを作成
        mock_page1 = MagicMock()
        mock_page2 = MagicMock()
        mock_convert.return_value = [mock_page1, mock_page2]
        
        test_pdf = os.path.join(self.test_dir, "test.pdf")
        with open(test_pdf, 'w') as f:
            f.write("dummy pdf")
        
        output_dir = os.path.join(self.test_dir, "images")
        
        # 画像変換を実行
        image_paths = self.processor.convert_to_images(test_pdf, output_dir)
        
        # 結果検証
        assert len(image_paths) == 2
        assert all(path.endswith('.png') for path in image_paths)
        assert mock_convert.called
        assert mock_save.call_count == 2
    
    @patch('fitz.open')
    def test_extract_text_blocks_success(self, mock_fitz_open):
        """テキストブロック抽出の成功ケース"""
        # モックページオブジェクトを設定
        mock_page = MagicMock()
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "bbox": [100, 100, 400, 150],
                    "lines": [
                        {
                            "spans": [
                                {"text": "定理 1.1", "size": 16},
                                {"text": " 実数の連続性", "size": 14}
                            ]
                        }
                    ]
                },
                {
                    "bbox": [100, 200, 500, 300],
                    "lines": [
                        {
                            "spans": [
                                {"text": "実数体は完備順序体である。", "size": 12}
                            ]
                        }
                    ]
                }
            ]
        }
        mock_page.rect.width = 595
        mock_page.rect.height = 842
        mock_page.rotation = 0
        
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz_open.return_value = mock_doc
        
        test_pdf = os.path.join(self.test_dir, "test.pdf")
        with open(test_pdf, 'w') as f:
            f.write("dummy pdf")
        
        # テキストブロック抽出を実行
        pages_info = self.processor.extract_text_blocks(test_pdf)
        
        # 結果検証
        assert len(pages_info) == 1
        page_info = pages_info[0]
        assert isinstance(page_info, PageInfo)
        assert page_info.page_number == 1
        assert page_info.width == 595
        assert page_info.height == 842
        assert len(page_info.text_blocks) == 2
    
    def test_detect_chapters_success(self):
        """章・節検出の成功ケース"""
        # テスト用ページ情報を作成
        pages_info = [
            PageInfo(
                page_number=1,
                width=595,
                height=842,
                rotation=0,
                text_blocks=[
                    {
                        "bbox": [100, 100, 400, 150],
                        "lines": [
                            {
                                "spans": [
                                    {"text": "第1章 実数論", "size": 18}
                                ]
                            }
                        ]
                    }
                ]
            ),
            PageInfo(
                page_number=5,
                width=595,
                height=842,
                rotation=0,
                text_blocks=[
                    {
                        "bbox": [100, 100, 400, 150],
                        "lines": [
                            {
                                "spans": [
                                    {"text": "第2章 連続性", "size": 18}
                                ]
                            }
                        ]
                    }
                ]
            )
        ]
        
        # 章検出を実行
        chapters = self.processor.detect_chapters(pages_info)
        
        # 結果検証
        assert len(chapters) == 2
        assert "第1章 実数論" in chapters
        assert "第2章 連続性" in chapters
        assert 1 in chapters["第1章 実数論"]
        assert 5 in chapters["第2章 連続性"]
    
    @patch('fitz.open')
    def test_get_page_layout_analysis(self, mock_fitz_open):
        """ページレイアウト解析テスト"""
        # モックページを設定
        mock_page = MagicMock()
        mock_page.rect.width = 595
        mock_page.rect.height = 842
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "bbox": [100, 100, 400, 150],
                    "lines": [
                        {
                            "spans": [
                                {"text": "通常のテキスト"}
                            ]
                        }
                    ]
                },
                {
                    "bbox": [100, 200, 400, 250],
                    "lines": [
                        {
                            "spans": [
                                {"text": "$\\int_0^1 x dx = \\frac{1}{2}$"}
                            ]
                        }
                    ]
                }
            ]
        }
        
        mock_doc = MagicMock()
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz_open.return_value = mock_doc
        
        test_pdf = os.path.join(self.test_dir, "test.pdf")
        with open(test_pdf, 'w') as f:
            f.write("dummy pdf")
        
        # レイアウト解析を実行
        layout = self.processor.get_page_layout_analysis(test_pdf, 1)
        
        # 結果検証
        assert layout["page_number"] == 1
        assert "dimensions" in layout
        assert "text_regions" in layout
        assert "formula_regions" in layout
        assert len(layout["text_regions"]) >= 1
        assert len(layout["formula_regions"]) >= 1


class TestValidatePDF:
    """validate_pdf 関数のテスト"""
    
    def setup_method(self):
        """テスト準備"""
        self.test_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """テスト後始末"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_validate_nonexistent_file(self):
        """存在しないファイルの検証"""
        result = validate_pdf("/nonexistent/file.pdf")
        assert result == False
    
    def test_validate_non_pdf_file(self):
        """PDF以外のファイルの検証"""
        text_file = os.path.join(self.test_dir, "test.txt")
        with open(text_file, 'w') as f:
            f.write("This is not a PDF")
        
        result = validate_pdf(text_file)
        assert result == False
    
    @patch('fitz.open')
    def test_validate_empty_pdf(self, mock_fitz_open):
        """ページが0のPDFの検証"""
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 0
        mock_doc.needs_pass = False
        mock_fitz_open.return_value = mock_doc
        
        test_pdf = os.path.join(self.test_dir, "empty.pdf")
        with open(test_pdf, 'w') as f:
            f.write("empty pdf")
        
        result = validate_pdf(test_pdf)
        assert result == False
    
    @patch('fitz.open')
    def test_validate_encrypted_pdf(self, mock_fitz_open):
        """暗号化PDFの検証"""
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 10
        mock_doc.needs_pass = True
        mock_fitz_open.return_value = mock_doc
        
        test_pdf = os.path.join(self.test_dir, "encrypted.pdf")
        with open(test_pdf, 'w') as f:
            f.write("encrypted pdf")
        
        result = validate_pdf(test_pdf)
        assert result == False
    
    @patch('fitz.open')
    def test_validate_valid_pdf(self, mock_fitz_open):
        """有効なPDFの検証"""
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 10
        mock_doc.needs_pass = False
        mock_fitz_open.return_value = mock_doc
        
        test_pdf = os.path.join(self.test_dir, "valid.pdf")
        with open(test_pdf, 'w') as f:
            f.write("valid pdf")
        
        result = validate_pdf(test_pdf)
        assert result == True


if __name__ == "__main__":
    pytest.main([__file__])