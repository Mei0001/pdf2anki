"""
PDF前処理ユーティリティモジュール

PDF文書の解析、画像変換、前処理機能を提供
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import fitz  # PyMuPDF
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)


@dataclass
class PDFMetadata:
    """PDF文書のメタデータ"""
    title: Optional[str]
    author: Optional[str]
    subject: Optional[str]
    creator: Optional[str]
    producer: Optional[str]
    creation_date: Optional[str]
    modification_date: Optional[str]
    page_count: int
    file_size: int
    encryption: bool


@dataclass
class PageInfo:
    """ページ情報"""
    page_number: int
    width: int
    height: int
    rotation: int
    text_blocks: List[Dict[str, Any]]
    image_path: Optional[str] = None


class PDFProcessor:
    """PDF処理の中核クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: PDF処理設定
        """
        self.config = config
        self.pdf_config = config.get('pdf', {})
        self.dpi = self.pdf_config.get('dpi', 300)
        self.image_format = self.pdf_config.get('image_format', 'PNG')
        self.preprocessing = self.pdf_config.get('preprocessing', {})
        
    def extract_metadata(self, pdf_path: str) -> PDFMetadata:
        """PDFメタデータを抽出
        
        Args:
            pdf_path: PDFファイルパス
            
        Returns:
            PDFメタデータ
        """
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            
            return PDFMetadata(
                title=metadata.get('title'),
                author=metadata.get('author'),
                subject=metadata.get('subject'),
                creator=metadata.get('creator'),
                producer=metadata.get('producer'),
                creation_date=metadata.get('creationDate'),
                modification_date=metadata.get('modDate'),
                page_count=len(doc),
                file_size=os.path.getsize(pdf_path),
                encryption=doc.needs_pass
            )
        except Exception as e:
            logger.error(f"Failed to extract metadata from {pdf_path}: {e}")
            raise
        finally:
            if 'doc' in locals():
                doc.close()
    
    def convert_to_images(self, pdf_path: str, output_dir: str) -> List[str]:
        """PDF→画像変換
        
        Args:
            pdf_path: PDFファイルパス
            output_dir: 出力ディレクトリ
            
        Returns:
            変換された画像ファイルのパスリスト
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # pdf2imageで変換
            pages = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                fmt=self.image_format.lower()
            )
            
            image_paths = []
            pdf_name = Path(pdf_path).stem
            
            for i, page in enumerate(pages, 1):
                image_filename = f"{pdf_name}_page_{i:03d}.{self.image_format.lower()}"
                image_path = output_path / image_filename
                
                # 前処理を適用
                if self.preprocessing:
                    page = self._preprocess_image(page)
                
                page.save(image_path, self.image_format)
                image_paths.append(str(image_path))
                
            logger.info(f"Converted {len(pages)} pages from {pdf_path}")
            return image_paths
            
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            raise
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """画像前処理
        
        Args:
            image: 元画像
            
        Returns:
            前処理済み画像
        """
        # PIL -> OpenCV変換
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_cv = img_array
            
        # ノイズ除去
        if self.preprocessing.get('denoise', False):
            img_cv = cv2.fastNlMeansDenoising(img_cv)
            
        # 歪み補正
        if self.preprocessing.get('deskew', False):
            img_cv = self._deskew_image(img_cv)
            
        # コントラスト強化
        if self.preprocessing.get('enhance_contrast', False):
            img_cv = self._enhance_contrast(img_cv)
            
        # OpenCV -> PIL変換
        if len(img_cv.shape) == 3:
            processed_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        else:
            processed_image = Image.fromarray(img_cv)
            
        return processed_image
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """画像の歪み補正
        
        Args:
            image: 入力画像
            
        Returns:
            補正済み画像
        """
        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # エッジ検出
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 直線検出
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = theta * 180 / np.pi
                if angle < 45:
                    angles.append(angle)
                elif angle > 135:
                    angles.append(angle - 180)
                    
            if angles:
                # 中央値を回転角として使用
                rotation_angle = np.median(angles)
                
                # 画像回転
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                
                return cv2.warpAffine(image, M, (w, h), 
                                    flags=cv2.INTER_CUBIC, 
                                    borderMode=cv2.BORDER_REPLICATE)
        
        return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """コントラスト強化
        
        Args:
            image: 入力画像
            
        Returns:
            コントラスト強化済み画像
        """
        # CLAHEを使用した局所コントラスト強化
        if len(image.shape) == 3:
            # カラー画像の場合はLAB色空間に変換
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # L チャンネルのみにCLAHEを適用
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # チャンネル結合して元の色空間に戻す
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # グレースケール画像の場合
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def extract_text_blocks(self, pdf_path: str) -> List[PageInfo]:
        """PDFからテキストブロック情報を抽出
        
        Args:
            pdf_path: PDFファイルパス
            
        Returns:
            ページ情報のリスト
        """
        try:
            doc = fitz.open(pdf_path)
            pages_info = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # テキストブロック抽出
                text_blocks = page.get_text("dict")["blocks"]
                
                # 画像以外のテキストブロックのみ抽出
                text_only_blocks = []
                for block in text_blocks:
                    if "lines" in block:  # テキストブロック
                        text_only_blocks.append({
                            "bbox": block["bbox"],  # [x0, y0, x1, y1]
                            "lines": block["lines"]
                        })
                
                page_info = PageInfo(
                    page_number=page_num + 1,
                    width=int(page.rect.width),
                    height=int(page.rect.height),
                    rotation=page.rotation,
                    text_blocks=text_only_blocks
                )
                
                pages_info.append(page_info)
                
            logger.info(f"Extracted text blocks from {len(pages_info)} pages")
            return pages_info
            
        except Exception as e:
            logger.error(f"Failed to extract text blocks: {e}")
            raise
        finally:
            if 'doc' in locals():
                doc.close()
    
    def detect_chapters(self, pages_info: List[PageInfo]) -> Dict[str, List[int]]:
        """章・節の検出
        
        Args:
            pages_info: ページ情報のリスト
            
        Returns:
            章・節とページ番号のマッピング
        """
        chapters = {}
        current_chapter = None
        
        for page_info in pages_info:
            for block in page_info.text_blocks:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        font_size = span["size"]
                        
                        # 章・節のパターンを検出（日本語・英語対応）
                        chapter_patterns = [
                            r'^第\s*[0-9０-９]+\s*章',  # 第1章
                            r'^Chapter\s+\d+',         # Chapter 1
                            r'^[0-9０-９]+\.\s*',      # 1. タイトル
                        ]
                        
                        import re
                        for pattern in chapter_patterns:
                            if re.match(pattern, text, re.IGNORECASE):
                                # フォントサイズが大きい場合は章として扱う
                                if font_size > 14:  # 閾値は調整可能
                                    current_chapter = text
                                    if current_chapter not in chapters:
                                        chapters[current_chapter] = []
                                    chapters[current_chapter].append(page_info.page_number)
                                    break
        
        logger.info(f"Detected {len(chapters)} chapters/sections")
        return chapters
    
    def get_page_layout_analysis(self, pdf_path: str, page_num: int) -> Dict[str, Any]:
        """特定ページのレイアウト解析
        
        Args:
            pdf_path: PDFファイルパス
            page_num: ページ番号（1から開始）
            
        Returns:
            レイアウト解析結果
        """
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num - 1]
            
            # レイアウト解析
            layout_analysis = {
                "page_number": page_num,
                "dimensions": {
                    "width": page.rect.width,
                    "height": page.rect.height
                },
                "text_regions": [],
                "formula_regions": [],
                "figure_regions": [],
                "table_regions": []
            }
            
            # テキスト領域の分析
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    bbox = block["bbox"]
                    text_content = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text_content += span["text"]
                    
                    # 数式パターンの検出
                    math_patterns = [
                        r'\$.*?\$',  # インライン数式
                        r'\\.*?\\',  # LaTeX コマンド
                        r'[∫∑∏∆∇]',  # 数学記号
                    ]
                    
                    import re
                    is_formula = any(re.search(pattern, text_content) for pattern in math_patterns)
                    
                    region_info = {
                        "bbox": bbox,
                        "content": text_content,
                        "confidence": 1.0
                    }
                    
                    if is_formula:
                        layout_analysis["formula_regions"].append(region_info)
                    else:
                        layout_analysis["text_regions"].append(region_info)
            
            return layout_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze page layout: {e}")
            raise
        finally:
            if 'doc' in locals():
                doc.close()


def validate_pdf(pdf_path: str) -> bool:
    """PDFファイルの妥当性チェック
    
    Args:
        pdf_path: PDFファイルパス
        
    Returns:
        妥当性の判定結果
    """
    try:
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return False
            
        if not pdf_path.lower().endswith('.pdf'):
            logger.error(f"File is not a PDF: {pdf_path}")
            return False
            
        # PDFを開いて基本チェック
        doc = fitz.open(pdf_path)
        
        if len(doc) == 0:
            logger.error(f"PDF has no pages: {pdf_path}")
            doc.close()
            return False
            
        # 暗号化チェック
        if doc.needs_pass:
            logger.warning(f"PDF is password protected: {pdf_path}")
            doc.close()
            return False
            
        doc.close()
        return True
        
    except Exception as e:
        logger.error(f"PDF validation failed: {e}")
        return False