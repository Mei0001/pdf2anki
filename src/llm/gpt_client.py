"""
OpenAI GPT-4o クライアントモジュール

OCR結果からLaTeX形式への変換とカード生成機能
"""

import os
import logging
import time
import json
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import asyncio
from pathlib import Path

# OpenAI クライアントのインポート
try:
    from openai import OpenAI
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available. Please install with: pip install openai")

logger = logging.getLogger(__name__)


@dataclass 
class ConversionRequest:
    """変換リクエスト"""
    text: str
    page_number: int
    source_info: Dict[str, Any]
    context: Optional[str] = None


@dataclass
class ConversionResult:
    """変換結果"""
    original_text: str
    latex_content: str
    card_type: str  # "定義", "定理", "命題", "補題", "例", "注意" etc.
    title: str
    confidence: float
    processing_time: float
    error_message: Optional[str] = None


@dataclass 
class SelfRefineResult:
    """Self-refine処理結果"""
    final_content: str
    refinements: List[Dict[str, Any]]
    success: bool
    total_attempts: int


class GPTClient:
    """OpenAI GPT-4o クライアントクラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: LLM設定
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library is not installed. Please install with: pip install openai")
        
        self.config = config
        self.llm_config = config.get('llm', {})
        self.openai_config = self.llm_config.get('openai', {})
        self.self_refine_config = self.llm_config.get('self_refine', {})
        
        # APIキーの設定
        api_key = config.get('api_keys', {}).get('openai')
        if not api_key:
            raise ValueError("OpenAI API key is required. Please set it in config or environment variable OPENAI_API_KEY")
        
        self.client = OpenAI(api_key=api_key)
        
        # モデル設定
        self.model = self.openai_config.get('model', 'gpt-4o')
        self.max_tokens = self.openai_config.get('max_tokens', 4000)
        self.temperature = self.openai_config.get('temperature', 0.1)
        self.timeout = self.openai_config.get('timeout', 60)
        
        # キャッシュ設定
        self.cache_enabled = config.get('cache', {}).get('llm_cache', True)
        self.cache_dir = Path(config.get('cache', {}).get('directory', 'cache'))
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # プロンプトテンプレートの読み込み
        self._load_prompt_templates()
        
        logger.info(f"GPT Client initialized with model: {self.model}")
    
    def _load_prompt_templates(self):
        """プロンプトテンプレートを読み込み"""
        # プロンプトファイルのパス
        prompt_dir = Path(__file__).parent.parent.parent / "config" / "prompts"
        
        # デフォルトプロンプト
        self.prompts = {
            'latex_conversion': """あなたは数学専門のLaTeX変換エキスパートです。

与えられたOCRテキストを正確なLaTeX形式に変換してください。

【入力】: OCRで認識されたテキスト（数式や数学記号が含まれる可能性があります）
【出力】: 正確なLaTeX形式のテキスト

変換の際の注意点:
1. 数学記号は適切なLaTeX記法を使用
2. 数式は適切な環境（$...$ or $$...$$）で囲む
3. 定理、定義、命題などの構造を認識して適切にフォーマット
4. 日本語テキストは自然な表現に補正
5. OCRエラーを推測して修正

入力テキスト:
{text}

LaTeX形式で出力してください:""",
            
            'card_extraction': """あなたは数学教材の分析エキスパートです。

与えられたテキストから数学的概念（定義、定理、命題、補題、系、例、注意など）を抽出し、フラッシュカード形式に整理してください。

【入力】: LaTeX形式の数学テキスト
【出力】: JSON形式の構造化データ

出力形式:
{
  "cards": [
    {
      "type": "定義|定理|命題|補題|系|例|注意",
      "title": "概念の名前",
      "content": "LaTeX形式の内容",
      "front": "フラッシュカードの表面（質問）",
      "back": "フラッシュカードの裏面（答え）",
      "confidence": 0.0-1.0
    }
  ]
}

入力テキスト:
{text}

JSON形式で出力してください:""",
            
            'self_refine': """以下のLaTeX コードを検証し、エラーがあれば修正してください。

【検証項目】:
1. LaTeX構文の正確性
2. 数式記法の適切性
3. 日本語表現の自然さ
4. 論理的整合性

【入力LaTeX】:
{latex_content}

【コンパイルエラー（あれば）】:
{errors}

修正が必要な場合は修正版を、問題がない場合は元のコードをそのまま出力してください:"""
        }
        
        # カスタムプロンプトファイルが存在する場合は読み込み
        try:
            for prompt_name in self.prompts.keys():
                prompt_file = prompt_dir / f"{prompt_name}.txt"
                if prompt_file.exists():
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        self.prompts[prompt_name] = f.read()
                    logger.info(f"Loaded custom prompt: {prompt_name}")
        except Exception as e:
            logger.warning(f"Failed to load custom prompts: {e}")
    
    def convert_to_latex(self, request: ConversionRequest) -> ConversionResult:
        """OCRテキストをLaTeX形式に変換
        
        Args:
            request: 変換リクエスト
            
        Returns:
            変換結果
        """
        start_time = time.time()
        
        # キャッシュチェック
        if self.cache_enabled:
            cached_result = self._load_conversion_cache(request.text)
            if cached_result:
                logger.info("Loaded LaTeX conversion from cache")
                return cached_result
        
        try:
            # プロンプト生成
            prompt = self.prompts['latex_conversion'].format(text=request.text)
            
            # コンテキスト情報の追加
            if request.context:
                prompt += f"\n\nコンテキスト情報:\n{request.context}"
            
            # GPT-4oで変換
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "あなたは数学専門のLaTeX変換エキスパートです。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout
            )
            
            latex_content = response.choices[0].message.content.strip()
            
            # Self-refine処理（有効な場合）
            if self.self_refine_config.get('enabled', True):
                refine_result = self._self_refine_latex(latex_content)
                latex_content = refine_result.final_content
            
            processing_time = time.time() - start_time
            
            result = ConversionResult(
                original_text=request.text,
                latex_content=latex_content,
                card_type="unknown",  # 後で抽出
                title="",  # 後で抽出
                confidence=0.9,  # 成功時のデフォルト信頼度
                processing_time=processing_time
            )
            
            # キャッシュ保存 
            if self.cache_enabled:
                self._save_conversion_cache(request.text, result)
            
            logger.info(f"LaTeX conversion completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"LaTeX conversion failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            return ConversionResult(
                original_text=request.text,
                latex_content="",
                card_type="error",
                title="",
                confidence=0.0,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def extract_cards(self, latex_text: str, source_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """LaTeXテキストからカード情報を抽出
        
        Args:
            latex_text: LaTeX形式のテキスト
            source_info: ソース情報（ページ番号、章など）
            
        Returns:
            抽出されたカード情報のリスト
        """
        try:
            # プロンプト生成
            prompt = self.prompts['card_extraction'].format(text=latex_text)
            
            # GPT-4oでカード抽出
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "あなたは数学教材の分析エキスパートです。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout
            )
            
            # 生のレスポンス取得
            raw_response = response.choices[0].message.content.strip()
            logger.debug(f"Raw GPT response for card extraction:\n{raw_response}")
            
            # JSON抽出の試行
            json_text = self._extract_json_from_response(raw_response)
            logger.debug(f"Extracted JSON text:\n{json_text}")
            
            # JSON解析
            card_data = self._parse_json_safely(json_text)
            
            if not card_data:
                logger.warning("Failed to parse JSON, attempting fallback extraction")
                return self._fallback_card_extraction(raw_response, source_info)
            
            cards = card_data.get('cards', [])
            
            # カードデータの検証
            valid_cards = []
            for i, card in enumerate(cards):
                if self._validate_card_data(card):
                    # ソース情報を各カードに追加
                    card.update(source_info)
                    valid_cards.append(card)
                else:
                    logger.warning(f"Skipping invalid card at index {i}: {card}")
            
            logger.info(f"Extracted {len(valid_cards)} valid cards from LaTeX text")
            return valid_cards
            
        except Exception as e:
            logger.error(f"Card extraction failed: {e}")
            logger.error(f"Input LaTeX text (first 200 chars): {latex_text[:200]}...")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []
    
    def _extract_json_from_response(self, response_text: str) -> str:
        """GPTレスポンスからJSON部分を抽出
        
        Args:
            response_text: GPTからの生レスポンス
            
        Returns:
            抽出されたJSON文字列
        """
        import re
        
        # 複数のパターンでJSON抽出を試行
        patterns = [
            # ```json ... ``` パターン
            r'```json\s*\n(.*?)\n```',
            # ``` ... ``` パターン（jsonタグなし）
            r'```\s*\n(.*?)\n```',
            # { ... } パターン（改行を含む）
            r'(\{[^{}]*"cards"[^{}]*\[[^\]]*\][^{}]*\})',
            # シンプルな { ... } パターン（単一行）
            r'(\{.*?"cards".*?\})',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if matches:
                json_candidate = matches[0].strip()
                logger.debug(f"JSON extraction pattern matched: {pattern}")
                return json_candidate
        
        # パターンマッチが失敗した場合、レスポンス全体を試用
        logger.warning("No JSON pattern matched, using full response")
        return response_text
    
    def _parse_json_safely(self, json_text: str) -> Optional[Dict[str, Any]]:
        """安全なJSON解析（エラー回復機能付き）
        
        Args:
            json_text: JSON文字列
            
        Returns:
            解析されたJSONデータ、失敗時はNone
        """
        if not json_text.strip():
            logger.warning("Empty JSON text provided")
            return None
        
        # 最初の解析試行
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parsing failed: {e}")
            logger.debug(f"Failed JSON text: {json_text[:500]}...")
        
        # JSON修復を試行
        cleaned_json = self._clean_json_text(json_text)
        try:
            return json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed after cleanup: {e}")
            logger.debug(f"Cleaned JSON text: {cleaned_json[:500]}...")
        
        # 部分的なJSON抽出を試行
        partial_json = self._extract_partial_json(json_text)
        if partial_json:
            try:
                return json.loads(partial_json)
            except json.JSONDecodeError as e:
                logger.warning(f"Partial JSON parsing failed: {e}")
        
        return None
    
    def _clean_json_text(self, json_text: str) -> str:
        """JSON文字列のクリーンアップ
        
        Args:
            json_text: 元のJSON文字列
            
        Returns:
            クリーンアップされたJSON文字列
        """
        import re
        
        # 先頭・末尾の不要な文字を除去
        cleaned = json_text.strip()
        
        # 文字化けした引用符を修正
        cleaned = re.sub(r'[""]', '"', cleaned)  # スマートクオートを通常の引用符に
        cleaned = re.sub(r'[''`]', "'", cleaned)  # スマートアポストロフィを通常のアポストロフィに
        
        # 不正な改行を除去
        cleaned = re.sub(r'\n\s*"', ' "', cleaned)
        
        # 末尾のカンマを修正
        cleaned = re.sub(r',\s*}', '}', cleaned)
        cleaned = re.sub(r',\s*]', ']', cleaned)
        
        # 不完全な文字列の修復
        cleaned = re.sub(r'"([^"]*)\n([^"]*)"', r'"\1 \2"', cleaned)
        
        return cleaned
    
    def _extract_partial_json(self, text: str) -> Optional[str]:
        """部分的なJSONの抽出を試行
        
        Args:
            text: 元のテキスト
            
        Returns:
            抽出できた部分JSON、失敗時はNone
        """
        import re
        
        # "cards"キーを含む部分的なJSONを探す
        cards_pattern = r'"cards"\s*:\s*\[([^\]]*)\]'
        cards_match = re.search(cards_pattern, text, re.DOTALL)
        
        if cards_match:
            try:
                # 最小限のJSON構造を作成
                cards_content = cards_match.group(1)
                partial_json = f'{{"cards": [{cards_content}]}}'
                return partial_json
            except Exception as e:
                logger.debug(f"Failed to create partial JSON: {e}")
        
        return None
    
    def _validate_card_data(self, card: Dict[str, Any]) -> bool:
        """カードデータの検証
        
        Args:
            card: カードデータ
            
        Returns:
            検証結果
        """
        required_fields = ['type', 'title', 'content', 'front', 'back', 'confidence']
        
        for field in required_fields:
            if field not in card:
                logger.warning(f"Missing required field '{field}' in card data")
                return False
            
            if not card[field] and field != 'confidence':  # confidenceは0でも有効
                logger.warning(f"Empty value for required field '{field}' in card data")
                return False
        
        # 信頼度の範囲チェック
        try:
            confidence = float(card['confidence'])
            if not (0.0 <= confidence <= 1.0):
                logger.warning(f"Invalid confidence value: {confidence} (must be 0.0-1.0)")
                return False
        except (ValueError, TypeError):
            logger.warning(f"Invalid confidence value type: {card['confidence']}")
            return False
        
        # カードタイプの検証
        valid_types = ['定義', '定理', '命題', '補題', '系', '例', '注意']
        if card['type'] not in valid_types:
            logger.warning(f"Invalid card type: {card['type']}")
            # 型が無効でも処理を継続（警告のみ）
        
        return True
    
    def _fallback_card_extraction(self, response_text: str, source_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """フォールバックカード抽出（JSON解析失敗時）
        
        Args:
            response_text: GPTからの生レスポンス
            source_info: ソース情報
            
        Returns:
            抽出できたカード情報のリスト
        """
        logger.info("Attempting fallback card extraction from response text")
        
        # 基本的なパターンマッチングでカード情報を抽出
        import re
        cards = []
        
        # 定義、定理などのパターンを探す
        concept_patterns = [
            r'(定義|定理|命題|補題|系|例|注意)[:：]\s*([^\n]+)',
            r'(Definition|Theorem|Proposition|Lemma|Corollary|Example|Remark)[:：]\s*([^\n]+)',
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            for match in matches:
                concept_type = match[0]
                concept_content = match[1].strip()
                
                if concept_content:
                    card = {
                        'type': concept_type,
                        'title': concept_content[:50] + '...' if len(concept_content) > 50 else concept_content,
                        'content': concept_content,
                        'front': f"{concept_type}について説明してください",
                        'back': concept_content,
                        'confidence': 0.5  # フォールバック抽出では低い信頼度
                    }
                    card.update(source_info)
                    cards.append(card)
        
        if cards:
            logger.info(f"Fallback extraction found {len(cards)} potential cards")
        else:
            logger.warning("Fallback extraction found no cards")
        
        return cards
    
    def _self_refine_latex(self, latex_content: str) -> SelfRefineResult:
        """LaTeXコードのSelf-refine処理
        
        Args:
            latex_content: 初期LaTeXコード
            
        Returns:
            改善結果
        """
        max_attempts = self.self_refine_config.get('max_attempts', 3)
        refinements = []
        current_content = latex_content
        
        for attempt in range(max_attempts):
            try:
                # LaTeX検証（簡易）
                errors = self._validate_latex(current_content)
                
                if not errors:
                    # エラーがない場合は完了
                    return SelfRefineResult(
                        final_content=current_content,
                        refinements=refinements,
                        success=True,
                        total_attempts=attempt + 1
                    )
                
                # 改善プロンプト
                refine_prompt = self.prompts['self_refine'].format(
                    latex_content=current_content,
                    errors="\n".join(errors)
                )
                
                # GPT-4oで改善
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "LaTeX専門家として、コードを検証・修正してください。"},
                        {"role": "user", "content": refine_prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0.1,  # より確定的な出力のため低温度
                    timeout=self.timeout
                )
                
                refined_content = response.choices[0].message.content.strip()
                
                refinements.append({
                    'attempt': attempt + 1,
                    'errors': errors,
                    'original': current_content,
                    'refined': refined_content
                })
                
                current_content = refined_content
                
            except Exception as e:
                logger.warning(f"Self-refine attempt {attempt + 1} failed: {e}")
                break
        
        return SelfRefineResult(
            final_content=current_content,
            refinements=refinements,
            success=False,  # 最大回数に達した
            total_attempts=max_attempts
        )
    
    def _validate_latex(self, latex_content: str) -> List[str]:
        """LaTeXコードの簡易検証
        
        Args:
            latex_content: LaTeXコード
            
        Returns:
            エラーリスト
        """
        errors = []
        
        # 基本的な構文チェック
        import re
        
        # 括弧の対応チェック
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        for char in latex_content:
            if char in brackets:
                stack.append(brackets[char])
            elif stack and char == stack[-1]:
                stack.pop()
        
        if stack:
            errors.append("Unmatched brackets detected")
        
        # 数式環境の対応チェック
        math_delimiters = [
            (r'\$', r'\$'),  # インライン数式
            (r'\$\$', r'\$\$'),  # ディスプレイ数式
            (r'\\begin\{equation\}', r'\\end\{equation\}'),
            (r'\\begin\{align\}', r'\\end\{align\}'),
            (r'\\begin\{eqnarray\}', r'\\end\{eqnarray\}')
        ]
        
        for start_delim, end_delim in math_delimiters:
            starts = len(re.findall(start_delim, latex_content))
            ends = len(re.findall(end_delim, latex_content))
            if starts != ends:
                errors.append(f"Unmatched {start_delim} environment")
        
        # 不正なLaTeXコマンドチェック
        invalid_patterns = [
            r'\\[a-zA-Z]+\s*\{[^}]*$',  # 閉じられていないコマンド
            r'\$[^$]*$',  # 閉じられていない数式
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, latex_content):
                errors.append(f"Invalid LaTeX pattern: {pattern}")
        
        return errors
    
    def process_batch(self, requests: List[ConversionRequest]) -> List[ConversionResult]:
        """バッチ変換処理
        
        Args:
            requests: 変換リクエストのリスト
            
        Returns:
            変換結果のリスト
        """
        results = []
        
        logger.info(f"Starting batch LaTeX conversion for {len(requests)} requests")
        
        for i, request in enumerate(requests, 1):
            try:
                result = self.convert_to_latex(request)
                results.append(result)
                
                # 進捗ログ
                if i % 5 == 0 or i == len(requests):
                    logger.info(f"Processed {i}/{len(requests)} conversions")
                
                # レート制限対策（必要に応じて）
                if i < len(requests):
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Failed to process request {i}: {e}")
                # エラーがあっても続行
                continue
        
        logger.info(f"Batch conversion completed: {len(results)}/{len(requests)} successful")
        return results
    
    def _get_conversion_cache_path(self, text: str) -> Path:
        """変換キャッシュパスを生成"""
        import hashlib
        
        cache_key = hashlib.md5(text.encode()).hexdigest()
        return self.cache_dir / f"latex_conv_{cache_key}.json"
    
    def _save_conversion_cache(self, text: str, result: ConversionResult):
        """変換結果をキャッシュに保存"""
        try:
            cache_path = self._get_conversion_cache_path(text)
            
            cache_data = {
                'original_text': result.original_text,
                'latex_content': result.latex_content,
                'card_type': result.card_type,
                'title': result.title,
                'confidence': result.confidence,
                'processing_time': result.processing_time,
                'error_message': result.error_message,
                'timestamp': time.time()
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save conversion cache: {e}")
    
    def _load_conversion_cache(self, text: str) -> Optional[ConversionResult]:
        """キャッシュから変換結果を読み込み"""
        try:
            cache_path = self._get_conversion_cache_path(text)
            
            if not cache_path.exists():
                return None
            
            # キャッシュの有効期限チェック（30日）
            cache_age = time.time() - cache_path.stat().st_mtime
            if cache_age > 30 * 24 * 3600:  # 30日
                cache_path.unlink()  # 古いキャッシュを削除
                return None
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            return ConversionResult(
                original_text=cache_data['original_text'],
                latex_content=cache_data['latex_content'],
                card_type=cache_data['card_type'],
                title=cache_data['title'],
                confidence=cache_data['confidence'],
                processing_time=cache_data['processing_time'],
                error_message=cache_data.get('error_message')
            )
            
        except Exception as e:
            logger.warning(f"Failed to load conversion cache: {e}")
            return None
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """使用統計を取得"""
        try:
            # 簡易的な統計（実際のAPI使用量は別途OpenAIダッシュボードで確認）
            cache_files = list(self.cache_dir.glob("latex_conv_*.json")) if self.cache_enabled else []
            
            return {
                'model': self.model,
                'cache_enabled': self.cache_enabled,
                'cached_conversions': len(cache_files),
                'self_refine_enabled': self.self_refine_config.get('enabled', True),
                'max_tokens': self.max_tokens,
                'temperature': self.temperature
            }
            
        except Exception as e:
            logger.error(f"Failed to get usage statistics: {e}")
            return {}