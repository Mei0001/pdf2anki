"""
GPT-4oクライアントモジュールのテスト
"""

import os
import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# プロジェクトルートをsys.pathに追加
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.gpt_client import GPTClient, ConversionRequest, ConversionResult, SelfRefineResult


class TestGPTClient:
    """GPTClient クラスのテスト"""
    
    def setup_method(self):
        """各テストメソッド実行前の準備"""
        self.test_dir = tempfile.mkdtemp()
        
        # テスト用設定
        self.test_config = {
            'api_keys': {
                'openai': 'test-api-key'
            },
            'llm': {
                'primary_model': 'gpt-4o',
                'openai': {
                    'model': 'gpt-4o',
                    'max_tokens': 4000,
                    'temperature': 0.1,
                    'timeout': 60
                },
                'self_refine': {
                    'enabled': True,
                    'max_attempts': 3,
                    'compile_check': True
                }
            },
            'cache': {
                'llm_cache': True,
                'directory': os.path.join(self.test_dir, 'cache')
            }
        }
        
    def teardown_method(self):
        """各テストメソッド実行後のクリーンアップ"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_client_initialization_without_api_key(self):
        """APIキーなしでの初期化エラーテスト"""
        config_without_key = {
            'api_keys': {},
            'llm': {'openai': {}}
        }
        
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            GPTClient(config_without_key)
    
    @patch('src.llm.gpt_client.OPENAI_AVAILABLE', False)
    def test_client_initialization_without_openai(self):
        """OpenAIライブラリなしでの初期化エラーテスト"""
        with pytest.raises(ImportError, match="OpenAI library is not installed"):
            GPTClient(self.test_config)
    
    @patch('src.llm.gpt_client.OPENAI_AVAILABLE', True)
    @patch('openai.OpenAI')
    def test_client_initialization_success(self, mock_openai):
        """正常な初期化テスト"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        client = GPTClient(self.test_config)
        
        assert client.model == 'gpt-4o'
        assert client.max_tokens == 4000
        assert client.temperature == 0.1
        assert client.cache_enabled == True
        assert mock_openai.called
    
    @patch('src.llm.gpt_client.OPENAI_AVAILABLE', True)  
    @patch('openai.OpenAI')
    def test_convert_to_latex_success(self, mock_openai):
        """LaTeX変換成功テスト"""
        # モックレスポンスの設定
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "\\begin{theorem}\n実数の連続性について\n\\end{theorem}"
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        client = GPTClient(self.test_config)
        
        # 変換リクエスト
        request = ConversionRequest(
            text="定理1.1 実数の連続性について",
            page_number=1,
            source_info={'chapter': '第1章'}
        )
        
        result = client.convert_to_latex(request)
        
        # 結果検証
        assert isinstance(result, ConversionResult)
        assert result.original_text == request.text
        assert "\\begin{theorem}" in result.latex_content
        assert result.confidence > 0.0
        assert result.error_message is None
    
    @patch('src.llm.gpt_client.OPENAI_AVAILABLE', True)
    @patch('openai.OpenAI')
    def test_convert_to_latex_with_error(self, mock_openai):
        """LaTeX変換エラーテスト"""
        # APIエラーをシミュレート
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        client = GPTClient(self.test_config)
        
        request = ConversionRequest(
            text="テスト文字列",
            page_number=1,
            source_info={}
        )
        
        result = client.convert_to_latex(request)
        
        # エラー結果の検証
        assert result.card_type == "error"
        assert result.confidence == 0.0
        assert result.error_message == "API Error"
        assert result.latex_content == ""
    
    @patch('src.llm.gpt_client.OPENAI_AVAILABLE', True)
    @patch('openai.OpenAI')
    def test_extract_cards_success(self, mock_openai):
        """カード抽出成功テスト"""
        # モックレスポンス（JSON形式）
        mock_response_content = """```json
{
  "cards": [
    {
      "type": "定理",
      "title": "中間値定理",
      "content": "関数$f$が閉区間$[a,b]$で連続...",
      "front": "中間値定理を述べよ",
      "back": "関数$f$が閉区間$[a,b]$で連続で...",
      "confidence": 0.95
    },
    {
      "type": "定義",
      "title": "連続関数",
      "content": "関数$f$が点$a$で連続...",
      "front": "連続関数の定義は？",
      "back": "関数$f$が点$a$で連続とは...",
      "confidence": 0.90
    }
  ]
}
```"""
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = mock_response_content
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        client = GPTClient(self.test_config)
        
        # カード抽出実行
        latex_text = "\\begin{theorem}中間値定理\\end{theorem}"
        source_info = {'page': 1, 'chapter': '第1章'}
        
        cards = client.extract_cards(latex_text, source_info)
        
        # 結果検証
        assert len(cards) == 2
        
        first_card = cards[0]
        assert first_card['type'] == '定理'
        assert first_card['title'] == '中間値定理'
        assert first_card['confidence'] == 0.95
        assert first_card['page'] == 1  # source_infoが追加されていることを確認
        
        second_card = cards[1]
        assert second_card['type'] == '定義'
        assert second_card['title'] == '連続関数'
    
    @patch('src.llm.gpt_client.OPENAI_AVAILABLE', True)
    @patch('openai.OpenAI')
    def test_extract_cards_invalid_json(self, mock_openai):
        """無効なJSON応答でのカード抽出テスト"""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Invalid JSON response"
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        client = GPTClient(self.test_config)
        
        cards = client.extract_cards("test latex", {})
        
        # 空リストが返されることを確認
        assert cards == []
    
    @patch('src.llm.gpt_client.OPENAI_AVAILABLE', True)
    @patch('openai.OpenAI')
    def test_validate_latex(self, mock_openai):
        """LaTeX検証テスト"""
        mock_openai.return_value = MagicMock()
        client = GPTClient(self.test_config)
        
        # 正常なLaTeXコード
        valid_latex = r"\begin{theorem}$x + y = z$\end{theorem}"
        errors = client._validate_latex(valid_latex)
        assert len(errors) == 0
        
        # 括弧の不一致
        invalid_latex1 = r"\begin{theorem}$x + y = z\end{theorem}"  # $が閉じられていない
        errors1 = client._validate_latex(invalid_latex1)
        assert len(errors1) > 0
        
        # ブレースの不一致
        invalid_latex2 = r"\begin{theorem $x + y = z$\end{theorem}"  # {が閉じられていない
        errors2 = client._validate_latex(invalid_latex2)
        assert len(errors2) > 0
    
    @patch('src.llm.gpt_client.OPENAI_AVAILABLE', True)
    @patch('openai.OpenAI')
    def test_self_refine_latex(self, mock_openai):
        """Self-refine処理テスト"""
        # 改善されたLaTeXを返すモックレスポンス
        mock_response = MagicMock()
        mock_response.choices[0].message.content = r"\begin{theorem}$x + y = z$\end{theorem}"
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        client = GPTClient(self.test_config)
        
        # エラーのあるLaTeXコード
        problematic_latex = r"\begin{theorem $x + y = z\end{theorem}"  # {が閉じられていない
        
        result = client._self_refine_latex(problematic_latex)
        
        assert isinstance(result, SelfRefineResult)
        assert result.final_content != problematic_latex  # 改善されていることを確認
        assert len(result.refinements) > 0
        assert result.total_attempts > 0
    
    @patch('src.llm.gpt_client.OPENAI_AVAILABLE', True)
    @patch('openai.OpenAI')
    def test_process_batch(self, mock_openai):
        """バッチ処理テスト"""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = r"\begin{theorem}テスト\end{theorem}"
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        client = GPTClient(self.test_config)
        
        # バッチリクエスト
        requests = [
            ConversionRequest("テスト1", 1, {}),
            ConversionRequest("テスト2", 2, {}),
            ConversionRequest("テスト3", 3, {})
        ]
        
        results = client.process_batch(requests)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.original_text == f"テスト{i+1}"
            assert result.latex_content == r"\begin{theorem}テスト\end{theorem}"
    
    @patch('src.llm.gpt_client.OPENAI_AVAILABLE', True)
    @patch('openai.OpenAI')
    def test_cache_functionality(self, mock_openai):
        """キャッシュ機能のテスト"""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = r"\begin{theorem}キャッシュテスト\end{theorem}"
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        client = GPTClient(self.test_config)
        
        request = ConversionRequest("テスト", 1, {})
        
        # 最初の変換（APIが呼ばれる）
        result1 = client.convert_to_latex(request)
        api_call_count = mock_client.chat.completions.create.call_count
        
        # 同じリクエストでの2回目の変換（キャッシュから取得）
        result2 = client.convert_to_latex(request)
        
        # APIが再度呼ばれていないことを確認
        assert mock_client.chat.completions.create.call_count == api_call_count
        
        # 結果が同じことを確認
        assert result1.latex_content == result2.latex_content
    
    @patch('src.llm.gpt_client.OPENAI_AVAILABLE', True)
    @patch('openai.OpenAI')
    def test_get_usage_statistics(self, mock_openai):
        """使用統計取得テスト"""
        mock_openai.return_value = MagicMock()
        client = GPTClient(self.test_config)
        
        stats = client.get_usage_statistics()
        
        assert isinstance(stats, dict)
        assert 'model' in stats
        assert 'cache_enabled' in stats
        assert 'self_refine_enabled' in stats
        assert stats['model'] == 'gpt-4o'
        assert stats['cache_enabled'] == True


if __name__ == "__main__":
    pytest.main([__file__])