"""
設定読み込み機能のテスト
"""

import os
import pytest  
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

# プロジェクトルートをsys.pathに追加
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import ConfigLoader, get_config, reset_config


class TestConfigLoader:
    """ConfigLoader クラスのテスト"""
    
    def setup_method(self):
        """各テストメソッド実行前の準備"""
        reset_config()
        self.test_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """各テストメソッド実行後のクリーンアップ"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        reset_config()
    
    def test_load_valid_config(self):
        """有効な設定ファイルの読み込みテスト"""
        # テスト用設定ファイルを作成
        config_content = """
api_keys:
  openai: "test-openai-key"
  anthropic: "test-anthropic-key"

ocr:
  engine: "paddle"
  paddle:
    use_gpu: false
    lang: "ja+en"

llm:
  primary_model: "gpt-4o"
  openai:
    model: "gpt-4o"
    temperature: 0.1

pdf:
  dpi: 300
  image_format: "PNG"
  preprocessing:
    denoise: true

card_generation:
  format: "obsidian"
  extract_types:
    - "定義"
    - "定理"

logging:
  level: "INFO"
"""
        config_path = Path(self.test_dir) / "settings.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        # ConfigLoaderを初期化
        loader = ConfigLoader(str(config_path))
        
        # 設定値の検証
        assert loader.get_api_key('openai') == "test-openai-key"
        assert loader.get_api_key('anthropic') == "test-anthropic-key"
        assert loader.get('ocr.engine') == "paddle"
        assert loader.get('llm.primary_model') == "gpt-4o"
        assert loader.get('pdf.dpi') == 300
        assert loader.get('nonexistent.key', 'default') == 'default'
    
    def test_environment_variable_overrides(self):
        """環境変数による設定上書きテスト"""
        # テスト用設定ファイル
        config_content = """
api_keys:
  openai: "original-key"

ocr:
  engine: "paddle"

llm:
  primary_model: "gpt-4o"

pdf:
  dpi: 300

logging:
  level: "INFO"

card_generation:
  format: "obsidian"
"""
        config_path = Path(self.test_dir) / "settings.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        # 環境変数を設定
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'env-openai-key',
            'ANTHROPIC_API_KEY': 'env-anthropic-key',
            'PDF2ANKI_OCR_ENGINE': 'pix2text',
            'PDF2ANKI_LLM_MODEL': 'claude-3.5-sonnet',
            'PDF2ANKI_DPI': '200',
            'PDF2ANKI_LOG_LEVEL': 'DEBUG'
        }):
            loader = ConfigLoader(str(config_path))
            
            # 環境変数による上書きを検証
            assert loader.get_api_key('openai') == 'env-openai-key'
            assert loader.get_api_key('anthropic') == 'env-anthropic-key'
            assert loader.get('ocr.engine') == 'pix2text'
            assert loader.get('llm.primary_model') == 'claude-3.5-sonnet'
            assert loader.get('pdf.dpi') == 200
            assert loader.get('logging.level') == 'DEBUG'
    
    def test_config_validation_success(self):
        """設定検証の成功ケース"""
        config_content = """
api_keys:
  openai: "valid-key"

ocr:
  engine: "paddle"

llm:
  primary_model: "gpt-4o"

pdf:
  dpi: 300
  image_format: "PNG"

card_generation:
  format: "obsidian"

logging:
  level: "INFO"
"""
        config_path = Path(self.test_dir) / "settings.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        # 例外が発生しないことを確認
        loader = ConfigLoader(str(config_path))
        assert loader.config is not None
    
    def test_config_validation_missing_api_key(self):
        """APIキー未設定時の検証エラー"""
        config_content = """
api_keys:
  openai: ""
  anthropic: ""

ocr:
  engine: "paddle"

llm:
  primary_model: "gpt-4o"

pdf:
  dpi: 300

card_generation:
  format: "obsidian"
"""
        config_path = Path(self.test_dir) / "settings.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        # 検証エラーが発生することを確認
        with pytest.raises(ValueError, match="At least one API key"):
            ConfigLoader(str(config_path))
    
    def test_config_validation_invalid_ocr_engine(self):
        """無効なOCRエンジンの検証エラー"""
        config_content = """
api_keys:
  openai: "valid-key"

ocr:
  engine: "invalid-engine"

llm:
  primary_model: "gpt-4o"

pdf:
  dpi: 300

card_generation:
  format: "obsidian"
"""
        config_path = Path(self.test_dir) / "settings.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        with pytest.raises(ValueError, match="Invalid OCR engine"):
            ConfigLoader(str(config_path))
    
    def test_config_validation_invalid_dpi(self):
        """無効なDPI設定の検証エラー"""
        config_content = """
api_keys:
  openai: "valid-key"

ocr:
  engine: "paddle"

llm:
  primary_model: "gpt-4o"

pdf:
  dpi: 1000  # 無効なDPI値

card_generation:
  format: "obsidian"
"""
        config_path = Path(self.test_dir) / "settings.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        with pytest.raises(ValueError, match="Invalid DPI setting"):
            ConfigLoader(str(config_path))
    
    def test_get_specific_configs(self):
        """特定設定取得メソッドのテスト"""
        config_content = """
api_keys:
  openai: "test-key"

ocr:
  engine: "paddle"
  paddle:
    use_gpu: true

llm:
  primary_model: "gpt-4o"
  openai:
    temperature: 0.2

pdf:
  dpi: 150
  preprocessing:
    denoise: false

card_generation:
  extract_types:
    - "定義"

output:
  structure: "by_type"

logging:
  level: "DEBUG"
"""
        config_path = Path(self.test_dir) / "settings.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        loader = ConfigLoader(str(config_path))
        
        # 各設定取得メソッドをテスト
        ocr_config = loader.get_ocr_config()
        assert ocr_config['engine'] == 'paddle'
        assert ocr_config['paddle']['use_gpu'] == True
        
        llm_config = loader.get_llm_config()
        assert llm_config['primary_model'] == 'gpt-4o'
        assert llm_config['openai']['temperature'] == 0.2
        
        pdf_config = loader.get_pdf_config()
        assert pdf_config['dpi'] == 150
        assert pdf_config['preprocessing']['denoise'] == False
        
        card_config = loader.get_card_generation_config()
        assert '定義' in card_config['extract_types']
        
        output_config = loader.get_output_config()
        assert output_config['structure'] == 'by_type'
    
    def test_missing_config_file(self):
        """設定ファイルが見つからない場合のテスト"""
        nonexistent_path = Path(self.test_dir) / "nonexistent.yaml"
        
        with pytest.raises(FileNotFoundError):
            ConfigLoader(str(nonexistent_path))
    
    def test_malformed_yaml(self):
        """不正なYAMLファイルのテスト"""
        config_path = Path(self.test_dir) / "malformed.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("api_keys:\n  openai: [unclosed bracket")
        
        with pytest.raises(Exception):  # YAML解析エラー
            ConfigLoader(str(config_path))


class TestGlobalConfig:
    """グローバル設定インスタンスのテスト"""
    
    def setup_method(self):
        """テスト準備"""
        reset_config()
        self.test_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """テスト後始末"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        reset_config()
    
    def test_global_config_singleton(self):
        """グローバル設定のシングルトン動作テスト"""
        # テスト用設定ファイル
        config_content = """
api_keys:
  openai: "test-key"
ocr:
  engine: "paddle"
llm:
  primary_model: "gpt-4o"
pdf:
  dpi: 300
card_generation:
  format: "obsidian"
"""
        config_path = Path(self.test_dir) / "settings.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        # 複数回取得して同じインスタンスが返されることを確認
        config1 = get_config(str(config_path))
        config2 = get_config()  # 2回目はパス指定なし
        
        assert config1 is config2
        assert config1.get_api_key('openai') == "test-key"


if __name__ == "__main__":
    pytest.main([__file__])