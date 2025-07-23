"""
設定ファイル読み込みユーティリティ

YAML設定の読み込み、環境変数の処理、設定検証機能を提供
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

logger = logging.getLogger(__name__)


class ConfigLoader:
    """設定ファイル読み込みクラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: 設定ファイルパス（未指定時はデフォルトパスを使用）
        """
        if config_path is None:
            # プロジェクトルートからconfig/settings.yamlを探す
            self.config_path = self._find_default_config()
        else:
            self.config_path = Path(config_path)
            
        self.config = self._load_config()
        self._apply_env_overrides()
        self._validate_config()
    
    def _find_default_config(self) -> Path:
        """デフォルト設定ファイルを探す"""
        # 現在のファイルから相対的にプロジェクトルートを探す
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent  # src/utils/ から ../../
        
        config_path = project_root / "config" / "settings.yaml"
        
        if not config_path.exists():
            # settings.yamlがない場合はexampleをコピー
            example_path = project_root / "config" / "settings.example.yaml"
            if example_path.exists():
                logger.warning(f"settings.yaml not found. Please copy {example_path} to {config_path} and configure API keys.")
                # 設定ファイルのテンプレートを使用（APIキーは空）
                return example_path
            else:
                raise FileNotFoundError(f"No configuration file found. Expected: {config_path}")
                
        return config_path
    
    def _load_config(self) -> Dict[str, Any]:
        """YAML設定ファイルを読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {self.config_path}: {e}")
            raise
    
    def _apply_env_overrides(self):
        """環境変数による設定の上書き"""
        # API キーの環境変数チェック
        env_api_keys = {
            'openai': os.getenv('OPENAI_API_KEY'),
            'anthropic': os.getenv('ANTHROPIC_API_KEY')
        }
        
        for key, value in env_api_keys.items():
            if value:
                if 'api_keys' not in self.config:
                    self.config['api_keys'] = {}
                self.config['api_keys'][key] = value
                logger.debug(f"Applied environment variable for {key} API key")
        
        # その他の環境変数
        env_overrides = {
            'PDF2ANKI_LOG_LEVEL': ('logging', 'level'),
            'PDF2ANKI_OCR_ENGINE': ('ocr', 'engine'),
            'PDF2ANKI_LLM_MODEL': ('llm', 'primary_model'),
            'PDF2ANKI_DPI': ('pdf', 'dpi'),
            'PDF2ANKI_CACHE_DIR': ('cache', 'directory'),
        }
        
        for env_var, (section, key) in env_overrides.items():
            value = os.getenv(env_var)
            if value is not None:
                if section not in self.config:
                    self.config[section] = {}
                
                # 型変換
                if key == 'dpi':
                    try:
                        value = int(value)
                    except ValueError:
                        logger.warning(f"Invalid DPI value from environment: {value}")
                        continue
                elif key == 'level':
                    value = value.upper()
                    
                self.config[section][key] = value
                logger.debug(f"Applied environment override: {env_var} = {value}")
    
    def _validate_config(self):
        """設定の妥当性チェック"""
        errors = []
        
        # 必須セクションのチェック
        required_sections = ['api_keys', 'ocr', 'llm', 'pdf', 'card_generation']
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Missing required section: {section}")
        
        # API キーのチェック
        if 'api_keys' in self.config:
            api_keys = self.config['api_keys']
            if not api_keys.get('openai') and not api_keys.get('anthropic'):
                errors.append("At least one API key (OpenAI or Anthropic) is required")
        
        # OCRエンジンのチェック
        if 'ocr' in self.config:
            ocr_engine = self.config['ocr'].get('engine')
            if ocr_engine not in ['paddle', 'pix2text']:
                errors.append(f"Invalid OCR engine: {ocr_engine}. Must be 'paddle' or 'pix2text'")
        
        # LLMモデルのチェック
        if 'llm' in self.config:
            llm_model = self.config['llm'].get('primary_model')
            if llm_model not in ['gpt-4o', 'claude-3.5-sonnet']:
                errors.append(f"Invalid LLM model: {llm_model}")
        
        # PDFの設定チェック
        if 'pdf' in self.config:
            dpi = self.config['pdf'].get('dpi', 300)
            if not isinstance(dpi, int) or dpi < 72 or dpi > 600:
                errors.append(f"Invalid DPI setting: {dpi}. Must be between 72 and 600")
                
            image_format = self.config['pdf'].get('image_format', 'PNG')
            if image_format not in ['PNG', 'JPEG', 'TIFF']:
                errors.append(f"Invalid image format: {image_format}")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Configuration validation passed")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """設定値を取得（ドット記法でネストしたキーをサポート）
        
        Args:
            key_path: 設定キーのパス（例: "llm.openai.model"）
            default: デフォルト値
            
        Returns:
            設定値
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_api_key(self, service: str) -> Optional[str]:
        """APIキーを取得
        
        Args:
            service: サービス名（"openai" or "anthropic"）
            
        Returns:
            APIキー（設定されていない場合はNone）
        """
        return self.get(f'api_keys.{service}')
    
    def get_ocr_config(self) -> Dict[str, Any]:
        """OCR設定を取得"""
        return self.get('ocr', {})
    
    def get_llm_config(self) -> Dict[str, Any]:
        """LLM設定を取得"""
        return self.get('llm', {})
    
    def get_pdf_config(self) -> Dict[str, Any]:
        """PDF処理設定を取得"""
        return self.get('pdf', {})
    
    def get_card_generation_config(self) -> Dict[str, Any]:
        """カード生成設定を取得"""
        return self.get('card_generation', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """出力設定を取得"""
        return self.get('output', {})
    
    def setup_logging(self):
        """ログ設定のセットアップ"""
        log_config = self.get('logging', {})
        log_level = log_config.get('level', 'INFO')
        log_file = log_config.get('file')
        
        # ログレベル設定
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # ログフォーマット
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # ルートロガー設定
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        
        # コンソールハンドラー
        if not root_logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # ファイルハンドラー（設定されている場合）
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                file_handler = logging.FileHandler(log_path, encoding='utf-8')
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
                logger.info(f"Logging to file: {log_path}")
            except Exception as e:
                logger.warning(f"Failed to setup file logging: {e}")


# グローバル設定インスタンス
_global_config = None


def get_config(config_path: Optional[str] = None) -> ConfigLoader:
    """グローバル設定インスタンスを取得
    
    Args:
        config_path: 設定ファイルパス（初回のみ有効）
        
    Returns:
        ConfigLoaderインスタンス
    """
    global _global_config
    
    if _global_config is None:
        _global_config = ConfigLoader(config_path)
        _global_config.setup_logging()
    
    return _global_config


def reset_config():
    """グローバル設定をリセット（テスト用）"""
    global _global_config
    _global_config = None