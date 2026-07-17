"""项目配置文件"""
import os
from dotenv import load_dotenv

load_dotenv()

# 系统用 LLM 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "")

# 模拟学生用 LLM 配置
SIMULATOR_API_KEY = os.getenv("SIMULATOR_API_KEY", "")
SIMULATOR_BASE_URL = os.getenv("SIMULATOR_BASE_URL", "")
SIMULATOR_MODEL = os.getenv("SIMULATOR_MODEL", "")

# 数据存储目录
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Semantic Scholar API Key（可选）
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

# 代理配置（可选）
HTTP_PROXY = os.getenv("HTTP_PROXY", "")
HTTPS_PROXY = os.getenv("HTTPS_PROXY", "")
