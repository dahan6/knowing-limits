#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    echo "正在创建虚拟环境..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install/update dependencies if requirements changed
pip install -q -r requirements.txt

echo "启动知识库服务..."
python app.py
