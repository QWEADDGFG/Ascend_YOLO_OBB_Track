#!/bin/bash
# chmod +x run_compile.sh
set -e  # 遇到错误立即退出

echo "[INFO] 开始编译源代码..."
cd build
cmake ..
make
echo "[INFO] 编译完成."