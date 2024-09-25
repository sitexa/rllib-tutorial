#!/bin/bash

# 进入文件所在目录
# cd /path/to/your/folder || exit

# 重命名 "best" 文件
best_count=0
for file in best_*.gif; do
    # 使用printf将计数器格式化为三位数
    new_name=$(printf "best_%03d.gif" "$best_count")
    mv -- "$file" "$new_name"
    best_count=$((best_count + 1))
done

# 重命名 "worst" 文件
worst_count=0
for file in worst_*.gif; do
    # 使用printf将计数器格式化为三位数
    new_name=$(printf "worst_%03d.gif" "$worst_count")
    mv -- "$file" "$new_name"
    worst_count=$((worst_count + 1))
done
