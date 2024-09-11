#!/bin/bash

# 进入文件所在目录
cd wandb_gifs/edc13_00000_media/videos/env_runners || exit

# 创建输出目录
mkdir -p best worst

# 将文件分类到对应的目录
for file in best_*.gif; do
    mv "$file" best/
done

for file in worst_*.gif; do
    mv "$file" worst/
done

# 生成文件列表
cd best || exit
for f in *.gif; do echo "file '$PWD/$f'" >> file_list.txt; done

# 合成 best.mp4
ffmpeg -f concat -safe 0 -i file_list.txt -vsync vfr -pix_fmt yuv420p ../best.mp4

# 生成文件列表
cd ../worst || exit
for f in *.gif; do echo "file '$PWD/$f'" >> file_list.txt; done

# 合成 worst.mp4
ffmpeg -f concat -safe 0 -i file_list.txt -vsync vfr -pix_fmt yuv420p ../worst.mp4
