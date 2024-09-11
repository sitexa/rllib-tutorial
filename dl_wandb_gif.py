import wandb
import os
import requests

# 设置您的 wandb API 密钥
wandb.login(key="9c5b48cc3eb8716aa51b2eb3d0237b4cc5b962fa")

# 设置项目名称和实体（通常是您的用户名或组织名）
project_name = "rllib-tuto"
entity = "sitexa-com"

# 设置下载目录
download_dir = "wandb_gifs"
os.makedirs(download_dir, exist_ok=True)

# 获取项目中的所有运行
api = wandb.Api()
runs = api.runs(f"{entity}/{project_name}")

for run in runs:
    # 获取运行中的所有文件
    files = run.files()
    
    for file in files:
        # 检查文件是否为 .gif
        if file.name.endswith('.gif'):
            # 构建下载 URL
            url = f"https://api.wandb.ai/files/{entity}/{project_name}/{run.id}/{file.name}"
            
            # 下载文件
            response = requests.get(url, headers={"Authorization": f"Bearer {wandb.api.api_key}"})
            
            if response.status_code == 200:
                # 创建保存文件的完整路径
                file_path = os.path.join(download_dir, f"{run.id}_{file.name}")
                
                # 确保文件的目录存在
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # 保存文件
                file_path = os.path.join(download_dir, f"{run.id}_{file.name}")
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded: {file_path}")
            else:
                print(f"Failed to download: {file.name}")

print("Download complete!")