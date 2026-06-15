# 数据迁移方案

## 第一步：本地导出数据库

```bash
# 等爬虫跑完后执行：
docker exec retail-sentiment-alpha-postgres-1 \
  pg_dump -U alpha -d sentiment_alpha \
  -Fc --no-owner --no-acl \
  -f /tmp/sentiment_alpha.dump

# 从容器拷出来
docker cp retail-sentiment-alpha-postgres-1:/tmp/sentiment_alpha.dump ./sentiment_alpha.dump

# 上传到服务器
scp sentiment_alpha.dump user@your-server:/home/user/
```

> `-Fc` 是压缩格式，导入用 `pg_restore`；比纯 SQL 快、体积小。

---

## 第二步：服务器部署 PostgreSQL

```bash
# 方式 A：Docker（推荐，与本机一致）
scp docker-compose.yml user@your-server:/home/user/retail/
ssh user@your-server
cd /home/user/retail
docker compose up -d

# 方式 B：直接装 PostgreSQL 16
sudo apt install postgresql-16
sudo -u postgres createuser alpha -P   # 密码 alpha123
sudo -u postgres createdb sentiment_alpha -O alpha
```

---

## 第三步：服务器导入数据

```bash
# Docker 方式
docker exec -i retail-postgres-1 \
  pg_restore -U alpha -d sentiment_alpha \
  --no-owner --no-acl \
  < sentiment_alpha.dump

# 直接 PostgreSQL 方式
pg_restore -U alpha -d sentiment_alpha \
  -h localhost --no-owner --no-acl \
  sentiment_alpha.dump
```

---

## 第四步：同步项目代码

```bash
# 方式 A：git（推荐）
git init && git add -A && git commit -m "init"
git remote add origin git@github.com:your/repo.git
git push
# 服务器上 git clone

# 方式 B：rsync
rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude '*.dump' \
  /Users/oleander/Projects/retail-sentiment-alpha/ \
  user@your-server:/home/user/retail/
```

---

## 第五步：服务器环境配置

```bash
# 安装 Python 依赖
cd /home/user/retail
pip install uv
uv sync

# 配置 .env
cat > .env << 'EOF'
DATABASE_URL=postgresql://alpha:alpha123@localhost:5432/sentiment_alpha
TGB_COOKIE=你的淘股吧Cookie（如果还要爬）
EOF
```

---

## 第六步：验证

```bash
# 测试数据库连接
uv run python -c "
from crawlers.config import engine
with engine.connect() as c:
    r = c.execute('SELECT count(*) FROM posts')
    print(f'Posts: {r.scalar()}')
"

# 有 GPU 的话，NLP 可以用 CUDA 加速
uv run python -c "
from nlp.sentiment import run_sentiment_pipeline
n = run_sentiment_pipeline(limit=10, device='cuda')
print(f'Test scored: {n}')
"

# 无 GPU，照常用 CPU
uv run python -c "
from nlp.sentiment import run_sentiment_pipeline
n = run_sentiment_pipeline(limit=10)
print(f'Test scored: {n}')
"

# 跑全流程
uv run python run_pipeline.py
```

---

## 注意事项

| 问题 | 说明 |
|---|---|
| **FinBERT 模型** | 首次运行会自动下载 ~410MB，服务器上也要下载一次 |
| **baostock** | 服务器 IP 可能被限速，market_data.py 已有 0.2s 间隔 |
| **TGB_COOKIE** | 如果不再爬淘股吧可以为空，东方财富不需要 Cookie |
| **GPU 加速** | 服务器有 NVIDIA GPU → NLP 用 `device='cuda'` 能快 10 倍 |
| **端口** | Docker PG 映射 5432，服务器上确认端口不冲突 |
