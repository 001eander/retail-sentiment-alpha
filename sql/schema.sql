-- 淘股吧帖子原始数据
CREATE TABLE IF NOT EXISTS posts (
    id              SERIAL PRIMARY KEY,
    platform        VARCHAR(20) NOT NULL DEFAULT 'taoguba',
    stock_code      VARCHAR(10) NOT NULL,
    stock_name      VARCHAR(20),
    title           TEXT,
    content         TEXT,
    author          VARCHAR(50),
    post_time       TIMESTAMP NOT NULL,
    read_count      INT DEFAULT 0,
    reply_count     INT DEFAULT 0,
    url             TEXT UNIQUE NOT NULL,
    sentiment       REAL,           -- FinBERT [−1, 1]
    fetched_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_posts_stock_time ON posts(stock_code, post_time);

-- 日频因子
CREATE TABLE IF NOT EXISTS daily_factors (
    stock_code               VARCHAR(10) NOT NULL,
    trade_date               DATE NOT NULL,
    post_volume_anomaly      REAL,     -- 关注度异常
    sentiment_score          REAL,     -- 情绪均值 (FinBERT)
    sentiment_divergence     REAL,     -- 情绪分歧 (日 std)
    interaction_intensity    REAL,     -- 互动热度
    PRIMARY KEY (stock_code, trade_date)
);

-- 行情
CREATE TABLE IF NOT EXISTS market_daily (
    stock_code   VARCHAR(10) NOT NULL,
    trade_date   DATE NOT NULL,
    open         REAL,
    high         REAL,
    low          REAL,
    close        REAL,
    pre_close    REAL,
    volume       BIGINT,
    amount       REAL,        -- 成交额
    turnover     REAL,        -- 换手率
    PRIMARY KEY (stock_code, trade_date)
);

-- 沪深300 指数
CREATE TABLE IF NOT EXISTS index_daily (
    idx_code     VARCHAR(10) NOT NULL DEFAULT '000300',
    trade_date   DATE NOT NULL,
    close        REAL,
    PRIMARY KEY (idx_code, trade_date)
);
