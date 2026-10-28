# HVAC Energy Agent — FastAPI 后端镜像
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 依赖先行（利用层缓存）
COPY requirements.txt .
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn -r requirements.txt \
    || pip install -r requirements.txt

# 业务代码（调研表模板/skills 为运行必需资产）
COPY core ./core
COPY config ./config
COPY agent ./agent
COPY evaluation ./evaluation
COPY apps/api ./apps/api
COPY simulation ./simulation
COPY quote_module ./quote_module
COPY .claude/skills .claude/skills
COPY "00_sample/模板测算表" "00_sample/模板测算表"

# P2 改造：生产以 Alembic 迁移建表（不再依赖 create_all），镜像需带迁移资产
COPY alembic.ini ./
COPY migrations ./migrations

# tmp：会话工作目录（生产挂卷）；data：SQLite 开发库（生产用 Postgres 不需要）
# P0 加固：以非 root 运行（容器被攻破时不获得宿主 root）
RUN mkdir -p tmp data \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
VOLUME ["/app/tmp"]

USER appuser
EXPOSE 8000

# 健康检查：探活端点匿名可达
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status==200 else 1)"

# P2 改造：启动前执行数据库迁移（幂等），再起服务
CMD ["sh", "-c", "alembic upgrade head && uvicorn apps.api.main:app --host 0.0.0.0 --port 8000"]
