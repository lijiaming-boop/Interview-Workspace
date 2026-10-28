# HVAC Energy Agent — Next.js 前端镜像
FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json* ./
# 国内网络优先 npmmirror，海外/失败回落官方源
RUN npm ci --registry=https://registry.npmmirror.com \
    || npm ci --registry=https://registry.npmjs.org \
    || npm install
COPY . .
ARG NEXT_PUBLIC_API_BASE=/api
ENV NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE
# rewrites 目标在 next build 时即写入 routes-manifest.json，必须在构建期可解析
ARG API_INTERNAL_URL=http://api:8000
ENV API_INTERNAL_URL=$API_INTERNAL_URL
RUN npm run build

FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app ./
EXPOSE 3000
CMD ["npm", "run", "start"]
