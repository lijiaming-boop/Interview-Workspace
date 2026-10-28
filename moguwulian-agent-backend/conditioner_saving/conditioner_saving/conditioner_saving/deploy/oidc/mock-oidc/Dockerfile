# 测试 OIDC Provider（迷你 mock，授权码模式 + RS256）
# 零第三方依赖，base 镜像与 web 共用 node:20-alpine（无需额外拉取）
FROM node:20-alpine
WORKDIR /app
RUN addgroup -S oidc && adduser -S oidc -G oidc
COPY server.js ./
USER oidc
EXPOSE 8080
CMD ["node", "server.js"]
