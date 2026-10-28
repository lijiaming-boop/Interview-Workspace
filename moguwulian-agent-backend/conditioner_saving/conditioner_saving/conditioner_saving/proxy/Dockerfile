# 认证网关镜像（OIDC 登录 + 身份头 HMAC 签名）
# P1 认证闭环：proxy 是唯一入口，将签名后的身份头经 web(BFF) 透传给 api
FROM golang:1.25-alpine AS build
# goproxy.cn 国内模块代理，direct 兜底（海外网络同样可用）
ENV GOPROXY=https://goproxy.cn,direct
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY main.go .
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o /out/conditioner-proxy main.go

FROM alpine:3.20
RUN adduser -D -u 10001 proxy
COPY --from=build /out/conditioner-proxy /usr/local/bin/conditioner-proxy
USER proxy
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/conditioner-proxy"]
