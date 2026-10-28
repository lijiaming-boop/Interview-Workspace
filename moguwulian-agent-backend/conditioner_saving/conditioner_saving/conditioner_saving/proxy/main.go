package main

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/coreos/go-oidc/v3/oidc"
	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
	"golang.org/x/oauth2"
)

// OAuth 2.0 配置
var (
	oauthServerURL     string
	clientID           string
	clientSecret       string
	redirectURI        string
	backendURL         string
	cookieSecret       []byte
	sessionCookieName  string
	userInfoCookieName string
	proxyPort          string
)

// 客户端可伪造、必须由网关剥离再重新注入的身份头
var identityHeaders = []string{"X-User-Info", "X-User-Info-Sig"}

// OAuth2 配置
var (
	oauth2Config *oauth2.Config
	provider     *oidc.Provider
	verifier     *oidc.IDTokenVerifier
)

// UserInfo 用户信息结构
type UserInfo struct {
	Sub               string `json:"sub"`
	Name              string `json:"name"`
	Email             string `json:"email"`
	Avatar            string `json:"avatar"`
	Picture           string `json:"picture"` // 有些 OIDC provider 使用 picture 而不是 avatar
	LoginName         string `json:"loginName"`
	PreferredUsername string `json:"preferred_username"` // OIDC 标准字段
	DisplayName       string `json:"displayName"`
	Id                string `json:"id"`
	// 角色声明（API 的 ACL 依赖 role/roles 判定特权角色；IdP 不发则为空）
	Role  string   `json:"role"`
	Roles []string `json:"roles"`
}

// GetDisplayName 获取用户显示名称（优先级：DisplayName > Name > PreferredUsername > LoginName > Email > Sub）
func (u *UserInfo) GetDisplayName() string {
	if u.DisplayName != "" {
		return u.DisplayName
	}
	if u.Name != "" {
		return u.Name
	}
	if u.PreferredUsername != "" {
		return u.PreferredUsername
	}
	if u.LoginName != "" {
		return u.LoginName
	}
	if u.Email != "" {
		return u.Email
	}
	return u.Sub
}

// GetAvatar 获取用户头像（优先级：Avatar > Picture）
func (u *UserInfo) GetAvatar() string {
	if u.Avatar != "" {
		return u.Avatar
	}
	return u.Picture
}

func init() {
	// 加载 .env 文件（仅开发环境）
	godotenv.Load()

	// 从环境变量读取配置
	oauthServerURL = getEnv("OAUTH_SERVER_URL", "https://h-oauth.mogulinker.com")
	clientID = getEnv("CLIENT_ID", "")
	clientSecret = getEnv("CLIENT_SECRET", "")
	redirectURI = getEnv("REDIRECT_URI", "http://localhost:8080/oauth2callback")
	// P1 认证闭环：认证后的下一跳是 Next.js BFF（web），身份头经其
	// rewrites 透传给 api；保留 STREAMLIT_BACKEND 作为旧部署兼容回退
	backendURL = getEnv("BACKEND_URL", getEnv("STREAMLIT_BACKEND", "http://localhost:8501"))
	cookieSecret = []byte(getEnv("COOKIE_SECRET", "change-this-secret-in-production"))
	sessionCookieName = getEnv("SESSION_COOKIE_NAME", "session_token")
	userInfoCookieName = getEnv("USER_INFO_COOKIE_NAME", "user_info")
	proxyPort = getEnv("PROXY_PORT", "8080")

	if len(cookieSecret) == 0 || string(cookieSecret) == "change-this-secret-in-production" {
		log.Printf("⚠️  COOKIE_SECRET 未设置或仍为默认值——身份 Cookie 签名可被离线伪造。" +
			"生产部署必须显式设置（docker-compose 已强制与 API 的 AUTH_SECRET 一致）")
	}

	// 初始化 OIDC Provider，带重试机制
	ctx := context.Background()
	var err error
	retryInterval := 3 * time.Second
	maxRetries := 10 // 最多重试10次，避免无限重试

	for i := 0; i < maxRetries; i++ {
		provider, err = oidc.NewProvider(ctx, oauthServerURL)
		if err == nil {
			log.Printf("Successfully initialized OIDC provider")
			break
		}

		if i < maxRetries-1 {
			log.Printf("Failed to initialize OIDC provider (attempt %d/%d): %v. Retrying in %v...",
				i+1, maxRetries, err, retryInterval)
			time.Sleep(retryInterval)
		} else {
			panic(fmt.Sprintf("Failed to initialize OIDC provider after %d attempts: %v", maxRetries, err))
		}
	}

	// 配置 OAuth2
	oauth2Config = &oauth2.Config{
		ClientID:     clientID,
		ClientSecret: clientSecret,
		RedirectURL:  redirectURI,
		Scopes:       []string{oidc.ScopeOpenID, "email", "profile"},
		Endpoint:     provider.Endpoint(),
	}

	// 配置 ID Token 验证器
	verifier = provider.Verifier(&oidc.Config{
		ClientID: clientID,
	})
}

func main() {
	r := gin.Default()

	// 静态文件和健康检查
	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})

	// OAuth2 回调处理
	r.GET("/oauth2callback", handleCallback)

	// 登录路由
	r.GET("/login", handleLogin)

	// 登出路由
	r.GET("/logout", handleLogout)

	// 认证中间件
	authMiddleware := AuthMiddleware()

	// 反向代理到认证后的后端（web/BFF）- 所有其他请求
	r.NoRoute(authMiddleware, proxyToBackend())

	// 启动服务
	fmt.Println("===========================================")
	fmt.Println("  Conditioner Saving Proxy Server")
	fmt.Println("===========================================")
	fmt.Printf("  OAuth Server: %s\n", oauthServerURL)
	fmt.Printf("  Backend: %s\n", backendURL)
	fmt.Printf("  Redirect URI: %s\n", redirectURI)
	fmt.Printf("  Port: %s\n", proxyPort)
	fmt.Println("===========================================")
	fmt.Printf("  Server starting on http://localhost:%s\n", proxyPort)
	fmt.Println("===========================================")

	if err := r.Run(":" + proxyPort); err != nil {
		panic(fmt.Sprintf("Failed to start server: %v", err))
	}
}

// getEnv 获取环境变量，如果不存在则返回默认值
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// handleLogin 处理登录请求
func handleLogin(c *gin.Context) {
	state := generateRandomString()
	c.SetCookie("oauth_state", state, 300, "/", "", false, true)

	nonce := generateRandomString()
	c.SetCookie("oauth_nonce", nonce, 300, "/", "", false, true)

	authURL := oauth2Config.AuthCodeURL(state, oidc.Nonce(nonce))
	c.Redirect(http.StatusTemporaryRedirect, authURL)
}

// handleCallback 处理 OAuth 回调
func handleCallback(c *gin.Context) {
	state := c.Query("state")
	code := c.Query("code")

	// 验证 state
	savedState, err := c.Cookie("oauth_state")
	if err != nil || state != savedState {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid state"})
		return
	}

	// 获取 nonce
	savedNonce, _ := c.Cookie("oauth_nonce")

	ctx := context.Background()

	// 交换 token
	token, err := oauth2Config.Exchange(ctx, code)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to exchange token: %v", err)})
		return
	}

	// 验证并提取 ID Token
	rawIDToken, ok := token.Extra("id_token").(string)
	if !ok {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "no id_token in token response"})
		return
	}

	idToken, err := verifier.Verify(ctx, rawIDToken)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to verify id_token: %v", err)})
		return
	}

	// 验证 nonce
	if idToken.Nonce != savedNonce {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "invalid nonce"})
		return
	}

	// 获取用户信息 - 优先从 userinfo endpoint 获取完整信息
	var userInfo UserInfo
	userInfoEndpoint := oauthServerURL + "/api/userinfo"
	log.Printf("Fetching user info from: %s", userInfoEndpoint)
	resp, err := oauth2Config.Client(ctx, token).Get(userInfoEndpoint)
	if err != nil {
		// 如果 userinfo endpoint 失败，尝试从 ID Token 获取
		log.Printf("Failed to get userinfo from endpoint: %v, falling back to ID token claims", err)
		if err := idToken.Claims(&userInfo); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to get user info: %v", err)})
			return
		}
	} else {
		defer resp.Body.Close()

		// 读取响应体用于调试
		var rawResponse map[string]interface{}
		if err := json.NewDecoder(resp.Body).Decode(&rawResponse); err != nil {
			// 如果解码失败，尝试从 ID Token 获取
			log.Printf("Failed to decode userinfo: %v, falling back to ID token claims", err)
			if err := idToken.Claims(&userInfo); err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to get user info: %v", err)})
				return
			}
		} else {
			log.Printf("Raw userinfo response: %+v", rawResponse)

			// 将 map 转换为 JSON 再解析到 UserInfo 结构体
			jsonBytes, _ := json.Marshal(rawResponse)
			if err := json.Unmarshal(jsonBytes, &userInfo); err != nil {
				log.Printf("Failed to unmarshal userinfo: %v", err)
			}
		}
	}

	log.Printf("User info retrieved: sub=%s, name=%s, email=%s, loginName=%s, preferredUsername=%s, displayName=%s",
		userInfo.Sub, userInfo.Name, userInfo.Email, userInfo.LoginName, userInfo.PreferredUsername, userInfo.DisplayName)

	// 生成会话 token（使用 access token 作为会话标识）
	sessionToken := token.AccessToken

	// 设置会话 cookie（不设置过期时间，浏览器关闭自动清除）
	c.SetCookie(sessionCookieName, sessionToken, 0, "/", "", false, true)

	// P1 认证闭环修复（审查报告）：身份 Cookie 必须由服务端签名。
	// 原实现把明文 JSON 放进浏览器可修改的 user_info Cookie，网关随后
	// 为"浏览器提交的任何内容"生成 HMAC——等于给伪造身份盖章。
	// 现在写入 base64url(payload).hmac(payload)，密钥只在服务端，
	// 浏览器篡改/伪造身份 Cookie 将在签名校验处失败并回退登录。
	signedCookie, err := encodeSignedIdentityCookie(userInfo)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to encode identity cookie: %v", err)})
		return
	}
	c.SetCookie(userInfoCookieName, signedCookie, 0, "/", "", false, false)

	// 重定向到应用首页
	c.Redirect(http.StatusTemporaryRedirect, "/")
}

// handleLogout 处理登出请求
func handleLogout(c *gin.Context) {
	c.SetCookie(sessionCookieName, "", -1, "/", "", false, true)
	c.SetCookie(userInfoCookieName, "", -1, "/", "", false, false)

	// 重定向到登录页
	c.Redirect(http.StatusTemporaryRedirect, "/")
}

// AuthMiddleware 认证中间件
func AuthMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// 排除 OAuth 回调和登录路由
		if strings.HasPrefix(c.Request.URL.Path, "/oauth2callback") ||
			strings.HasPrefix(c.Request.URL.Path, "/login") ||
			strings.HasPrefix(c.Request.URL.Path, "/health") {
			c.Next()
			return
		}

		// 检查会话 cookie
		sessionToken, err := c.Cookie(sessionCookieName)
		if err != nil || sessionToken == "" {
			// 未登录，重定向到登录页
			c.Redirect(http.StatusTemporaryRedirect, "/login")
			c.Abort()
			return
		}

		// 获取并校验签名的身份 Cookie
		userInfoJSON, err := c.Cookie(userInfoCookieName)
		if err != nil || userInfoJSON == "" {
			// 用户信息不存在，清除会话
			c.SetCookie(sessionCookieName, "", -1, "/", "", false, true)
			c.Redirect(http.StatusTemporaryRedirect, "/login")
			c.Abort()
			return
		}

		// P1 认证闭环修复：签名校验——浏览器伪造/篡改的身份 Cookie 在此被拒
		userInfo, ok := decodeSignedIdentityCookie(userInfoJSON)
		if !ok {
			// 校验失败，清除会话
			c.SetCookie(sessionCookieName, "", -1, "/", "", false, true)
			c.SetCookie(userInfoCookieName, "", -1, "/", "", false, false)
			c.Redirect(http.StatusTemporaryRedirect, "/login")
			c.Abort()
			return
		}

		// 将用户信息存入上下文
		c.Set("user_info", userInfo)
		c.Set("session_token", sessionToken)

		c.Next()
	}
}

// proxyToBackend 反向代理到认证后的后端（web/BFF）
func proxyToBackend() gin.HandlerFunc {
	remote, err := url.Parse(backendURL)
	if err != nil {
		panic(fmt.Sprintf("Failed to parse backend URL: %v", err))
	}

	proxy := httputil.NewSingleHostReverseProxy(remote)
	originalDirector := proxy.Director
	proxy.Director = func(req *http.Request) {
		originalDirector(req)
		req.Host = remote.Host
		req.Header.Set("X-Forwarded-Host", req.Host)
		req.Header.Set("X-Forwarded-Proto", "http")

		// P1 认证闭环：先剥离客户端自带的身份头（防头注入），再注入
		// 网关签名后的身份。客户端无法伪造：签名密钥只在网关/API。
		for _, h := range identityHeaders {
			req.Header.Del(h)
		}
		if userInfo, exists := getuserInfoFromRequest(req); exists {
			userInfoJSON, _ := json.Marshal(userInfo)
			payload := string(userInfoJSON)
			req.Header.Set("X-User-Info", payload)
			// P0 安全改造：对身份头做 HMAC-SHA256 签名（与 API 的 AUTH_SECRET 配对）。
			// 未签名的 X-User-Info 会被 API 拒绝——客户端无法伪造身份。
			req.Header.Set("X-User-Info-Sig", signPayload(payload))
		}
	}

	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		fmt.Printf("Proxy error: %v\n", err)
		w.WriteHeader(http.StatusBadGateway)
		w.Write([]byte(fmt.Sprintf("Backend error: %v", err)))
	}

	return func(c *gin.Context) {
		proxy.ServeHTTP(c.Writer, c.Request)
	}
}

// getuserInfoFromRequest 从请求中获取（签名校验通过的）用户信息
func getuserInfoFromRequest(r *http.Request) (UserInfo, bool) {
	cookie, err := r.Cookie(userInfoCookieName)
	if err != nil || cookie.Value == "" {
		return UserInfo{}, false
	}
	return decodeSignedIdentityCookie(cookie.Value)
}

// encodeSignedIdentityCookie 把用户信息编码为 base64url(payload).hmac 形式
func encodeSignedIdentityCookie(userInfo UserInfo) (string, error) {
	payload, err := json.Marshal(userInfo)
	if err != nil {
		return "", err
	}
	sig := signPayload(string(payload))
	return base64.RawURLEncoding.EncodeToString(payload) + "." + sig, nil
}

// decodeSignedIdentityCookie 校验并解码签名的身份 Cookie：
// 签名不匹配（浏览器伪造/篡改）返回 false
func decodeSignedIdentityCookie(value string) (UserInfo, bool) {
	var userInfo UserInfo
	parts := strings.SplitN(value, ".", 2)
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return userInfo, false
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[0])
	if err != nil {
		return userInfo, false
	}
	// 常量时间比较，防时序侧信道
	if !hmac.Equal([]byte(signPayload(string(payload))), []byte(parts[1])) {
		return userInfo, false
	}
	if err := json.Unmarshal(payload, &userInfo); err != nil {
		return userInfo, false
	}
	return userInfo, true
}

// signPayload 计算身份载荷的 HMAC-SHA256 签名（hex）。
// 密钥即 COOKIE_SECRET，必须与后端 API 的 AUTH_SECRET 一致——
// 两侧共享密钥后，客户端直接向 API 发未签名/错签名的身份头将被 401 拒绝。
func signPayload(payload string) string {
	mac := hmac.New(sha256.New, cookieSecret)
	mac.Write([]byte(payload))
	return hex.EncodeToString(mac.Sum(nil))
}

// generateRandomString 生成随机字符串
func generateRandomString() string {
	b := make([]byte, 32)
	for i := range b {
		b[i] = byte(time.Now().UnixNano() >> (i * 8) & 0xFF)
	}
	return fmt.Sprintf("%x", b)[:32]
}
