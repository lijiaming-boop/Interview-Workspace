// 迷你 OIDC Provider（仅测试用途）：授权码模式 + RS256 ID Token + userinfo。
// 面向 proxy 网关（go-oidc / golang.org/x/oauth2）的最小实现面：
//   - /.well-known/openid-configuration：issuer 取自请求 Host（与 OAUTH_SERVER_URL 全等即可通过 go-oidc 校验）
//   - /auth → 登录表单 → /login-submit → 302 回 redirect_uri?code&state
//   - /token：client_secret_basic 与 client_secret_post 都接受（go-oauth2 自动探测）
//   - /jwks.json、/userinfo、/api/userinfo（网关按 Casdoor 习惯先打 /api/userinfo）
// 默认账号 test / test123456（realm 角色 sr = API 特权角色），可用 MOCK_USERS 覆盖。
const http = require("http");
const crypto = require("crypto");

const PORT = Number(process.env.PORT || 8080);
const CLIENT_ID = process.env.CLIENT_ID || "hvac-proxy";
const CLIENT_SECRET = process.env.CLIENT_SECRET || "hvac-proxy-secret";
const REDIRECT_URI = process.env.REDIRECT_URI || "";
const USERS = JSON.parse(process.env.MOCK_USERS || "null") || [
  { username: "test", password: "test123456", roles: ["sr"],
    name: "Test User", email: "test@hvac.local" },
];

const { publicKey, privateKey } = crypto.generateKeyPairSync("rsa", { modulusLength: 2048 });
const KID = "mock-key-1";
const b64u = (s) => Buffer.from(s).toString("base64url");

function signJwt(claims) {
  const header = b64u(JSON.stringify({ alg: "RS256", typ: "JWT", kid: KID }));
  const payload = b64u(JSON.stringify(claims));
  const sig = crypto.createSign("RSA-SHA256")
    .update(`${header}.${payload}`).sign(privateKey, "base64url");
  return `${header}.${payload}.${sig}`;
}

const authRequests = new Map(); // state -> {nonce, redirect_uri, exp}
const codes = new Map();        // code  -> {user, nonce, redirect_uri, exp}
const accessTokens = new Map(); // token -> {user, exp}
const TTL = 10 * 60 * 1000;
setInterval(() => {
  const now = Date.now();
  for (const m of [authRequests, codes, accessTokens]) {
    for (const [k, x] of m) if (x.exp < now) m.delete(k);
  }
}, 60_000).unref();

const originOf = (req) => `http://${req.headers.host}`;

function readBody(req) {
  return new Promise((resolve, reject) => {
    let buf = "";
    req.on("data", (c) => { buf += c; if (buf.length > 1e6) req.destroy(); });
    req.on("end", () => resolve(buf));
    req.on("error", reject);
  });
}

const json = (res, code, obj) => {
  res.writeHead(code, { "Content-Type": "application/json" });
  res.end(JSON.stringify(obj));
};
const redirect = (res, location) => {
  res.writeHead(302, { Location: location });
  res.end();
};

const LOGIN_PAGE = (action) => `<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>Sign in · Test IdP</title>
<style>body{font-family:system-ui;display:grid;place-items:center;height:100vh;margin:0;background:#f5f6f8}
form{background:#fff;padding:32px;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,.08);min-width:280px}
h1{font-size:18px;margin:0 0 16px}input{width:100%;box-sizing:border-box;padding:9px;margin:4px 0 12px;border:1px solid #ccc;border-radius:6px}
button{width:100%;padding:10px;border:0;border-radius:6px;background:#2563eb;color:#fff;font-size:14px;cursor:pointer}
p{font-size:12px;color:#888}</style></head><body>
<form method="post" action="${action}">
<h1>HVAC Energy Agent 登录（测试 IdP）</h1>
<input name="username" placeholder="用户名" value="test" autofocus>
<input name="password" type="password" placeholder="密码" value="test123456">
<button type="submit">Sign In</button>
<p>测试账号：test / test123456</p>
</form></body></html>`;

const server = http.createServer(async (req, res) => {
  const u = new URL(req.url, originOf(req));
  const p = u.pathname;

  if (p === "/health") return json(res, 200, { status: "ok" });

  if (p === "/.well-known/openid-configuration") {
    const issuer = originOf(req);
    return json(res, 200, {
      issuer,
      authorization_endpoint: `${issuer}/auth`,
      token_endpoint: `${issuer}/token`,
      userinfo_endpoint: `${issuer}/userinfo`,
      jwks_uri: `${issuer}/jwks.json`,
      response_types_supported: ["code"],
      grant_types_supported: ["authorization_code"],
      subject_types_supported: ["public"],
      id_token_signing_alg_values_supported: ["RS256"],
      scopes_supported: ["openid", "email", "profile"],
      token_endpoint_auth_methods_supported: ["client_secret_basic", "client_secret_post"],
      claims_supported: ["sub", "name", "email", "preferred_username", "roles"],
    });
  }

  if (p === "/jwks.json") {
    const j = publicKey.export({ format: "jwk" });
    return json(res, 200, { keys: [
      { kty: j.kty, n: j.n, e: j.e, kid: KID, alg: "RS256", use: "sig" },
    ] });
  }

  if (p === "/auth") {
    const clientId = u.searchParams.get("client_id");
    const responseMode = u.searchParams.get("response_type");
    const redirectUri = u.searchParams.get("redirect_uri") || "";
    const state = u.searchParams.get("state");
    const nonce = u.searchParams.get("nonce");
    if (clientId !== CLIENT_ID) return json(res, 400, { error: "invalid client_id" });
    if (responseMode !== "code") return json(res, 400, { error: "unsupported response_type" });
    if (REDIRECT_URI && redirectUri !== REDIRECT_URI) {
      return json(res, 400, { error: "redirect_uri mismatch" });
    }
    if (!state || !nonce) return json(res, 400, { error: "missing state/nonce" });
    authRequests.set(state, { nonce, redirect_uri: redirectUri, exp: Date.now() + TTL });
    return sendHtml(res, LOGIN_PAGE(`${originOf(req)}/login-submit?state=${encodeURIComponent(state)}`));
  }

  if (p === "/login-submit" && req.method === "POST") {
    const body = new URLSearchParams(await readBody(req));
    const state = u.searchParams.get("state");
    const pending = state && authRequests.get(state);
    if (!pending) return json(res, 400, { error: "unknown or expired state" });
    const user = USERS.find(
      (x) => x.username === body.get("username") && x.password === body.get("password"),
    );
    if (!user) return sendHtml(res, LOGIN_PAGE(`${originOf(req)}/login-submit?state=${state}`), 401);
    authRequests.delete(state);
    const code = crypto.randomBytes(24).toString("base64url");
    codes.set(code, { user, nonce: pending.nonce, redirect_uri: pending.redirect_uri,
      exp: Date.now() + TTL });
    const sep = pending.redirect_uri.includes("?") ? "&" : "?";
    return redirect(res, `${pending.redirect_uri}${sep}code=${code}&state=${encodeURIComponent(state)}`);
  }

  if (p === "/token" && req.method === "POST") {
    const body = new URLSearchParams(await readBody(req));
    let clientId = body.get("client_id");
    let clientSecret = body.get("client_secret");
    const auth = req.headers.authorization || "";
    if (auth.startsWith("Basic ")) {
      const [id, sec] = Buffer.from(auth.slice(6), "base64").toString().split(":");
      clientId = clientId || id;
      clientSecret = clientSecret || sec;
    }
    if (clientId !== CLIENT_ID || clientSecret !== CLIENT_SECRET) {
      return json(res, 401, { error: "invalid_client" });
    }
    if (body.get("grant_type") !== "authorization_code") {
      return json(res, 400, { error: "unsupported_grant_type" });
    }
    const rec = codes.get(body.get("code") || "");
    if (!rec || rec.exp < Date.now()
        || (body.get("redirect_uri") && body.get("redirect_uri") !== rec.redirect_uri)) {
      return json(res, 400, { error: "invalid_grant" });
    }
    codes.delete(body.get("code"));
    const now = Math.floor(Date.now() / 1000);
    const idToken = signJwt({
      iss: originOf(req),
      sub: rec.user.username,
      aud: CLIENT_ID,
      nonce: rec.nonce,
      iat: now,
      exp: now + 3600,
      name: rec.user.name,
      email: rec.user.email,
      preferred_username: rec.user.username,
      roles: rec.user.roles || [],
    });
    const accessToken = crypto.randomBytes(32).toString("base64url");
    accessTokens.set(accessToken, { user: rec.user, exp: Date.now() + TTL });
    return json(res, 200, {
      access_token: accessToken,
      token_type: "Bearer",
      expires_in: 3600,
      id_token: idToken,
    });
  }

  if ((p === "/userinfo" || p === "/api/userinfo")) {
    const auth = req.headers.authorization || "";
    const rec = auth.startsWith("Bearer ") ? accessTokens.get(auth.slice(7)) : null;
    if (!rec || rec.exp < Date.now()) {
      return json(res, 401, { error: "invalid_token" });
    }
    return json(res, 200, {
      sub: rec.user.username,
      name: rec.user.name,
      email: rec.user.email,
      preferred_username: rec.user.username,
      roles: rec.user.roles || [],
    });
  }

  json(res, 404, { error: "not_found" });
});

function sendHtml(res, html, code = 200) {
  res.writeHead(code, { "Content-Type": "text/html; charset=utf-8" });
  res.end(html);
}

server.listen(PORT, () => {
  console.log(`mock-oidc listening on :${PORT}  issuer=<derived from Host>  client=${CLIENT_ID}`);
});
