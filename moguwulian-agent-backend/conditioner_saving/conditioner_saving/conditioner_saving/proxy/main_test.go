package main

import (
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"
)

func jsonMarshal(v interface{}) ([]byte, error) {
	return json.Marshal(v)
}

func TestSignedIdentityCookieRoundTrip(t *testing.T) {
	cookieSecret = []byte("test-secret")
	original := UserInfo{Sub: "user-1", Name: "张三", Email: "u@example.com", Role: "user"}

	encoded, err := encodeSignedIdentityCookie(original)
	if err != nil {
		t.Fatalf("encode failed: %v", err)
	}
	decoded, ok := decodeSignedIdentityCookie(encoded)
	if !ok {
		t.Fatalf("valid signed cookie failed verification")
	}
	if decoded.Sub != original.Sub || decoded.Name != original.Name {
		t.Fatalf("round-trip mismatch: %+v vs %+v", decoded, original)
	}
}

func TestSignedIdentityCookieRejectsTampering(t *testing.T) {
	cookieSecret = []byte("test-secret")
	original := UserInfo{Sub: "user-1", Name: "张三"}

	encoded, err := encodeSignedIdentityCookie(original)
	if err != nil {
		t.Fatalf("encode failed: %v", err)
	}

	// 1. 篡改 payload（提权攻击：改 sub/role）
	sig := encoded[strings.Index(encoded, ".")+1:]
	for _, forged := range []UserInfo{
		{Sub: "admin-user", Name: "hacker", Role: "admin"},
		{Sub: "user-2", Name: "张三"},
	} {
		forgedJSON := mustJSON(t, forged)
		forgedEncoded := base64.RawURLEncoding.EncodeToString(forgedJSON)
		if _, ok := decodeSignedIdentityCookie(forgedEncoded + "." + sig); ok {
			t.Fatalf("tampered payload accepted (sub=%s)", forged.Sub)
		}
	}

	// 2. 原始明文 JSON Cookie（旧格式，无签名）必须被拒绝
	plaintext := `{"sub":"admin-user","role":"admin"}`
	if _, ok := decodeSignedIdentityCookie(plaintext); ok {
		t.Fatal("unsigned plaintext cookie accepted")
	}

	// 3. 错误密钥签名的 Cookie 必须被拒绝
	cookieSecret = []byte("other-secret")
	if _, ok := decodeSignedIdentityCookie(encoded); ok {
		t.Fatal("cookie signed with different secret accepted")
	}
	cookieSecret = []byte("test-secret")

	// 4. 格式损坏必须被拒绝
	for _, bad := range []string{"", "abc", "abc.def", ".", "a.b.c"} {
		if _, ok := decodeSignedIdentityCookie(bad); ok {
			t.Fatalf("malformed cookie accepted: %q", bad)
		}
	}
}

func mustJSON(t *testing.T, v interface{}) []byte {
	t.Helper()
	// 复用 main.go 的 json 序列化路径
	b, err := jsonMarshal(v)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}
	return b
}
