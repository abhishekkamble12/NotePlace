"""SSRF tests. Real local HTTP servers; the only seam is the test harness allowing ITS OWN origin through the loopback ban.
Everything else in app.core.fetcher is the real code. xfail(strict) = a real, reproduced hole (AUDIT.md section 6, 'SSRF-safe fetcher')."""
import asyncio
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import pytest

import app.core.fetcher as fetcher
from app.core.fetcher import SSRFSecurityError, fetch_url_snapshot, is_ip_allowed, validate_url_security

pytestmark = pytest.mark.sec
PAGE = "<html><body><article>" + "<p>The Acme lab reported a peak efficiency of 33.9 percent for the cell.</p>" * 30 + "</article></body></html>"


class Srv:
    def __init__(self, handler_cls):
        self.hits: list[str] = []
        self.sent = 0
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        self.httpd.owner = self
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        o = self.server.owner
        o.hits.append(self.path)
        if self.path == "/redir":
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{o.target_port}/secret")
            self.end_headers()
        elif self.path == "/big":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(20 * 1024 * 1024))
            self.end_headers()
            try:
                for _ in range(320):
                    self.wfile.write(b"x" * 65536)
                    o.sent += 65536
            except OSError:
                pass
        else:
            ctype = "image/png" if self.path == "/png" else "text/html; charset=utf-8"
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


@pytest.fixture
def servers(monkeypatch):
    a, b = Srv(H), Srv(H)
    a.target_port = b.port
    real = fetcher.validate_url_security

    def allow_only_a(url):  # harness seam: A is "the public site"; everything else (incl. B) stays under the real rules
        u = urlparse(url)
        return "127.0.0.1" if u.hostname == "127.0.0.1" and u.port == a.port else real(url)

    monkeypatch.setattr(fetcher, "validate_url_security", allow_only_a)
    yield a, b
    a.close()
    b.close()


# ---------- pure validator matrix (expected = blocked by construction) ----------
@pytest.mark.parametrize("url", [
    "http://127.0.0.1/", "http://localhost/", "http://localhost./", "http://[::1]/", "http://[::ffff:127.0.0.1]/",
    "http://[::ffff:7f00:1]/", "http://2130706433/", "http://0x7f000001/", "http://0177.0.0.1/", "http://127.1/",
    "http://169.254.169.254/latest/meta-data/", "http://[fd00::1]/", "http://[fe80::1]/", "http://10.0.0.1/",
    "http://172.16.0.1/", "http://192.168.1.1/", "http://100.64.0.1/", "http://0.0.0.0/", "http://user@127.0.0.1/",
    "http://127.0.0.1:80@evil.example/".replace("evil.example", "127.0.0.1"),
    "file:///etc/passwd", "ftp://example.com/", "gopher://127.0.0.1/", "javascript:alert(1)", "http:///nohost",
])
def test_validator_blocks_dangerous_urls(url):
    with pytest.raises(SSRFSecurityError):
        validate_url_security(url)


def test_validator_allows_a_public_ip_literal_positive_control():
    assert validate_url_security("http://93.184.216.34/page") == "93.184.216.34"
    assert is_ip_allowed("93.184.216.34") and is_ip_allowed("2606:4700:4700::1111")


# ---------- fetcher behaviour against real servers ----------
@pytest.mark.asyncio
async def test_harness_positive_control_fetch_works(servers):
    a, _ = servers
    domain, h, text = await fetch_url_snapshot(f"http://127.0.0.1:{a.port}/ok")
    assert "33.9" in text and len(h) == 64


@pytest.mark.asyncio
async def test_redirect_to_blocked_host_is_rejected(servers):
    a, _ = servers
    with pytest.raises(SSRFSecurityError):
        await fetch_url_snapshot(f"http://127.0.0.1:{a.port}/redir")


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, reason="AUDIT s6 SSRF: redirect hops are fetched BEFORE only the final URL is validated; the internal host receives the request")
async def test_redirect_target_is_never_contacted(servers):
    a, b = servers
    with pytest.raises(SSRFSecurityError):
        await fetch_url_snapshot(f"http://127.0.0.1:{a.port}/redir")
    assert b.hits == [], f"internal host was contacted: {b.hits}"


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, reason="AUDIT s6 SSRF: 2 MB cap is enforced after the whole body is downloaded")
async def test_oversize_body_is_cut_off_near_the_cap(servers):
    a, _ = servers
    with pytest.raises(Exception):
        await fetch_url_snapshot(f"http://127.0.0.1:{a.port}/big")
    assert a.sent < 4 * 1024 * 1024, f"server pushed {a.sent} bytes before the client stopped"


@pytest.mark.asyncio
async def test_oversize_body_is_rejected_at_all(servers):
    a, _ = servers
    with pytest.raises(ValueError):
        await fetch_url_snapshot(f"http://127.0.0.1:{a.port}/big")


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, reason="AUDIT s6 SSRF: no content-type allowlist; image/png is parsed as text")
async def test_non_text_content_type_is_rejected(servers):
    a, _ = servers
    with pytest.raises(Exception):
        await fetch_url_snapshot(f"http://127.0.0.1:{a.port}/png")


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, reason="AUDIT s6 SSRF: DNS is resolved once for validation and again by httpx (rebinding TOCTOU)")
async def test_dns_rebinding_does_not_reach_loopback(monkeypatch):
    a = Srv(H)
    a.target_port = 0
    seen = {"validate": 0, "connect": 0}
    real_sock = socket.getaddrinfo
    real_loop = asyncio.base_events.BaseEventLoop.getaddrinfo

    def check_time(host, *args, **kw):  # resolver used by validate_url_security: hostile DNS answers "public"
        if host == "rebind.test":
            seen["validate"] += 1
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]
        return real_sock(host, *args, **kw)

    async def connect_time(self, host, port, **kw):  # resolver used by httpx/asyncio: the same name now answers loopback
        if (host.decode() if isinstance(host, bytes) else host) == "rebind.test":  # httpx passes bytes
            seen["connect"] += 1
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]
        return await real_loop(self, host, port, **kw)

    monkeypatch.setattr(socket, "getaddrinfo", check_time)
    monkeypatch.setattr(asyncio.base_events.BaseEventLoop, "getaddrinfo", connect_time)
    try:
        try:
            await fetch_url_snapshot(f"http://rebind.test:{a.port}/ok")
        except Exception:
            pass
        assert seen["validate"] >= 1 and seen["connect"] >= 1, f"simulation did not engage both resolvers: {seen}"
        assert a.hits == [], "request reached the loopback server after DNS flipped"
    finally:
        a.close()
