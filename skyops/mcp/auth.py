"""Lecture 7.4: authorisation for the SkyOps MCP server (an OAuth 2.1 resource server).

What the server checks on every request (spec 2026-07-28, basic/authorization):
  - a bearer token in the Authorization header (never the query string); none or invalid -> 401
  - issued by the authorization server we trust (`iss`), not expired (`exp`)
  - issued FOR THIS SERVER: the audience (`aud`, RFC 8707 resource indicator) is our canonical URI.
    A token minted for any other resource is refused, even if everything else is valid. Accepting it
    would be token passthrough's first half; forwarding it downstream would be the second.
  - the scopes the operation needs; missing -> 403 with WWW-Authenticate error="insufficient_scope" and
    the full set of scopes the operation needs, in one challenge.

What a client must check (lecture 7.4, `check_authorization_response`): the `iss` returned with the
authorization code, against the issuer it recorded from validated metadata (RFC 9207), BEFORE sending
the code to any token endpoint.

The course mints HS256 test tokens with a local key so everything runs offline. A real deployment verifies
RS256/ES256 signatures against the authorization server's published keys (JWKS); only `_decode` changes.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import jwt
from mcp.server.auth.provider import AccessToken

ISSUER = "https://auth.skyops.example"            # the authorization server SkyOps trusts (synthetic)
RESOURCE = "https://mcp.skyops.example/mcp"        # flight-ops' canonical URI = the audience it accepts
CREW_RESOURCE = "https://crew.skyops.example/mcp"  # crew-ops' canonical URI
TEST_KEY = "skyops-course-test-signing-key-not-secret-0123456789"   # offline tests only, never in production

# Scopes per operation. Reads are cheap to grant; each write has its own scope (least privilege).
TOOL_SCOPES = {
    "list_departures": ["flights:read"],
    "get_flight_status": ["flights:read"],
    "get_crew_status": ["crew:read"],
    "notify_passengers": ["passengers:notify"],
    "issue_voucher": ["vouchers:issue"],
}
RESOURCE_SCOPES = {"skyops://bookings/*": ["bookings:read"]}
BASELINE = ["flights:read"]                        # scopes_supported: the minimum for basic use


def mint(sub: str, scopes: list[str], *, aud: str = RESOURCE, iss: str = ISSUER, ttl_s: int = 600,
         client_id: str = "skyops-agent", key: str = TEST_KEY, now: float | None = None,
         act: str | None = None) -> str:
    """Test tokens only: what an authorization server would issue after the user consents.
    `act` names the party acting on the caller's behalf after a token exchange (RFC 8693 `act` claim)."""
    t = int(now if now is not None else time.time())
    claims = {"iss": iss, "aud": aud, "sub": sub, "client_id": client_id, "scope": " ".join(scopes),
              "iat": t, "exp": t + ttl_s}
    if act:
        claims["act"] = {"sub": act}
    return jwt.encode(claims, key, algorithm="HS256")


@dataclass
class JwtVerifier:
    """The SDK's TokenVerifier protocol: return an AccessToken, or None for 'not valid here' (-> 401)."""
    issuer: str = ISSUER
    audience: str = RESOURCE
    key: str = TEST_KEY
    jwks_url: str | None = None          # production: the authorization server's published keys (asymmetric)
    leeway_s: int = 30
    rejected: list[str] = field(default_factory=list)      # reasons, for logs and tests; never sent to the client

    def _decode(self, token: str) -> dict[str, Any]:
        if self.jwks_url:
            key, algs = jwt.PyJWKClient(self.jwks_url).get_signing_key_from_jwt(token).key, ["RS256", "ES256"]
        else:
            key, algs = self.key, ["HS256"]
        return jwt.decode(token, key, algorithms=algs, audience=self.audience, issuer=self.issuer,
                          leeway=self.leeway_s, options={"require": ["exp", "iss", "aud", "sub"]})

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            claims = self._decode(token)
        except jwt.InvalidAudienceError:
            self.rejected.append("wrong_audience")
            return None
        except jwt.InvalidIssuerError:
            self.rejected.append("wrong_issuer")
            return None
        except jwt.ExpiredSignatureError:
            self.rejected.append("expired")
            return None
        except jwt.PyJWTError as e:
            self.rejected.append(type(e).__name__)
            return None
        return AccessToken(token=token, client_id=claims.get("client_id", "unknown"),
                           scopes=claims.get("scope", "").split(), expires_at=claims["exp"],
                           resource=self.audience, subject=claims["sub"],
                           claims={"iss": claims["iss"], **({"act": claims["act"]} if "act" in claims else {})})


def verifier_from_env(audience: str = RESOURCE, *, dev: bool = False) -> JwtVerifier:
    """Lecture 7.10: the verifier a deployed server uses. SKYOPS_JWKS_URL (and SKYOPS_ISSUER) name the real
    authorization server. Without them the server refuses to start, unless --dev asks for the course test key,
    which is published in this repository and so proves nothing about who signed a token."""
    url = os.environ.get("SKYOPS_JWKS_URL")
    if url:
        return JwtVerifier(issuer=os.environ.get("SKYOPS_ISSUER", ISSUER), audience=audience, jwks_url=url)
    if not dev:
        raise SystemExit("refusing to start: set SKYOPS_JWKS_URL (and SKYOPS_ISSUER), or pass --dev to accept "
                         "tokens signed with the course test key")
    return JwtVerifier(audience=audience)


def required_scopes(method: str, name: str | None, baseline: list[str] = BASELINE,
                    tool_scopes: dict[str, list[str]] | None = None) -> list[str]:
    """Scopes an MCP request needs, from its method and Mcp-Name. Discovery and lists need the server's baseline.
    `tool_scopes` lets another server (Section 9's procedures server) bring its own table; default TOOL_SCOPES."""
    from fnmatch import fnmatch
    if method == "tools/call" and name:
        return (tool_scopes or TOOL_SCOPES).get(name, ["admin"])   # unknown tool: nobody gets it by accident
    if method == "resources/read" and name:
        return next((s for p, s in RESOURCE_SCOPES.items() if fnmatch(name, p)), ["admin"])
    return baseline


def check_authorization_response(params: dict[str, str], recorded_issuer: str, iss_supported: bool) -> str:
    """Client side, lecture 7.4: RFC 9207 as the spec's table states it. Returns the code, or raises.

    supported + present -> compare exactly · supported + absent -> reject
    not supported + present -> compare exactly · not supported + absent -> proceed
    Exact string comparison: no case folding, no trailing-slash or port normalisation. On a mismatch,
    don't act on (or display) any error fields in the response either.
    """
    iss = params.get("iss")
    if iss is None and iss_supported:
        raise PermissionError("authorization response has no iss, but the server advertises it: reject")
    if iss is not None and iss != recorded_issuer:
        raise PermissionError("authorization response iss does not match the recorded issuer: possible mix-up")
    if "error" in params:
        raise PermissionError(f"authorization failed: {params['error']}")
    return params["code"]
