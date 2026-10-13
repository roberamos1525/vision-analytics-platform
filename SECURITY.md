# Security

## Reporting

Please report vulnerabilities privately to the maintainer (Robert Ramos)
rather than opening a public issue. Include reproduction steps and the
affected version.

## Security model

The service is designed to run on a trusted host or private network next to
the cameras it processes.

| Concern | Control |
| --- | --- |
| Authentication | API keys (`X-API-Key` / Bearer), min. 16 chars, constant-time comparison against every configured key. Authentication is **on by default** when the server is started from configuration; disabling it requires an explicit `VAP_AUTH__ENABLED=false`. |
| Authorisation | Roles `viewer` < `operator` < `admin`; model uploads/activation/deletion require `admin`. |
| Network exposure | Binds to `127.0.0.1` by default. Put a TLS-terminating reverse proxy in front for remote access. |
| Request size | Body limits enforced on the byte stream (not only `Content-Length`): `max_request_bytes` (2 MiB) for JSON, `max_upload_bytes` for model uploads. |
| Image bombs | Image headers are inspected before decoding; images above `max_image_pixels` are rejected (413). |
| File access | File and image-sequence sources resolve paths and must stay inside configured `media.roots` (traversal rejected with 403). Template images use the same check. Snapshot downloads are confined to the snapshot directory. |
| SSRF | Webhook URLs must be http(s), may not embed credentials, and by default may not resolve to private, loopback, link-local, reserved or multicast addresses (`webhooks.allow_private_networks`). Redirects are not followed. |
| Webhook integrity | `X-VAP-Signature: sha256=HMAC(secret, "<timestamp>.<body>")` plus `X-VAP-Timestamp` for replay windows on the receiver side. |
| Secrets | API keys and the webhook secret come from environment variables; `vap config` redacts them; RTSP credentials are redacted in API responses and logs (they are stored in the local SQLite database — protect `state_dir`). |
| CSV exports | Cells beginning with `= + - @` (non-numeric) are prefixed with `'` to prevent formula injection. |
| Model artefacts | SHA-256 computed while streaming; optional expected checksum; re-verification before a model is loaded by a pipeline. Models are not executed except through onnxruntime. |
| Errors | Unexpected exceptions return a generic 500 envelope; details only go to the server log. |

## Known gaps

- No rate limiting or account lockout on failed authentication.
- API keys are stored in plain text in the environment/config (no hashing at rest).
- No TLS in-process; use a reverse proxy.
- The SQLite database and snapshot directory are not encrypted.
