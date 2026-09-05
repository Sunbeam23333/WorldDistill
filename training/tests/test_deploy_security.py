import ast
import concurrent.futures
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SECURITY_PATH = ROOT / "inference/lightx2v/deploy/common/security.py"
AUTH_PATH = ROOT / "inference/lightx2v/deploy/server/auth.py"
SERVER_PATH = ROOT / "inference/lightx2v/deploy/server/__main__.py"
WORKER_PATH = ROOT / "inference/lightx2v/deploy/worker/__main__.py"
HUB_PATH = ROOT / "inference/lightx2v/deploy/worker/hub.py"
PARALLEL_RUNNER_PATH = ROOT / "inference/lightx2v/deploy/worker/parallel_model_runner.py"
S3_PATH = ROOT / "inference/lightx2v/deploy/data_manager/s3_data_manager.py"
ALIYUN_PATH = ROOT / "inference/lightx2v/deploy/common/aliyun.py"
FACE_DETECTOR_PATH = ROOT / "inference/lightx2v/deploy/common/face_detector.py"
ASR_PATH = ROOT / "inference/lightx2v/deploy/common/volcengine_asr.py"
TTS_PATH = ROOT / "inference/lightx2v/deploy/common/volcengine_tts.py"
PODCASTS_PATH = ROOT / "inference/lightx2v/deploy/common/podcasts.py"
UTILS_PATH = ROOT / "inference/lightx2v/deploy/common/utils.py"
VA_CONTROLLER_PATH = ROOT / "inference/lightx2v/deploy/common/va_controller.py"
PODCAST_FRONTEND_PATH = ROOT / "inference/lightx2v/deploy/server/frontend/src/views/PodcastGenerate.vue"
FRONTEND_UTILS_PATH = ROOT / "inference/lightx2v/deploy/server/frontend/src/utils/other.js"
FRONTEND_APP_PATH = ROOT / "inference/lightx2v/deploy/server/frontend/src/App.vue"
FRONTEND_ROUTER_PATH = ROOT / "inference/lightx2v/deploy/server/frontend/src/router/index.js"


def _logger_calls(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if isinstance(node.func.value, ast.Name) and node.func.value.id == "logger":
            yield node


def _load_security_module():
    spec = importlib.util.spec_from_file_location("worlddistill_deploy_security", SECURITY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "   ",
        "change-me",
        "worker-secret-key-change-in-production",
        "your-secret-key-change-in-production",
        "replace-with-a-long-random-value",
    ],
)
def test_required_deploy_secret_rejects_missing_or_placeholder(monkeypatch, value):
    security = _load_security_module()
    if value is None:
        monkeypatch.delenv("TEST_DEPLOY_SECRET", raising=False)
    else:
        monkeypatch.setenv("TEST_DEPLOY_SECRET", value)

    with pytest.raises(RuntimeError, match="TEST_DEPLOY_SECRET"):
        security.require_secret_env("TEST_DEPLOY_SECRET")


def test_required_deploy_secret_accepts_explicit_value(monkeypatch):
    security = _load_security_module()
    monkeypatch.setenv("TEST_DEPLOY_SECRET", "test-only-random-looking-value-123")
    assert security.require_secret_env("TEST_DEPLOY_SECRET") == "test-only-random-looking-value-123"


def test_optional_oauth_config_rejects_partial_configuration(monkeypatch):
    security = _load_security_module()
    monkeypatch.setenv("TEST_OAUTH_ID", "client-id")
    monkeypatch.delenv("TEST_OAUTH_SECRET", raising=False)

    with pytest.raises(RuntimeError, match="TEST_OAUTH_SECRET"):
        security.optional_env_group("Test OAuth", ("TEST_OAUTH_ID", "TEST_OAUTH_SECRET"))


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"Authorization": "Bearer test-token"}, "test-token"),
        ({"authorization": "bearer lower-case-scheme"}, "lower-case-scheme"),
        ({"Authorization": "Basic abc"}, None),
        ({"Authorization": "Bearer    "}, None),
        ({}, None),
    ],
)
def test_bearer_token_is_extracted_only_from_headers(headers, expected):
    security = _load_security_module()
    assert security.bearer_token_from_headers(headers) == expected


def test_safe_url_host_removes_credentials_paths_and_queries():
    security = _load_security_module()
    value = "http://alice:super-secret@proxy.example:8443/private?token=jwt-secret"

    assert security.safe_url_host(value) == "proxy.example"
    assert security.safe_url_host("proxy.internal:3128") == "proxy.internal"
    assert security.safe_url_host(None) is None


def test_cors_defaults_to_same_origin_and_accepts_only_exact_allowlist(monkeypatch):
    security = _load_security_module()
    monkeypatch.delenv("LIGHTX2V_CORS_ALLOWED_ORIGINS", raising=False)

    assert security.cors_allowed_origins() == []
    assert security.cors_allowed_origins(
        "https://app.example.com,http://localhost:5173,https://app.example.com"
    ) == ["https://app.example.com", "http://localhost:5173"]


@pytest.mark.parametrize(
    "value",
    [
        "*",
        "null",
        "https://user:password@app.example.com",
        "https://app.example.com/path",
        "https://app.example.com?debug=true",
        "https://app.example.com#fragment",
        "file:///tmp/frontend",
    ],
)
def test_cors_rejects_wildcards_and_non_origins(value):
    security = _load_security_module()
    with pytest.raises(RuntimeError, match="CORS|origin"):
        security.cors_allowed_origins(value)


def test_server_error_details_are_never_returned_to_clients():
    security = _load_security_module()
    internal = RuntimeError("database-password-and-private-host")

    assert security.public_error_detail(internal, 400) is internal
    assert security.public_error_detail(internal, 500) == "Internal server error"
    assert "database-password" not in security.public_error_detail(internal, 503)


def test_oauth_state_is_browser_bound_provider_bound_and_one_time():
    security = _load_security_module()
    now = [100.0]
    store = security.OAuthStateStore(ttl_seconds=600, clock=lambda: now[0])

    github_state = store.issue("github", "https://app.example.com/")
    google_state = store.issue("google", "https://app.example.com/")
    assert github_state != google_state
    assert security.oauth_state_cookie_name(github_state) != security.oauth_state_cookie_name(google_state)

    # A state without the browser's HttpOnly cookie cannot burn the transaction.
    assert security.consume_browser_bound_oauth_state(store, github_state, "wrong", "github") is None
    record = security.consume_browser_bound_oauth_state(store, github_state, github_state, "github")
    assert record is not None
    assert record.provider == "github"
    assert record.redirect_uri == "https://app.example.com/"

    # The record was atomically removed before any provider exchange.
    assert security.consume_browser_bound_oauth_state(store, github_state, github_state, "github") is None
    assert security.consume_browser_bound_oauth_state(store, google_state, google_state, "github") is None
    assert security.consume_browser_bound_oauth_state(store, google_state, google_state, "google") is None


def test_oauth_state_expires_and_concurrent_consumers_have_one_winner():
    security = _load_security_module()
    now = [10.0]
    store = security.OAuthStateStore(ttl_seconds=5, clock=lambda: now[0])
    expired_state = store.issue("github", "https://app.example.com/")
    now[0] = 15.0
    assert security.consume_browser_bound_oauth_state(store, expired_state, expired_state) is None

    now[0] = 20.0
    state = store.issue("google", "https://app.example.com/")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _: security.consume_browser_bound_oauth_state(store, state, state),
                range(8),
            )
        )
    assert sum(record is not None for record in results) == 1


def test_oauth_and_cors_are_integrated_without_request_host_trust():
    auth_source = AUTH_PATH.read_text(encoding="utf-8")
    server_source = SERVER_PATH.read_text(encoding="utf-8")
    frontend_source = FRONTEND_UTILS_PATH.read_text(encoding="utf-8")
    app_source = FRONTEND_APP_PATH.read_text(encoding="utf-8")
    router_source = FRONTEND_ROUTER_PATH.read_text(encoding="utf-8")

    assert '"GITHUB_REDIRECT_URI"' in auth_source
    assert '"redirect_uri": redirect_uri' in auth_source
    assert "request.base_url" not in server_source
    assert "urlencode(" in server_source
    assert '"state": state' in server_source
    assert "httponly=True" in server_source
    assert 'samesite="lax"' in server_source
    assert "consume_oauth_state(state, cookie_state" in server_source
    assert '@app.get("/auth/callback/oauth")' in server_source
    assert "allow_origins=cors_allowed_origins()" in server_source
    assert 'allow_origins=["*"]' not in server_source

    combined_frontend = frontend_source + app_source + router_source
    assert "loginSource" not in combined_frontend
    assert "callbackParams.set('state', state)" in frontend_source
    assert "/auth/callback/oauth?" in frontend_source
    assert "new URLSearchParams(window.location.search)" in app_source
    assert "const state = urlParams.get('state')" in app_source
    assert "console.log(data)" not in frontend_source


def test_all_dynamic_server_errors_flow_through_the_public_redaction_boundary():
    server_source = SERVER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(server_source)

    assert 'public_error_detail(f"error: {e}!", code)' in server_source
    assert "public_error_detail(exc.detail, exc.status_code)" in server_source

    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            continue
        if call.func.id != "JSONResponse":
            continue
        status_keyword = next((item for item in call.keywords if item.arg == "status_code"), None)
        if status_keyword is None or not isinstance(status_keyword.value, ast.Constant):
            continue
        if status_keyword.value.value != 500 or not call.args:
            continue

        content = call.args[0]
        rendered = ast.unparse(content)
        # Direct 500 responses may use a static operational message or the
        # shared generic constant, but never values from an exception or an
        # upstream provider response.
        forbidden = ("str(e)", "asr_result", "clone_result", "error_msg", "traceback")
        assert not any(value in rendered for value in forbidden), rendered


def test_deploy_entrypoints_require_secrets_without_predictable_defaults():
    auth_source = AUTH_PATH.read_text(encoding="utf-8")
    worker_source = WORKER_PATH.read_text(encoding="utf-8")
    combined = auth_source + worker_source

    assert 'require_secret_env("JWT_SECRET_KEY")' in auth_source
    assert combined.count('require_secret_env("WORKER_SECRET_KEY")') == 2
    assert "worker-secret-key-change-in-production" not in combined
    assert "your-secret-key-change-in-production" not in combined


def test_auth_logger_calls_do_not_reference_sensitive_values():
    tree = ast.parse(AUTH_PATH.read_text(encoding="utf-8"))
    forbidden = ("client_secret", "jwt_secret", "worker_secret", "oauth code", "{code}", "{proxy}")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "logger":
            continue
        rendered = ast.unparse(node).lower()
        assert not any(fragment in rendered for fragment in forbidden), rendered


def test_deploy_logger_calls_do_not_emit_credentials_or_private_payloads():
    forbidden_by_path = {
        S3_PATH: ("self.config", "self.aws_access_key_id", "self.aws_secret_access_key"),
        ALIYUN_PATH: ("phone_number", "verify_code", "logger.info(f'{prefix}: {res}"),
        ASR_PATH: ("request_payload", "file_url", "logger.info(f'result:", "logger.warning(f'asr recognition failed: {result}"),
        TTS_PATH: ("logger.info(f'volcengine tts params: {params}", "logger.info(f\"volcengine tts params: {params}"),
        PODCASTS_PATH: ("json.dumps(req_params", "logger.debug(f'sending: {msg}", "logger.debug(f'request params:"),
        HUB_PATH: ("json.dumps(config", "logger.info(f'run params:", "logger.info(f'config:"),
        PARALLEL_RUNNER_PATH: ("json.dumps(config", "logger.info(f'clip {name} config:"),
        VA_CONTROLLER_PATH: ("self.stream_config",),
        SERVER_PATH: ("user_response", "request.image[:", "request.audio[:", "request.video[:", "logger.info(f'{params}", "logger.info(f'args: {args}"),
    }

    for path, forbidden in forbidden_by_path.items():
        for call in _logger_calls(path):
            rendered = ast.unparse(call).lower()
            assert not any(fragment.lower() in rendered for fragment in forbidden), f"{path}: {rendered}"


@pytest.mark.parametrize("path", [FACE_DETECTOR_PATH, ASR_PATH, TTS_PATH, PODCASTS_PATH])
def test_proxy_values_are_sanitized_before_logging(path):
    proxy_names = {"proxy", "http_proxy", "https_proxy"}
    for call in _logger_calls(path):
        referenced = {node.id for node in ast.walk(call) if isinstance(node, ast.Name)}
        attributes = {ast.unparse(node) for node in ast.walk(call) if isinstance(node, ast.Attribute)}
        references_proxy_value = bool(referenced & proxy_names) or "self.proxy" in attributes
        if references_proxy_value:
            assert "safe_url_host(" in ast.unparse(call), ast.unparse(call)


def test_http_exception_logs_only_the_request_path():
    for call in _logger_calls(SERVER_PATH):
        rendered = ast.unparse(call)
        if "request.url" in rendered:
            assert "request.url.path" in rendered, rendered
            assert "query_params" not in rendered, rendered


def test_query_string_authentication_is_disabled_end_to_end():
    server_source = SERVER_PATH.read_text(encoding="utf-8")
    frontend_source = PODCAST_FRONTEND_PATH.read_text(encoding="utf-8")
    frontend_utils_source = FRONTEND_UTILS_PATH.read_text(encoding="utf-8")

    assert "verify_user_access_from_query" not in server_source
    assert '.query_params.get("token")' not in server_source
    assert ".query_params.get('token')" not in server_source
    assert "Query-string tokens are disabled" in server_source
    assert "Query-string WebSocket tokens are disabled" in server_source
    assert "bearer_token_from_headers(request.headers)" in server_source
    assert "bearer_token_from_headers(websocket.headers)" in server_source
    assert "websocket.receive_json()" in server_source
    assert "access_log=False" in server_source

    combined_frontend = frontend_source + frontend_utils_source
    assert "?token=" not in combined_frontend
    assert "&token=" not in combined_frontend
    assert "type: 'authenticate', access_token: token" in frontend_source
    assert "fetchProtectedAssetUrl" in frontend_utils_source


def test_resource_download_logs_never_reference_the_full_url():
    tree = ast.parse(UTILS_PATH.read_text(encoding="utf-8"))
    fetch_resource = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "fetch_resource")
    for call in ast.walk(fetch_resource):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        if isinstance(call.func.value, ast.Name) and call.func.value.id == "logger":
            referenced_names = {node.id for node in ast.walk(call) if isinstance(node, ast.Name)}
            assert "url" not in referenced_names, ast.unparse(call)
