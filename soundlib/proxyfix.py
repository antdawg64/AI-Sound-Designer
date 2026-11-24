"""Make the Modal client work behind a corporate / cloud HTTPS proxy.

Modal's client checks certificates against the `certifi` bundle and ignores
SSL_CERT_FILE. When a proxy re-signs traffic, point certifi at the bundle the
machine was told to trust. This never turns certificate checking off.
"""

import os


def apply() -> None:
    bundle = os.environ.get("SSL_CERT_FILE")
    if not bundle or not os.path.exists(bundle):
        return
    try:
        import certifi
    except ImportError:
        return
    certifi.where = lambda: bundle  # type: ignore[assignment]
