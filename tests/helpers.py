class FakeResponse:
    """Stand-in for requests.Response, used to mock Session.request in tests."""

    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (str(payload) if payload is not None else "")
        self.content = b"1" if (payload is not None or text) else b""

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload
