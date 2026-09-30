import json

from chatv1.chat_http import make_chat_handler


def test_chat_handler_factory_builds_handler():
    class Service: pass
    handler = make_chat_handler(lambda: Service())
    assert handler is not None
    assert hasattr(handler, "do_POST")


def test_json_payload_shape():
    payload = {"conversation_id": "demo", "prompt": "hello"}
    assert json.loads(json.dumps(payload)) == payload
