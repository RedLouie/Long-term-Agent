import json



def test_chat_app(request):
    data = json.loads(request.body)
    message = data.get("message")
    username = data.get("username")