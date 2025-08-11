import json
import logging
import sqlite3

from django.shortcuts import render, redirect
from django.contrib import auth
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
from django.contrib.auth.decorators import login_required
from .models import ChatSession, ChatMessage
from . import models
from .agent import LongTermMemoryAgent
from django.utils import timezone
from .small_agent import Small_Agent

db_path = "Red_memory3.db"
agent = LongTermMemoryAgent()
small_agent = Small_Agent()


def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        # 验证用户凭证[8,7](@ref)
        # user = auth.authenticate(request, username=username, password=password)
        #
        # if user:
        #     auth.login(request, user)  # 用户登录[8](@ref)
        #     return redirect('/chat/')  # 跳转到聊天页面
        if agent.get_user_password(username) == password:
            return redirect(f'/chat/?username={username}')
        else:
            # 传递错误信息到模板[6,2](@ref)
            return render(request, 'login.html', {'error': '用户名或密码错误'})

    return render(request, 'login.html')


def register(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        # 用户名校验[6](@ref)
        if User.objects.filter(username=username).exists():
            return render(request, 'register.html', {
                'error': '用户名已存在',
                'username': username
            })

        # 创建用户[5](@ref)
        User.objects.create(
            username=username,
            password=password
        )
        return redirect('/')  # 注册成功后重定向到登录页[10](@ref)

    return render(request, 'register.html')

def get_session_messages(request,session_id):
    messaages = ChatMessage.objects.filter(session_id=session_id).order_by('last_accessed')
    data = [
        {
            'role': msg.role,
            'content': msg.message,
            'timestamp': msg.last_accessed.strftime('')
        }
        for msg in messaages
    ]
    return JsonResponse({'status': True, 'messages': data})

@csrf_exempt
def create_session(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            username=data.get('username')
            user = User.objects.get(username=username)
            session = ChatSession.objects.create(user=user, last_accessed=timezone.now(),summary="新会话")
            return JsonResponse({
                'status': True,
                'session_id': session.id,
                'last_accessed' : session.last_accessed.strftime('%Y-%m-%d %H:%M:%S'),
                'summary': session.summary
            })
        except Exception as e:
            return JsonResponse({'status': False, 'error': str(e)})
    return JsonResponse({'status': False, 'error': 'Invalid request method'})
def chat(request):
    # 获取最新一条消息的完整方案
    username = request.GET.get('username')
    user = User.objects.get(username=username)
    sessions = ChatSession.objects.filter(user=user).order_by('-last_accessed')
    messages = []
    # 获取用户的历史对话（如果需要）
    context = {
        'username': username,
        'user_id': user.id,
        'sessions': sessions,
        'messages': messages,
        'session_id': '',
    }
    return render(request, 'chat.html', context)


from django.views.decorators.http import require_http_methods, require_POST
from django.contrib.auth.decorators import login_required
import json
@ensure_csrf_cookie
@require_POST
def chat_api(request):
    try:
        if request.method == 'POST':
            data = json.loads(request.body)
            message = data.get("message")
            username = data.get("username")
            session_id = data.get("session_id")
            user = User.objects.get(username=username)
            session = ChatSession.objects.filter(id=session_id)
            user_id = agent.search_user(username)
            # if not ChatMessage.objects.filter(session=session).exists():
            #     session.summary = small_agent.small_chat(message)
            #     session.save()
            # ChatMessage.objects.create(
            #     session=session,
            #     user=user,
            #     role='user',
            #     message=message,
            #     timestamp=timezone.now()
            # )

            if not message:
                return JsonResponse({'username': username, 'status': 'error', 'message': '消息不能为空'}, status=400)

            # 调用 agent 的对话接口
            bot_response = agent.chat_agent(message, user_id)

            return JsonResponse({
                'status': True,
                'response': bot_response
            })

    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)



from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from .models import User, ChatSession, ChatMessage
@csrf_exempt
def delete_session(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            session_id = data.get('session_id')
            session = ChatSession.objects.filter(id=session_id).first()
            if session:
                session.delete()
                return JsonResponse({'status': 'success'})
            else:
                return JsonResponse({'status': 'error', 'message': '未找到会话'}, status=404)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': '无效请求'}, status=400)


@csrf_exempt
def test(request):
    return render(request, '提交测试.html')

def test_ajax(request):
    data=json.loads(request.body)
    name=data.get('name')
    age=data.get('age')
    print(name)
    print(age)
    username = data.get('username')
    print(username)
    print(request.POST)
    print(request.GET)
    n1=request.POST.get('message')
    print(n1)
    data={"status":"success","data":[1,2,3,4]}
    return JsonResponse(data,n1)