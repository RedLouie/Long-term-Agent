import hashlib
from django.db import models
from django.contrib.auth.models import AbstractUser, UserManager

class CustomUser(AbstractUser):
    # 添加自定义字段（可选）

    # 必须显式指定 UserManager
    objects = UserManager()  # 👈 关键配置


class User(models.Model):
    username = models.CharField(unique=True, max_length=100)
    password = models.CharField(unique=True, max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

class ChatSession(models.Model):
    STATUS_CHOICES = [('active', 'Active'), ('closed', 'Closed')]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True,null=True)
    last_accessed = models.DateTimeField(auto_now=True)
    summary = models.TextField()
    status = models.CharField(max_length=10, choices=STATUS_CHOICES)
    metadata = models.TextField(null=True)


class ChatMessage(models.Model):
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System')
    ]

    message_id = models.CharField(
        max_length=32,
        unique=True,  # 如需唯一性约束
        editable=False,  # 防止人工修改
        default=hashlib.sha256(b'').hexdigest()  # 临时默认值
    )
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE)
    user = models.ForeignKey(User, null=True, on_delete=models.SET_NULL)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True,null=True)
    last_accessed = models.DateTimeField(auto_now=True)
    embedding = models.BinaryField()
    weight = models.FloatField(default=1.0)
    metadata = models.TextField(null=True)

    class Meta:
        db_table = 'chat_app_chatmessage'



