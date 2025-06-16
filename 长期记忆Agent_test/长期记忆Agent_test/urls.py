"""
URL configuration for 长期记忆Agent_test project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from chat_app import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.login),
    path("chat/", views.chat),
    path("register/", views.register),
    path("chat_api/", views.chat_api),
    path("test/",views.test),
    path("test_ajax/",views.test_ajax),
    path('get_session_messages/<int:session_id>/', views.get_session_messages),
    path('create_session/', views.create_session),
    path('delete_session/', views.delete_session, name='delete_session'),


]
