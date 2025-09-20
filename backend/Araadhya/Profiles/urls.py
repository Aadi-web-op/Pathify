from django.urls import path, include
from rest_framework.routers import DefaultRouter

from Profiles.views import UserDetail, ProfileDetail

router = DefaultRouter()
router.register('users', UserDetail, basename='users')
router.register('profiles', ProfileDetail, basename='profiles')

urlpatterns = [
    path('', include(router.urls)),
]