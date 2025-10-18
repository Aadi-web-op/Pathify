from django.contrib.auth.models import User
from rest_framework import viewsets, permissions

from Profiles.models import Profile
from Profiles.serializers import UserSerializer, ProfileSerializer


class IsUser(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        return request.user == obj


class IsOwner(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        return request.user == obj.user


# Create your views here.
class UserDetail(viewsets.ModelViewSet):
    permission_classes = [permissions.IsAuthenticated, IsUser]

    allowed_methods = ['get']

    queryset = User.objects.all()
    serializer_class = UserSerializer


class ProfileDetail(viewsets.ModelViewSet):
    permission_classes = [permissions.IsAuthenticated, IsOwner]

    allowed_methods = ['get', 'post']

    queryset = Profile.objects.all()
    serializer_class = ProfileSerializer