from django.contrib.auth.models import User
from rest_framework import serializers

from Courses.models import Certification
from Courses.serializers import CertificateSerializer
from Profiles.models import Profile, Skills, Task


class PhoneNumberSerializer(serializers.ModelSerializer):
    class Meta:
        model = Profile
        fields = 'phone_number'


class UserSerializer(serializers.ModelSerializer):
    phone_number = PhoneNumberSerializer(read_only=False)

    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name', 'phone_number']


class SkillsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Skills
        fields = ['programming', 'creativity', 'teamwork', 'leadership', 'communication', 'problem_solving']


class TaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = Task
        fields = ['title', 'description', 'priority', 'status', 'due_date']


class CertificationSerializer(serializers.ModelSerializer):
    certificate = CertificateSerializer(read_only=True)

    class Meta:
        model = Certification
        fields = ['certificate', '']


class ProfileSerializer(serializers.ModelSerializer):

    class UsernameSerializer(serializers.ModelSerializer):
        class Meta:
            model = User
            fields = ['username']

    skills = SkillsSerializer(many=True, read_only=True)
    tasks = TaskSerializer(many=True, read_only=True)
    username = UsernameSerializer(read_only=True)

    class Meta:
        model = Profile
        fields = ['username', 'skills', 'tasks']
