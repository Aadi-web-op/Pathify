from django.contrib.auth.models import User
from django.db import models

from Profiles.models import Profile


# Create your models here.
class Course(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    duration = models.DurationField()


class Certificate(models.Model):
    title = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    course = models.OneToOneField(Course, on_delete=models.CASCADE, related_name='certificate', blank=True, null=True)


class CourseEnrollment(models.Model):
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name='enrolled_users')
    profile = models.ForeignKey(Profile, on_delete=models.CASCADE, related_name='courses')
    start_date = models.DateField()
    end_date = models.DateField(blank=True, null=True)


class Phase(models.Model):
    name = models.CharField(max_length=100)
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name='phases')
    description = models.TextField(blank=True, null=True)
    duration = models.DurationField()


class PhaseEnrollment(models.Model):
    phase = models.ForeignKey(Phase, on_delete=models.CASCADE)
    course = models.ForeignKey(CourseEnrollment, on_delete=models.CASCADE, related_name='phases')
    start_date = models.DateField(blank=True, null=True)
    end_date = models.DateField(blank=True, null=True)


class Certification(models.Model):
    profile = models.ForeignKey(Profile, on_delete=models.CASCADE, related_name='certifications')
    certificate = models.ForeignKey(Certificate, on_delete=models.CASCADE)
    score = models.PositiveIntegerField(default=0, blank=True, null=True)
    completed_on = models.DateField(blank=True, null=True)

