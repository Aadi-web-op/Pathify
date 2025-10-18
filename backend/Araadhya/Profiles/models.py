from django.contrib.auth.models import User
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    phone_number = models.BigIntegerField(validators=[MaxValueValidator(9999999999), MinValueValidator(1000000000)])


# Create your models here.
class Task(models.Model):
    profile = models.ForeignKey(Profile, on_delete=models.CASCADE, related_name='tasks')
    title = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    due_date = models.DateField()
    priority = models.IntegerField(choices=[[1, "High"], [2, "Medium"], [3, "Low"]])
    status = models.IntegerField(choices=[[1, "Active"], [2, "Completed"], [3, "Expired"]])

    def __str__(self):
        return self.title


class Skills(models.Model):
    profile = models.OneToOneField(Profile, on_delete=models.CASCADE, related_name='skills')
    programming = models.PositiveIntegerField(validators=[MaxValueValidator(100)])
    creativity = models.PositiveIntegerField(validators=[MaxValueValidator(100)])
    leadership = models.PositiveIntegerField(validators=[MaxValueValidator(100)])
    teamwork = models.PositiveIntegerField(validators=[MaxValueValidator(100)])
    communication = models.PositiveIntegerField(validators=[MaxValueValidator(100)])
    problem_solving = models.PositiveIntegerField(validators=[MaxValueValidator(100)])


