from django.contrib import admin

from Profiles.models import Profile, Task, Skills

# Register your models here.
admin.site.register(Profile)
admin.site.register(Task)
admin.site.register(Skills)
