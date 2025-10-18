from django.contrib import admin

from Courses.models import Course, Certificate, CourseEnrollment, PhaseEnrollment, Phase, Certification

# Register your models here.
admin.site.register(Course)
admin.site.register(Certificate)
admin.site.register(CourseEnrollment)
admin.site.register(PhaseEnrollment)
admin.site.register(Phase)
admin.site.register(Certification)