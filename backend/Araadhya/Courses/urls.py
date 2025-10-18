from django.urls import path

from Courses.views import CourseList, CourseDetail, CourseEnrollmentList, CourseEnrollmentDetail

urlpatterns = [
    path('course/', CourseList.as_view(), name='course-list'),
    path('course/<int:pk>/', CourseDetail.as_view(), name='course-detail'),
    path('enrolled_courses/', CourseEnrollmentList.as_view(), name='course-enrollment-list'),
    path('enrolled_coursrs/<int:pk>/', CourseEnrollmentDetail.as_view(), name='course-enrollment-detail'),
]