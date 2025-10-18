from rest_framework import status
from rest_framework.permissions import IsAuthenticated, BasePermission
from rest_framework.response import Response
from rest_framework.views import APIView

from Courses.models import Course, CourseEnrollment, Phase, PhaseEnrollment
from Courses.serializers import CourseSerializer, PhaseSerializer, CourseEnrollmentSerializer


class CourseList(APIView):
    def get(self, request):
        courses = Course.objects.all()
        serializer = CourseSerializer(courses, many=True)

        return Response(serializer.data)

class CourseDetail(APIView):
    def get(self, request, pk):
        course = Course.objects.get(pk=pk)
        serializer = CourseSerializer(course)

        return Response()


class CourseEnrollmentList(APIView):
    permission_classes = (IsAuthenticated,)

    def get(self, request):
        courses = CourseEnrollment.objects.filter(enrollment=request.user)
        serializer = CourseEnrollmentSerializer(courses, many=True)

        return Response(serializer.data)

    def post(self, request):
        serializer = CourseEnrollmentSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class CourseEnrollmentDetail(APIView):
    permission_classes = (IsAuthenticated,)

    def get(self, request, pk):
        course = CourseEnrollment.objects.get(pk=pk)

        if course.profile.user != request.user:
            return Response(status=status.HTTP_403_FORBIDDEN)

        serializer = CourseSerializer(course)

        return Response(serializer.data)

    def post(self, request, pk):
        course = CourseEnrollment.objects.get(pk=pk)

        if course.profile.user != request.user:
            return Response(status=status.HTTP_403_FORBIDDEN)

        serializer = CourseSerializer(course)

        return Response(serializer.data)

    def put(self, request, pk):
        course = CourseEnrollment.objects.get(pk=pk)

        if course.profile.user != request.user:
            return Response(status=status.HTTP_403_FORBIDDEN)

        serializer = CourseSerializer(course, data=request.data)

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        course = CourseEnrollment.objects.get(pk=pk)

        if course.profile.user != request.user:
            return Response(status=status.HTTP_403_FORBIDDEN)

        course.delete()

        return Response(status=status.HTTP_204_NO_CONTENT)