from rest_framework import serializers

from Courses.models import Phase, Certificate, Course, CourseEnrollment, PhaseEnrollment


class PhaseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Phase
        fields = ['name', 'description', 'duration']


class PhaseEnrollmentSerializer(serializers.ModelSerializer):
    phase = PhaseSerializer()

    class Meta:
        model = PhaseEnrollment
        fields = ['phase', 'start_date', 'end_date']



class CertificateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Certificate
        fields = ['title', 'description']


class CourseSerializer(serializers.ModelSerializer):
    phase = PhaseSerializer(many=True)
    certificate = CertificateSerializer()

    class Meta:
        model = Course
        fields = ['name', 'description', 'phase', 'certificate', 'duration']


class CourseEnrollmentSerializer(serializers.ModelSerializer):
    course = CourseSerializer()
    phases = PhaseEnrollmentSerializer()

    class Meta:
        model = CourseEnrollment
        fields = ['course', 'phases', 'start_date', 'end_date']
