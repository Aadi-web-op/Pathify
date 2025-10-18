from django.urls import path, include

from Users.views import GoogleLoginView, UserViewSet

urlpatterns = [
    path('', include('dj_rest_auth.urls')),
    path('registration/', include('dj_rest_auth.registration.urls')),
    path('google/', GoogleLoginView.as_view(), name='google_login'),
    path('users/', UserViewSet.as_view({'get': 'list'})),
]