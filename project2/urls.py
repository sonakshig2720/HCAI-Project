from django.urls import path
from .views import index

app_name = 'project2'
urlpatterns = [
    # The index now shows BOTH Task 1 & Task 2
    path('', index, name='index'),
]
