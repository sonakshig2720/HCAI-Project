from django.urls import path
from . import views

app_name = 'project3'

urlpatterns = [
    path('', views.index, name='index'),
    path('tree/', views.decision_tree_view, name='decision_tree'),
    path('logistic/', views.logistic_regression_view, name='logistic_regression'),
    path('counterfactual/', views.counterfactual_view, name='counterfactual'),
]
