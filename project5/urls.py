from django.urls import path
from . import views

app_name = "project5"

urlpatterns = [
    path("", views.index, name="index"),
    path("train/", views.train, name="train"),
    path("sample/", views.sample, name="sample"),
    path("compare/", views.compare, name="compare"),
    path("fit_reward/", views.fit_reward, name="fit_reward"),
    path("rlhf_retrain/", views.rlhf_retrain, name="rlhf_retrain"),
    path("reset/", views.reset, name="reset"),
    path("reshuffle/", views.reshuffle, name="reshuffle"),  # 👈 NEW
]
