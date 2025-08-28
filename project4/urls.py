from django.urls import path
from . import views

app_name = 'project4'
urlpatterns = [
    path('',            views.index,         name='index'),
    path('download/',   views.download_doc,  name='download_doc'),
    path('consent/',    views.consent,       name='consent'),
    path('pre/',        views.pre_survey,    name='pre'),
    path('study/',      views.study,         name='study'),
    path('restart/', views.restart_study, name='restart'),   # NEW
    path('post/',       views.post_survey,   name='post'),
    path('debrief/',    views.debrief,       name='debrief'),
    path('details/',   views.details,      name='details'),  # simple summary page
]
