# from django.http import HttpResponse


# def index(request):
#     return HttpResponse("Hello, world. You're at the polls index.")

from django.http import HttpResponse
from django.template import loader


def index(request):
    template = loader.get_template("home/index.html")
    
    
    students = [
        {"name": "Sonakshi Godavarthy", "matriculation": "612535"},
        {"name": "Krishnan Lakshmanan Ananya", "matriculation": "611778"}
    ]
    
    projects = [
        {"name": "Home", "url_name": "home:index"},
        {"name": "Project 1", "url_name": "project1:index"},
        {"name": "Project 2", "url_name": "project2:index"},
        {"name": "Project 3", "url_name": "project3:index"},
        {"name": "Project 4", "url_name": "project4:index"},
        {"name": "Project 5", "url_name": "project5:index"},
    ]

    
    context = { 
        "students": students, 
        "projects": projects, 
    }
    
    return HttpResponse(template.render(context, request))