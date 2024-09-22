
from django.shortcuts import render
from app.models import City


def showlist(request):
    results = City.objects.all
    return render(request, "home.html", {"showcity": results})