from django.shortcuts import render
from django.http import JsonResponse
from django.http import HttpResponse

from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def show(request):
    if(request.method=="POST"):
        print("post request is made")
        data = dict(request.POST)
        return JsonResponse(data)
    return HttpResponse("hello")