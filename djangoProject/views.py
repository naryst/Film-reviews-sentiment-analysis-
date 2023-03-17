from django.http import HttpResponseRedirect
from django.shortcuts import render

from djangoProject.forms import DataForm
from model import model_inference


def get_data(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = DataForm(request.POST)
        # check whether it's valid:
        print(form.data['review'])
        if form.is_valid():
            print('valid form')
            return HttpResponseRedirect('/thanks/')

    # if a GET (or any other method) we'll create a blank form
    else:
        form = DataForm()
    return render(request, 'page_front.html', {'form': form})


model = model_inference.load_model()


def analysis(request):
    global model
    assert request.method == 'POST'
    form = DataForm(request.POST)
    data = form.data['review']
    res = model_inference.predict(model, data)
    if res > 5:
        res = str(res) + ', this review is positive'
    else:
        res = str(res) + ', this review in negative'
    return render(request, 'analysis_page.html', {'data': res})
