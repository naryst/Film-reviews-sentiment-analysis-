from django import forms


class DataForm(forms.Form):
    text = forms.CharField()
