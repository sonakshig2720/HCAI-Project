from django import forms

class TrainForm(forms.Form):
    episodes = forms.IntegerField(min_value=1, initial=300, label="Episodes")
    gamma = forms.FloatField(min_value=0.0, max_value=0.999, initial=0.99, label="Discount (γ)")
    lr = forms.FloatField(min_value=1e-5, initial=1e-3, label="Learning rate")
    max_steps = forms.IntegerField(min_value=1, initial=50, label="Max steps / episode")
    seed = forms.IntegerField(min_value=0, initial=0, label="Random seed")
