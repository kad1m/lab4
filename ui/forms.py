from django import forms


class BackEndForm(forms.Form):
  CHOICES_FORM = [
    ('Адитивна форма', 'Адитивна форма'),
    ('Мультиплікативна форма', 'Мультиплікативна форма'),
    ('ARMAX', 'ARMAX')]

  CHOICES_POLYNOM = [
    ('Чебишова', 'Чебишова'),
    ('Лежандра', 'Лежандра'),
    ('Лаґерра', 'Лаґерра'),
    ('Ерміта', 'Ерміта'),
  ]

  CHOICES_FUNCTION = [
    ('Нормоване значення', 'Нормоване значення'),
    ('Середнє арифметичне', 'Середнє арифметичне')
  ]

  dictionary = {
    'dimension1': {
      'class': "form-control",
      'id': "dimension1",

      'placeholder': "Розмірність X1...",
    },
    'dimension2': {
      'class': "form-control",
      'id': "dimension2",
      'placeholder': "Розмірність X2...",

    },
    'dimension3': {
      'class': "form-control",
      'id': "dimension3",
      'placeholder': "Розмірність X3...",
    },
    'dimension4': {
      'class': "form-control",
      'id': "dimension4",
      'placeholder': "Розмірність Y...",
    },
    'form_fz': {
      'class': "radio_button_controller form-select mb-3",
      'id': "form_fz",
    },
    'polynom': {
      'class': "radio_button_controller form-select mb-3",
      'id': "polynom",
    },

    'degree_of_polynomials_x1': {
      'class': "form-control",
      'id': "degree_of_polynomials1",
      'placeholder': "Степінь поліномів для x1",
    },

    'degree_of_polynomials_x2': {
      'class': "form-control",
      'id': "degree_of_polynomials2",
      'placeholder': "Степінь поліномів для x2",
    },

    'degree_of_polynomials_x3': {
      'class': "form-control",
      'id': "degree_of_polynomials3",
      'placeholder': "Степінь поліномів для x3",
    },

    'weight': {
      'class': "form-control mb-3",
      'id': "weight",
      'placeholder': "Ваги цільових функцій",
    },

    'lambda': {
      'class': "form-control form-check-input",
      'id': "lambda",
    },

    'sample_size': {
      'class': "form-control",
      'id': "sample_size",
      'placeholder': "Розмір вибірки",
    },

    'prediction_step': {
      'class': "form-control",
      'id': "prediction_step",
      'placeholder': "Крок прогнозування",

    }

  }
  dimension1 = forms.CharField(required=True, label='Розмірність х1',
                               widget=forms.TextInput(attrs=dictionary['dimension1']), initial=4)
  dimension2 = forms.CharField(required=True, label='Розмірність х2',
                               widget=forms.TextInput(attrs=dictionary['dimension2']), initial=2)
  dimension3 = forms.CharField(required=True, label='Розмірність х3',
                               widget=forms.TextInput(attrs=dictionary['dimension3']), initial=3)
  dimension4 = forms.CharField(required=True, label='Розмірність y',
                               widget=forms.TextInput(attrs=dictionary['dimension4']), initial=3)

  form_fz = forms.ChoiceField(choices=CHOICES_FORM, label='Форма ФЗ', widget=forms.Select(attrs=dictionary['form_fz']))
  polynom = forms.ChoiceField(choices=CHOICES_POLYNOM, label='Тип поліномів',
                              widget=forms.Select(attrs=dictionary['polynom']))

  degree_of_polynomials_x1 = forms.CharField(required=True, label='Степінь полінома для х1',
                                             widget=forms.TextInput(attrs=dictionary['degree_of_polynomials_x1']))
  degree_of_polynomials_x2 = forms.CharField(required=True, label='Степінь полінома для х2',
                                             widget=forms.TextInput(attrs=dictionary['degree_of_polynomials_x2']))
  degree_of_polynomials_x3 = forms.CharField(required=True, label='Степінь полінома для х3',
                                             widget=forms.TextInput(attrs=dictionary['degree_of_polynomials_x3']))

  weight = forms.ChoiceField(choices=CHOICES_FUNCTION, label='Ваги цільових функцій',
                             widget=forms.Select(attrs=dictionary['weight']))
  sample_size = forms.CharField(required=True, label='Розмір вибірки',
                                widget=forms.TextInput(attrs=dictionary['sample_size']))

  check_lambda = forms.BooleanField(required=False, label='Визначення λ з трьох систем рівнянь',
                                    widget=forms.CheckboxInput(attrs=dictionary['lambda']))

  prediction_step = forms.CharField(required=True, label='Крок прогнозовання',
                                    widget=forms.TextInput(attrs=dictionary['prediction_step']))
