import os

from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views import View
from django.conf import settings

from ui.backend.interactive_graph import PlotlyInteractiveGraph
from ui.backend.predictor import Predictor
from ui.backend.read_file import ReadFile
from ui.forms import BackEndForm

temp = []
path_to_file = os.path.join(settings.BASE_DIR, 'data.txt')


class UIView(View):
    template_name = 'index.html'

    def get(self, request):
        data = ReadFile(path_to_file).get_data()
        context = {'form': BackEndForm, 'plot_div': self.get_graph_plot(data)}
        return render(request, self.template_name, context=context)

    def post(self, request):
        form = BackEndForm(request.POST)
        if form.is_valid():
            if len(temp) > 0:
                temp.clear()
            data = form.cleaned_data
            print(data)
            predictor = Predictor(
                path=path_to_file,
                x1_dim=int(data['dimension1']),
                x2_dim=int(data['dimension2']),
                x3_dim=int(data['dimension3']),
                y_dim=int(data['dimension4']),
                x1_deg=int(data['degree_of_polynomials_x1']),
                x2_deg=int(data['degree_of_polynomials_x2']),
                x3_deg=int(data['degree_of_polynomials_x3']),
                weight_method=data['weight'],
                lambda_option=data['check_lambda'],
                poly_type=data['polynom'],
                samples=int(data['sample_size']),
                pred_steps=int(data['prediction_step']),
                form=data['form_fz'],
                ar_order=int(data.get('ar_order')) if data.get('ar_order') else None,
                ma_order=int(data.get('ma_order')) if data.get('ma_order') else None
            )
            temp.append(predictor)
            data, fig = get_next_predict(predictor)
        #context = {'form': form}
            context = {
                'form': form,
                'data': data,
                'plot_div': fig
            }
            return render(request, 'index.html', context)
        return HttpResponse('NO')

        #return render(request, self.template_name, context=context)

    def get_graph_plot(self, data):
        plot_div = PlotlyInteractiveGraph.create_simple_graph(
            data=data,
        )
        return plot_div


def get_next_predict(predictor):
    try:
        g = predictor.predict()
        data, fig = next(g)


    except StopIteration:
        data = 'Конец данных'
        fig = None
    except IndexError:
        data = 'Нет еще данных, пожалуйста, заполните форму'
        fig = None
    return data, fig

