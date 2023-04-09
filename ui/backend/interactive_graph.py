from plotly.subplots import make_subplots
from plotly.offline import plot

import plotly.graph_objs as go


class PlotlyInteractiveGraph:

    @staticmethod
    def create_simple_graph(
            data: list[dict],
    ):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        data_time = [float(i.get('time')) for i in data]
        y1_data = [float(i.get('Y1')) for i in data]

        fig.add_trace(
            go.Scatter(x=data_time[:60], y=y1_data[:60], name=f"Statistic by inst/lead", line=dict(color="#030BF2")),
            secondary_y=False,
        )
        # Add figure title
        fig.update_layout(
            title_text="Напруга в бортовій мережі"
        )
        # Set x-axis title
        fig.update_xaxes(title_text="time")
        # Set y-axes titles
        fig.update_yaxes(title_text=f"<b>Y1</b>", secondary_y=False)
        # fig.update_yaxes(title_text="<b>Unique clicks</b>", secondary_y=True)

        plot_div = plot(fig, output_type='div')

        return plot_div

    @staticmethod
    def make_three_figures(timestamps, data, future_timestamps, predicted, danger_levels, labels, height):
        fig = make_subplots(rows=3, cols=1, subplot_titles=labels)

        for i in range(3):
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=data[:, i],
                    mode='lines',
                    line={'color': 'black'},
                    name=f'Вибірка Y{i + 1}'),
                row=i + 1, col=1
            )
            fig.add_vline(
                x=timestamps[-1],
                line_dash='dash',
                line_color='gray'
            )
            fig.add_trace(
                go.Scatter(
                    x=future_timestamps,
                    y=predicted[:, i],
                    mode='lines',
                    line={'color': '#5fe0de'},
                    name=f'Прогноз Y{i + 1}'),
                row=i + 1, col=1
            )
            if data[:, i].min() <= danger_levels[i][0]:
                fig.add_hline(
                    y=danger_levels[i][0],
                    line_color='#ffd894',
                    row=i + 1, col=1
                )
                fig.add_hline(
                    y=danger_levels[i][1],
                    line_color='#ffdbdb',
                    row=i + 1, col=1
                )

        fig.update_layout(
            showlegend=False,
            height=height
        )

        plot_div = plot(fig, output_type='div')
        return plot_div
