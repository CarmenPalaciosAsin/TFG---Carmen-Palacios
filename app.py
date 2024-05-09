import pandas as pd
import dash
from dash import Dash, html, dcc, Output, Input
import datetime
import plotly.graph_objs as go
import refinitiv.data as rd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
import numpy as np

# Carga el DataFrame
df = pd.read_csv(r'C:\Users\Carmen (TFG)\Documents\Bases de datos\Base_Final.csv')

app = Dash(__name__)

# Define estilos CSS
table_style = {'border': '1px solid black', 'border-collapse': 'collapse', 'margin': 'auto'}
cell_style = {'border': '1px solid black', 'padding': '5px'}
header_cell_style = {**cell_style, 'font-weight': 'bold', 'textAlign': 'left'}

# Layout de la aplicación
app.layout = html.Div([
    html.H1('Comparador de Fondos de Inversión', style={'textAlign': 'center'}),
    dcc.Dropdown(
        options=[{'label': f"{row['Nombre']} ({row['ISIN']})", 'value': row['ISIN']} for index, row in df.iterrows()],
        multi=True,
        id='dropdown-selection',
        placeholder='Selecciona un fondo'
    ),
    html.Div(
        style={'textAlign': 'center', 'margin-top': '10px'},
        children=[html.Button('YTD', id='btn-ytd', n_clicks=0),
                  html.Button('1 Año', id='btn-1y', n_clicks=0),
                  html.Button('3 Años', id='btn-3y', n_clicks=0),
                  html.Button('5 Años', id='btn-5y', n_clicks=0)]
    ),
    dcc.Graph(id='graph-content'),
    html.Div(id='table-content', style={'margin-bottom': '20px'}),
    html.H2('Ratios', style={'textAlign': 'center'}),
    dcc.Tabs(
        id='tabs-ratios',
        value='1año',
        children=[dcc.Tab(label='1 Año', value='1año'),
                  dcc.Tab(label='3 Años', value='3año'),
                  dcc.Tab(label='5 Años', value='5año')]
    ),
    html.Div(id='tabla-ratios-content'),
    html.Div(id='recommendation-content', style={'margin-top': '20px', 'textAlign': 'center'})
])

# Preprocesamiento y modelo
def preprocess_data(df):
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_features])
    pca = PCA(n_components=0.95)
    principal_components = pca.fit_transform(scaled_data)
    return scaler, pca, principal_components

def train_autoencoder(data):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(data.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(data.shape[1], activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(data, data, epochs=50, batch_size=32)
    return model

# Cargar y preparar datos antes de iniciar el servidor
scaler, pca, principal_components = preprocess_data(df)
autoencoder_model = train_autoencoder(principal_components)

@app.callback(
    [Output('graph-content', 'figure'),
     Output('table-content', 'children'),
     Output('tabla-ratios-content', 'children')],
    [Input('dropdown-selection', 'value'),
     Input('tabs-ratios', 'value'),
     Input('btn-ytd', 'n_clicks'),
     Input('btn-1y', 'n_clicks'),
     Input('btn-3y', 'n_clicks'),
     Input('btn-5y', 'n_clicks')]
)
def update_graph(selected_funds, selected_tab, btn_ytd, btn_1y, btn_3y, btn_5y):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'btn-ytd' in changed_id:
        start_date = datetime.datetime(datetime.datetime.now().year, 1, 1)
    elif 'btn-1y' in changed_id:
        start_date = datetime.datetime.now() - datetime.timedelta(days=360)
    elif 'btn-3y' in changed_id:
        start_date = datetime.datetime.now() - datetime.timedelta(days=1080)
    elif 'btn-5y' in changed_id:
        start_date = datetime.datetime.now() - datetime.timedelta(days=1800)
    else:
        start_date = datetime.datetime.now() - datetime.timedelta(days=360)

    end_date = datetime.datetime.now()

    if not selected_funds:
        return {}, [], []

    rd.open_session()
    list_RIC = []

    for isin in selected_funds:
        # Aquí deberías mapear los ISIN a los RIC usando tu DataFrame 'df'
        ric = df.loc[df['ISIN'] == isin, 'RIC'].iloc[0]
        list_RIC.append(ric)

    df_price_history = rd.get_history(
        universe=list_RIC,
        fields=['TR.FundNAV'],
        start=start_date,
        end=end_date,
        interval='daily'
    )

    for ric in df_price_history.columns:
        df_price_history[f'{ric}_%'] = (df_price_history[ric] / df_price_history[ric].iloc[0] - 1) * 100

    fig = go.Figure()

    colores = ['silver', 'lightblue', 'lightgreen', 'pink', 'yellow']

    ric_names = dict(zip(df['RIC'], df['Nombre']))

    columns_to_plot = [column for column in df_price_history.columns if column.endswith('%')]

    for column, color in zip(columns_to_plot, colores):
        ric = column.split('_')[0]
        name = ric_names.get(ric, ric)
        fig.add_trace(go.Scatter(
            x=df_price_history.index,
            y=df_price_history[column],
            mode='lines',
            name=name,
            line=dict(color=color),
            hovertemplate='%{y:.2f}%<br>Valor Liquidativo: %{customdata:.2f}',
            customdata=df_price_history[ric]
        ))

    fig.update_layout(
        title={
            'text': 'Historial de Precios',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Fecha',
        yaxis_title='Valor Liquidativo',
        xaxis=dict(rangeslider=dict(visible=True)),
        template='plotly_white'
    )

    # Crear la tabla con los valores proporcionados
    df_RIC = df.loc[df['RIC'].isin(list_RIC)]
    df_RIC = df_RIC.fillna('')
    df_RIC = df_RIC.replace('No Registrado', '')
    df_RIC.set_index('Nombre', inplace=True)

    df_basicos = df_RIC[['ISIN', 'Gestora', 'Moneda', 'Categoria', 'Ratio de Sostenibilidad', 'Valoracion Lipper',
                         'MiFID II', 'SRRI', 'Valor Liquidativo', 'Patrimonio']]
    df_basicos = df_basicos.rename(columns={'Categoria': 'Categoría',
                                             'Valoracion Lipper': 'Valoración Lipper'})
    df_basicos['Valor Liquidativo'] = df_basicos['Valor Liquidativo'].apply(lambda x: f'{x:,.2f} €' if x else x)
    df_basicos['Patrimonio'] = df_basicos['Patrimonio'].apply(lambda x: f'{x:,.2f} €' if x else x)
    df_basicos = df_basicos.T

    # Imprimir el DataFrame directamente en un Div
    tabla_basicos = html.Div([
        html.H2('Datos Básicos', style={'textAlign': 'center'}),
        html.Table(
            # Cabecera de la tabla
            [html.Tr([html.Th('Nombre', style=header_cell_style)] +
                     [html.Th(col, style=header_cell_style) for col in df_basicos.columns])] +
            # Contenido de la tabla
            [html.Tr([html.Td(df_basicos.index[i], style=cell_style)] +
                     [html.Td(df_basicos[col].iloc[i], style=cell_style) for col in df_basicos.columns]) for i in
             range(len(df_basicos))]
            , style=table_style),
    ])

    df_rentabilidad = df_RIC[['Rentabilidad YTD', 'Rentabilidad 6 Meses', 'Rentabilidad 1 Year', 'Rentabilidad 3 Year',
                               'Rentabilidad 5 Year', 'Rentabilidad 10 Year']]
    df_rentabilidad = df_rentabilidad.rename(columns={'Rentabilidad 1 Year': 'Rentabilidad 1 Año',
                                                       'Rentabilidad 3 Year': 'Rentabilidad 3 Años',
                                                       'Rentabilidad 5 Year': 'Rentabilidad 5 Años',
                                                       'Rentabilidad 10 Year': 'Rentabilidad 10 Años'})
    df_rentabilidad[['Rentabilidad YTD', 'Rentabilidad 6 Meses', 'Rentabilidad 1 Año', 'Rentabilidad 3 Años',
                     'Rentabilidad 5 Años', 'Rentabilidad 10 Años']] = df_rentabilidad[
        ['Rentabilidad YTD', 'Rentabilidad 6 Meses', 'Rentabilidad 1 Año', 'Rentabilidad 3 Años',
         'Rentabilidad 5 Años', 'Rentabilidad 10 Años']].map(lambda x: f'{x:.2f}%' if isinstance(x, (int, float)) else x)
    df_rentabilidad = df_rentabilidad.T

    tabla_rentabilidad = html.Div([
        html.H2('Rentabilidad', style={'textAlign': 'center'}),
        html.Table(
            # Cabecera de la tabla
            [html.Tr([html.Th('Nombre', style=header_cell_style)] +
                     [html.Th(col, style=header_cell_style) for col in df_rentabilidad.columns])] +
            # Contenido de la tabla
            [html.Tr([html.Td(df_rentabilidad.index[i], style=cell_style)] +
                     [html.Td(df_rentabilidad[col].iloc[i], style=cell_style) for col in df_rentabilidad.columns]) for
             i in range(len(df_rentabilidad))]
            , style=table_style),
    ])

    df_RIC = df_RIC.rename(columns={'Volatilidad 1 Year': 'Volatilidad 1 Año',
                                    'Volatilidad 3 Year': 'Volatilidad 3 Año',
                                    'Volatilidad 5 Year': 'Volatilidad 5 Año',
                                    'Maxima caida 1 Year': 'Máxima caída 1 Año',
                                    'Maxima caida 3 Year': 'Máxima caída 3 Año',
                                    'Maxima caida 5 Year': 'Máxima caída 5 Año',
                                    'Alpha 1 Year': 'Alpha 1 Año',
                                    'Alpha 3 Year': 'Alpha 3 Año',
                                    'Alpha 5 Year': 'Alpha 5 Año',
                                    'Beta 1 Year': 'Beta 1 Año',
                                    'Beta 3 Year': 'Beta 3 Año',
                                    'Beta 5 Year': 'Beta 5 Año',
                                    'Sharpe Ratio 1 Year': 'Sharpe Ratio 1 Año',
                                    'Sharpe Ratio 3 Year': 'Sharpe Ratio 3 Año',
                                    'Sharpe Ratio 5 Year': 'Sharpe Ratio 5 Año',
                                    'R^2 1 Year': 'R^2 1 Año',
                                    'R^2 3 Year': 'R^2 3 Año',
                                    'R^2 5 Year': 'R^2 5 Año',
                                    'Tracking Error 1 Year': 'Tracking Error 1 Año',
                                    'Tracking Error 3 Year': 'Tracking Error 3 Año',
                                    'Tracking Error 5 Year': 'Tracking Error 5 Año',
                                    'Correlacion 1 Year': 'Correlación 1 Año',
                                    'Correlacion 3 Year': 'Correlación 3 Año',
                                    'Correlacion 5 Year': 'Correlación 5 Año',
                                    'Information Ratio 1 Year': 'Ratio de Información 1 Año',
                                    'Information Ratio 3 Year': 'Ratio de Información 3 Año',
                                    'Information Ratio 5 Year': 'Ratio de Información 5 Año'})

    intervalos = ['1 Año', '3 Año', '5 Año']
    ratios = ['Volatilidad', 'Máxima caída', 'Alpha', 'Beta', 'Sharpe Ratio', 'R^2', 'Tracking Error', 'Correlación',
              'Ratio de Información']

    dfs_ratios = {}

    for intervalo in intervalos:
        cols = [f'{ratio} {intervalo}' for ratio in ratios]
        df_ratios = df_RIC[cols].T
        dfs_ratios[f'df_ratios_{intervalo.replace(" ", "").lower()}'] = df_ratios

    # Tabla de ratios seleccionada
    selected_df_ratios = dfs_ratios.get(f'df_ratios_{selected_tab}', pd.DataFrame())

    for col in selected_df_ratios.columns:
        selected_df_ratios[col] = selected_df_ratios[col].apply(lambda x: f'{x:.2f}' if isinstance(x, (int, float)) else x)

    tabla_ratios = html.Div([
        html.Table(
            # Cabecera de la tabla
            [html.Tr([html.Th('Nombre', style=header_cell_style)] +
                     [html.Th(col, style=header_cell_style) for col in selected_df_ratios.columns])] +
            # Contenido de la tabla
            [html.Tr([html.Td(selected_df_ratios.index[i], style=cell_style)] +
                     [html.Td(selected_df_ratios[col].iloc[i], style=cell_style) for col in selected_df_ratios.columns]) for
             i in range(len(selected_df_ratios))]
            , style={'border': '1px solid black',  # Añade bordes a la tabla
                     'border-collapse': 'collapse',  # Combina los bordes de las celdas,
                     'margin': 'auto',  # Centra la tabla en la página,
                     'margin-top': '20px',
                    }),
    ])

    df_comercializacion = df_RIC[['Inversion Minima', 'Comision de Gestion', 'Comision de Reembolso', 'Gastos Corrientes']]
    df_comercializacion = df_comercializacion.rename(columns={'Inversion Minima': 'Inversión Mínima',
                                                              'Comision de Gestion': 'Comisión de Gestión',
                                                              'Comision de Reembolso': 'Comisión de Reembolso'})
    df_comercializacion['Inversión Mínima'] = df_comercializacion['Inversión Mínima'].apply(lambda x: f'{x:,.2f} €'.format(x))
    df_comercializacion[['Comisión de Gestión', 'Comisión de Reembolso', 'Gastos Corrientes']] = df_comercializacion[
        ['Comisión de Gestión', 'Comisión de Reembolso', 'Gastos Corrientes']].map(
        lambda x: f'{x:.2f} %'.format(x))
    df_comercializacion = df_comercializacion.T

    tabla_comercializacion = html.Div([
        html.H2('Comercialización', style={'textAlign': 'center'}),
        html.Table(
            # Cabecera de la tabla
            [html.Tr([html.Th('Nombre', style=header_cell_style)] +
                     [html.Th(col, style=header_cell_style) for col in df_comercializacion.columns])] +
            # Contenido de la tabla
            [html.Tr([html.Td(df_comercializacion.index[i], style=cell_style)] +
                     [html.Td(df_comercializacion[col].iloc[i], style=cell_style) for col in df_comercializacion.columns]) for
             i in range(len(df_comercializacion))]
            , style=table_style),
    ])

    return fig, [tabla_basicos, tabla_rentabilidad, tabla_comercializacion], tabla_ratios

@app.callback(
    Output('recommendation-content', 'children'),
    [Input('dropdown-selection', 'value')]
)
def update_recommendation(selected_funds):
    if not selected_funds:
        return html.Div("Por favor, selecciona uno o más fondos para obtener una recomendación.")

    # Filtra df para obtener solo los fondos seleccionados
    selected_df = df[df['ISIN'].isin(selected_funds)]
    
    # Preprocesamiento de los datos seleccionados
    numeric_features = selected_df.select_dtypes(include=['float64', 'int64']).columns
    selected_df[numeric_features] = selected_df[numeric_features].fillna(selected_df[numeric_features].mean())
    scaled_selected_funds = scaler.transform(selected_df[numeric_features])
    
    # Aplicar PCA a los fondos seleccionados
    pca_selected_funds = pca.transform(scaled_selected_funds)
    
    # Predicción con el autoencoder para reconstruir las características
    predicted_features = autoencoder_model.predict(pca_selected_funds)
    
    # Calcula la distancia euclidiana entre los fondos reconstruidos y los originales
    distances = np.linalg.norm(pca_selected_funds - predicted_features, axis=1)
    
    # Encontrar el índice del fondo más cercano en términos de características reconstruidas
    recommended_index = np.argmin(distances)
    recommended_fund = selected_df.iloc[recommended_index]
    
    # Devuelve la información del fondo recomendado
    recommended_content = html.Div([
        html.H1("Fondo de inversión recomendado:", style={'textAlign': 'center'}),
        html.P(f"Nombre: {recommended_fund['Nombre']}"),
        html.P(f"ISIN: {recommended_fund['ISIN']}")
    ])
    
    return recommended_content

if __name__ == '__main__':
    app.run_server(debug=True, port=6051)
