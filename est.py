import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- Configuração da Página Streamlit ---
# REMOVIDO o argumento 'icon' para compatibilidade com versões mais antigas do Streamlit
st.set_page_config(layout="wide", page_title="DOE PFF - Lubrificantes Industriais")


# --- Funções Auxiliares para Codificação de Fatores ---
# Converte valores categóricos selecionados para seus códigos -1 ou +1
def get_coded_value_categorical(factor_name, selected_value):
    if factor_name == "Tipo de Antioxidante":
        return -1 if selected_value == "Amínico" else 1
    elif factor_name == "Tipo de Antidesgaste":
        return -1 if selected_value == "Com Zinco" else 1
    elif factor_name == "Tipo de Óleo Base":
        return -1 if selected_value == "Leve/Médio" else 1
    return 0


# Converte valores numéricos selecionados para seus códigos -1 (min), 0 (centro), +1 (max)
def get_coded_value_numerical(factor_name, selected_value):
    if factor_name == "Quantidade de Antioxidante":  # Min=0.2, Centro=0.5, Max=0.8
        # Ensure selected_value is treated as float for calculation
        return (float(selected_value) - 0.5) / 0.3
    elif factor_name == "Quantidade de Antidesgaste":  # Min=0.5, Centro=1.75, Max=3.0
        return (float(selected_value) - 1.75) / 1.25
    elif factor_name == "Razão de Óleo Base":  # Min=20, Centro=50, Max=80
        return (float(selected_value) - 50) / 30
    return 0


# --- Implementação dos Modelos (Baseado nos achados do Artigo) ---

# Modelo preditivo para o Índice de Acidez (IAT) - Baseado na Equação do Artigo
# Assume que A, B, C, D, E, F são os valores codificados (-1, 0, +1)
def calculate_iat(A_coded, B_coded, C_coded, D_coded, E_coded, F_coded):
    # Coeficientes da equação linear do artigo para Acidez
    # ÍNDICE DE ACIDEZ = 1.92 + 0.03 * A – 0.01 * B + 1.28 * C + 0.99 * D + 0.01 * E + 0.01 * F + 0.02 * AB + 0.03 * AC + 0.05 * AF – 0.03 * BD + 0.91 * BF + 0.03 * ABD – 0.02 * ABF ± 0.17

    iat = (
            1.92
            + 0.03 * A_coded
            - 0.01 * B_coded
            + 1.28 * C_coded
            + 0.99 * D_coded
            + 0.01 * E_coded
            + 0.01 * F_coded
            + 0.02 * (A_coded * B_coded)
            + 0.03 * (A_coded * C_coded)
            + 0.05 * (A_coded * F_coded)
            - 0.03 * (B_coded * D_coded)
            + 0.91 * (B_coded * F_coded)
            + 0.03 * (A_coded * B_coded * D_coded)  # Interação ABD
            - 0.02 * (A_coded * B_coded * F_coded)  # Interação ABF
    )
    # O artigo menciona +/- 0.17 como erro, o que representa uma faixa de incerteza.
    error_range = 0.17
    return iat, error_range


# Modelo qualitativo para Viscosidade - Baseado nas tendências discutidas no artigo
# O artigo afirmou que não foi possível propor um modelo preditivo linear válido devido à curvatura.
# Esta função reflete as tendências dominantes e a interação A x D.
def calculate_viscosity_qualitative(A_coded, B_val_actual, C_coded, D_val_actual, E_coded, F_val_actual):
    # Valores base aproximados da Tabela 18 do artigo para Leve/Médio vs Pesado/Médio
    base_viscosity = 0.0
    if E_coded == -1:  # Leve/Médio
        base_viscosity = 40.0  # Aproximado do range de 30-50 para Leve/Médio
    else:  # Pesado/Médio
        base_viscosity = 70.0  # Aproximado do range de 60-80 para Pesado/Médio

    # Efeito da Quantidade de Antidesgaste (D_val_actual)
    # "o aumento na concentração de antidesgaste promove uma redução na viscosidade do lubrificante"
    # Adicionando um pequeno efeito negativo direto da quantidade, escalado.
    D_val_scaled_for_effect = (D_val_actual - 0.5) / 2.5  # Escala de 0 a 1 para o range de D
    base_viscosity += -3 * D_val_scaled_for_effect  # Penalidade por aumentar D

    # Interação A (Tipo de Antioxidante) x D (Quant. de Antidesgaste) - Figura 40 do artigo
    # Isso é um ajuste sobre a base.
    if A_coded == -1:  # Amínico
        # "viscosidade do lubrificante assume um valor máximo no nível inferior da variável D"
        # "diminuindo seu valor conforme o aumento da quantidade de antidesgaste"
        base_viscosity += -8 * D_val_scaled_for_effect  # Acentua a queda com D_val_actual
    else:  # Fenólico
        # "viscosidade assume o menor valor no nível inferior da variável D"
        # "aumentando seu valor conforme o aumento do nível de antidesgaste"
        base_viscosity += 8 * D_val_scaled_for_effect  # Acentua o aumento com D_val_actual

    # Adicionar uma pequena variação aleatória para simular ruído/variabilidade
    noise = np.random.uniform(-1, 1)
    predicted_viscosity = base_viscosity + noise

    # Garantir que a viscosidade esteja dentro de um range razoável observado no artigo (30-87 cSt)
    return max(28, min(89, predicted_viscosity))


# Modelo qualitativo para Desgaste - Baseado nas conclusões do artigo
# O artigo afirmou que não foi possível propor um modelo preditivo significativo para o desgaste.
# Esta função ilustra a importância dos aditivos e as tendências qualitativas.
def calculate_desgaste_qualitative(selected_tipo_antidesgaste, D_val_actual):
    # O artigo afirma que o óleo base sem aditivo tem desgaste de 0.771 mm
    # e a média com aditivos é de 0.409 mm.

    # Se "Sem Aditivo Antidesgaste" fosse uma opção, o valor seria 0.771.
    # Como o usuário SEMPRE seleciona um tipo de aditivo (Com Zinco ou Sem Zinco),
    # o valor inicial será próximo da média com aditivos.
    desgaste = 0.409

    # C_coded = -1 (Com Zinco), +1 (Sem Zinco)
    C_coded = get_coded_value_categorical("Tipo de Antidesgaste", selected_tipo_antidesgaste)

    # "o emprego do Antidesgaste sem Zinco, por conciliar proteção contra desgaste e baixa acidez."
    # Isso sugere que "Sem Zinco" resulta em um desgaste ligeiramente menor.
    if C_coded == -1:  # Com Zinco
        desgaste += 0.015  # Um pouco mais de desgaste
    else:  # Sem Zinco
        desgaste -= 0.005  # Um pouco menos de desgaste

    # "bastando acrescentar esse aditivo em sua menor proporção (0,5 %), para conseguir a proteção desejada."
    # Isso implica que, uma vez que o aditivo é adicionado, a quantidade D_val_actual não tem um grande efeito
    # de redução adicional no desgaste. Podemos adicionar uma pequena penalidade se a quantidade for muito baixa (mas ainda presente).
    if D_val_actual < 0.75:  # Se a quantidade for próxima do mínimo (0.5), talvez um leve aumento ou sem mudança
        desgaste += 0.005  # Pequeno aumento para ilustrar que o mínimo é já eficaz, mas não "super" eficaz.

    # Adicionar uma pequena variação aleatória para simular ruído/variabilidade
    noise = np.random.uniform(-0.005, 0.005)
    predicted_desgaste = desgaste + noise

    # Garantir que o desgaste esteja dentro de um range razoável observado no artigo (0.348 - 0.428 com aditivo)
    return max(0.34, min(0.43, predicted_desgaste))


# --- Título e Introdução do Aplicativo ---
st.title("🧪 Otimização de Formulações de Lubrificantes com PFF")
st.markdown(
    "Uma aplicação interativa para explorar os resultados de um experimento de **Planejamento Fatorial Fracionado (PFF)** na formulação de óleos lubrificantes industriais.")
st.info(
    "Esta ferramenta simula os resultados discutidos na dissertação de mestrado de Marcelo O.Q. de Almeida (2019), demonstrando a aplicação do DOE para análise de fatores e respostas. Modelos preditivos para acidez, e tendências qualitativas para viscosidade e desgaste.")

# --- Sidebar para Seleção dos Fatores ---
st.sidebar.header("⚙️ Fatores do Experimento")
st.sidebar.markdown("Ajuste os níveis de cada fator para observar seu impacto nas propriedades do lubrificante.")

# Fator A: Tipo de Antioxidante
tipo_antioxidante_options = ["Amínico", "Fenólico"]
selected_tipo_antioxidante = st.sidebar.selectbox("Tipo de Antioxidante (A)", tipo_antioxidante_options, key="sel_A")
A_coded = get_coded_value_categorical("Tipo de Antioxidante", selected_tipo_antioxidante)

# Fator B: Quantidade de Antioxidante
quant_antioxidante_options_values = [0.2, 0.5, 0.8]
selected_quant_antioxidante = st.sidebar.select_slider("Quantidade de Antioxidante (B) [% peso]",
                                                       options=quant_antioxidante_options_values,
                                                       value=0.5, key="sel_B")
B_coded = get_coded_value_numerical("Quantidade de Antioxidante", selected_quant_antioxidante)

# Fator C: Tipo de Antidesgaste
tipo_antidesgaste_options = ["Com Zinco", "Sem Zinco"]
selected_tipo_antidesgaste = st.sidebar.selectbox("Tipo de Antidesgaste (C)", tipo_antidesgaste_options, key="sel_C")
C_coded = get_coded_value_categorical("Tipo de Antidesgaste", selected_tipo_antidesgaste)

# Fator D: Quantidade de Antidesgaste
quant_antidesgaste_options_values = [0.5, 1.75, 3.0]
selected_quant_antidesgaste = st.sidebar.select_slider("Quantidade de Antidesgaste (D) [% peso]",
                                                       options=quant_antidesgaste_options_values,
                                                       value=1.75, key="sel_D")
D_coded = get_coded_value_numerical("Quantidade de Antidesgaste", selected_quant_antidesgaste)

# Fator E: Tipo de Óleo Base
tipo_oleo_base_options = ["Leve/Médio", "Pesado/Médio"]
selected_tipo_oleo_base = st.sidebar.selectbox("Tipo de Óleo Base (E)", tipo_oleo_base_options, key="sel_E")
E_coded = get_coded_value_categorical("Tipo de Óleo Base", selected_tipo_oleo_base)

# Fator F: Razão de Óleo Base
razao_oleo_base_options_values = [20, 50, 80]
selected_razao_oleo_base = st.sidebar.select_slider("Razão de Óleo Base (F) [%]",
                                                    options=razao_oleo_base_options_values,
                                                    value=50, key="sel_F")
F_coded = get_coded_value_numerical("Razão de Óleo Base", selected_razao_oleo_base)

st.sidebar.markdown("---")
st.sidebar.write("ℹ️ Quantidade de Anticorrosivo: **1.5% peso (fixo)**")
st.sidebar.markdown("---")

# --- Cálculo das Respostas com base nas Seleções do Usuário ---
# Corrigido: As chaves do dicionário devem corresponder aos nomes dos parâmetros de calculate_iat
current_coded_params_for_iat = {
    'A_coded': A_coded, 'B_coded': B_coded, 'C_coded': C_coded,
    'D_coded': D_coded, 'E_coded': E_coded, 'F_coded': F_coded
}

iat_predicted, iat_error = calculate_iat(**current_coded_params_for_iat)
viscosity_predicted = calculate_viscosity_qualitative(A_coded, selected_quant_antioxidante, C_coded,
                                                      selected_quant_antidesgaste, E_coded, selected_razao_oleo_base)
desgaste_predicted = calculate_desgaste_qualitative(selected_tipo_antidesgaste, selected_quant_antidesgaste)

st.header("📊 Resultados Preditos (Simulados)")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Índice de Acidez Total (IAT)")
    st.markdown(f"**{iat_predicted:.2f} ± {iat_error:.2f} mg KOH/g**")
    st.success(f"**Modelo estatisticamente significativo** com R² = 0.9938.")

with col2:
    st.subheader("Viscosidade Cinemática (40°C)")
    st.markdown(f"**{viscosity_predicted:.2f} cSt**")
    st.warning(
        "Modelo qualitativo: **Curvatura significativa** no estudo original. Predição baseada em tendências observadas no artigo.")

with col3:
    st.subheader("Desgaste por 4 Esferas")
    st.markdown(f"**{desgaste_predicted:.3f} mm**")
    st.error(
        "Modelo qualitativo: **Sem modelo preditivo significativo** no estudo original. Valor ilustrativo de tendências e comparação com óleo base.")

st.markdown("---")

st.header("📈 Análise de Efeitos e Interações")
st.markdown(
    "Estes gráficos ilustram o impacto dos fatores nas respostas. Para o IAT, eles refletem o modelo linear. Para viscosidade e desgaste, as tendências são baseadas nas discussões e observações qualitativas do artigo.")

# --- Gráfico de Perturbação para IAT ---
st.subheader("1. Gráfico de Perturbação para Índice de Acidez (IAT)")
st.info(
    "Este gráfico mostra como a acidez predita muda quando cada fator é variado individualmente de seu valor mais baixo para o mais alto (ou entre categorias), enquanto os outros fatores são mantidos nos valores selecionados atualmente na sidebar.")

perturbation_data = []

# Definir os fatores e seus níveis para a perturbação
# Cada tupla agora inclui o nome exato do parâmetro em calculate_iat
factors_for_perturbation = [
    ("Tipo de Antioxidante (A)", tipo_antioxidante_options, "categorical", "A_coded"),
    ("Quantidade de Antioxidante (B)", quant_antioxidante_options_values, "numerical", "B_coded"),
    ("Tipo de Antidesgaste (C)", tipo_antidesgaste_options, "categorical", "C_coded"),
    ("Quantidade de Antidesgaste (D)", quant_antidesgaste_options_values, "numerical", "D_coded"),
    ("Tipo de Óleo Base (E)", tipo_oleo_base_options, "categorical", "E_coded"),
    ("Razão de Óleo Base (F)", razao_oleo_base_options_values, "numerical", "F_coded"),
]

# Recalcular IAT para cada fator em seus extremos (mantendo os outros no valor selecionado)
for factor_full_name, levels, factor_type, iat_param_name in factors_for_perturbation:
    if factor_type == "categorical":
        for level_val_str in levels:
            temp_params = current_coded_params_for_iat.copy()
            # Obter o valor codificado para o nível específico do fator sendo perturbado
            temp_params[iat_param_name] = get_coded_value_categorical(factor_full_name.split(' (')[0], level_val_str)
            predicted_val, _ = calculate_iat(**temp_params)
            perturbation_data.append({
                "Fator": factor_full_name,
                "Nível": level_val_str,
                "Acidez Predita": predicted_val,
            })
    else:  # Fatores Numéricos (Min e Max para a linha)
        # Usar min e max dos níveis reais para calcular os pontos da linha
        min_val_coded = get_coded_value_numerical(factor_full_name.split(' (')[0], levels[0])
        max_val_coded = get_coded_value_numerical(factor_full_name.split(' (')[0], levels[-1])

        # Ponto Mínimo do fator
        temp_params_min = current_coded_params_for_iat.copy()
        temp_params_min[iat_param_name] = min_val_coded
        predicted_min, _ = calculate_iat(**temp_params_min)
        perturbation_data.append({
            "Fator": factor_full_name,
            "Nível": str(levels[0]),  # Converter para string para o eixo X
            "Acidez Predita": predicted_min,
        })

        # Ponto Máximo do fator
        temp_params_max = current_coded_params_for_iat.copy()
        temp_params_max[iat_param_name] = max_val_coded
        predicted_max, _ = calculate_iat(**temp_params_max)
        perturbation_data.append({
            "Fator": factor_full_name,
            "Nível": str(levels[-1]),  # Converter para string para o eixo X
            "Acidez Predita": predicted_max,
        })

df_perturbation = pd.DataFrame(perturbation_data)

fig_perturbation = go.Figure()

# Adicionar as linhas de perturbação
for factor in df_perturbation['Fator'].unique():
    df_factor = df_perturbation[df_perturbation['Fator'] == factor]
    fig_perturbation.add_trace(go.Scatter(x=df_factor['Nível'], y=df_factor['Acidez Predita'],
                                          mode='lines+markers', name=factor,
                                          marker=dict(size=8)))

# Adicionar o ponto da seleção atual para cada linha
# Usamos os valores atualmente selecionados (reais) para os nomes dos níveis no eixo X
current_iat_val_at_selection, _ = calculate_iat(**current_coded_params_for_iat)

# Mapear os valores selecionados (reais) para as chaves usadas no gráfico de perturbação (Nível)
# Garante que os valores numéricos sejam convertidos para string para corresponder ao eixo X
current_selection_display_map = {
    "Tipo de Antioxidante (A)": selected_tipo_antioxidante,
    "Quantidade de Antioxidante (B)": str(selected_quant_antioxidante),
    "Tipo de Antidesgaste (C)": selected_tipo_antidesgaste,
    "Quantidade de Antidesgaste (D)": str(selected_quant_antidesgaste),
    "Tipo de Óleo Base (E)": selected_tipo_oleo_base,
    "Razão de Óleo Base (F)": str(selected_razao_oleo_base),
}

for factor_full_name in df_perturbation['Fator'].unique():
    fig_perturbation.add_trace(go.Scatter(
        x=[current_selection_display_map[factor_full_name]],
        y=[current_iat_val_at_selection],
        mode='markers',
        marker=dict(color='black', size=10, symbol='x'),
        name=f"Seleção Atual ({factor_full_name.split(' (')[1][0]})",
        showlegend=False  # Não mostrar na legenda principal para evitar repetição
    ))

fig_perturbation.update_layout(
    xaxis_title="Nível do Fator (Mínimo/Máximo ou Categoria)",
    yaxis_title="Acidez Predita (mg KOH/g)",
    title="Gráfico de Perturbação para Índice de Acidez (IAT)",
    legend_title="Fator",
    hovermode="x unified"
)
st.plotly_chart(fig_perturbation, use_container_width=True)

st.markdown("---")

# --- Gráfico de Interação Qualitativa (A x D) para Viscosidade ---
st.subheader("2. Interação Qualitativa: Tipo de Antioxidante (A) vs. Quantidade de Antidesgaste (D) na Viscosidade")
st.warning(
    "Este gráfico representa visualmente a interação discutida no artigo (Figura 40), utilizando valores aproximados do modelo qualitativo para a viscosidade. **Lembre-se da curvatura significativa** no estudo original para esta resposta.")

interaction_data = []
quant_antidesgaste_levels_actual = [0.5, 3.0]  # Níveis reais Mínimo e Máximo para Quant. de Antidesgaste (D)
tipo_antioxidante_levels_str = ["Amínico", "Fenólico"]  # Níveis para Tipo de Antioxidante (A)

for A_val_str in tipo_antioxidante_levels_str:
    A_coded_for_visc = get_coded_value_categorical("Tipo de Antioxidante", A_val_str)
    for D_val_float in quant_antidesgaste_levels_actual:
        visc = calculate_viscosity_qualitative(A_coded_for_visc, selected_quant_antioxidante,
                                               C_coded, D_val_float, E_coded, selected_razao_oleo_base)
        interaction_data.append({
            "Tipo de Antioxidante": A_val_str,
            "Quantidade de Antidesgaste": D_val_float,
            "Viscosidade Predita": visc
        })

df_interaction = pd.DataFrame(interaction_data)

fig_interaction = px.line(df_interaction, x="Quantidade de Antidesgaste", y="Viscosidade Predita",
                          color="Tipo de Antioxidante", markers=True,
                          title="Interação A x D para Viscosidade (Qualitativo)",
                          hover_data={"Quantidade de Antidesgaste": True, "Viscosidade Predita": ':.2f',
                                      "Tipo de Antioxidante": True})
fig_interaction.update_layout(
    xaxis_title="Quantidade de Antidesgaste (% peso)",
    yaxis_title="Viscosidade Predita (cSt)",
    legend_title="Tipo de Antioxidante",
    hovermode="x unified"
)
st.plotly_chart(fig_interaction, use_container_width=True)

st.markdown("---")

# --- Comparativo Qualitativo de Desgaste ---
st.subheader("3. Comparativo Qualitativo de Desgaste por 4 Esferas")
st.warning(
    "Esta seção ilustra a importância fundamental da presença de aditivos antidesgaste, conforme as conclusões do artigo. **Lembre-se que não foi encontrado um modelo preditivo significativo para o desgaste** no estudo original.")

desgaste_comparison_data = [
    {"Condição": "Óleo Base sem Aditivo Antidesgaste", "Desgaste (mm)": 0.771},
    {"Condição": "Com Aditivo Antidesgaste (Média Observada)", "Desgaste (mm)": 0.409},
    {"Condição": f"Sua Seleção: {selected_tipo_antidesgaste} ({selected_quant_antidesgaste}%)",
     "Desgaste (mm)": desgaste_predicted}
]
df_desgaste_comparison = pd.DataFrame(desgaste_comparison_data)

fig_desgaste = px.bar(df_desgaste_comparison, x="Condição", y="Desgaste (mm)",
                      title="Comparativo de Desgaste (Qualitativo)",
                      color="Condição",
                      color_discrete_map={
                          "Óleo Base sem Aditivo Antidesgaste": "red",
                          "Com Aditivo Antidesgaste (Média Observada)": "lightgray",
                          f"Sua Seleção: {selected_tipo_antidesgaste} ({selected_quant_antidesgaste}%)": "blue"
                      })
fig_desgaste.update_layout(yaxis_title="Desgaste (mm)")
st.plotly_chart(fig_desgaste, use_container_width=True)

st.markdown("---")
st.markdown("### 📚 Referência")
st.markdown(
    "Almeida, M. O. Q. (2019). *Avaliação das Influências de Variáveis Composicionais na Formulação de Óleos Lubrificantes Industriais Usando Design de Experimentos*. Dissertação de Mestrado, Universidade Federal do Rio de Janeiro.")

# --- Rodapé ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray; font-size: small;'>"
    "Desenvolvida por <b>Diógenes Oliveira</b> e <b>Carlos Marins</b>, "
    "alunos do Mestrado em Engenharia Ambiental da UERJ, "
    "para a disciplina de Design of Experiments (DOE) ministrada pelo "
    "Prof. Dr. Nilo Antônio de Souza Sampaio. © 2026."
    "</p>",
    unsafe_allow_html=True
)
