import streamlit as st
import pandas as pd
from scipy.stats import f_oneway, ttest_rel, ttest_ind, levene
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações gerais do Streamlit
st.set_page_config(layout="wide", page_title="Análise Estatística de Exercícios")

# --- Inicialização do Session State (CORRIGIDO) ---
# Garante que 'user_custom_name' existe antes de ser usado.
if 'user_custom_name' not in st.session_state:
    st.session_state.user_custom_name = "Diógenes Oliveira" # Pode ser seu nome real ou um placeholder

# --- Conteúdo Principal da Aplicação ---
st.title("📊 Aplicação de Análise Estatística para Engenheiros")
st.write(f"Olá, {st.session_state.user_custom_name}! Esta aplicação demonstra a resolução dos exercícios de testes estatísticos que você me apresentou, utilizando Python e Streamlit.")
st.write("Explore as análises e visualize os resultados de cada cenário.")

st.markdown("---")

## Função para o Exercício 1: ANOVA (Comparação de 3 Métodos de Ensino)
def run_exercise_1():
    st.header("1. Comparação de Três Métodos de Ensino (ANOVA)")
    st.markdown("""
    **Problema:** Uma escola está testando três métodos diferentes de ensino para melhorar o desempenho dos alunos em matemática.
    Após 1 semestre, 5 alunos de cada grupo foram submetidos à mesma prova de matemática, com pontuação de 0 a 100.
    O objetivo é verificar se há diferença significativa no desempenho médio dos alunos entre os três métodos de ensino (5% de significância).
    """)

    # Dados do problema
    data = {
        'Metodo': ['A'] * 5 + ['B'] * 5 + ['C'] * 5,
        'Pontuacao': [70, 65, 60, 72, 68, 80, 85, 75, 78, 82, 90, 95, 85, 88, 92]
    }
    df = pd.DataFrame(data)

    st.subheader("Dados Brutos:")
    st.dataframe(df)

    st.subheader("Visualização dos Dados (Box Plot):")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Metodo', y='Pontuacao', data=df, ax=ax1, palette='viridis')
    ax1.set_title('Desempenho dos Alunos por Método de Ensino')
    ax1.set_xlabel('Método de Ensino')
    ax1.set_ylabel('Pontuação')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig1)
    st.write("O box plot nos permite visualizar a distribuição das pontuações para cada método, incluindo medianas, quartis e possíveis outliers. Isso dá uma primeira impressão das diferenças.")

    st.subheader("Análise Estatística: ANOVA")
    st.markdown("""
    Para comparar as médias de três ou mais grupos independentes, utilizamos a **Análise de Variância (ANOVA)**.

    *   **Hipótese Nula (H₀):** As médias de desempenho dos alunos são iguais para os três métodos de ensino (μA = μB = μC).
    *   **Hipótese Alternativa (H₁):** Pelo menos uma das médias de desempenho dos alunos é diferente das outras.
    *   **Nível de Significância (α):** 0.05 (5%).
    """)

    # Preparar os dados para ANOVA
    metodo_a = df[df['Metodo'] == 'A']['Pontuacao']
    metodo_b = df[df['Metodo'] == 'B']['Pontuacao']
    metodo_c = df[df['Metodo'] == 'C']['Pontuacao']

    # Realizar o teste ANOVA
    f_statistic, p_value = f_oneway(metodo_a, metodo_b, metodo_c)

    st.write(f"**Estatística F da ANOVA:** `{f_statistic:.2f}`")
    st.write(f"**Valor p da ANOVA:** `{p_value:.3f}`")

    st.subheader("Interpretação dos Resultados da ANOVA:")
    if p_value < 0.05:
        st.success(f"Como o valor p ({p_value:.3f}) é menor que o nível de significância (0.05), rejeitamos a Hipótese Nula.")
        st.write("Isso indica que há uma **diferença estatisticamente significativa** no desempenho médio dos alunos entre os métodos de ensino. Ou seja, pelo menos um método difere dos outros.")
        st.subheader("Teste Post-Hoc (Tukey HSD):")
        st.markdown("Para identificar *quais* grupos são diferentes entre si, realizamos um Teste Post-Hoc, como o **Teste de Tukey HSD**.")
        tukey_result = pairwise_tukeyhsd(endog=df['Pontuacao'], groups=df['Metodo'], alpha=0.05)
        st.write(tukey_result)
        st.markdown("""
        Na tabela do Tukey HSD:
        *   `reject=True` indica que há uma diferença significativa entre o par de grupos.
        *   `reject=False` indica que não há evidência de diferença significativa entre o par de grupos.
        """)
    else:
        st.info(f"Como o valor p ({p_value:.3f}) é maior ou igual ao nível de significância (0.05), não rejeitamos a Hipótese Nula.")
        st.write("Isso sugere que não há evidência suficiente para afirmar que existe uma diferença significativa no desempenho médio dos alunos entre os métodos de ensino.")
    st.markdown("---")

## Função para o Exercício 2: Teste t Pareado (Programa de Reforço)
def run_exercise_2():
    st.header("2. Efetividade de um Programa de Reforço (Teste t Pareado)")
    st.markdown("""
    **Problema:** Uma escola aplica um programa intensivo de reforço por 8 semanas para melhorar a nota em matemática de 10 alunos. As notas foram registradas antes (pré) e depois (pós) do programa.
    O objetivo é verificar se o Programa foi efetivo (5% de significância).
    """)

    # Dados do problema
    data_prog = {
        'Aluno': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Pre': [65, 70, 68, 72, 60, 75, 69, 71, 66, 64],
        'Pos': [70, 74, 72, 78, 65, 80, 73, 75, 70, 68]
    }
    df_prog = pd.DataFrame(data_prog)
    df_prog['Diferenca'] = df_prog['Pos'] - df_prog['Pre']

    st.subheader("Dados Brutos:")
    st.dataframe(df_prog)

    st.subheader("Visualização dos Dados (Mudança Individual Pré-Pós):")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for index, row in df_prog.iterrows():
        ax2.plot([0, 1], [row['Pre'], row['Pos']], color='gray', linestyle='-', marker='o', alpha=0.6)
        ax2.text(0, row['Pre'], f"({row['Pre']})", ha='right', va='center', fontsize=8, color='blue')
        ax2.text(1, row['Pos'], f"({row['Pos']})", ha='left', va='center', fontsize=8, color='green')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Pré-Programa', 'Pós-Programa'])
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_title('Mudança na Pontuação de Matemática por Aluno')
    ax2.set_ylabel('Pontuação')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig2)
    st.write("Este gráfico mostra a mudança individual para cada aluno. Uma linha ascendente indica melhora, descendente indica piora.")


    st.subheader("Análise Estatística: Teste t para Amostras Pareadas")
    st.markdown("""
    Para comparar as médias de duas amostras relacionadas (os mesmos indivíduos medidos antes e depois de uma intervenção), utilizamos o **Teste t para Amostras Pareadas**.

    *   **Hipótese Nula (H₀):** A média das diferenças (Pós - Pré) é igual a zero (μ_diff = 0). O programa não teve efeito.
    *   **Hipótese Alternativa (H₁):** A média das diferenças (Pós - Pré) é maior que zero (μ_diff > 0). O programa foi efetivo em aumentar as notas (teste unilateral).
    *   **Nível de Significância (α):** 0.05 (5%).
    """)

    # Realizar o Teste t Pareado
    t_statistic, p_value_bilateral = ttest_rel(df_prog['Pos'], df_prog['Pre'])

    # Para teste unilateral (H1: Pos > Pre), dividimos o p-value bilateral por 2
    # e verificamos a direção da média das diferenças.
    mean_diff = df_prog['Diferenca'].mean()
    if mean_diff > 0:
        p_value_unilateral = p_value_bilateral / 2
    else:
        p_value_unilateral = 1 - (p_value_bilateral / 2) # Se a média da diferença for negativa ou zero, não há suporte para H1

    st.write(f"**Estatística t do Teste Pareado:** `{t_statistic:.2f}`")
    st.write(f"**Valor p (unilateral) do Teste Pareado:** `{p_value_unilateral:.3f}`")

    st.subheader("Interpretação dos Resultados do Teste t Pareado:")
    if p_value_unilateral < 0.05 and mean_diff > 0:
        st.success(f"Como o valor p unilateral ({p_value_unilateral:.3f}) é menor que o nível de significância (0.05) e a média das diferenças é positiva ({mean_diff:.2f}), rejeitamos a Hipótese Nula.")
        st.write("Isso indica que o programa de reforço foi **estatisticamente efetivo** em aumentar as notas dos alunos.")
    else:
        st.info(f"Como o valor p unilateral ({p_value_unilateral:.3f}) é maior ou igual ao nível de significância (0.05), ou a média das diferenças não é positiva, não rejeitamos a Hipótese Nula.")
        st.write("Isso sugere que não há evidência suficiente para afirmar que o programa de reforço foi efetivo em aumentar as notas.")
    st.markdown("---")

## Função para o Exercício 3: Teste t Independente (Grupo Experimental vs. Controle)
def run_exercise_3():
    st.header("3. Avaliação de Novo Método de Ensino (Grupo Experimental vs. Controle - Teste t Independente)")
    st.markdown("""
    **Problema:** Uma escola quer avaliar se um novo método de ensino (Grupo A — método experimental) aumenta as notas de matemática em comparação ao método tradicional (Grupo B — controle). Foram selecionados 8 alunos aleatoriamente para cada grupo e, após o curso, aplicou-se um teste padronizado de matemática.
    Nível de significância: α = 0,05.
    """)

    # Dados do problema
    grupo_a = [78, 82, 85, 90, 74, 88, 79, 84]
    grupo_b = [72, 75, 70, 68, 74, 69, 73, 71]

    # Criar DataFrame para visualização
    df_grupos = pd.DataFrame({
        'Grupo': ['A'] * len(grupo_a) + ['B'] * len(grupo_b),
        'Pontuacao': grupo_a + grupo_b
    })

    st.subheader("Dados Brutos:")
    st.write("Grupo A (Experimental):", grupo_a)
    st.write("Grupo B (Controle):", grupo_b)
    st.dataframe(df_grupos)

    st.subheader("Visualização dos Dados (Box Plot):")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Grupo', y='Pontuacao', data=df_grupos, ax=ax3, palette='coolwarm')
    ax3.set_title('Comparação de Pontuações entre Grupo Experimental (A) e Controle (B)')
    ax3.set_xlabel('Grupo de Ensino')
    ax3.set_ylabel('Pontuação')
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig3)
    st.write("O box plot permite comparar as distribuições de pontuação e as medianas de ambos os grupos.")

    st.subheader("Análise Estatística: Teste t para Amostras Independentes")
    st.markdown("""
    Para comparar as médias de dois grupos independentes, utilizamos o **Teste t para Amostras Independentes**. Antes, verificamos a homogeneidade das variâncias com o Teste de Levene.

    *   **Hipótese Nula (H₀):** A média de pontuação do Grupo A é menor ou igual à média de pontuação do Grupo B (μA ≤ μB). Não há aumento nas notas.
    *   **Hipótese Alternativa (H₁):** A média de pontuação do Grupo A é maior que a média de pontuação do Grupo B (μA > μB). O novo método aumenta as notas (teste unilateral).
    *   **Nível de Significância (α):** 0.05 (5%).
    """)

    st.markdown("### Verificação de Homogeneidade das Variâncias (Teste de Levene):")
    stat_levene, p_levene = levene(grupo_a, grupo_b)
    st.write(f"**Estatística de Levene:** `{stat_levene:.2f}`")
    st.write(f"**Valor p do Teste de Levene:** `{p_levene:.3f}`")

    equal_variances = p_levene >= 0.05
    if equal_variances:
        st.info("Como o valor p do Teste de Levene é maior ou igual a 0.05, não rejeitamos a hipótese nula de variâncias iguais. O Teste t será executado assumindo variâncias iguais.")
    else:
        st.warning("Como o valor p do Teste de Levene é menor que 0.05, rejeitamos a hipótese nula de variâncias iguais. O Teste t será executado utilizando a correção de Welch (variâncias desiguais).")

    st.markdown("### Realizando o Teste t Independente:")
    t_statistic, p_value_bilateral = ttest_ind(grupo_a, grupo_b, equal_var=equal_variances)

    mean_a = sum(grupo_a) / len(grupo_a)
    mean_b = sum(grupo_b) / len(grupo_b)

    # Para teste unilateral (H1: media_A > media_B)
    if mean_a > mean_b:
        p_value_unilateral = p_value_bilateral / 2
    else:
        p_value_unilateral = 1 - (p_value_bilateral / 2) # Não há suporte para H1 na direção desejada

    st.write(f"**Estatística t do Teste Independente:** `{t_statistic:.2f}`")
    st.write(f"**Valor p (unilateral) do Teste Independente:** `{p_value_unilateral:.3f}`")

    st.subheader("Interpretação dos Resultados do Teste t Independente:")
    if p_value_unilateral < 0.05 and mean_a > mean_b:
        st.success(f"Como o valor p unilateral ({p_value_unilateral:.3f}) é menor que o nível de significância (0.05) e a média do Grupo A ({mean_a:.2f}) é maior que a do Grupo B ({mean_b:.2f}), rejeitamos a Hipótese Nula.")
        st.write("Isso indica que o novo método de ensino (Grupo A) **aumentou as notas de forma estatisticamente significativa** em comparação ao método tradicional (Grupo B).")
    else:
        st.info(f"Como o valor p unilateral ({p_value_unilateral:.3f}) é maior ou igual ao nível de significância (0.05), ou a média do Grupo A não é maior que a do Grupo B, não rejeitamos a Hipótese Nula.")
        st.write("Isso sugere que não há evidência suficiente para afirmar que o novo método de ensino (Grupo A) aumenta as notas em comparação ao método tradicional (Grupo B).")
    st.markdown("---")

# Execução das funções dos exercícios
if __name__ == "__main__":
    run_exercise_1()
    run_exercise_2()
    run_exercise_3()

    st.sidebar.title("Sobre a Aplicação")
    st.sidebar.info(
        """
        Esta aplicação foi criada para demonstrar a aplicação de diferentes testes estatísticos
        (ANOVA, Teste t Pareado, Teste t Independente) utilizando dados de exemplos.

        Desenvolvido em Python com as bibliotecas:
        - Streamlit (para a interface web)
        - Pandas (para manipulação de dados)
        - Matplotlib e Seaborn (para visualização)
        - SciPy e Statsmodels (para os testes estatísticos)

        Espero que seja útil em seus estudos e projetos, Diógenes!
        """
    )