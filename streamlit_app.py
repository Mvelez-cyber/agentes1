import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from pathlib import Path
import json
import asyncio

st.set_page_config(page_title="Análisis de Oportunidad Académica", layout="wide")

# -------------------------------------------------------------
# Funciones de apoyo
# -------------------------------------------------------------

def cargar_datos():
    """Carga los datasets de SNIES desde la fuente pública."""
    maestro = pd.read_parquet('https://robertohincapie.com/data/snies/MAESTRO.parquet')
    oferta = pd.read_parquet('https://robertohincapie.com/data/snies/OFERTA.parquet')
    programas = pd.read_parquet('https://robertohincapie.com/data/snies/PROGRAMAS.parquet')
    ies = pd.read_parquet('https://robertohincapie.com/data/snies/IES.parquet')
    return maestro, oferta, programas, ies


def analizar_programa(programa_nombre: str):
    """Ejecuta el flujo de análisis SNIES para el programa indicado."""
    maestro, oferta, programas, ies = cargar_datos()
    requerido = set(programa_nombre.lower().split())
    programa = set(programa_nombre.lower().split())
    n = len(programa)

    equivalentes = []
    for prg in programas['PROGRAMA_ACADEMICO'].unique():
        prg2 = str(prg).lower().split()
        indice = len(set(programa).intersection(prg2)) / len(programa)
        if indice >= (n - 1) / n and len(requerido.intersection(prg2)) == len(requerido):
            equivalentes.append(prg)

    programas2 = programas[programas['PROGRAMA_ACADEMICO'].isin(equivalentes)]
    snies2 = list(programas2['CODIGO_SNIES'].unique())
    maestro2 = maestro[maestro['CODIGO_SNIES'].isin(snies2)]

    maestro3 = maestro2.merge(programas, on='CODIGO_SNIES', how='left')
    maestro4 = maestro3.merge(oferta, on=['CODIGO_SNIES', 'PERIODO'], how='left')

    return maestro4, programas2


def generar_graficas(maestro4: pd.DataFrame, programa_nombre: str):
    """Genera las principales gráficas de análisis de oportunidad."""
    figuras = []

    # 1. Número de programas e instituciones en el tiempo
    NprogNies = maestro4.groupby(by='PERIODO').agg({'CODIGO_INSTITUCION_x':'nunique', 'CODIGO_SNIES':'nunique'})
    fig1, ax1 = plt.subplots()
    NprogNies.plot(ax=ax1)
    ax1.set_title(f"Programas e Instituciones en el tiempo - {programa_nombre}")
    ax1.set_xlabel('Periodo')
    ax1.set_ylabel('Cantidad')
    ax1.grid(True)
    figuras.append(fig1)

    # 2. Costo del programa vs promedio de matriculados
    maestro4['PROXY_PER'] = maestro4['PROXY_PER'].astype(int)
    df = maestro4[(maestro4['PROXY_PER']>=20211) & (maestro4['PROXY_PER']<=20242)].copy()
    df['Nombre_ies'] = df['INSTITUCION']+' - '+df['PROGRAMA_ACADEMICO']
    df = df[df['PROCESO']=='MATRICULADOS'].copy()
    df['CANTIDAD'] = df['CANTIDAD'].astype(int)
    df = df[['MATRICULA','CANTIDAD','Nombre_ies','PERIODO']].dropna()
    df = df[df['MATRICULA']!='null'].copy()
    df['MATRICULA'] = df['MATRICULA'].astype(float)
    df2 = df.groupby(by='Nombre_ies').agg({'MATRICULA':'last', 'CANTIDAD':'mean'})

    fig2, ax2 = plt.subplots()
    ax2.scatter(df2['CANTIDAD'], df2['MATRICULA'])
    for i, txt in enumerate(df2.index):
        ax2.text(df2['CANTIDAD'].iloc[i], df2['MATRICULA'].iloc[i], str(txt), fontsize=8, ha='center')
    ax2.set_xlabel('Promedio de matriculados')
    ax2.set_ylabel('Valor último de matrícula')
    ax2.set_title('Costo vs Matrícula promedio')
    ax2.grid(True)
    figuras.append(fig2)

    # 3. Valor de matrículas en el tiempo
    valor = pd.pivot_table(df, index='Nombre_ies', columns='PERIODO', values='MATRICULA', aggfunc='mean', fill_value=0)
    fig3, ax3 = plt.subplots()
    valor.T.plot(ax=ax3)
    ax3.set_title('Valor de matrícula en el tiempo')
    ax3.set_ylabel('Valor ($)')
    ax3.grid(True)
    figuras.append(fig3)

    # 4. Programas por departamento y ciudad
    df_geo = maestro4[(maestro4['PROXY_PER']>=20211) & (maestro4['PROXY_PER']<=20242)].copy()
    df_geo['Nombre_ies'] = df_geo['INSTITUCION']+' - '+df_geo['PROGRAMA_ACADEMICO']
    df_geo = df_geo[df_geo['PROCESO']=='MATRICULADOS'].copy()
    df_geo['CANTIDAD'] = df_geo['CANTIDAD'].astype(int)
    porDpto = df_geo.groupby('DEPARTAMENTO_PROGRAMA').agg({'CODIGO_SNIES':'nunique'}).sort_values(by='CODIGO_SNIES', ascending=False)
    porMpio = df_geo.groupby('MUNICIPIO_PROGRAMA').agg({'CODIGO_SNIES':'nunique'}).sort_values(by='CODIGO_SNIES', ascending=False)

    fig4, ax4 = plt.subplots(1,2, figsize=(10,4))
    porDpto.plot.bar(ax=ax4[0], legend=False, title='Programas por departamento')
    porMpio.plot.bar(ax=ax4[1], legend=False, title='Programas por municipio')
    figuras.append(fig4)

    # 5. Número de estudiantes en el tiempo
    maestro4 = maestro4[maestro4['CANTIDAD']!='null']
    maestro4['CANTIDAD'] = maestro4['CANTIDAD'].astype(int)
    num = pd.pivot_table(maestro4, index='PERIODO', columns='PROCESO', values='CANTIDAD', fill_value=0, aggfunc='sum')
    fig5, axes = plt.subplots(len(num.columns), 1, sharex=True, figsize=(8, 8))
    for i,col in enumerate(num.columns):
        axes[i].plot(num[col])
        axes[i].set_title(col)
        axes[i].grid(True)
    plt.tight_layout()
    figuras.append(fig5)

    return figuras


# -------------------------------------------------------------
# Interfaz Streamlit
# -------------------------------------------------------------

st.title("📊 Análisis de Oportunidad de Programas Académicos (SNIES + Agentes)")
st.markdown("Este sistema integra el análisis de oferta académica con agentes inteligentes para la Universidad.")

programa = st.text_input("Nombre del programa a analizar", value="Doctorado Ciencias Sociales")
ejecutar = st.button("Ejecutar análisis completo")

if ejecutar:
    with st.spinner('Procesando información de SNIES...'):
        maestro4, programas2 = analizar_programa(programa)
        figuras = generar_graficas(maestro4, programa)

    st.success("Análisis SNIES completado. Resultados:")
    st.write(f"**Programas equivalentes encontrados:** {len(programas2)}")

    for fig in figuras:
        st.pyplot(fig)

    # Crear resumen
    resumen = f"Se encontraron {len(programas2)} programas equivalentes al término '{programa}'. "\
              f"El análisis muestra variación en matrícula y costo, con presencia en {maestro4['DEPARTAMENTO_PROGRAMA'].nunique()} departamentos."
    st.info(resumen)

    # Integración con agentes (modo demostrativo)
    st.markdown("### 🔎 Análisis semántico con Agente")
    st.write("Aquí se podría integrar un agente tipo Planner o ReAct para buscar tendencias nacionales e internacionales.")

    ejemplo_prompt = f"Analiza las tendencias del programa '{programa}' a nivel global y nacional. Identifica palabras clave, instituciones y enfoques emergentes."
    st.code(ejemplo_prompt, language='markdown')

    # Exportar PowerPoint
    from pptx import Presentation
    from pptx.util import Inches

    ppt_button = st.button("Generar reporte PowerPoint")
    if ppt_button:
        prs = Presentation()
        slide_title = prs.slides.add_slide(prs.slide_layouts[0])
        slide_title.shapes.title.text = f"Análisis de Oportunidad - {programa}"
        slide_title.placeholders[1].text = "Reporte generado con datos SNIES y agentes."

        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = "Resumen del Análisis"
        slide.placeholders[1].text = resumen

        for i, fig in enumerate(figuras, start=1):
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            slide = prs.slides.add_slide(prs.slide_layouts[6])
            left, top = Inches(1), Inches(1)
            slide.shapes.add_picture(BytesIO(buf.getvalue()), left, top, height=Inches(5))
            buf.close()

        output_path = Path('/mnt/data') / f"reporte_{programa.replace(' ','_')}.pptx"
        prs.save(str(output_path))
        st.success(f"Reporte generado: {output_path}")
        st.markdown(f"[Descargar reporte PowerPoint]({output_path})")

st.caption("Desarrollado como integración del lector SNIES con un proyecto de agentes para análisis de oportunidad académica.")
