# planner_executor_agent.py
from strands import Agent, tool
from strands_tools import http_request
from mcp_tools import get_position, wikipedia_search, duckduckgo_search, tavily_search
from model_config import get_configured_model
from typing import Dict, Any
from pydantic import BaseModel, Field
from typing import List, Optional
import requests
from bs4 import BeautifulSoup
#from dotenv import load_dotenv
import asyncio

"""
Patron de diseño planeador-ejecutor. En este modelo, el planeador pone tareas que el ejecutor realiza. 
"""

@tool
def fetch_url(url: str, max_chars: int = 4000) -> str:
    """
    Descarga una página y retorna texto visible (recortado).
    """
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    return text[:max_chars]

EXECUTOR_INSTRUCTIONS = """
Eres un EXECUTOR. Tu trabajo es resolver subtareas CONCRETAS que te delega un Planner.
Sigue este patrón simple de verificación:
- Si necesitas fuentes, usa primero la herramienta de búsqueda web para localizar URLs confiables.
- Luego, usa fetch_url para extraer el contenido clave y verificar.
- Devuelve SIEMPRE una respuesta breve, precisa y con 1–3 URLs como evidencia.
No inventes datos. Si hay incertidumbre, dilo explícitamente.
Formato de salida recomendado (texto):
- Hallazgos clave en 3–6 viñetas.
- Fuentes: lista de URLs.
"""

@tool
def delegate_to_executor(subtask: str) -> str:
    """
    Ejecuta la subtarea con el EXECUTOR y devuelve su salida final.
    """
    try:
        agent = Agent(
            model=get_configured_model(),
            system_prompt=EXECUTOR_INSTRUCTIONS,
            tools=[tavily_search, fetch_url]
        )
        response = agent(f"Please process the following search request: {subtask}")
        return str(response)
    except Exception as e:
        return f"Search agent error: {str(e)}"

# ----------------------------
# MODELO PARA PARSEAR EL INFORME FINAL (opcional)
# ----------------------------
class ProgramItem(BaseModel):
    program_name: Optional[str] = Field(None, description="Nombre del programa")
    university: Optional[str] = Field(None, description="Universidad")
    country: Optional[str] = Field(None, description="País")
    url: Optional[str] = Field(None, description="URL oficial o principal")
    courses_examples: List[str] = Field(default_factory=list, description="Curso(s) representativos si están disponibles")
    tuition: Optional[str] = Field(None, description="Costo (monto+moneda+periodicidad) si está disponible")
    intake_per_year: Optional[str] = Field(None, description="Ingreso/aforo anual si está disponible")
    sources: List[str] = Field(default_factory=list)

class FinalReport(BaseModel):
    input_program: str
    input_description: str
    coverage: dict
    items: List[ProgramItem]
    insights: List[str]

# ----------------------------
# PLANNER AGENT
# ----------------------------
PLANNER_INSTRUCTIONS = """
Eres un PLANNER. Tu objetivo es:
1) Descomponer la solicitud del usuario en subtareas claras.
2) Delegar cada subtarea al EXECUTOR usando la herramienta delegate_to_executor.
3) Integrar la información en un informe final estructurado, con cobertura local, nacional e internacional.

Reglas:
- Empieza definiendo 4–8 subtareas que cubran: búsqueda local, nacional e internacional; syllabus/plan de estudios; costo (tuition/fees); ingreso/cupos (intake/enrollment); y tendencias del nombre del programa.
- Para cada subtarea, haz:
  Thought: explica por qué esa subtarea es necesaria.
  Action: delegate_to_executor{"subtask": "..."}
  Observation: captura el resumen devuelto por el EXECUTOR.
- Tras cubrir suficientes resultados (≥6 programas únicos o ≥2 por nivel geográfico), sintetiza.

Salida final:
Devuelve un JSON que cumpla EXACTAMENTE este esquema (usa lenguaje claro):
{
  "input_program": "...",
  "input_description": "...",
  "coverage": {"local": int, "national": int, "international": int},
  "items": [
    {
      "program_name": "...",
      "university": "...",
      "country": "...",
      "url": "...",
      "courses_examples": ["...", "..."],
      "tuition": "...",
      "intake_per_year": "...",
      "sources": ["...", "..."]
    }
  ],
  "insights": ["...", "...", "..."]
}

Notas:
- Incluye SIEMPRE "sources" por ítem (aunque sea 1 URL).
- Si un campo no aparece en la web, déjalo vacío o pon "No disponible".
- No incluyas el texto de Thought/Action/Observation en el JSON final.
- Mantén el informe conciso y verificable.
"""


if __name__=='__main__':
    user_program = "Doctorado en Ciencias Sociales"
    user_desc = "El Doctorado en Ciencias Sociales se orienta a la formación de investigadores con capacidad de interpretar las subjetividades, la cultura y la sociedad colombiana y latinoamericana de forma crítica, interdisciplinaria y compleja, construyendo conocimiento e investigando en el área de las ciencias sociales, con énfasis en las perspectivas latinoamericanas, planteando discusiones desde el ámbito académico que permitan establecer diálogos con saberes alternativos y que aporten a la transformación social y humana. "
    programas_equivalentes = 'Programa 1: Universidad: Universidad Nacional Colombia, Programa: Doctorado Ciencias Humanas Sociales, Ubicación o ciudad: Bogota, D.C.. Programa 2: Universidad: Universidad Nacional Colombia, Programa: Doctorado Ciencias Humanas Sociales, Ubicación o ciudad: Medellin. Programa 3: Universidad: Universidad Antioquia, Programa: Doctorado Ciencias Sociales, Ubicación o ciudad: Medellin. Programa 4: Universidad: Pontificia Universidad Javeriana, Programa: Doctorado Ciencias Sociales Humanas, Ubicación o ciudad: Bogota, D.C.. Programa 5: Universidad: Universidad Externado Colombia, Programa: Doctorado Estudios Sociales, Ubicación o ciudad: Bogota, D.C.. Programa 6: Universidad: Universidad Pontificia Bolivariana, Programa: Doctorado Ciencias Sociales, Ubicación o ciudad: Medellin. Programa 7: Universidad: Universidad Norte, Programa: Doctorado Ciencias Sociales, Ubicación o ciudad: Barranquilla. Programa 8: Universidad: Universidad Distrital-Francisco Jose Caldas, Programa: Doctorado Estudios Sociales, Ubicación o ciudad: Bogota, D.C.. Programa 9: Universidad: Fundacion Universidad Bogota Jorge Tadeo Lozano, Programa: Doctorado Estudios Sociales, Ubicación o ciudad: Bogota, D.C.. Programa 10: Universidad: Tecnologico Antioquia, Programa: Doctorado Educacion Estudios Sociales, Ubicación o ciudad: Medellin. Programa 11: Universidad: Colegio Mayor Señora Rosario, Programa: Doctorado Estudios Sociales, Ubicación o ciudad: Bogota, D.C.. Programa 12: Universidad: Universidad Industrial Santander, Programa: Doctorado Ciencias Sociales Humanas, Ubicación o ciudad: Bucaramanga. Programa 13: Universidad: Universidad Salle, Programa: Doctorado Estudios Sociales Religion, Ubicación o ciudad: Bogota, D.C.. Programa 14: Universidad: Universidad Sergio Arboleda, Programa: Doctorado Ciencias Sociales Humanas, Ubicación o ciudad: Bogota, D.C.'

    prompt = f"""
Quiero mapear programas similares a: "{user_program}".
Descripción corta: "{user_desc}".
Una lista de programas que considero equivalentes son: "{programas_equivalentes}"

Tareas:
- Encontrar programas locales (Colombia), nacionales e internacionales (EE.UU./Europa).
- Para cada programa: nombre, universidad, sitio web, cursos representativos, costo (si existe) y estudiantes que ingresan (si existe).
- Si no encuentras suficiente información para un país o región, amplía el rango de búsqueda.
- Al final, analiza si el nombre del programa aparece en búsquedas/tendencias (usa términos ES/EN si ayuda).

Devuélveme el JSON final con el esquema indicado.
"""
    try:
        planner = Agent(
            model= get_configured_model(),
            name="Planner",
            system_prompt=PLANNER_INSTRUCTIONS,
            tools=[delegate_to_executor],  # El Planner solo puede delegar (no busca directo)
        )

        response = planner(prompt)
        print(response)
    except Exception as e:
        print( f"Search agent error: {str(e)}")