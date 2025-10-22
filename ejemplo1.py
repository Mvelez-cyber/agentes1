# -*- coding: utf-8 -*-
import asyncio, threading
from agents import Agent, Runner, ModelSettings, WebSearchTool, function_tool
from dotenv import load_dotenv
from typing import List, Tuple, Any
from pydantic import BaseModel, Field, ValidationError
load_dotenv() #Carga de la clave de acceso de OpenAI
import json, re
class JsonExtractionError(Exception):
    pass

def extract_json_obj(text: str) -> Tuple[Any, str]:
    """
    Extrae SOLO el objeto JSON de una respuesta que puede incluir un bloque
    ```json ... ``` y/o texto adicional. Devuelve:
      - obj: el objeto Python parseado (dict/list)
      - raw: el string JSON exacto extraído

    Estrategia:
      1) Si hay bloque ```json ... ```, usa su contenido.
      2) Si no, busca el PRIMER objeto JSON balanceado recorriendo { ... }.
      3) Intenta json.loads; si falla, lanza JsonExtractionError con pista.
    """
    if not isinstance(text, str):
        raise TypeError("text debe ser str")

    # 1) Intentar con bloque ```json ... ```
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = None
    if fence:
        candidate = fence.group(1).strip()

    # 2) Si no hay bloque, buscar primer objeto JSON balanceado
    if candidate is None:
        # Buscar primer '{'
        start = text.find("{")
        if start == -1:
            raise JsonExtractionError("No se encontró ninguna llave '{' en el texto.")
        # Recorrer contando llaves y respetando strings/escapes
        i = start
        depth = 0
        in_string = False
        escape = False
        end = None
        while i < len(text):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            i += 1

        if end is None:
            raise JsonExtractionError("No se encontró un objeto JSON balanceado (faltan llaves de cierre).")
        candidate = text[start:end+1].strip()

    # 3) Parsear JSON estrictamente
    try:
        obj = json.loads(candidate)
        return obj, candidate
    except json.JSONDecodeError as e:
        # Pista útil para depurar
        context = candidate[max(0, e.pos-60): e.pos+60]
        msg = (
            f"Error al parsear JSON: {e}\n"
            f"Contexto cercano a la posición {e.pos}:\n---\n{context}\n---"
        )
        raise JsonExtractionError(msg)



class Respuesta_marcas(BaseModel):
    anio_separacion: str = Field(..., description="Año de la separación")
    motivo: str = Field(..., description="Motivo factual de la separación")
    marca_original: str = Field(..., description="Nombre de la marca original antes de separarse")
    marcas_resultantes: List[str] = Field(default_factory=list, description="Marcas tras la separación")

# Función para validar la respuesta y determinar si está completa o no
#@function_tool
def validar(info: dict) -> bool:
   #print(type(info))
   #print(info.keys())
   try:
    flag_anio=len(info['anio_separacion'])>0 
    print('Año: ',flag_anio)
    flag_motivo=len(info['motivo'])>0
    print('Motivo: ',flag_motivo)
    flag_marca=len(info['marca_original'])>0
    print('Marca Original: ',flag_marca)
    flag_marcas=len(info['marcas_resultantes'])>0 
    print('Marcas resultantes:',flag_marcas)
    flag=flag_anio and flag_motivo and flag_marca and flag_marcas
    print('Se procede a validar las marcas nuevas y los dueños')
    for d2 in info['marcas_resultantes']:
       flag=flag and len(d2['marca'])>0 and len(d2['dueños'])>0 
    if(not(flag)): 
        print('En la validación, falta alguno de los datos')
    return flag
   except Exception as e:
       print('Ocurrió una excepción en la validación: ',e)
       return False
# Creamos un agente básico que trata de responder la pregunta. 
agente1 = Agent(
   name="Buscador de noticias",
   instructions="""Eres un agente conocedor de información de mercado, que puede 
   extraer y presentar información acerca de la separación de una marca tradicional en antioquia
   que se separó en dos marcas nuevas. 
   Tu respuesta debe estar en formato JSON, incluyendo los siguientes campos: 
   anio_separacion, que indica el año en que ocurrió la separación
   motivo, que describe la razón por la cuál la dos marcas se separaron
   marca_original, corresponde a la marca que compartían las empresas inicialmente y
   marcas_resultantes, que indica las marcas nuevas y los dueños dueños que quedaron en cada una de las marcas nuevas. 
   Esto debe ser una lista de diccionarios con campos: marca, dueños por cada marca nueva
   Presenta respuestas factuales, sin inventar información que no se conoce. 
   Si no conoces la información, deja el campo vacío en el json de respuesta. 
   """,
   tools=[],
   model="gpt-4.1-nano",
   model_settings=ModelSettings(
       temperature=0.0,  # Lower for more deterministic outputs (0.0-2.0)
       #max_tokens=1024,  # Maximum length of response
   ),
)



# Creamos un agente básico que trata de responder la pregunta. 
agente2 = Agent(
   name="Buscador de noticias",
   instructions="""Eres un agente conocedor de información de mercado, que puede 
   extraer y presentar información acerca de la separación de una marca tradicional en antioquia
   que se separó en dos marcas nuevas. 
   Tu respuesta debe estar en formato JSON, incluyendo los siguientes campos: 
   anio_separacion, que indica el año en que ocurrió la separación
   motivo, que describe la razón por la cuál la dos marcas se separaron
   marca_original, corresponde a la marca que compartían las empresas inicialmente y
   marcas_resultantes, que indica las marcas nuevas y los dueños dueños que quedaron en cada una de las marcas nuevas. 
   Esto debe ser una lista de diccionarios con campos: marca, dueños por cada marca nueva
   Presenta respuestas factuales, sin inventar información que no se conoce. 
   Si no conoces la información, deja el campo vacío en el json de respuesta. 
<<<<<<< Updated upstream
   Todos los campos del objeto json deben ser cadena de texto aunque sean numéricos
=======
   Todos los campos del objeto JSON deben ser cadena de texto aunque sean numericos.
>>>>>>> Stashed changes
   """,
   tools=[WebSearchTool(search_context_size='low')],
   model="gpt-4.1",
   model_settings=ModelSettings(
       temperature=0.0,  # Lower for more deterministic outputs (0.0-2.0)
       #max_tokens=1024,  # Maximum length of response
   ),
)

# Creamos un agente básico que trata de responder la pregunta. 
agente3 = Agent(
   name="Generador de reporte",
   instructions="""Eres un agente conocedor de información de mercado, que puede 
   extraer y presentar información acerca de la separación de una marca tradicional en antioquia
   que se separó en dos marcas nuevas. 
   Vas a recibir una información respecto a la separación de una marca antioqueña. Complementa lo necesario con una busqueda web 
   y genera un reporte tipo noticia, que especifique la información que se te pasa y otra que pueda complementar
   """,
   tools=[WebSearchTool(search_context_size='low')],
   model="gpt-4.1",
   model_settings=ModelSettings(
       temperature=0.8,  # Lower for more deterministic outputs (0.0-2.0)
       #max_tokens=1024,  # Maximum length of response
   ),
)
async def main():
    try:
        result = await asyncio.wait_for(
            Runner.run(agente1, "Dime por qué se separó el supermercado la vaquita en la vaquita y supermu"),
            timeout=30)
        #print("Resultado de la primera consulta: ",result.final_output)
        data_json, tmp=extract_json_obj(result.final_output)
        #print(data_json.keys())
        validacion=validar(data_json)
        print('Validación primera respuesta: ', validacion)
        if(not(validacion)):
            print("No fue una respuesta completa, se pasa a un segundo agente")
            result2 = await asyncio.wait_for(
                Runner.run(agente2, "Dime por qué se separó el supermercado la vaquita en la vaquita y supermu"),
                timeout=30)
            #print("Resultado de la segunda consulta: ",result2.final_output)
            json_data, candidate=extract_json_obj(result2.final_output)
            print(json_data)
            validacion=validar(json_data)
            print('Validación segunda respuesta: ', validacion)

            if(validacion):
                result3 = await asyncio.wait_for(
                    Runner.run(agente3, json.dumps(json_data, ensure_ascii=False, indent=2)),
                    timeout=30)
                print('-------------------------------')
                print(result3.final_output)
    except Exception as e:
        print('Ocurrió un error :',e)
    
    # Function calls itself,
    # Looping in smaller pieces,
    # Endless by design.

asyncio.run(main())

