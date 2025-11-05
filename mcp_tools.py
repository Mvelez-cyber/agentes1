import httpx
import wikipedia
import asyncio
import json
from typing import Dict, Any
from strands import tool
import requests 
import os
from dotenv import load_dotenv

# Carga automáticamente las variables desde el archivo .env
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

@tool
def tavily_search(query: str, search_depth: str = "basic") -> str:
    """
    Usa la API de Tavily para hacer una búsqueda web contextual.
    search_depth puede ser 'basic' o 'advanced'.
    """
    url = "https://api.tavily.com/search"
    payload = {
        "query": query,
        "search_depth": search_depth,
        "include_answer": True,
        "max_results": 5
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TAVILY_API_KEY}"
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    data = response.json()
    answer = data.get("answer", "")
    results = data.get("results", [])

    summary = f"**Respuesta Tavily:** {answer}\n\n"
    for r in results:
        summary += f"- [{r['title']}]({r['url']})\n"
    return summary

@tool
def wikipedia_search(query: str) -> Dict[str, Any]:
    """Search Wikipedia for information
    
    Args:
        query: Search query
        
    Returns:
        Dictionary containing search results
    """
    try:
        # Korean first, fallback to English if failed
        try:
            wikipedia.set_lang("es")
            page = wikipedia.page(query)
        except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
            wikipedia.set_lang("en")
            page = wikipedia.page(query)
        
        # Limit summary text (500 characters)
        summary = page.summary
        if len(summary) > 500:
            summary = summary[:500] + "..."
        
        return {
            "success": True,
            "title": page.title,
            "summary": summary,
            "url": page.url
        }
        
    except wikipedia.exceptions.DisambiguationError as e:
        return {
            "success": False,
            "error": "Multiple results found",
            "options": e.options[:5]  # Top 5 only
        }
         
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@tool
def duckduckgo_search(query: str) -> Dict[str, Any]:
    """Search DuckDuckGo for information
    
    Args:
        query: Search query
        
    Returns:
        Dictionary containing search results
    """
    try:
        async def fetch_search_results():
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.duckduckgo.com/",
                    params={
                        "q": query,
                        "format": "json",
                        "no_html": "1",
                        "skip_disambig": "1"
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # If Abstract information exists
                    if data.get("Abstract"):
                        return {
                            "success": True,
                            "title": data.get("Heading", query),
                            "summary": data["Abstract"],
                            "url": data.get("AbstractURL", "")
                        }
                    
                    # If Definition information exists
                    elif data.get("Definition"):
                        return {
                            "success": True,
                            "title": query,
                            "summary": data["Definition"],
                            "url": data.get("DefinitionURL", "")
                        }
                    
                    # If related topics exist
                    elif data.get("RelatedTopics"):
                        topics = data["RelatedTopics"][:3]
                        summaries = []
                        for topic in topics:
                            if isinstance(topic, dict) and topic.get("Text"):
                                summaries.append(topic["Text"])
                        
                        if summaries:
                            return {
                                "success": True,
                                "title": query,
                                "summary": " | ".join(summaries)
                            }
                
                return {"success": False, "error": "No results found"}
        
        return asyncio.run(fetch_search_results())
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@tool
def get_position(location: str) -> Dict[str, Any]:
    """Get latitude and longitude coordinates for a given location name
    
    Args:
        location: The name of the location to get coordinates for
        
    Returns:
        Dictionary containing coordinates and location information
    """
    try:
        # Using OpenStreetMap Nominatim API for geocoding
        async def fetch_coordinates():
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={
                        "q": location,
                        "format": "json",
                        "limit": 1
                    },
                    headers={
                        "User-Agent": "StrandsAgents/1.0",
                        "Accept": "application/json",
                        "Accept-Charset": "utf-8"
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200: 
                    data = response.json()  
                    if data:
                        result = data[0]
                        return {
                            "success": True,
                            "latitude": float(result["lat"]),
                            "longitude": float(result["lon"]),
                            "display_name": result.get("display_name", location)
                        }
                
                return {"success": False, "error": "Location not found"}
        
        # Execute async function
        return asyncio.run(fetch_coordinates())
        
    except Exception as e:
        return {"success": False, "error": str(e)}
 
 

# Test code
if __name__ == "__main__":
    print("🧪 MCP Tools Test")
    print("-" * 50)
    
    # TODO: Write test code for each function in Lab 2
    print("💡 Consulta en duckduck go")
    result=duckduckgo_search('Artificial Intelligence')
    print(f"Success: {result.get('success', False)}")
    print(f"Title: {result.get('title', False)}")
    print(f"Summary: {result.get('summary', False)}")
    print(f"url: {result.get('url', False)}")
    
    # Example:
    print("\n📚 Wikipedia Search Test:")
    result = wikipedia_search("Inteligencia artificial")
    print(f"Success: {result.get('success', False)}")
    print(f"Title: {result.get('title', False)}")
    print(f"Summary: {result.get('summary', False)}")
    print(f"url: {result.get('url', False)}")
    
    print("\n📚 Position get test")
    result=get_position('Guatapé, Antioquia')
    print(result)

    print('\nConsulta de Tavily')
    result=tavily_search('Universidad Pontificia Bolivariana')
    print(result)
