# planner_executor_agent.py
from strands import Agent, tool
from strands_tools import http_request
from model_config import get_configured_model

if __name__=='__main__':
    try:
        planner = Agent(
            model= get_configured_model(),
            name="Planner",
            system_prompt="Responde de manera concisa la pregunta que se te hace sin inventar información",
        )

        response = planner("Qué conoces sobre la UPB Medellín?")
        print(response)
    except Exception as e:
        print( f"Search agent error: {str(e)}")