import os
import json
from typing import List, Dict, Union
from openai import OpenAI
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)


# Defining the Worker class that will take care of the role of the worker
class Worker:
    def __init__(self, role: str, responsibilities: List[str]):
        self.role = role
        self.responsibilities = responsibilities
        self.knowledge = []


    def perform_task(self, task: str) -> str:
        prompt = f"You are a {self.role}. Your responsibilities are {', '.join(self.responsibilities)}. "
        prompt += f"Your current knowledge: {' '.join(self.knowledge)}. "
        prompt += f"Your task: {task}. Provide your response."


        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        return response.choices[0].message.content


    def add_knowledge(self, info: str):
        self.knowledge.append(info)


class AgentSystem:
    def __init__(self):
        self.workers: Dict[str, Worker] = {}


    def analyze_task(self, task: str) -> List[Dict[str, Union[str, List[str]]]]:
        prompt = f"Analyze the following task and suggest necessary roles to complete it, along with their responsibilities: {task}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        content = response.choices[0].message.content.strip()
       
        try:
            # Try to parse as JSON
            parsed_content = json.loads(content)
            if isinstance(parsed_content, list):
                return parsed_content
            elif isinstance(parsed_content, dict):
                return [parsed_content]
        except json.JSONDecodeError:
            # If not JSON, create a default structure
            pass
       
        # If parsing failed or content is a string, create a default structure
        return [{"role": "General Assistant", "responsibilities": [content if content else "Complete the given task"]}]


    def create_workers(self, roles: List[Dict[str, Union[str, List[str]]]]):
        for role in roles:
            if isinstance(role, dict):
                role_name = role.get('role', 'Assistant')
                responsibilities = role.get('responsibilities', ['Complete tasks as assigned'])
                if isinstance(responsibilities, str):
                    responsibilities = [responsibilities]
            else:
                role_name = 'Assistant'
                responsibilities = ['Complete tasks as assigned']
           
            self.workers[role_name] = Worker(role_name, responsibilities)


    def execute_task(self, task: str) -> str:
        roles = self.analyze_task(task)
        self.create_workers(roles)


        subtasks = self.break_down_task(task)
        results = []


        for subtask in subtasks:
            assigned_role = self.assign_subtask(subtask)
            if assigned_role not in self.workers:
                assigned_role = list(self.workers.keys())[0]  # Assign to the first worker by default if role not found to avoid any error, we can pick other too
            result = self.workers[assigned_role].perform_task(subtask)
            results.append(result)
            self.share_knowledge(result)


        return self.aggregate_results(results)


    def break_down_task(self, task: str) -> List[str]:
        prompt = f"Break down the following task into subtasks. Respond with a Python list of strings: {task}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        content = response.choices[0].message.content.strip()
       
        try:
            # Try to parse as Python list
            return eval(content)
        except:
            # If parsing fails, return the task as a single subtask
            return [task]


    def assign_subtask(self, subtask: str) -> str:
        if not self.workers:
            return "General Assistant"
       
        prompt = f"Assign the following subtask to the most appropriate role: {subtask}. Available roles: {', '.join(self.workers.keys())}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        assigned_role = response.choices[0].message.content.strip()
       
        # If the assigned role doesn't exist, return the first available role
        return assigned_role if assigned_role in self.workers else list(self.workers.keys())[0]


    def share_knowledge(self, info: str):
        for worker in self.workers.values():
            worker.add_knowledge(info)


    def aggregate_results(self, results: List[str]) -> str:
        prompt = f"Aggregate and synthesize the following results into a coherent output: {' '.join(results)}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        return response.choices[0].message.content


def main():
    system = AgentSystem()
    task = input("Enter the task you want to complete: ")
    result = system.execute_task(task)
    print(f"Final output: {result}")


if __name__ == "__main__":
    main()
