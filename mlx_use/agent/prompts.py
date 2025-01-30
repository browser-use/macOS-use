from datetime import datetime
from typing import List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from mlx_use.agent.views import ActionResult, AgentStepInfo

class SystemPrompt:
    def __init__(self, action_description: str, current_date: datetime, max_actions_per_step: int = 10):
        self.default_action_description = action_description
        self.current_date = current_date
        self.max_actions_per_step = max_actions_per_step

    def important_rules(self) -> str:
        return f"""
1. RESPONSE FORMAT: You must ALWAYS respond with valid JSON in this exact format:
{{
  "actions": [
    {{"open_app": {{"app_name": "calculator"}}}},
    {{"click_element": {{"index": 0}}}},
    {{"input_text": {{"index": 1, "text": "Hello", "submit": true}}}}

  ]
}}

2. ACTION SEQUENCING:
- First ALWAYS open required app using open_app
- Then perform UI interactions
- Use maximum {self.max_actions_per_step} actions per sequence
- Actions are executed in the order they appear in the list

3. APP HANDLING:
- App names must be lowercase (e.g. 'calculator' not 'Calculator')
- Never assume apps are already open
- If app fails to open, retry open_app action

4. ELEMENT INTERACTION:
- Only use indexes that exist in the provided element list
- Each element has a unique index number (e.g. "0: Button: Submit")
- Elements are refreshed after each action execution
- Use input_text with submit=True for text fields needing Enter submission (e.g. "input_text": "index": 1, "text": "Hello", "submit": true)


5. ERROR RECOVERY:
- If "No PID" error occurs: Retry open_app action
- If element not found: Verify app is open and element index exists
- If multiple failures occur: Start over with open_app
- If text input fails: Verify element is a text field
- If submit fails: Try click_element on submit button instead

6. TASK COMPLETION:
- Use the 'done' action when task is complete
- Don't hallucinate actions
- Include all task results in the 'done' action text
- If stuck after 3 attempts, use 'done' with error details
"""

    def input_format(self) -> str:
        return """
INPUT STRUCTURE:
1. Current App: Active macOS application (or "None" if none open)
2. UI Elements: List in format:
   [index] ElementType: Description
   Example:
   [0] Button: Close
   [1] TextField: Search (submit)
3. Previous Results: Outcomes of last executed actions
"""

    def get_system_message(self) -> SystemMessage:
        time_str = self.current_date.strftime('%Y-%m-%d %H:%M')
        return SystemMessage(content=f"""You are a macOS automation agent. Current time: {time_str}

{self.input_format()}

{self.important_rules()}

AVAILABLE ACTIONS:
{self.default_action_description}

Respond ONLY with valid JSON matching the specified format!""")

class AgentMessagePrompt:
    def __init__(
        self,
        state: str,
        result: Optional[List[ActionResult]] = None,
        include_attributes: list[str] = [],
        max_error_length: int = 400,
        step_info: Optional[AgentStepInfo] = None,
    ):
        self.state = state
        self.result = result
        self.max_error_length = max_error_length
        self.include_attributes = include_attributes
        self.step_info = step_info

    def get_user_message(self) -> HumanMessage:
        step_info_str = f"Step {self.step_info.step_number + 1}/{self.step_info.max_steps}\n" if self.step_info else ""
        
        state_description = f"""{step_info_str}
CURRENT APPLICATION STATE:
{self.state}
"""

        if self.result:
            for i, result in enumerate(self.result):
                if result.extracted_content:
                    state_description += f"\nACTION RESULT {i+1}: {result.extracted_content}"
                if result.error:
                    error = result.error[-self.max_error_length :]
                    state_description += f"\nACTION ERROR {i+1}: ...{error}"

        return HumanMessage(content=state_description)