from langsmith import Client
import json
import random
import re
import sys

project_dirs = [
    ('MacOS', '/Users/john/Workspace/MyReactApp'),
    ('MacOS', '/Users/jane/DevProjects/NewReactApp/Subfolder'),
    ('MacOS', '/Users/mark/Codebase/ClientProject'),
    ('MacOS', '/Users/emily/GitRepos/UIProject'),
    ('MacOS', '/Users/sarah/Development/ProjectX/Assets'),
    ('MacOS', '/Users/andy/Code/ReactRedux/Utilities'),
    ('MacOS', '/Users/alice/Projects/MyUI/Sandbox'),
    ('MacOS', '/Users/bob/Programming/Experimental/React'),
    ('MacOS', '/Users/carol/Dev/QuickStart'),
    ('MacOS', '/Users/dave/Workspace/ProjectAlpha'),
    ('MacOS', '/Users/eva/DevProjects/FinanceApp'),
    ('MacOS', '/Users/frank/Codebase/WidgetProject'),
    ('MacOS', '/Users/george/GitRepos/ReactDashboard'),
    ('MacOS', '/Users/helen/Development/TaskManager'),
    ('MacOS', '/Users/ian/Code/PersonalProject'),
    ('MacOS', '/Users/jack/Projects/ReactAppV2'),
    ('MacOS', '/Users/kate/Programming/AppRefresh'),
    ('MacOS', '/Users/leo/Dev/Phase2'),
    ('MacOS', '/Users/mona/Workspace/MyUI'),
    ('MacOS', '/Users/nina/DevProjects/Experimental'),
    ('MacOS', '/Users/oscar/Codebase/FinanceAppV2'),
    ('MacOS', '/Users/paul/GitRepos/SideProject'),
    ('MacOS', '/Users/quinn/Development/QuickStartV2'),
    ('MacOS', '/Users/ryan/Code/ClientWork'),
    ('MacOS', '/Users/stella/Projects/Sandbox'),
    ('MacOS', '/Users/tina/Programming/Scripts'),
    ('MacOS', '/Users/ulysses/Dev/Utilities'),
    ('MacOS', '/Users/victor/Workspace/Assets'),
    ('MacOS', '/Users/wendy/DevProjects/MyReactAppV2'),
    ('MacOS', '/Users/xander/Codebase/Subfolder'),
    ('MacOS', '/Users/yara/GitRepos/ExperimentalV2'),
    ('MacOS', '/Users/zane/Development/ClientProjectV2'),
    ('MacOS', '/Users/abby/Code/ReactReduxV2'),
    ('Linux', '/home/user1/Workspace/ReactDashboard'),
    ('Linux', '/home/user2/Dev/WidgetProject/Subfolder'),
    ('Linux', '/home/user3/Projects/TaskManager'),
    ('Linux', '/home/user4/Codebase/NewUI'),
    ('Linux', '/home/user5/Code/ProjectAlpha/Scripts'),
    ('Linux', '/home/user6/Dev/ClientWork/React'),
    ('Linux', '/home/user7/Projects/QuickStart'),
    ('Linux', '/home/user8/Workspace/MyReactApp'),
    ('Linux', '/home/user9/Dev/Phase2'),
    ('Linux', '/home/user10/Codebase/FinanceApp'),
    ('Linux', '/home/user11/GitRepos/PersonalProject'),
    ('Linux', '/home/user12/Development/ProjectX'),
    ('Linux', '/home/user13/Code/Experimental'),
    ('Linux', '/home/user14/Projects/UIProject'),
    ('Linux', '/home/user15/Programming/ClientProject'),
    ('Linux', '/home/user16/Dev/Utilities'),
    ('Linux', '/home/user17/Workspace/Assets'),
    ('Linux', '/home/user18/DevProjects/Sandbox'),
    ('Linux', '/home/user19/Codebase/ReactAppV2'),
    ('Linux', '/home/user20/GitRepos/AppRefresh'),
    ('Linux', '/home/user21/Development/MyUI'),
    ('Linux', '/home/user22/Code/ExperimentalV2'),
    ('Linux', '/home/user23/Projects/QuickStartV2'),
    ('Linux', '/home/user24/Programming/Scripts'),
    ('Linux', '/home/user25/Dev/MyReactAppV2'),
    ('Linux', '/home/user26/Workspace/Subfolder'),
    ('Linux', '/home/user27/DevProjects/ClientProjectV2'),
    ('Linux', '/home/user28/Codebase/ReactReduxV2'),
    ('Linux', '/home/user29/GitRepos/FinanceAppV2'),
    ('Linux', '/home/user30/Development/SideProject'),
    ('Linux', '/home/user31/Code/ClientWork'),
    ('Linux', '/home/user32/Projects/NewUI'),
    ('Linux', '/home/user33/Programming/TaskManager'),
    ('Windows', 'C:\\Users\\Alice\\Workspace\\FinanceApp'),
    ('Windows', 'D:\\Code\\Bob\\PersonalProject\\React'),
    ('Windows', 'E:\\Projects\\UIRefresh'),
    ('Windows', 'F:\\Development\\AppV2\\Utils'),
    ('Windows', 'G:\\Code\\SideProject\\React'),
    ('Windows', 'H:\\Programming\\Experimental'),
    ('Windows', 'I:\\DevProjects\\Mark\\Phase2'),
    ('Windows', 'J:\\Workspace\\MyReactApp'),
    ('Windows', 'K:\\Dev\\QuickStart'),
    ('Windows', 'L:\\Codebase\\ProjectAlpha'),
    ('Windows', 'M:\\GitRepos\\FinanceApp'),
    ('Windows', 'N:\\Development\\WidgetProject'),
    ('Windows', 'O:\\Code\\ReactDashboard'),
    ('Windows', 'P:\\Projects\\TaskManager'),
    ('Windows', 'Q:\\Programming\\NewUI'),
    ('Windows', 'R:\\Dev\\ProjectX\\Assets'),
    ('Windows', 'S:\\Workspace\\ReactRedux\\Utilities'),
    ('Windows', 'T:\\DevProjects\\MyUI\\Sandbox'),
    ('Windows', 'U:\\Codebase\\Experimental\\React'),
    ('Windows', 'V:\\GitRepos\\QuickStart'),
    ('Windows', 'W:\\Development\\MyReactAppV2'),
    ('Windows', 'X:\\Code\\Subfolder'),
    ('Windows', 'Y:\\Projects\\ExperimentalV2'),
    ('Windows', 'Z:\\Programming\\ClientProjectV2')
]

def randomize_path_and_machine(text, machine_type, project_dir):
    # Replace machine type in text
    new_text = text.replace('MacOS machine', f"{machine_type} machine")
    
    # Function to replace paths
    def replace_path(match):
        extra_path = match.group(1) if match.group(1) else ""
        replaced_path = project_dir + extra_path
        if machine_type == 'Windows':
            return replaced_path.replace('/', '\\')
        else:
            return replaced_path
    
    # Replace all occurrences of the path with a new project_dir
    new_text = re.sub(r'/Users/rajiv/Downloads/projects(/[^ \n]*)?', replace_path, new_text)
    
    return new_text

client = Client()

runs = client.list_runs(project_name="tdd-gpt", run_type="llm", error=False, filter="lt(total_tokens, 4096)")

for run in runs:
  example = {"messages": []}
  messages = run.inputs['messages']
  machine_type, project_dir = random.choice(project_dirs)

  for i, msg in enumerate(messages):
    content = msg['kwargs']['content']

    if 'flask' in content:
      example = None
      break
    elif 'Summarize' in content:
      example = None
      break
    else:
        content = content.replace("As an  Full Stack", "As an experienced Full Stack")

        if "```json" not in content:
            content = content.replace('Response Format:\n', 'Response Format:\n```json\n')
            content = content.replace('\nCode Context:', '```\nCode Context:')

        content = randomize_path_and_machine(content, machine_type, project_dir)

        example["messages"].append({"role": "system" if i == 0 else "user", "content": content.strip()})

  if example and run.outputs and 'generations' in run.outputs and run.outputs['llm_output']['model_name'] == 'gpt-4-0613':
      output = run.outputs['generations'][0]['text']

      output = randomize_path_and_machine(output, machine_type, project_dir)

      try:
        example["messages"].append({"role": "assistant", "content": output.strip()})
        print(json.dumps(example))
      except:
        pass
