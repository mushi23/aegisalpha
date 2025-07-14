import argparse
import yaml
import sys

def update_task_status(yaml_path, section_name, task_name, new_status):
    with open(yaml_path, 'r') as f:
        roadmap = yaml.safe_load(f)
    found = False
    for section in roadmap['sections']:
        if section['name'].lower() == section_name.lower():
            for task in section['tasks']:
                if task['name'].lower() == task_name.lower():
                    task['status'] = new_status
                    found = True
                    break
    if not found:
        print(f"❌ Task '{task_name}' in section '{section_name}' not found.")
        sys.exit(1)
    with open(yaml_path, 'w') as f:
        yaml.dump(roadmap, f, sort_keys=False)
    print(f"✅ Updated '{task_name}' in section '{section_name}' to status '{new_status}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', required=True, help='Path to roadmap YAML file')
    parser.add_argument('--section', required=True, help='Section name (e.g., Meta Modeling)')
    parser.add_argument('--task', required=True, help='Task name (e.g., LightGBM alternative)')
    parser.add_argument('--status', required=True, help='New status (e.g., done, partial, not_started)')
    args = parser.parse_args()
    update_task_status(args.yaml, args.section, args.task, args.status) 