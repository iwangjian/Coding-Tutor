from flask import Flask, render_template, request, jsonify
import json
import os
import re

app = Flask(__name__)

# Set the data path
DATA_PATH = "data/example_eval_pairs.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    eval_pairs = json.load(f)

if not os.path.exists("results"):
    os.makedirs("results")
results_file = os.path.join("results", "eval_results.json")

# Add after loading eval_pairs
completed_results = {}
if os.path.exists(results_file):
    with open(results_file, "r", encoding="utf-8") as f:
        eval_results = json.load(f)
    for result in eval_results:
        completed_results[f"eval_idx_{result['eval_idx']}"] = result

def process_text(text):
    """Process text, wrap code blocks with <pre><code> tags"""
    # Find code blocks in text (parts that start and end with ```)
    # Support both 'python' and 'Python' code blocks
    code_pattern = r'```(?:python|Python)?\n(.*?)```'

    def replace_code(match):
        code = match.group(1).strip()
        return f'<pre><code class="python">{code}</code></pre>'
    
    # replace '\n\n' to '\n'
    text = re.sub(r'```\n\n', '```', text)
    text = re.sub(r'```\n', '```', text)
    text = re.sub(r'\n\n', '\n', text)
    # Replace all code blocks
    processed_text = re.sub(code_pattern, replace_code, text, flags=re.DOTALL)
    return processed_text

@app.route("/")
def index():
    current_index = request.args.get("index", 1, type=int)
    total_pairs = len(eval_pairs)
    current_pair = eval_pairs[current_index - 1]
    
    # Get completed indices
    completed_indices = [i+1 for i, pair in enumerate(eval_pairs) 
                        if f"eval_idx_{pair['eval_idx']}" in completed_results]
    
    left_dialogue = []
    right_dialogue = []
    
    # Process model_1's conversation
    for turn in current_pair["model_1"]["conversation"]:
        for role, text in turn.items():
            left_dialogue.append({
                "speaker": role.capitalize(),
                "text": process_text(text)
            })
            
    # Process model_2's conversation
    for turn in current_pair["model_2"]["conversation"]:
        for role, text in turn.items():
            right_dialogue.append({
                "speaker": role.capitalize(),
                "text": process_text(text)
            })
    
    dialogues = [{
        "left": left_dialogue,
        "right": right_dialogue
    }]
    
    task_name = current_pair["namespace"]
    task_desc = current_pair["task"]
    # Automatically remove extra spaces and indentations from task_desc
    task_desc = re.sub(r'\n\s*', '\n', task_desc)
    task_desc = process_text(f"```python\n{task_desc}\n```")
    
    dependency_all = current_pair["dependency_all"]
    reference_steps = current_pair["reference_steps"]
    student_level = current_pair["student_level"]
    student_prior = current_pair["student_prior"]
    student_dependency = current_pair["student_dependency"]
    
    return render_template("index.html", 
                         dialogues=dialogues,
                         current_index=current_index,
                         total_pairs=total_pairs,
                         task_name=task_name,
                         task_desc=task_desc,
                         dependency_all=dependency_all,
                         reference_steps=reference_steps,
                         student_level=student_level,
                         student_prior=student_prior,
                         student_dependency=student_dependency,
                         completed_indices=completed_indices
                         )

@app.route("/submit", methods=["POST"])
def submit():
    data = request.json
    current_index = data.pop("current_index", 1)
    
    # Get current eval pair
    current_pair = eval_pairs[current_index - 1]
    
    # Create new eval result object
    result = current_pair.copy()
    
    # Process eval result
    human_eval = {
        "proactivity": data["proactivity"].replace("left", "model_1").replace("right", "model_2"),
        "adaptability": data["adaptability"].replace("left", "model_1").replace("right", "model_2"),
        "coherence": data["coherence"].replace("left", "model_1").replace("right", "model_2"),
        "justification": data["justification"]
    }
    
    result["human_eval"] = human_eval
    
    # Update in-memory results
    completed_key = f"eval_idx_{result['eval_idx']}"
    completed_results[completed_key] = result

    # Sort by eval_idx
    saved_results = []
    for k, v in completed_results.items():
        saved_results.append(v)
    saved_results = sorted(saved_results, key=lambda x: x['eval_idx'])
    
    # Rewrite entire results file
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(saved_results, f, indent=4)
        f.flush()
    
    all_completed = all(
        f"eval_idx_{pair['eval_idx']}" in completed_results 
        for pair in eval_pairs
    )
    
    if all_completed:
        return jsonify({"message": "All evaluations completed!", "completed": True})

    next_index = current_index + 1
    if next_index > len(eval_pairs):
        next_index = 1
    
    completed_indices = [i+1 for i, pair in enumerate(eval_pairs) 
                        if f"eval_idx_{pair['eval_idx']}" in completed_results]
    
    # Prepare next dialogue data
    current_pair = eval_pairs[next_index - 1]
    left_dialogue = []
    right_dialogue = []
    
    for turn in current_pair["model_1"]["conversation"]:
        for role, text in turn.items():
            left_dialogue.append({
                "speaker": role.capitalize(),
                "text": process_text(text)
            })
            
    for turn in current_pair["model_2"]["conversation"]:
        for role, text in turn.items():
            right_dialogue.append({
                "speaker": role.capitalize(),
                "text": process_text(text)
            })
    
    task_name = current_pair["namespace"]
    task_desc = current_pair["task"]
    # Automatically remove extra spaces and indentations from task_desc
    task_desc = re.sub(r'\n\s*', '\n', task_desc)
    task_desc = process_text(f"```python\n{task_desc}\n```")

    dependency_all = current_pair["dependency_all"]
    reference_steps = current_pair["reference_steps"]
    student_level = current_pair["student_level"]
    student_prior = current_pair["student_prior"]
    student_dependency = current_pair["student_dependency"]
    
    return jsonify({
        "message": f"Submission successful! Continue to {next_index} / {len(eval_pairs)} task.",
        "completed": False,
        "next_data": {
            "dialogues": [{
                "left": left_dialogue,
                "right": right_dialogue
            }],
            "current_index": next_index,
            "total_pairs": len(eval_pairs),
            "task_name": task_name,
            "task_desc": task_desc,
            "dependency_all": dependency_all,
            "reference_steps": reference_steps,
            "student_level": student_level,
            "student_prior": student_prior,
            "student_dependency": student_dependency,
            "completed_indices": completed_indices
        }
    })

@app.route("/get_case", methods=["POST"])
def get_case():
    data = request.json
    current_index = data.get("index", 1)
    total_pairs = len(eval_pairs)
    current_pair = eval_pairs[current_index - 1]
    
    # Get completed indices
    completed_indices = [i+1 for i, pair in enumerate(eval_pairs) 
                        if f"eval_idx_{pair['eval_idx']}" in completed_results]
    
    left_dialogue = []
    right_dialogue = []
    
    # Process model_1's conversation
    for turn in current_pair["model_1"]["conversation"]:
        for role, text in turn.items():
            left_dialogue.append({
                "speaker": role.capitalize(),
                "text": process_text(text)
            })
            
    # Process model_2's conversation
    for turn in current_pair["model_2"]["conversation"]:
        for role, text in turn.items():
            right_dialogue.append({
                "speaker": role.capitalize(),
                "text": process_text(text)
            })
    
    task_name = current_pair["namespace"]
    task_desc = current_pair["task"]
    task_desc = re.sub(r'\n\s*', '\n', task_desc)
    task_desc = process_text(f"```python\n{task_desc}\n```")
    
    return jsonify({
        "dialogues": [{
            "left": left_dialogue,
            "right": right_dialogue
        }],
        "current_index": current_index,
        "total_pairs": total_pairs,
        "task_name": task_name,
        "task_desc": task_desc,
        "dependency_all": current_pair["dependency_all"],
        "reference_steps": current_pair["reference_steps"],
        "student_level": current_pair["student_level"],
        "student_prior": current_pair["student_prior"],
        "student_dependency": current_pair["student_dependency"],
        "completed_indices": completed_indices
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
