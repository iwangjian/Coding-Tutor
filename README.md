<h2 align="center"> Training Turn-by-Turn Verifiers for Dialogue Tutoring Agents: <br> The Curious Case of LLMs as Your Coding Tutors </h2>


<p align="center">
📃<a href="https://arxiv.org/abs/2502.13311">arXiv</a>
•
🤗<a href="https://huggingface.co/papers/2502.13311"> Huggingface</a>
</p>


This work explores the potential of LLMs as task-tutoring agents. We take coding tutoring as a representative scenario, then propose Trace-and-Verify (**Traver**), an effective agent workflow that incorporates knowledge tracing and turn-by-turn verification, to tackle key challenges in coding tutoring. While this work focuses on coding tutoring as an example, the proposed method extends beyond coding to other task-tutoring scenarios, where the tutor must adapt content to users' varying levels of background knowledge. We further introduce Dialogue for Coding Tutoring (**DICT**), a novel evaluation protocol combining student simulation and coding tests to assess tutoring performance. Such automated evaluation is critical for developing task-tutoring agents as it supports a systematic development and evaluation cycle.

<p align="center">
  <img src="./assets/overview.png" width="95%" alt="overview">
</p>


## News
- [2025.06.02] 🔥 Update detailed instructions for quickstart.
- [2025.05.16] 🔥 Our work is accepted to the [ACL Findings 2025](https://2025.aclweb.org/).
- [2025.02.18] 🚀 Release this work on [arXiv](https://arxiv.org/abs/2502.13311).


## Environment Setup

1. Please follow the instructions of [EvoCodeBench](https://github.com/seketeam/EvoCodeBench) to download EvoCodeBench-2403 dataset, then build the execution environment based on [Environment Setup](https://github.com/seketeam/EvoCodeBench?tab=readme-ov-file#environment-setup). Note that this will cost a few hours to build the execution environment.

2. For this `Coding-Tutor` project, please install the required packages as follows:
    ```bash
    conda create -n coding-tutor python=3.10
    conda activate coding-tutor
    pip install -r requirements.txt
    ```


## Usage
Note: Please set your own Azure API key, data path, and model path based on `scripts/run/`.

### Running Baselines
```bash
# Start the student engine
bash scripts/run/run_engine_student.sh

# (Optional) Start the tutor engine for open-source backbone models
bash scripts/run/run_engine_tutor.sh

# Simulate tutoring dialogues
bash scripts/run/run_base.sh
```

### Running the Trace-and-Verify Workflow
```bash
# Start the student engine
bash scripts/run/run_engine_student.sh

# (Optional) Start the tutor engine for open-source backbone models
bash scripts/run/run_engine_tutor.sh

# Prepare the verifier data
bash scripts/run/prepare_verifier_data.sh

# Train the verifier
bash scripts/run/run_verifier.sh

# Simulate tutoring dialogues
bash scripts/run/run_traver.sh
```

### Running Coding Tests for Simulated Students
```bash
# Run pre-test
bash scripts/run/run_pretest.sh

# Run student code generation
bash scripts/run/run_code_gen.sh

# Run coding test metrics
bash scripts/run/run_coding_test.sh
```

### Automatic Evaluation
```python
# Evaluate pre-test performance
python scripts/eval/eval_pretest.py

# Evaluate tutoring outcome performance
python scripts/eval/eval_TOR.py --pretest_dir ${pretest_dir} --posttest_dir ${posttest_dir}

# Draw tutoring outcome curves
python scripts/eval/eval_TOC.py --tutor_settings ${tutor_settings} --tutor_models ${tutor_models}
```

### Human Evaluation
Please refer to the `human_eval` folder.


## Released Models and Results
For the trained verifier models, please download from 🤗 Hugging Face [jwanglvy/Verifier-7B](https://huggingface.co/jwanglvy/Verifier-7B).

For the simulated dialogues, please refer to the `output/dialogue.zip`. Below is the data format:
```
[
    {
        "namespace": "easyvolcap.utils.gl_utils.Quad.upload_to_texture",
        "conversation": [
            {
                "tutor": "Hello! How familiar are you with OpenGL texture updates and the use of PyTorch tensors in Python?"
            },
            {
                "student": "Hello! I'm familiar with OpenGL texture updates and the use of PyTorch tensors in Python, but I'm still learning. I appreciate ..."
            },
            {
                "tutor": "Great start! In the function, when `w` and `h` are not provided or are zero, we should set them to the object's width and height (`self.W` and `self.H`). Also, ... How would you implement these steps in the function?"
            },
            ...
        ]
    }
    ...
]
```


For the evaluation results, please refer to the `output/student_pretest.zip` and `output/student_posttest.zip`.


## Tutoring Performance Comparison

<p align="center">
  <img src="./assets/eval_results.png" width="90%" alt="eval_results">
</p>

## Analysis of Simulated Students
Under a controlled setup, simulated students at different levels demonstrate distinct abilities in completing target coding tasks. Our DICT protocol serves as a feasible proxy for human evaluation, offering its advantages of scalability and cost-effectiveness for evaluating tutor agents.

<p align="center">
  <img src="./assets/simulated_students.png" width="70%" alt="simulated_students">
</p>

## Inference-Time Scaling with the Verifier

Our proposed Traver agent workflow with the trained verifier shows inference-time scaling for coding tutoring:
<p align="center">
  <img src="./assets/test_scaling.png" width="70%" alt="scaling">
</p>


## Citation
If you find the resources in this repository useful for your work, please kindly cite our work as:
```bibtex
@inproceedings{wang2025training,
  title={Training Turn-by-Turn Verifiers for Dialogue Tutoring Agents: The Curious Case of LLMs as Your Coding Tutors},
  author={Wang, Jian and Dai, Yinpei and Zhang, Yichi and Ma, Ziqiao and Li, Wenjie and Chai, Joyce},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2025},
  month={jul},
  year={2025}
}
```