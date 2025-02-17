document.addEventListener('DOMContentLoaded', function() {
  const submitButton = document.getElementById("submit-button");
  if (submitButton) {
    submitButton.addEventListener("click", function() {
      const form = document.getElementById("evaluation-form");
      const formData = new FormData(form);
      
      // Validate all radio buttons are selected
      const requiredFields = ['proactivity', 'adaptability', 'coherence'];
      let isValid = true;
      
      requiredFields.forEach(field => {
        if (!formData.get(field)) {
          isValid = false;
        }
      });
      
      if (!isValid) {
        alert("Please complete all evaluation items before submission.");
        return;
      }
      
      // Get current index
      const currentIndex = parseInt(document.querySelector('p').textContent.match(/(\d+)/)[0]);
      
      // Prepare data to submit
      const data = {
        current_index: currentIndex,
        proactivity: formData.get("proactivity"),
        adaptability: formData.get("adaptability"),
        coherence: formData.get("coherence"),
        justification: formData.get("justification") || ""
      };

      // Send data to server
      fetch("/submit", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      })
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(data => {
        alert(data.message);
        
        if (data.completed) {
          // If all evaluations are completed, redirect to a completion page or refresh current page
          window.location.reload();
          return;
        }
        
        // Update page content
        updatePageContent(data.next_data);
        
        // Reset form
        form.reset();
        
        // Reapply code highlighting
        document.querySelectorAll('pre code').forEach((block) => {
          hljs.highlightElement(block);
        });
      })
      .catch(error => {
        console.error("Error:", error);
        alert("Submission failed, please try again.");
      });
    });
  }

  // Case select event listener
  const caseSelect = document.getElementById('case-select');
  if (caseSelect) {
    caseSelect.addEventListener('change', function() {
      const selectedIndex = this.value;
      
      // Use POST request to get new case data
      fetch('/get_case', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          index: parseInt(selectedIndex)
        })
      })
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(data => {
        // Use updatePageContent function to update page content
        updatePageContent(data);
        
        // Reset form
        const form = document.getElementById('evaluation-form');
        if (form) {
          form.reset();
        }
      })
      .catch(error => {
        console.error('Error:', error);
      });
    });
  }

  // Reset case-select to Case 1 on page load
  if (caseSelect) {
    caseSelect.value = "1";
  }
});

function updatePageContent(data) {
    // Update current index display
    const indexDisplay = document.querySelector('p');
    if (indexDisplay) {
        indexDisplay.textContent = `You are currently at comparison ${data.current_index} / ${data.total_pairs}`;
    }
    
    // Update task name
    document.querySelector('#task-name').textContent = data.task_name;
    
    // Update task description
    const taskDesc = document.querySelector('#task-desc');
    if (taskDesc && data.task_desc) {
        taskDesc.innerHTML = data.task_desc;
    }
    
    // Update dependency information
    const dependencyAll = document.querySelector('#dependency-all');
    if (dependencyAll && data.dependency_all) {
        dependencyAll.textContent = data.dependency_all;
    }
    
    // Update reference steps
    const referenceSteps = document.querySelector('#reference-steps');
    if (referenceSteps && data.reference_steps) {
        referenceSteps.textContent = data.reference_steps;
    }

    // Update student information
    const studentLevel = document.querySelector('#student-level');
    const studentDependency = document.querySelector('#student-dependency');
    const studentPrior = document.querySelector('#student-prior');
    
    if (studentLevel && data.student_level) {
        studentLevel.textContent = data.student_level;
        if (data.student_level === "low-level") {
            studentDependency.style.display = 'none';
        } else {
            studentDependency.style.display = 'block';
            studentDependency.textContent = data.student_dependency;
        }
    }
    
    if (studentPrior && data.student_prior) {
        studentPrior.textContent = data.student_prior;
    }
    
    // Update dialogue content
    const leftContainer = document.querySelector('.dialogue-container:first-of-type .dialogue');
    const rightContainer = document.querySelector('.dialogue-container:last-of-type .dialogue');
    
    if (leftContainer && rightContainer && data.dialogues && data.dialogues[0]) {
        // Clear existing content
        leftContainer.innerHTML = '';
        rightContainer.innerHTML = '';
        
        // Add new dialogue content
        data.dialogues[0].left.forEach(line => {
            leftContainer.innerHTML += `
                <div class="message">
                    <div class="speaker">${line.speaker}</div>
                    <div class="text">${line.text}</div>
                </div>
            `;
        });
        
        data.dialogues[0].right.forEach(line => {
            rightContainer.innerHTML += `
                <div class="message">
                    <div class="speaker">${line.speaker}</div>
                    <div class="text">${line.text}</div>
                </div>
            `;
        });
    }
    
    // Update case select's selected value
    const caseSelect = document.getElementById('case-select');
    if (caseSelect) {
        caseSelect.value = data.current_index;
    }
    
    // Update case select's completion status
    if (caseSelect && data.completed_indices) {
        Array.from(caseSelect.options).forEach((option, index) => {
            const caseNumber = index + 1;
            const isCompleted = data.completed_indices.includes(caseNumber);
            option.text = `Case ${caseNumber} ${isCompleted ? '(Completed)' : ''}`;
        });
    }
    
    // Reapply code highlighting
    document.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightElement(block);
    });
}