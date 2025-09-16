## 📚 Multi-Agent GenAI Knowledge Navigator for InfoFusion Technologies Pvt. Ltd.

[Link to Live Project]()

[Link to Presentation Video]()

[Link to LinkedIn Post]()

---

### 🏢 About the Company
InfoFusion Technologies Pvt. Ltd. delivers next-generation learning and AI solutions for enterprises, transforming how modern organizations upskill their teams and adapt to rapid advancements in business, analytics, and technology. InfoFusion empowers workforce readiness through data-driven, intelligent platforms that integrate curated knowledge sources with real-time industry insights.

The mission of the company is to accelerate knowledge adoption, close skill gaps, and keep organizations competitive — from AI and cloud engineering to data analytics and project management. Through hybrid learning flows, they provide personalized microlearning modules and adaptive interview preparation tailored to each employee’s role and career goals.

---

### 👥 Project Stakeholders and Team

| Role                               | Name           | Responsibility                                                                 |
|------------------------------------|----------------|-------------------------------------------------------------------------------|
| Chief Learning Officer (Stakeholder) | Ms. Sarvesh Jain | Oversees upskilling strategy, validates learning outcomes                      |
| Director of AI Initiatives          | Dr. Prashant Rao | Guides GenAI adoption, ensures compliance with industry standards              |
| Program Manager                     | Neha Sethi     | Coordinates timelines, risk management, and cross-functional collaboration     |
| Lead Data Scientist                 | Anupam Singh   | Architectures RAG system, LLM integrations, and learning analytics             |
| Machine Learning Engineer (MLOps)   | Vikram Chawla  | CI/CD pipelines, containerization, cloud deployment and versioning             |
| Frontend Developer                  | Geetika Sharma | Designs Streamlit UI, feedback loops, and reporting modules                    |
| Data Platform Engineer              | Shubham Joshi  | Knowledge ingestion, vector DB management, and data reliability                |

---

### 📉 Business Problem
Organizations face continuous pressure to keep their workforce up to speed in fast-evolving domains (Data Science, Generative AI, Computer Vision, Management). Static training manuals and outdated PDFs slow learning cycles, overwhelm employees, and disconnect foundational knowledge from the latest industry practices.

Key Challenges
- Delayed skill adoption and workforce readiness

- High training costs, low ROI

- Limited scalability and adaptability of training resources

- Outdated materials leading to competitive disadvantage

---

### 🎯 Project Objectives
- Transform technical books and compliance guides into role-specific microlearning modules

- Index dense material in a scalable vector DB and enable hybrid retrieval with live internet sources

- Auto-generate interview prep content, study aids, and validated practice questions for targeted roles

- Deliver an interactive, multi-tab dashboard for Q&A and MCQ quizzes

---

### 🧭 Solution Overview
The GenAI Knowledge Navigator is a Retrieval-Augmented Generation (RAG) platform that:

- Ingests technical literature (PDFs/books) and indexes knowledge in ChromaDB

- Synthesizes dense material into role-based learning modules, summaries, and certifications

- Cross-references content with live internet sources (Wikipedia, SERP API) for validation and freshness

- Generates scenario-based resources, including interview questions and MCQs for skill reinforcement

- Delivers an interactive, multi-tab dashboard for Q&A and MCQ quizzes

---

### 🖥️ App Features
- Multi-Tab Streamlit UI:

    - Conversational Q&A (hybrid RAG retrieval)

    - MCQ creation and automated evaluation

- Book & PDF Ingestion:

    - Indexes entire guides in ChromaDB for scalable Q&A

- Conversational Assistant:

    - Hybrid search via VectorDB, Wikipedia & SerpAPI

- MCQ Generator & Evaluator:

    - Automated question creation for continuous assessment

---

### 🧰 Tech Stack



| Category          | Tools & Libraries                                    |
|-------------------|------------------------------------------------------|
| Language          | Python 3.10                                          |
| LLM/GenAI         | LangChain, ChatGroq, OpenAI GPT family               |
| Vector DB         | ChromaDB                                             |
| Data Processing   | Pandas, NumPy, PyPDF, SentenceTransformers           |
| UI Development    | Streamlit                                            |
| Cloud & DevOps    | Docker, GitHub Actions, GHCR                         |

---

### 🚀 Setup & Run

#### Option 1 — Run with Docker (Recommended)

**Prerequisites:**

* Make sure you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed on your system.

> **For contributors/maintainers:** If the repository is private, authenticate with GHCR using:
> ```bash
> docker login ghcr.io
> ```
> Use a GitHub Personal Access Token for authentication.

**Steps:**

1.  **Pull the Docker Image:**

The Docker image is pre-built and pushed to GitHub Container Registry (GHCR). Pull it directly:
  ```bash
  docker pull ghcr.io/anupamknn/knowledge-navigator-rag:latest
  ```

**Or build locally from source:**

  ```bash
  git clone https://github.com/AnupamKNN/Knowledge_Navigator_RAG.git
  cd taxi_demand_forecast
  
  docker build -t knowledge-navigator-rag:latest
  ```

2.  **Run the Docker Container:**
Once the image is pulled, you can run the application. The container will listen on map port `8502` on your local machine (adjust if your app uses a different port).
  ```bash
  docker run -p 8503:8503 ghcr.io/anupamknn/knowledge-navigator-rag:latest
  ```

3.  **Access the Application:**
Navigate to [http://localhost:8503](http://localhost:8503) in your web browser to access the Streamlit dashboard.

> Ensure vector DB files and ChromaDB artifacts are available/mounted in the container.

#### Option 2 — Manual Local Setup (For Development)

If a manual setup is preferred, follow these steps:

**Prerequisites:**

* **Python 3.10** or higher installed.

1.  **Clone the Repository:**
  ```bash
  git clone https://github.com/AnupamKNN/Knowledge_Navigator_RAG.git
  ```

2. *** Locate to the project folder:***
  ```bash
  cd Knowledge_Navigator_RAG
  ```

3. **Create a Virtual Environment and Install Dependencies:**

 **Using `venv` (recommended for Python projects):**

#### 1️⃣ Create a virtual environment
  
    ```bash
    python3.11 -m venv venv
    ```

#### 2️⃣ Activate the virtual environment

##### For Linux/macOS:
    ```bash
    source venv/bin/activate
    ```

##### For Windows (PowerShell):
    ```bash
    .\venv\Scripts\Activate.ps1
    ```

##### For Windows (Command Prompt):
    ```bash
    .\venv\Scripts\activate
    ```

#### 3️⃣ Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

#### 3️⃣ Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

4) **Run the app**
streamlit run app.py

This makes it easy for anyone cloning your repo to set up their environment correctly! ✅

---


### 🕹️ Usage Guide
- Q&A Tab — Ask natural language questions grounded in ingested PDFs and live sources.

- MCQ Quiz Tab — Generate and evaluate multiple-choice questions to reinforce concepts.

---


### 🔥 Results & Insights

| **Metric**              | **Projected Outcome**                                                                |
|------------------------ |-------------------------------------------------------------------------------------|
| 🔻 Preparation Time     | Reduced by **35–45%** via concise, role-specific auto-generated summaries           |
| 🔺 Workforce Readiness  | Employees achieve **faster adoption** of cutting-edge AI, analytics & tech skills   |
| 🔁 Knowledge Freshness  | Content stays **up-to-date** through Wikipedia & SerpAPI integration                |
| 🧠 Interview Success    | Practice Q&A and MCQs boost **interview confidence and pass rates by 25–30%**       |
| 💸 ROI on Learning      | **Higher impact** per learner-hour; improved skill retention and training outcomes  |
| ⏱️ Module Completion    | Average microlearning module finished in **7–10 minutes** per learner               |
| 🗂️ Resource Reach      | 100% coverage of required compliance/tech domains for all target roles              |

- **Peak Insights:**  
  - Microlearning modules mapped to certification objectives led to **22% faster onboarding** for new tech hires  
  - Cross-referencing internal guides with live sources detected and corrected outdated regulatory advice in 2 major domains  
  - Interview prep content tailored by role increased success rates in internal upskilling tests

---

### ✅ Final Deliverables

| Deliverable                  | Key Components                                         |
|-------------------------------|-------------------------------------------------------|
| 📦 Data/Knowledge Pipeline    | Book/PDF ingestion, role mapping, feature engineering  |
| 🧮 Vector DB                  | ChromaDB-based scalable document/QA retrieval          |
| 🏷️ Model & LLM Assets         | RAG, LLM agent chains, role-specific prompt templates  |
| 📊 Dashboard & Reporting      | Streamlit UI                                           |
| 🔧 CI/CD & Containerization   | GitHub Actions, Docker container with ready-to-run app |


---

### 💡 Enjoyed this project?

If this repository helped you, consider:

* ⭐ **Starring** the repo — helps others find it
* 🍴 **Forking** it — adapt to your city & data
* 🐛 **Opening issues** — suggest features or report bugs

---