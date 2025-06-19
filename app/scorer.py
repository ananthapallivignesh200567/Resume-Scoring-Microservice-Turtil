import os
import re
import json
import logging
import joblib
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("resume-scorer")

class ResumeScorer:
    """
    Main class for scoring resumes against various goals using ML models
    and skill-based analysis.
    """
    
    def __init__(self, config: Dict[str, Any], goals: Dict[str, List[str]]):
        """
        Initialize the ResumeScorer with configuration and goals data.
        
        Args:
            config: Configuration dictionary loaded from config.json
            goals: Dictionary of goals and their required skills
        """
        self.goals_data = goals
        self.synonym_map={
    "Object-Oriented Programming": ["OOP", "Object Oriented Design", "Object Oriented Prog", "OOPS"],
    "Data Structures": ["DSA", "Data Structs"],
    "Algorithms": ["Algo", "Algos"],
    "System Design": ["Architecture Design", "Software Design"],
    "CI/CD": ["Continuous Integration", "Continuous Deployment", "DevOps Pipelines"],
    "AWS": ["Amazon Web Services", "AWS Cloud", "AWS EC2", "AWS Lambda"],
    "REST APIs": ["RESTful APIs", "API Development", "REST Services"],
    "Microservices": ["Microservice Architecture", "Micro-services"],
    "Git": ["Version Control", "GitHub", "Gitlab"],
    "Linux": ["Unix", "Linux OS", "Ubuntu", "Red Hat"],
    "Docker": ["Containers", "Docker Engine", "Containerization"],
    "Kubernetes": ["K8s", "Kube"],
    "SQL": ["Structured Query Language", "MySQL", "PostgreSQL", "T-SQL"],
    "NoSQL": ["MongoDB", "Cassandra", "DynamoDB"],
    "Concurrency": ["Multithreading", "Parallel Programming", "Threading"],
    "Testing": ["Unit Testing", "Integration Testing", "Test Automation"],
    "Problem Solving": ["Coding Challenges", "Algorithmic Thinking"],
    "Networking": ["Computer Networks", "TCP/IP", "Networking Protocols"],
    "Performance Optimization": ["Performance Tuning", "Code Optimization", "Profiling"],
    "Scalability": ["Horizontal Scaling", "Vertical Scaling"],
    "Machine Learning": ["ML", "Supervised Learning", "Unsupervised Learning"],
    "Deep Learning": ["DL", "Neural Nets", "Neural Networks"],
    "Natural Language Processing": ["NLP", "Text Mining", "Text Processing"],
    "Computer Vision": ["CV", "Image Processing"],
    "Jupyter Notebooks": ["Jupyter", "IPython Notebook"],
    "Feature Engineering": ["Feature Extraction", "Feature Selection"],
    "Data Preprocessing": ["Data Cleaning", "Data Wrangling"],
    "Model Deployment": ["Model Serving", "Deploying Models", "ML Deployment"],
    "MLOps": ["ML Operations", "ModelOps"],
    "Experiment Tracking": ["MLflow", "Experiment Logging"],
    "Statistics": ["Statistical Analysis", "Descriptive Stats"],
    "Linear Algebra": ["Matrix Math", "Vectors and Matrices"],
    "Visualization": ["Data Viz", "Plots", "Graphs"],
    "Reinforcement Learning": ["RL", "Q-Learning", "Policy Gradients"],
    "Optimization": ["Mathematical Optimization", "Convex Optimization"],
    "Agile Methodologies": ["Agile", "Scrum", "Kanban"],
    "Wireframing": ["Mockups", "UI Sketches"],
    "Prototyping": ["Prototype Design", "Interactive Prototypes"],
    "User Research": ["UX Research", "User Studies"],
    "UI/UX Design": ["User Interface Design", "User Experience Design"],
    "Accessibility": ["A11y", "Web Accessibility"],
    "Responsive Design": ["Mobile Friendly Design", "Adaptive Design"]
    }

        self.config = config
        self.goals = goals
        self.models = {}
        self.shared_vectorizer = None
        
        # Load models and vectorizers for each supported goal
        self._load_models()
        
        logger.info(f"ResumeScorer initialized with {len(self.models)} models")
    
    def _load_models(self) -> None:
        """Load shared TF-IDF vectorizer and goal-specific models."""
        model_dir = os.path.join(os.path.dirname(__file__), "model")

        # Load shared vectorizer
        try:
            vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
            self.shared_vectorizer = joblib.load(vectorizer_path)
            logger.info(f"✅ Shared TF-IDF vectorizer loaded with {len(self.shared_vectorizer.vocabulary_)} features")
        except Exception as e:
            logger.error(f"❌ Failed to load shared TF-IDF vectorizer: {str(e)}")
            self.shared_vectorizer = None

        # Load goal-specific models
        for goal in self.config["model_goals_supported"]:
            try:
                goal_filename = goal.lower().replace(" ", "_")
                model_path = os.path.join(model_dir, f"{goal_filename}_model.pkl")
                self.models[goal] = joblib.load(model_path)
                logger.info(f"✅ Loaded model for goal: {goal}")
            except Exception as e:
                logger.error(f"❌ Failed to load model for goal {goal}: {str(e)}")

    
    def _extract_skills_from_resume(self, resume_text: str) -> List[str]:
        if isinstance(resume_text, list):
            resume_text = ", ".join(resume_text)
        
        all_skills = set()
        for skills in self.goals.values():
            all_skills.update(skills)

        resume_text_lower = resume_text.lower()
        normalized_resume = resume_text_lower

        # Fix: Replace all synonyms with canonical skills
        for canonical, synonyms in self.synonym_map.items():
            for synonym in synonyms:
                pattern = r'\b' + re.escape(synonym.lower()) + r'\b'
                normalized_resume = re.sub(pattern, canonical.lower(), normalized_resume)

        found_skills = []
        for skill in all_skills:
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, normalized_resume):
                found_skills.append(skill)

        return found_skills

    
    def _get_matched_missing_skills(self, found_skills: List[str], goal: str) -> tuple:
        """
        Compare found skills with required skills for the goal.
        
        Args:
            found_skills: List of skills found in the resume
            goal: The target goal
            
        Returns:
            Tuple of (matched_skills, missing_skills)
        """
        if goal not in self.goals:
            logger.warning(f"Goal '{goal}' not found in goals data, using default")
            goal = self.config["default_goal_model"]
            
        required_skills = self.goals.get(goal, [])
        
        # Find matched and missing skills
        matched_skills = [skill for skill in found_skills if skill in required_skills]
        missing_skills = [skill for skill in required_skills if skill not in found_skills]
        
        return matched_skills, missing_skills
    
    def _generate_learning_path(self, missing_skills: List[str]) -> List[str]:
        """
        Generate a personalized learning path based on missing skills.
        
        Args:
            missing_skills: List of skills missing from the resume
            
        Returns:
            List of suggested learning activities
        """
        learning_path = []
        
        # Skill-specific learning recommendations
        skill_recommendations = {
  "Java": {
    "path": "1. Java Basics: Variables, data types, operators, control structures → 2. OOP Concepts: Classes, objects, inheritance, polymorphism, encapsulation → 3. Collections Framework: ArrayList, HashMap, LinkedList, TreeMap → 4. Exception Handling: try-catch, custom exceptions → 5. Multithreading: Thread class, Runnable interface, synchronization → 6. I/O Operations: File handling, streams → 7. JDBC: Database connectivity → 8. Advanced: Generics, lambda expressions, streams API",
    "course": "Oracle Java Programming Complete Course (Udemy) or Java Programming Masterclass (Tim Buchalka)"
  },
  "Python": {
    "path": "1. Python Fundamentals: Syntax, variables, data types → 2. Control Flow: if/else, loops, functions → 3. Data Structures: Lists, dictionaries, tuples, sets → 4. OOP in Python: Classes, inheritance, methods → 5. File I/O and Exception Handling → 6. Libraries: requests, json, csv → 7. Advanced: decorators, generators, context managers → 8. Virtual environments and package management",
    "course": "Python for Everybody Specialization (Coursera - University of Michigan) or Complete Python Bootcamp (Udemy)"
  },
  "C++": {
    "path": "1. C++ Basics: Syntax, variables, I/O → 2. Control Structures: loops, conditionals → 3. Functions and Arrays → 4. Pointers and References → 5. OOP: Classes, objects, inheritance → 6. Memory Management: new, delete, smart pointers → 7. STL: vectors, maps, algorithms → 8. Templates and Exception Handling",
    "course": "C++ Programming Course - Beginner to Advanced (Udemy) or C++ Fundamentals (Pluralsight)"
  },
  "Data Structures": {
    "path": "1. Arrays and Strings: Basic operations, two-pointer technique → 2. Linked Lists: Singly, doubly, circular → 3. Stacks and Queues: Implementation, applications → 4. Trees: Binary trees, BST, traversals → 5. Graphs: Representation, BFS, DFS → 6. Hash Tables: Implementation, collision handling → 7. Heaps: Min/max heap, priority queues → 8. Advanced: Trie, segment trees",
    "course": "Data Structures and Algorithms Specialization (Coursera - UC San Diego) or Master the Coding Interview (Udemy)"
  },
  "Algorithms": {
    "path": "1. Time/Space Complexity: Big O notation → 2. Sorting: Bubble, merge, quick, heap sort → 3. Searching: Binary search, linear search → 4. Recursion and Backtracking → 5. Dynamic Programming: Memoization, tabulation → 6. Greedy Algorithms → 7. Graph Algorithms: Dijkstra, Floyd-Warshall → 8. String Algorithms: KMP, Rabin-Karp",
    "course": "Algorithms Specialization (Coursera - Stanford) or JavaScript Algorithms and Data Structures (freeCodeCamp)"
  },
  "System Design": {
    "path": "1. Scalability Basics: Horizontal vs vertical scaling → 2. Load Balancing: Types, algorithms → 3. Caching: Redis, Memcached, CDN → 4. Database Design: SQL vs NoSQL, sharding → 5. Microservices Architecture → 6. Message Queues: Kafka, RabbitMQ → 7. Monitoring and Logging → 8. Case Studies: Design Twitter, Uber, Netflix",
    "course": "Grokking the System Design Interview (Educative) or System Design Interview Course (Exponent)"
  },
  "Object-Oriented Programming": {
    "path": "1. OOP Principles: Encapsulation, inheritance, polymorphism, abstraction → 2. Classes and Objects: Constructor, destructor → 3. Inheritance: Single, multiple, hierarchical → 4. Polymorphism: Method overriding, overloading → 5. Abstract Classes and Interfaces → 6. Design Patterns: Singleton, Factory, Observer → 7. SOLID Principles → 8. UML Diagrams",
    "course": "Object Oriented Programming in Java (Coursera - Duke University) or OOP Concepts in C++ (Udemy)"
  },
  "SQL": {
    "path": "1. SQL Basics: SELECT, INSERT, UPDATE, DELETE → 2. Filtering: WHERE, LIKE, IN, BETWEEN → 3. Joins: INNER, LEFT, RIGHT, FULL OUTER → 4. Grouping: GROUP BY, HAVING, aggregate functions → 5. Subqueries and CTEs → 6. Window Functions: ROW_NUMBER, RANK, LEAD/LAG → 7. Indexes and Performance → 8. Stored Procedures and Triggers",
    "course": "SQL for Data Science (Coursera - UC Davis) or The Complete SQL Bootcamp (Udemy)"
  },
  "NoSQL": {
    "path": "1. NoSQL Fundamentals: Types, CAP theorem → 2. Document Databases: MongoDB basics, queries → 3. Key-Value Stores: Redis operations → 4. Column-Family: Cassandra basics → 5. Graph Databases: Neo4j introduction → 6. Data Modeling in NoSQL → 7. Consistency Models → 8. Performance and Scaling",
    "course": "MongoDB University (Free) or Introduction to NoSQL (Coursera - University of Illinois)"
  },
  "AWS": {
    "path": "1. AWS Fundamentals: IAM, regions, availability zones → 2. Compute: EC2, Lambda, ECS → 3. Storage: S3, EBS, EFS → 4. Databases: RDS, DynamoDB → 5. Networking: VPC, subnets, security groups → 6. Monitoring: CloudWatch, CloudTrail → 7. Deployment: CloudFormation, Elastic Beanstalk → 8. Security: KMS, Secrets Manager",
    "course": "AWS Certified Solutions Architect Associate (A Cloud Guru) or AWS Fundamentals (Coursera - AWS)"
  },
  "REST APIs": {
    "path": "1. HTTP Fundamentals: Methods, status codes, headers → 2. REST Principles: Stateless, resource-based → 3. API Design: URL structure, naming conventions → 4. Authentication: API keys, JWT, OAuth → 5. Data Formats: JSON, XML → 6. Error Handling and Status Codes → 7. API Documentation: Swagger/OpenAPI → 8. Testing: Postman, automated testing",
    "course": "REST APIs with Flask and Python (Udemy) or API Design and Fundamentals (Pluralsight)"
  },
  "Microservices": {
    "path": "1. Microservices Concepts: Monolith vs microservices → 2. Service Decomposition: Domain-driven design → 3. Communication: REST, gRPC, message queues → 4. Data Management: Database per service → 5. Configuration Management → 6. Service Discovery: Eureka, Consul → 7. Circuit Breaker Pattern → 8. Observability: Logging, monitoring, tracing",
    "course": "Microservices with Spring Boot and Spring Cloud (Udemy) or Building Microservices (Pluralsight)"
  },
  "Git": {
    "path": "1. Git Basics: init, add, commit, status → 2. Branching: create, switch, merge branches → 3. Remote Repositories: clone, push, pull, fetch → 4. Collaboration: merge conflicts, pull requests → 5. Advanced: rebase, cherry-pick, stash → 6. Git Flow: Feature branches, releases → 7. Hooks and Automation → 8. Best Practices: Commit messages, branching strategies",
    "course": "Git Complete: The definitive guide (Udemy) or Version Control with Git (Coursera - Atlassian)"
  },
  "CI/CD": {
    "path": "1. CI/CD Concepts: Continuous integration, deployment → 2. Build Automation: Maven, Gradle, npm → 3. Testing Integration: Unit, integration tests → 4. Pipeline Design: Stages, jobs, artifacts → 5. Deployment Strategies: Blue-green, canary → 6. Tools: Jenkins, GitLab CI, GitHub Actions → 7. Infrastructure as Code → 8. Monitoring and Rollback",
    "course": "Jenkins, From Zero To Hero (Udemy) or DevOps Culture and Mindset (Coursera - UC Davis)"
  },
  "Distributed Systems": {
    "path": "1. Distributed Systems Basics: Characteristics, challenges → 2. Consistency Models: ACID, BASE, CAP theorem → 3. Consensus Algorithms: Raft, Paxos → 4. Fault Tolerance: Replication, partitioning → 5. Load Balancing and Service Discovery → 6. Message Passing: Synchronous vs asynchronous → 7. Distributed Storage: Sharding, replication → 8. Monitoring and Debugging",
    "course": "Distributed Systems Concepts (MIT OpenCourseWare) or Cloud Computing Concepts (Coursera - University of Illinois)"
  },
  "Concurrency": {
    "path": "1. Concurrency Basics: Processes vs threads → 2. Thread Creation and Management → 3. Synchronization: Locks, mutexes, semaphores → 4. Race Conditions and Deadlocks → 5. Thread-Safe Data Structures → 6. Parallel Algorithms → 7. Async Programming: Futures, promises → 8. Performance Considerations",
    "course": "Parallel Programming in Java (Coursera - Rice University) or Concurrency in C++ (Pluralsight)"
  },
  "Testing": {
    "path": "1. Testing Fundamentals: Unit, integration, system testing → 2. Test-Driven Development (TDD) → 3. Testing Frameworks: JUnit, pytest, Jest → 4. Mocking and Stubbing → 5. Test Coverage and Metrics → 6. Automated Testing: Selenium, Cypress → 7. Performance Testing: Load, stress testing → 8. API Testing: Postman, REST Assured",
    "course": "Testing JavaScript with Jest (Testing Library) or Java Unit Testing with JUnit 5 (Udemy)"
  },
  "Problem Solving": {
    "path": "1. Problem Analysis: Understanding requirements → 2. Algorithmic Thinking: Breaking down problems → 3. Pattern Recognition: Common problem types → 4. Debugging Techniques: Systematic approach → 5. Code Optimization: Time and space complexity → 6. Design Thinking: Multiple solution approaches → 7. Documentation: Writing clear solutions → 8. Practice: LeetCode, HackerRank",
    "course": "Algorithmic Thinking with Python (edX - Rice University) or Problem Solving with Algorithms (Udemy)"
  },
  "Linux": {
    "path": "1. Linux Basics: File system, navigation → 2. File Operations: cp, mv, rm, chmod → 3. Text Processing: grep, sed, awk → 4. Process Management: ps, top, kill → 5. Networking: netstat, curl, wget → 6. Shell Scripting: Bash basics → 7. System Administration: cron, systemd → 8. Security: Users, permissions, SSH",
    "course": "Linux Command Line Basics (Udacity) or Linux Administration Bootcamp (Udemy)"
  },
  "Networking": {
    "path": "1. Network Fundamentals: OSI model, TCP/IP → 2. IP Addressing: IPv4, subnetting → 3. Protocols: HTTP/HTTPS, FTP, DNS → 4. Routing and Switching → 5. Network Security: Firewalls, VPNs → 6. Load Balancing → 7. Network Troubleshooting → 8. Monitoring Tools: Wireshark, ping, traceroute",
    "course": "Computer Networking (Coursera - University of Washington) or Networking Fundamentals (Pluralsight)"
  },
  "Docker": {
    "path": "1. Docker Basics: Containers vs VMs, installation → 2. Images and Containers: Building, running → 3. Dockerfile: Creating custom images → 4. Docker Compose: Multi-container applications → 5. Volumes and Networking → 6. Registry: Docker Hub, private registries → 7. Container Orchestration basics → 8. Best Practices: Security, optimization",
    "course": "Docker Mastery (Udemy - Bret Fisher) or Docker for Developers (Pluralsight)"
  },
  "Performance Optimization": {
    "path": "1. Performance Metrics: Latency, throughput, response time → 2. Profiling Tools: Code profilers, memory analyzers → 3. Database Optimization: Query optimization, indexing → 4. Caching Strategies: Application, database, CDN → 5. Code Optimization: Algorithm efficiency → 6. Memory Management: Garbage collection, leaks → 7. Load Testing: Identifying bottlenecks → 8. Monitoring: APM tools, metrics",
    "course": "Web Performance Optimization (Google Developers) or Java Performance Tuning (Pluralsight)"
  },
  "Scalability": {
    "path": "1. Scalability Principles: Horizontal vs vertical → 2. Load Distribution: Load balancers, CDNs → 3. Database Scaling: Replication, sharding → 4. Caching: Multi-level caching strategies → 5. Microservices: Service decomposition → 6. Asynchronous Processing: Message queues → 7. Auto-scaling: Dynamic resource allocation → 8. Performance Monitoring",
    "course": "Scalable Web Architecture (Coursera) or Building Scalable Applications (Pluralsight)"
  },
  "NumPy": {
    "path": "1. NumPy Basics: Arrays, data types → 2. Array Operations: Indexing, slicing → 3. Mathematical Operations: Broadcasting, vectorization → 4. Array Manipulation: Reshaping, concatenation → 5. Linear Algebra: Matrix operations → 6. Random Number Generation → 7. File I/O: Loading, saving arrays → 8. Performance Optimization",
    "course": "NumPy Tutorial (Kaggle Learn) or Scientific Computing with Python (freeCodeCamp)"
  },
  "Pandas": {
    "path": "1. Pandas Basics: Series, DataFrames → 2. Data Loading: CSV, Excel, JSON, SQL → 3. Data Exploration: info(), describe(), head() → 4. Data Selection: iloc, loc, boolean indexing → 5. Data Cleaning: Missing values, duplicates → 6. Data Transformation: apply(), map(), groupby() → 7. Merging and Joining: concat(), merge() → 8. Data Visualization integration",
    "course": "Pandas Tutorial (Kaggle Learn) or Data Analysis with Python (freeCodeCamp)"
  },
  "Scikit-learn": {
    "path": "1. ML Basics: Supervised vs unsupervised learning → 2. Data Preprocessing: StandardScaler, encoders → 3. Model Selection: train_test_split, cross-validation → 4. Classification: Logistic regression, SVM, Random Forest → 5. Regression: Linear, polynomial regression → 6. Clustering: K-means, hierarchical → 7. Model Evaluation: Metrics, confusion matrix → 8. Hyperparameter Tuning: GridSearch, RandomSearch",
    "course": "Machine Learning with Scikit-Learn (DataCamp) or Scikit-Learn Course (Kaggle Learn)"
  },
  "TensorFlow": {
    "path": "1. TensorFlow Basics: Tensors, operations → 2. Keras API: Sequential, Functional models → 3. Neural Networks: Dense layers, activation functions → 4. Training: Optimizers, loss functions → 5. Convolutional Networks: Conv2D, pooling → 6. Recurrent Networks: LSTM, GRU → 7. Model Deployment: SavedModel, TensorFlow Serving → 8. Advanced: Custom layers, training loops",
    "course": "TensorFlow Developer Certificate (Coursera - DeepLearning.AI) or TensorFlow 2.0 Complete Course (Udemy)"
  },
  "PyTorch": {
    "path": "1. PyTorch Fundamentals: Tensors, autograd → 2. Neural Networks: nn.Module, layers → 3. Training Loop: Forward pass, backpropagation → 4. Data Loading: DataLoader, transforms → 5. CNN: Convolutional layers, image classification → 6. RNN: LSTM, sequence modeling → 7. Transfer Learning: Pre-trained models → 8. Model Deployment: TorchScript, ONNX",
    "course": "PyTorch for Deep Learning (Udemy - Andrei Neagoie) or Deep Learning with PyTorch (Coursera - IBM)"
  },
  "Linear Algebra": {
    "path": "1. Vectors: Operations, dot product → 2. Matrices: Operations, multiplication → 3. Matrix Decomposition: LU, QR, SVD → 4. Eigenvalues and Eigenvectors → 5. Vector Spaces: Basis, dimension → 6. Linear Transformations → 7. Optimization: Gradient, Hessian → 8. Applications in ML",
    "course": "Linear Algebra (Khan Academy) or Mathematics for Machine Learning: Linear Algebra (Coursera - Imperial College)"
  },
  "Statistics": {
    "path": "1. Descriptive Statistics: Mean, median, variance → 2. Probability: Distributions, Bayes theorem → 3. Inferential Statistics: Hypothesis testing → 4. Confidence Intervals → 5. Regression Analysis: Linear, multiple → 6. ANOVA: Analysis of variance → 7. Non-parametric Tests → 8. Experimental Design",
    "course": "Introduction to Statistics (Coursera - Stanford) or Statistics Fundamentals (StatQuest YouTube)"
  },
  "Probability": {
    "path": "1. Basic Probability: Events, sample space → 2. Conditional Probability: Bayes theorem → 3. Random Variables: Discrete, continuous → 4. Probability Distributions: Normal, binomial, Poisson → 5. Expected Value and Variance → 6. Central Limit Theorem → 7. Markov Chains → 8. Bayesian Statistics",
    "course": "Introduction to Probability (edX - MIT) or Think Stats (Free online book)"
  },
  "Data Visualization": {
    "path": "1. Visualization Principles: Chart types, design → 2. Matplotlib: Basic plots, customization → 3. Seaborn: Statistical plots, themes → 4. Plotly: Interactive visualizations → 5. Dashboard Creation: Streamlit, Dash → 6. Advanced: Animations, 3D plots → 7. Best Practices: Color, accessibility → 8. Storytelling with Data",
    "course": "Data Visualization with Python (Coursera - IBM) or Data Visualization (Kaggle Learn)"
  },
  "Matplotlib": {
    "path": "1. Matplotlib Basics: Figure, axes, plots → 2. Line Plots: Customization, multiple lines → 3. Bar Charts: Horizontal, vertical, grouped → 4. Scatter Plots: Colors, sizes, annotations → 5. Histograms and Distributions → 6. Subplots: Multiple plots, layout → 7. Customization: Colors, styles, themes → 8. Saving and Exporting",
    "course": "Matplotlib Tutorial (Real Python) or Data Visualization with Matplotlib (DataCamp)"
  },
  "Seaborn": {
    "path": "1. Seaborn Basics: Statistical plots overview → 2. Distribution Plots: histplot, boxplot, violinplot → 3. Categorical Plots: barplot, countplot → 4. Relationship Plots: scatterplot, lineplot → 5. Matrix Plots: heatmap, clustermap → 6. Multi-plot Grids: FacetGrid, PairGrid → 7. Styling: Themes, color palettes → 8. Integration with Pandas",
    "course": "Seaborn Tutorial (Kaggle Learn) or Statistical Data Visualization (DataCamp)"
  },
  "Jupyter Notebooks": {
    "path": "1. Jupyter Basics: Installation, interface → 2. Notebook Structure: Cells, markdown → 3. Code Execution: Running cells, kernels → 4. Data Analysis Workflow → 5. Visualization Integration → 6. Magic Commands: %time, %matplotlib → 7. Extensions: Widgets, nbextensions → 8. Sharing: nbviewer, GitHub, exports",
    "course": "Jupyter Notebook Tutorial (DataCamp) or Jupyter Notebooks for Beginners (YouTube - Corey Schafer)"
  },
  "Feature Engineering": {
    "path": "1. Feature Engineering Concepts: Importance, types → 2. Handling Missing Data: Imputation strategies → 3. Categorical Encoding: One-hot, label, target → 4. Numerical Features: Scaling, binning → 5. Feature Creation: Polynomial, interactions → 6. Feature Selection: Univariate, recursive → 7. Dimensionality Reduction: PCA, t-SNE → 8. Domain-specific Features",
    "course": "Feature Engineering for Machine Learning (Coursera - University of Washington) or Feature Engineering (Kaggle Learn)"
  },
  "Data Preprocessing": {
    "path": "1. Data Understanding: Exploration, profiling → 2. Data Cleaning: Missing values, outliers → 3. Data Transformation: Normalization, standardization → 4. Encoding: Categorical variables → 5. Feature Scaling: MinMax, Standard, Robust → 6. Data Splitting: Train, validation, test → 7. Handling Imbalanced Data: SMOTE, undersampling → 8. Pipeline Creation",
    "course": "Data Preprocessing in Data Science (Coursera) or Data Cleaning (Kaggle Learn)"
  },
  "Machine Learning Algorithms": {
    "path": "1. Supervised Learning: Classification, regression → 2. Linear Models: Linear/Logistic regression → 3. Tree-based: Decision trees, Random Forest → 4. Ensemble Methods: Boosting, bagging → 5. Instance-based: KNN, SVM → 6. Unsupervised: Clustering, dimensionality reduction → 7. Neural Networks: MLPs, deep learning → 8. Model Selection and Evaluation",
    "course": "Machine Learning (Coursera - Andrew Ng) or Machine Learning A-Z (Udemy)"
  },
  "Neural Networks": {
    "path": "1. Perceptron: Single neuron, linear separability → 2. Multi-layer Perceptrons: Hidden layers → 3. Backpropagation: Gradient computation → 4. Activation Functions: ReLU, sigmoid, tanh → 5. Loss Functions: MSE, cross-entropy → 6. Optimization: SGD, Adam, RMSprop → 7. Regularization: Dropout, batch normalization → 8. Architecture Design",
    "course": "Neural Networks and Deep Learning (Coursera - DeepLearning.AI) or Deep Learning Fundamentals (Cognitive Class)"
  },
  "Deep Learning": {
    "path": "1. Deep Learning Fundamentals: Architecture, training → 2. Convolutional Networks: CNN, image processing → 3. Recurrent Networks: RNN, LSTM, GRU → 4. Attention Mechanisms: Self-attention, transformers → 5. Generative Models: VAE, GANs → 6. Transfer Learning: Pre-trained models → 7. Optimization: Advanced techniques → 8. Deployment: Model serving, edge computing",
    "course": "Deep Learning Specialization (Coursera - DeepLearning.AI) or Practical Deep Learning (fast.ai)"
  },
  "NLP": {
    "path": "1. Text Processing: Tokenization, cleaning → 2. Feature Extraction: TF-IDF, n-grams → 3. Language Models: Statistical, neural → 4. Word Embeddings: Word2Vec, GloVe → 5. Named Entity Recognition → 6. Sentiment Analysis → 7. Transformers: BERT, GPT → 8. Applications: Chatbots, translation",
    "course": "Natural Language Processing Specialization (Coursera - DeepLearning.AI) or NLP with Python (Udemy)"
  },
  "Computer Vision": {
    "path": "1. Image Basics: Pixels, channels, formats → 2. Image Processing: Filtering, enhancement → 3. Feature Detection: Edges, corners, SIFT → 4. Convolutional Networks: CNN architecture → 5. Object Detection: YOLO, R-CNN → 6. Image Segmentation: Semantic, instance → 7. Transfer Learning: Pre-trained models → 8. Advanced: GANs, style transfer",
    "course": "Computer Vision Basics (Coursera - University at Buffalo) or PyTorch Computer Vision (Udemy)"
  },
  "Experiment Tracking": {
    "path": "1. Experiment Tracking Basics: Importance, tools → 2. MLflow: Tracking, projects, models → 3. Weights & Biases: Experiment logging → 4. Metadata Management: Parameters, metrics → 5. Model Versioning: Registry, lifecycle → 6. Reproducibility: Environment, seeds → 7. Collaboration: Sharing experiments → 8. Integration: CI/CD, production",
    "course": "MLflow Tutorial (Official Documentation) or ML Experiment Tracking (YouTube - Made with ML)"
  },
  "Research Paper Reading": {
    "path": "1. Paper Structure: Abstract, introduction, methodology → 2. Critical Reading: Understanding contributions → 3. Related Work: Literature survey → 4. Methodology: Experimental design → 5. Results: Statistical significance → 6. Implementation: Code, reproducibility → 7. Note-taking: Systematic approach → 8. Writing Reviews: Peer review process",
    "course": "How to Read a Paper (YouTube - Two Minute Papers) or Research Methods in AI (Coursera)"
  },
  "Reinforcement Learning": {
    "path": "1. RL Fundamentals: Agent, environment, rewards → 2. Markov Decision Process: States, actions → 3. Value Functions: State, action values → 4. Policy Methods: Policy gradient → 5. Q-Learning: Temporal difference → 6. Deep RL: DQN, Actor-Critic → 7. Multi-agent RL → 8. Applications: Games, robotics",
    "course": "Reinforcement Learning Specialization (Coursera - University of Alberta) or Deep Reinforcement Learning (Udacity)"
  },
  "Bayesian Methods": {
    "path": "1. Bayesian Thinking: Prior, posterior, likelihood → 2. Bayes Theorem: Applications → 3. Conjugate Priors: Beta, Dirichlet → 4. MCMC: Metropolis-Hastings, Gibbs → 5. Variational Inference: Mean field → 6. Bayesian Networks: Graphical models → 7. Hierarchical Models → 8. Practical Applications",
    "course": "Bayesian Statistics (Coursera - Duke University) or Think Bayes (Free online book)"
  },
  "Mathematical Optimization": {
    "path": "1. Optimization Basics: Objective, constraints → 2. Linear Programming: Simplex method → 3. Convex Optimization: Convex sets, functions → 4. Gradient Methods: Steepest descent → 5. Constrained Optimization: Lagrange multipliers → 6. Stochastic Optimization: SGD variants → 7. Metaheuristics: Genetic algorithms → 8. Applications in ML",
    "course": "Convex Optimization (Stanford CS364A) or Optimization Methods (edX - MIT)"
  },
  "Distributed Training": {
    "path": "1. Distributed Computing: Parallelism types → 2. Data Parallelism: Model replication → 3. Model Parallelism: Parameter splitting → 4. Communication: AllReduce, parameter servers → 5. Frameworks: Horovod, DistributedDataParallel → 6. Synchronization: Sync vs async → 7. Fault Tolerance: Checkpointing → 8. Performance: Scaling laws",
    "course": "Distributed Deep Learning (YouTube - NVIDIA) or Parallel and Distributed Computing (Coursera)"
  },
  "Research Paper Writing": {
    "path": "1. Paper Structure: Standard format → 2. Abstract Writing: Concise summary → 3. Introduction: Problem, contribution → 4. Literature Review: Related work → 5. Methodology: Clear description → 6. Experiments: Design, results → 7. Writing Style: Clarity, precision → 8. Submission: Peer review process",
    "course": "Writing in the Sciences (Coursera - Stanford) or Academic Writing (YouTube - Academic English Now)"
  },
  "Experiment Design": {
    "path": "1. Experimental Method: Hypothesis, variables → 2. Control Groups: Randomization → 3. Sample Size: Power analysis → 4. Bias Reduction: Blinding, randomization → 5. Statistical Tests: Choosing appropriate tests → 6. A/B Testing: Online experiments → 7. Confounding Variables: Identification, control → 8. Ethical Considerations",
    "course": "Design of Experiments (Coursera - Arizona State) or Experimental Design (Khan Academy)"
  },
  "LaTeX": {
    "path": "1. LaTeX Basics: Document structure, compilation → 2. Text Formatting: Fonts, emphasis → 3. Document Classes: Article, report, book → 4. Mathematical Notation: Equations, symbols → 5. Figures and Tables: Insertion, captions → 6. Bibliography: BibTeX, citations → 7. Advanced: Custom commands, packages → 8. Collaboration: Overleaf, version control",
    "course": "LaTeX Tutorial (Overleaf Documentation) or Learn LaTeX in 30 minutes (Overleaf)"
  },
  "OpenAI Gym": {
    "path": "1. Gym Basics: Environment interface → 2. Environment Exploration: Observation, action spaces → 3. Basic Agents: Random, heuristic → 4. Environment Wrappers: Preprocessing → 5. Custom Environments: Creating new envs → 6. Integration: With RL algorithms → 7. Rendering: Visualization → 8. Advanced: Multi-agent environments",
    "course": "OpenAI Gym Tutorial (Official Documentation) or Reinforcement Learning with Gym (YouTube)"
  },
  "Hugging Face Transformers": {
    "path": "1. Transformers Library: Installation, basics → 2. Pre-trained Models: BERT, GPT, RoBERTa → 3. Tokenization: WordPiece, BPE → 4. Fine-tuning: Task-specific adaptation → 5. Pipeline API: Easy inference → 6. Custom Models: Architecture modification → 7. Training: Trainer class → 8. Deployment: Model hub, serving",
    "course": "Hugging Face Course (Free online) or NLP with Transformers (O'Reilly Book)"
  },
  "BERT": {
    "path": "1. BERT Architecture: Transformer encoder → 2. Pre-training: MLM, NSP objectives → 3. Tokenization: WordPiece, special tokens → 4. Fine-tuning: Classification, QA → 5. Input Representation: Embeddings → 6. Attention: Self-attention mechanism → 7. Variants: RoBERTa, DistilBERT → 8. Applications: Sentiment, NER",
    "course": "BERT Research Paper + Illustrated BERT (Jay Alammar Blog) or Advanced NLP with spaCy (DataCamp)"
  },
  "GANs": {
    "path": "1. GAN Basics: Generator, discriminator → 2. Training: Adversarial loss → 3. Architecture: DCGAN, improvements → 4. Mode Collapse: Problem, solutions → 5. Evaluation: FID, IS metrics → 6. Variants: cGAN, StyleGAN → 7. Applications: Image generation → 8. Advanced: Progressive GANs, Wasserstein GANs",
    "course": "Generative Adversarial Networks (Coursera - DeepLearning.AI) or GANs in Action (Manning Book)"
  },
  "CNNs": {
    "path": "1. CNN Basics: Convolution, pooling → 2. Architecture: LeNet, AlexNet, VGG → 3. Feature Maps: Filters, activation → 4. Training: Backpropagation through convolution → 5. Advanced: ResNet, Inception → 6. Applications: Image classification → 7. Transfer Learning: Pre-trained models → 8. Optimization: Data augmentation, regularization",
    "course": "Convolutional Neural Networks (Coursera - DeepLearning.AI) or CS231n Stanford CNN Course"
  },
  "RNNs": {
    "path": "1. RNN Basics: Sequential processing → 2. Vanilla RNN: Architecture, limitations → 3. LSTM: Long short-term memory → 4. GRU: Gated recurrent unit → 5. Bidirectional RNNs → 6. Applications: Language modeling → 7. Training: BPTT, gradient problems → 8. Advanced: Attention, Transformers",
    "course": "Sequence Models (Coursera - DeepLearning.AI) or Understanding LSTMs (Christopher Olah Blog)"
  },
  "GPT": {
    "path": "1. GPT Architecture: Transformer decoder → 2. Pre-training: Language modeling → 3. Tokenization: BPE encoding → 4. Fine-tuning: Task adaptation → 5. Prompt Engineering: Zero-shot, few-shot → 6. GPT Variants: GPT-2, GPT-3, GPT-4 → 7. Applications: Text generation → 8. Ethical Considerations: Bias, safety",
    "course": "GPT and Language Models (Hugging Face Course) or The Illustrated GPT-2 (Jay Alammar Blog)"
  },
  "Figma": {
    "path": "1. Figma Basics: Interface, tools → 2. Design Elements: Shapes, text, images → 3. Layouts: Frames, grids, constraints → 4. Components: Reusable elements → 5. Prototyping: Interactions, transitions → 6. Collaboration: Comments, sharing → 7. Design Systems: Styles, libraries → 8. Handoff: Developer specs",
    "course": "Figma UI/UX Design Essentials (Udemy) or Learn Figma (Figma Academy)"
  },
  "Sketch": {
    "path": "1. Sketch Basics: Artboards, tools → 2. Vector Design: Shapes, paths → 3. Symbols: Reusable components → 4. Text Styles: Typography system → 5. Libraries: Shared design systems → 6. Plugins: Extending functionality → 7. Prototyping: InVision integration → 8. Export: Assets, specifications",
    "course": "Sketch from A to Z (Udemy) or Sketch Master Course (DesignCourse YouTube)"
  },
  "Adobe XD": {
    "path": "1. XD Basics: Artboards, design tools → 2. Design: UI elements, layouts → 3. Prototyping: Interactions, micro-animations → 4. Components: States, variants → 5. Design Systems: CC Libraries → 6. Collaboration: Sharing, feedback → 7. Plugins: Third-party extensions → 8. Handoff: Developer mode",
    "course": "Adobe XD Tutorial (Adobe) or Complete Adobe XD Course (Udemy)"
  },
  "Wireframing": {
    "path": "1. Wireframing Basics: Purpose, fidelity levels → 2. Information Architecture: Site maps, user flows → 3. Layout: Grid systems, spacing → 4. Content Strategy: Hierarchy, placement → 5. Navigation: Menus, breadcrumbs → 6. Responsive: Mobile-first approach → 7. Tools: Balsamiq, Sketch, Figma → 8. Testing: Usability validation",
    "course": "UX Design: From Concept to Prototype (Coursera - University of California San Diego) or Wireframing Basics (IxDF)"
  },
  "Prototyping": {
    "path": "1. Prototype Types: Paper, digital, interactive → 2. Fidelity: Low, medium, high → 3. Tools: Figma, Adobe XD, Principle → 4. Interactions: Clicks, hovers, gestures → 5. Animations: Transitions, micro-interactions → 6. User Testing: Feedback collection → 7. Iteration: Design refinement → 8. Handoff: Development specs",
    "course": "Prototyping and Design (Coursera - University of Minnesota) or Advanced Prototyping (IxDF)"
  },
  "User Research": {
    "path": "1. Research Methods: Qualitative, quantitative → 2. User Interviews: Planning, conducting → 3. Surveys: Design, distribution → 4. Usability Testing: Moderated, unmoderated → 5. Analytics: Behavioral data → 6. Personas: User archetypes → 7. Journey Mapping: User experience → 8. Research Synthesis: Insights, recommendations",
    "course": "User Experience Research and Design (Coursera - University of Michigan) or User Research (IxDF)"
  },
  "Interaction Design": {
    "path": "1. IxD Principles: Affordances, signifiers → 2. Mental Models: User expectations → 3. Information Architecture: Structure, navigation → 4. Interface Design: Controls, feedback → 5. Micro-interactions: Details, animations → 6. Accessibility: Inclusive design → 7. Mobile Interactions: Touch, gestures → 8. Voice UI: Conversational design",
    "course": "Interaction Design Specialization (Coursera - UC San Diego) or Interaction Design (IxDF)"
  },
  "Design Systems": {
    "path": "1. Design System Basics: Components, patterns → 2. Design Tokens: Colors, typography, spacing → 3. Component Library: Reusable elements → 4. Documentation: Usage guidelines → 5. Governance: Maintenance, updates → 6. Implementation: Designer-developer handoff → 7. Scaling: Multi-product systems → 8. Tools: Storybook, Figma libraries",
    "course": "Design Systems (Figma Academy) or Building Design Systems (Frontend Masters)"
  },
  "Visual Design": {
    "path": "1. Design Principles: Balance, contrast, hierarchy → 2. Color Theory: Harmony, psychology → 3. Typography: Fonts, spacing, readability → 4. Layout: Grid systems, composition → 5. Imagery: Photography, illustration → 6. Branding: Identity, consistency → 7. Trends: Modern design patterns → 8. Critique: Design evaluation",
    "course": "Visual Design (CalArts Coursera) or The Fundamentals of Visual Design (CreativeLive)"
  },
  "Adobe Photoshop": {
    "path": "1. Photoshop Basics: Interface, tools → 2. Layers: Management, blending modes → 3. Selection: Tools, masking → 4. Photo Editing: Color correction, retouching → 5. Compositing: Multiple images → 6. Text: Typography, effects → 7. Filters: Creative effects → 8. Export: Web, print optimization",
    "course": "Photoshop CC Essential Training (LinkedIn Learning) or Photoshop for UX Design (Udemy)"
  },
  "Adobe Illustrator": {
    "path": "1. Illustrator Basics: Vector graphics, interface → 2. Drawing Tools: Pen tool, shapes → 3. Typography: Text tools, effects → 4. Colors: Swatches, gradients → 5. Symbols: Reusable graphics → 6. Effects: Appearances, styles → 7. Layout: Artboards, alignment → 8. Export: Multiple formats",
    "course": "Illustrator CC Essential Training (LinkedIn Learning) or Vector Illustration (Skillshare)"
  },
  "InVision": {
    "path": "1. InVision Basics: Prototyping platform → 2. Screens: Upload, organize → 3. Hotspots: Interactive areas → 4. Transitions: Screen connections → 5. Collaboration: Comments, feedback → 6. Presentation: Client sharing → 7. Inspect: Developer handoff → 8. Design System Manager: Component library",
    "course": "InVision Prototyping (InVision Learn) or Prototype with InVision (Udemy)"
  },
  "Usability Testing": {
    "path": "1. Testing Methods: Moderated, unmoderated → 2. Test Planning: Objectives, scenarios → 3. Participant Recruitment: Target users → 4. Test Execution: Facilitation, observation → 5. Data Collection: Metrics, feedback → 6. Analysis: Patterns, insights → 7. Reporting: Findings, recommendations → 8. Iteration: Design improvements",
    "course": "Usability Testing (IxDF) or User Testing Course (Coursera - University of Minnesota)"
  },
  "Heuristic Evaluation": {
    "path": "1. Nielsen's Heuristics: 10 usability principles → 2. Evaluation Process: Systematic review → 3. Severity Rating: Problem prioritization → 4. Documentation: Issues, recommendations → 5. Expert Review: Multiple evaluators → 6. Comparison: Before/after analysis → 7. Action Planning: Fix prioritization → 8. Validation: User testing confirmation",
    "course": "Heuristic Evaluation (IxDF) or UX Evaluation Methods (Coursera)"
  },
  "Accessibility Standards": {
    "path": "1. WCAG Guidelines: A, AA, AAA levels → 2. Perceivable: Alt text, captions → 3. Operable: Keyboard navigation → 4. Understandable: Clear language → 5. Robust: Compatible code → 6. Testing Tools: Screen readers, validators → 7. Design Considerations: Color, contrast → 8. Legal Requirements: ADA compliance",
    "course": "Web Accessibility (Coursera - University of Minnesota) or Accessibility (IxDF)"
  },
  "Responsive Design": {
    "path": "1. Mobile-First: Design approach → 2. Breakpoints: Screen size ranges → 3. Flexible Grids: Fluid layouts → 4. Flexible Images: Scalable media → 5. Media Queries: CSS responsive rules → 6. Touch Design: Finger-friendly interfaces → 7. Performance: Mobile optimization → 8. Testing: Cross-device validation",
    "course": "Responsive Web Design (freeCodeCamp) or Advanced Responsive Design (Frontend Masters)"
  },
  "Mobile Design": {
    "path": "1. Mobile Platforms: iOS, Android guidelines → 2. Screen Sizes: Adaptation strategies → 3. Touch Interactions: Gestures, feedback → 4. Navigation: Tabs, hamburger menus → 5. Typography: Readable text sizes → 6. Performance: Loading, optimization → 7. Offline: Connectivity considerations → 8. Testing: Device testing",
    "course": "Mobile Design Best Practices (Google UX Design Certificate) or iOS Design (Apple Human Interface Guidelines)"
  },
  "Web Design": {
    "path": "1. Web Fundamentals: HTML, CSS basics → 2. Layout: Grid, flexbox → 3. Typography: Web fonts, hierarchy → 4. Color: Web-safe colors, accessibility → 5. Images: Optimization, formats → 6. Navigation: Menus, breadcrumbs → 7. Forms: User-friendly design → 8. Performance: Speed optimization",
    "course": "Web Design for Everybody (Coursera - University of Michigan) or Modern Web Design (Udemy)"
  },
  "UX Writing": {
    "path": "1. UX Writing Basics: Microcopy, voice → 2. Content Strategy: Information architecture → 3. User Journey: Content mapping → 4. Interface Copy: Buttons, labels → 5. Error Messages: Helpful, clear → 6. Onboarding: Progressive disclosure → 7. Accessibility: Plain language → 8. Testing: Content validation",
    "course": "UX Writing Hub (Free course) or Content Design (Sarah Richards)"
  },
  "Spark": {
    "path": "1. Spark Basics: RDDs, DataFrames → 2. Spark SQL: Structured data processing → 3. Data Loading: Various formats → 4. Transformations: map, filter, join → 5. Actions: collect, save, count → 6. Performance: Caching, partitioning → 7. Streaming: Real-time processing → 8. Cluster: Deployment, monitoring",
    "course": "Apache Spark Essential Training (LinkedIn Learning) or Spark and Python for Big Data (Udemy)"
  },
  "Hadoop": {
    "path": "1. Hadoop Ecosystem: HDFS, MapReduce, YARN → 2. HDFS: Distributed file system → 3. MapReduce: Parallel processing → 4. Hive: SQL on Hadoop → 5. Pig: Data flow language → 6. HBase: NoSQL database → 7. Administration: Cluster management → 8. Integration: Spark, Kafka",
    "course": "Hadoop Platform and Application Framework (Coursera - UC San Diego) or The Ultimate Hands-On Hadoop (Udemy)"
  },
  "ETL Pipelines": {
    "path": "1. ETL Concepts: Extract, Transform, Load → 2. Data Sources: APIs, databases, files → 3. Data Quality: Validation, cleansing → 4. Transformations: Aggregations, joins → 5. Scheduling: Cron, workflow managers → 6. Monitoring: Pipeline health → 7. Error Handling: Retry, alerting → 8. Testing: Data validation",
    "course": "Building Data Engineering Pipelines (Coursera - Google Cloud) or ETL with Python (DataCamp)"
  },
  "Airflow": {
    "path": "1. Airflow Basics: DAGs, operators → 2. Task Dependencies: Workflow design → 3. Scheduling: Cron expressions → 4. Operators: Bash, Python, SQL → 5. Connections: External systems → 6. Monitoring: Web UI, logging → 7. Scaling: Celery, Kubernetes → 8. Best Practices: Testing, deployment",
    "course": "Apache Airflow: The Complete Guide (Udemy) or Data Engineering with Apache Airflow (DataTalks.Club)"
  },
  "Kafka": {
    "path": "1. Kafka Concepts: Topics, partitions, brokers → 2. Producers: Message publishing → 3. Consumers: Message consumption → 4. Streams: Real-time processing → 5. Connect: Data integration → 6. Schema Registry: Data schemas → 7. Administration: Cluster management → 8. Performance: Tuning, monitoring",
    "course": "Apache Kafka Series (Udemy - Stephane Maarek) or Kafka: The Definitive Guide (O'Reilly Book)"
  },
  "GCP": {
    "path": "1. GCP Basics: Projects, IAM, billing → 2. Compute: Compute Engine, App Engine → 3. Storage: Cloud Storage, persistent disks → 4. Databases: Cloud SQL, Firestore → 5. Big Data: BigQuery, Dataflow → 6. ML: AI Platform, AutoML → 7. Networking: VPC, load balancers → 8. Monitoring: Stackdriver",
    "course": "Google Cloud Platform Fundamentals (Coursera - Google Cloud) or GCP Associate Cloud Engineer (A Cloud Guru)"
  },
  "Azure": {
    "path": "1. Azure Basics: Subscriptions, resource groups → 2. Compute: Virtual machines, App Service → 3. Storage: Blob storage, file shares → 4. Databases: SQL Database, Cosmos DB → 5. Analytics: Synapse, Data Factory → 6. AI/ML: Cognitive Services, ML Studio → 7. Networking: Virtual networks → 8. Security: Key Vault, Active Directory",
    "course": "Microsoft Azure Fundamentals (Microsoft Learn) or Azure Administrator (Pluralsight)"
  },
  "BigQuery": {
    "path": "1. BigQuery Basics: Datasets, tables → 2. SQL: Standard SQL syntax → 3. Data Loading: Batch, streaming → 4. Query Optimization: Partitioning, clustering → 5. Data Types: Nested, repeated fields → 6. Functions: Built-in, user-defined → 7. ML: BigQuery ML → 8. Cost Management: Monitoring, optimization",
    "course": "BigQuery for Data Analysts (Coursera - Google Cloud) or BigQuery Fundamentals (A Cloud Guru)"
  },
  "Data Modeling": {
    "path": "1. Conceptual Modeling: Entities, relationships → 2. Logical Modeling: Normalization → 3. Physical Modeling: Tables, indexes → 4. Dimensional Modeling: Facts, dimensions → 5. Data Vault: Hubs, links, satellites → 6. NoSQL Modeling: Document, graph → 7. Schema Design: Evolution, versioning → 8. Tools: ERwin, Lucidchart",
    "course": "Data Modeling Fundamentals (Pluralsight) or The Data Warehouse Toolkit (Kimball Book)"
  },
  "Data Warehousing": {
    "path": "1. DW Concepts: OLTP vs OLAP → 2. Architecture: Staging, integration, presentation → 3. Dimensional Modeling: Star, snowflake → 4. ETL: Data integration processes → 5. Slowly Changing Dimensions → 6. Data Quality: Validation, cleansing → 7. Performance: Indexing, partitioning → 8. Modern: Cloud data warehouses",
    "course": "Data Warehousing for Business Intelligence (Coursera - University of Colorado) or Modern Data Warehouse (Udemy)"
  },
  "Redshift": {
    "path": "1. Redshift Basics: Architecture, clusters → 2. Data Loading: COPY command → 3. Query Performance: Distribution, sorting → 4. Compression: Column encoding → 5. Maintenance: VACUUM, ANALYZE → 6. Security: Encryption, VPC → 7. Monitoring: Performance insights → 8. Integration: S3, EMR, Glue",
    "course": "Amazon Redshift Deep Dive (A Cloud Guru) or Data Warehousing with Amazon Redshift (AWS Training)"
  },
  "Snowflake": {
    "path": "1. Snowflake Architecture: Multi-cluster, separation → 2. Data Loading: Bulk, continuous → 3. Virtual Warehouses: Scaling, suspension → 4. Data Sharing: Secure sharing → 5. Time Travel: Historical queries → 6. Cloning: Zero-copy cloning → 7. Security: End-to-end encryption → 8. Performance: Query optimization",
    "course": "Snowflake Essentials (Snowflake University) or Complete Snowflake Masterclass (Udemy)"
  },
  "S3": {
    "path": "1. S3 Basics: Buckets, objects → 2. Storage Classes: Standard, IA, Glacier → 3. Security: IAM, bucket policies → 4. Versioning: Object versions → 5. Lifecycle: Automatic transitions → 6. Replication: Cross-region → 7. Performance: Multipart upload → 8. Integration: CloudFront, Lambda",
    "course": "Amazon S3 Masterclass (Udemy) or AWS Storage Services (A Cloud Guru)"
  },
  "Glue": {
    "path": "1. Glue Concepts: Crawlers, catalog → 2. Data Catalog: Metadata management → 3. ETL Jobs: Python, Scala → 4. Crawlers: Schema discovery → 5. Transformations: Built-in functions → 6. Scheduling: Triggers, workflows → 7. Monitoring: Job metrics → 8. Integration: S3, Redshift, RDS",
    "course": "AWS Glue Tutorial (AWS Documentation) or AWS Data Analytics (A Cloud Guru)"
  },
  "Athena": {
    "path": "1. Athena Basics: Serverless queries → 2. Data Sources: S3, data formats → 3. SQL: Standard SQL syntax → 4. Partitioning: Performance optimization → 5. Compression: Cost reduction → 6. Workgroups: Query management → 7. Federation: External data sources → 8. Integration: QuickSight, Glue",
    "course": "Amazon Athena Deep Dive (A Cloud Guru) or Querying Data with Amazon Athena (AWS Training)"
  },
  "Parquet": {
    "path": "1. Parquet Format: Columnar storage → 2. Schema: Nested data structures → 3. Compression: Snappy, GZIP → 4. Encoding: Dictionary, RLE → 5. Partitioning: Directory structure → 6. Tools: PyArrow, Spark → 7. Performance: Query optimization → 8. Evolution: Schema evolution",
    "course": "Apache Parquet Tutorial (YouTube) or Working with Parquet Files (DataCamp)"
  },
  "Delta Lake": {
    "path": "1. Delta Lake Concepts: ACID transactions → 2. Versioning: Time travel → 3. Schema Evolution: Automatic handling → 4. Upserts: MERGE operations → 5. Streaming: Real-time processing → 6. Optimization: Z-ordering, compaction → 7. Integration: Spark, MLflow → 8. Governance: Data lineage",
    "course": "Delta Lake Tutorial (Databricks Academy) or Building Data Lakes with Delta Lake (Udemy)"
  },
  "DBT": {
    "path": "1. DBT Concepts: Analytics engineering → 2. Models: SQL transformations → 3. Testing: Data quality checks → 4. Documentation: Model documentation → 5. Macros: Reusable code → 6. Packages: External dependencies → 7. Deployment: Production workflows → 8. Lineage: Data dependencies",
    "course": "dbt Fundamentals (dbt Learn) or Analytics Engineering with dbt (DataTalks.Club)"
  },
  "Kubernetes": {
    "path": "1. K8s Basics: Pods, services, deployments → 2. Architecture: Master, nodes, etcd → 3. Networking: Services, ingress → 4. Storage: Volumes, persistent volumes → 5. Configuration: ConfigMaps, secrets → 6. Scaling: HPA, VPA → 7. Security: RBAC, network policies → 8. Monitoring: Prometheus, Grafana",
    "course": "Kubernetes for Absolute Beginners (Udemy - Mumshad Mannambeth) or Kubernetes the Hard Way (Kelsey Hightower)"
  },
  "Product Strategy": {
    "path": "1. Strategy Fundamentals: Vision, mission → 2. Market Analysis: Competition, trends → 3. Customer Research: Needs, pain points → 4. Value Proposition: Unique benefits → 5. Product-Market Fit: Validation → 6. Prioritization: Feature ranking → 7. Metrics: KPIs, success measures → 8. Communication: Stakeholder alignment",
    "course": "Product Strategy (Coursera - University of Virginia) or Become a Product Manager (Udemy)"
  },
  "Roadmapping": {
    "path": "1. Roadmap Types: Strategy, feature, technology → 2. Timeline Planning: Quarterly, annual → 3. Prioritization: Impact vs effort → 4. Stakeholder Input: Requirements gathering → 5. Resource Planning: Team capacity → 6. Dependencies: Technical, business → 7. Communication: Visual presentation → 8. Iteration: Regular updates",
    "course": "Product Roadmaps (ProductPlan) or Product Management Fundamentals (LinkedIn Learning)"
  },
  "Agile Methodologies": {
    "path": "1. Agile Principles: Manifesto, values → 2. Scrum Framework: Roles, events → 3. Sprint Planning: Backlog, estimation → 4. Daily Standups: Progress, blockers → 5. Sprint Review: Demo, feedback → 6. Retrospectives: Process improvement → 7. Kanban: Visual workflow → 8. Scaling: SAFe, LeSS",
    "course": "Agile Development (Coursera - University of Virginia) or Scrum Master Certification (Scrum Alliance)"
  },
  "Scrum": {
    "path": "1. Scrum Framework: Roles, events, artifacts → 2. Product Owner: Backlog management → 3. Scrum Master: Process facilitation → 4. Development Team: Cross-functional → 5. Sprint Planning: Goal setting → 6. Daily Scrum: Synchronization → 7. Sprint Review: Inspect and adapt → 8. Sprint Retrospective: Continuous improvement",
    "course": "Professional Scrum Master (Scrum.org) or Scrum Fundamentals (Pluralsight)"
  },
  "Stakeholder Management": {
    "path": "1. Stakeholder Identification: Mapping influence → 2. Communication Planning: Frequency, format → 3. Expectation Management: Alignment → 4. Conflict Resolution: Negotiation → 5. Influence Strategies: Persuasion → 6. Feedback Collection: Regular check-ins → 7. Reporting: Status updates → 8. Relationship Building: Trust, credibility",
    "course": "Stakeholder Management (PMI) or Influencing People (Dale Carnegie)"
  },
  "User Stories": {
    "path": "1. Story Format: As a, I want, so that → 2. Acceptance Criteria: Definition of done → 3. Story Mapping: User journey → 4. Estimation: Story points, planning poker → 5. Splitting: Vertical slicing → 6. Backlog Management: Prioritization → 7. Refinement: Grooming sessions → 8. Testing: Acceptance testing",
    "course": "User Story Mapping (Jeff Patton Book) or Writing Effective User Stories (Coursera)"
  },
  "Market Research": {
    "path": "1. Research Methods: Primary, secondary → 2. Competitor Analysis: SWOT, positioning → 3. Customer Surveys: Design, analysis → 4. Focus Groups: Qualitative insights → 5. Market Sizing: TAM, SAM, SOM → 6. Trend Analysis: Industry trends → 7. Data Sources: Industry reports → 8. Synthesis: Actionable insights",
    "course": "Market Research (Coursera - UC Davis) or Competitive Analysis (Product School)"
  },
  "Data Analysis": {
    "path": "1. Data Types: Quantitative, qualitative → 2. Statistical Analysis: Descriptive, inferential → 3. Visualization: Charts, dashboards → 4. Tools: Excel, SQL, Python → 5. A/B Testing: Experimental design → 6. Cohort Analysis: User behavior → 7. Funnel Analysis: Conversion → 8. Insights: Actionable recommendations",
    "course": "Data Analysis for Business (Coursera - Duke University) or Google Data Analytics Certificate"
  },
  "A/B Testing": {
    "path": "1. Experiment Design: Hypothesis, variables → 2. Sample Size: Statistical power → 3. Randomization: Control, treatment → 4. Metrics: Primary, secondary → 5. Statistical Significance: p-values → 6. Effect Size: Practical significance → 7. Tools: Optimizely, Google Optimize → 8. Analysis: Results interpretation",
    "course": "A/B Testing (Udacity) or Experimentation for Improvement (Coursera - McMaster)"
  },
  "Customer Interviews": {
    "path": "1. Interview Planning: Objectives, questions → 2. Participant Recruitment: Target segments → 3. Interview Techniques: Open-ended questions → 4. Active Listening: Probing follow-ups → 5. Bias Avoidance: Leading questions → 6. Documentation: Notes, recordings → 7. Analysis: Pattern identification → 8. Synthesis: Customer insights",
    "course": "Talking to Humans (Steve Blank Book) or Customer Interview Techniques (Product School)"
  },
  "JIRA": {
    "path": "1. JIRA Basics: Projects, issues → 2. Issue Types: Stories, bugs, tasks → 3. Workflows: Status transitions → 4. Boards: Scrum, Kanban → 5. Reporting: Burndown, velocity → 6. Customization: Fields, screens → 7. Administration: Permissions, schemes → 8. Integration: Confluence, Bitbucket",
    "course": "JIRA Fundamentals (Atlassian University) or Mastering JIRA (Udemy)"
  },
  "Confluence": {
    "path": "1. Confluence Basics: Spaces, pages → 2. Content Creation: Rich text editor → 3. Collaboration: Comments, mentions → 4. Templates: Standardized formats → 5. Macros: Dynamic content → 6. Organization: Labels, navigation → 7. Permissions: Space, page security → 8. Integration: JIRA, other tools",
    "course": "Confluence Fundamentals (Atlassian University) or Effective Documentation (Confluence)"
  },
  "Product Lifecycle Management": {
    "path": "1. Lifecycle Stages: Introduction, growth, maturity → 2. Strategy: Stage-specific approaches → 3. Metrics: Stage-appropriate KPIs → 4. Innovation: New product development → 5. Portfolio Management: Product mix → 6. Sunset Planning: End-of-life → 7. Resource Allocation: Investment decisions → 8. Market Dynamics: Competitive response",
    "course": "Product Lifecycle Management (Coursera) or Strategic Product Management (Product School)"
  },
  "KPI Definition": {
    "path": "1. KPI Fundamentals: Objectives, measures → 2. SMART Criteria: Specific, measurable → 3. Leading vs Lagging: Predictive indicators → 4. Business Alignment: Strategic objectives → 5. Metric Selection: Relevant measures → 6. Benchmarking: Industry standards → 7. Tracking: Dashboards, reports → 8. Action Planning: Performance improvement",
    "course": "KPI Development (Coursera) or Data-Driven Product Management (Product School)"
  },
  "Go-to-Market Strategy": {
    "path": "1. Market Analysis: Target segments → 2. Value Proposition: Unique benefits → 3. Positioning: Competitive differentiation → 4. Pricing Strategy: Models, tiers → 5. Distribution Channels: Sales, partnerships → 6. Marketing: Campaigns, messaging → 7. Launch Planning: Timeline, milestones → 8. Success Metrics: Revenue, adoption",
    "course": "Go-to-Market Strategy (Product School) or Launch (Jeff Walker Book)"
  },
  "Competitive Analysis": {
    "path": "1. Competitor Identification: Direct, indirect → 2. Feature Analysis: Functionality comparison → 3. Pricing Analysis: Models, positioning → 4. Market Position: Strengths, weaknesses → 5. Customer Feedback: Reviews, testimonials → 6. SWOT Analysis: Strategic assessment → 7. Monitoring: Ongoing intelligence → 8. Strategic Response: Competitive moves",
    "course": "Competitive Intelligence (Coursera) or Market Research for Competitive Analysis (Udemy)"
  },
  "C#": {
    "path": "1. C# Basics: Syntax, variables, types → 2. OOP: Classes, inheritance, polymorphism → 3. Collections: Lists, dictionaries, arrays → 4. Exception Handling: try-catch, custom exceptions →"
  },
  
  "Go": {
    "path": "1. Syntax Basics: Variables, types, functions → 2. Goroutines: Concurrent programming → 3. Channels: Communication between goroutines → 4. Packages: Modules, imports, exports → 5. Error Handling: Error interface, panic/recover → 6. Interfaces: Implicit implementation → 7. Structs: Data structures, methods → 8. Testing: Unit tests, benchmarks → 9. HTTP Programming: Web servers, clients → 10. Standard Library: fmt, io, net packages",
    "course": "Go: The Complete Developer's Guide (Udemy - Stephen Grider) or Learn Go Programming (Codecademy)"
  },
  "Algorithms": {
    "path": "1. Time/Space Complexity: Big O notation → 2. Sorting: Quicksort, mergesort, heapsort → 3. Searching: Binary search, DFS, BFS → 4. Dynamic Programming: Memoization, tabulation → 5. Greedy Algorithms: Optimization problems → 6. Graph Algorithms: Shortest path, MST → 7. String Algorithms: Pattern matching, KMP → 8. Divide and Conquer: Problem decomposition → 9. Backtracking: Constraint satisfaction",
    "course": "Algorithms Specialization (Coursera - Stanford) or Introduction to Algorithms (MIT OpenCourseWare)"
  },
  "Operating Systems": {
    "path": "1. OS Concepts: Processes, threads, scheduling → 2. Memory Management: Virtual memory, paging → 3. File Systems: Inodes, directories, permissions → 4. Synchronization: Mutexes, semaphores → 5. Deadlocks: Detection, prevention, avoidance → 6. I/O Systems: Device drivers, interrupt handling → 7. System Calls: Kernel interface → 8. Security: Access control, authentication → 9. Distributed Systems: Network protocols",
    "course": "Operating Systems (Coursera - University of California San Diego) or Operating Systems: Three Easy Pieces (Book + Videos)"
  },
  "Concurrency": {
    "path": "1. Thread Fundamentals: Creation, lifecycle, scheduling → 2. Synchronization Primitives: Locks, semaphores, monitors → 3. Atomic Operations: Compare-and-swap, memory barriers → 4. Producer-Consumer: Bounded buffer problem → 5. Reader-Writer: Concurrent access patterns → 6. Deadlock Prevention: Ordering, timeouts → 7. Lock-free Programming: Wait-free data structures → 8. Parallel Algorithms: Fork-join, map-reduce → 9. Memory Models: Consistency, ordering",
    "course": "Parallel Programming (Coursera - University of Illinois) or Java Concurrency in Practice (Book)"
  },
  "Machine Learning Basics": {
    "path": "1. Supervised Learning: Classification, regression → 2. Unsupervised Learning: Clustering, dimensionality reduction → 3. Model Evaluation: Cross-validation, metrics → 4. Feature Engineering: Selection, transformation → 5. Overfitting: Bias-variance tradeoff → 6. Linear Models: Regression, logistic regression → 7. Tree-based Models: Decision trees, random forest → 8. Neural Networks: Perceptron, backpropagation → 9. Model Selection: Hyperparameter tuning",
    "course": "Machine Learning (Coursera - Andrew Ng) or Introduction to Statistical Learning (Stanford Online)"
  },
  "Big-O Analysis": {
    "path": "1. Growth Functions: Constant, linear, quadratic → 2. Asymptotic Notation: Big O, Omega, Theta → 3. Time Complexity: Algorithm analysis → 4. Space Complexity: Memory usage analysis → 5. Best/Worst/Average Case: Scenario analysis → 6. Recursive Algorithms: Recurrence relations → 7. Amortized Analysis: Average performance → 8. Comparison: Algorithm selection criteria → 9. Optimization: Complexity reduction techniques",
    "course": "Algorithms and Data Structures (Coursera - UC San Diego) or Algorithm Design Manual (Book)"
  },
  "Coding Interviews": {
    "path": "1. Problem-solving Framework: Understand, plan, code, test → 2. Data Structures: Arrays, linked lists, trees, graphs → 3. Algorithm Patterns: Two pointers, sliding window → 4. Dynamic Programming: Optimal substructure → 5. System Design: Scalability, trade-offs → 6. Behavioral Questions: STAR method → 7. Code Quality: Clean, readable, efficient → 8. Time Management: Prioritization, optimization → 9. Mock Interviews: Practice, feedback",
    "course": "Cracking the Coding Interview (Book) or LeetCode Premium or AlgoExpert"
  },
  "Software Engineering Principles": {
    "path": "1. SOLID Principles: Single responsibility, open-closed → 2. Design Patterns: Creational, structural, behavioral → 3. Code Quality: Clean code, refactoring → 4. Version Control: Git workflows, branching → 5. Testing: Unit, integration, end-to-end → 6. Documentation: API docs, README files → 7. Code Reviews: Best practices, feedback → 8. Agile Methodologies: Scrum, Kanban → 9. Technical Debt: Management, prevention",
    "course": "Software Engineering (Coursera - University of Alberta) or Clean Code (Book - Robert Martin)"
  },
  ".NET": {
    "path": "1. .NET Ecosystem: Framework vs Core vs 5+ → 2. C# Fundamentals: OOP, generics, LINQ → 3. ASP.NET Core: MVC, Web API, middleware → 4. Entity Framework: ORM, migrations, queries → 5. Dependency Injection: IoC container, lifetimes → 6. Authentication: Identity, JWT, OAuth → 7. Testing: xUnit, Moq, integration tests → 8. Deployment: Docker, Azure, CI/CD → 9. Performance: Profiling, optimization",
    "course": "Complete ASP.NET Core Developer Course (Udemy) or .NET Application Architecture Guides (Microsoft)"
  },
  "Azure": {
    "path": "1. Cloud Fundamentals: IaaS, PaaS, SaaS → 2. Compute Services: VMs, App Service, Functions → 3. Storage: Blob, File, Table, Queue → 4. Databases: SQL Database, Cosmos DB → 5. Networking: Virtual networks, load balancers → 6. Identity: Active Directory, authentication → 7. DevOps: Azure DevOps, CI/CD pipelines → 8. Monitoring: Application Insights, Log Analytics → 9. Security: Key Vault, security center",
    "course": "Microsoft Azure Fundamentals (Microsoft Learn) or Azure Developer Associate (Pluralsight)"
  },
  "Cloud Computing": {
    "path": "1. Cloud Models: Public, private, hybrid → 2. Service Models: IaaS, PaaS, SaaS → 3. Virtualization: Hypervisors, containers → 4. Scalability: Auto-scaling, load balancing → 5. Storage: Object, block, file storage → 6. Networking: VPC, CDN, DNS → 7. Security: IAM, encryption, compliance → 8. Cost Management: Pricing models, optimization → 9. Multi-cloud: Vendor lock-in, portability",
    "course": "Cloud Computing Concepts (Coursera - University of Illinois) or AWS Cloud Foundations (A Cloud Guru)"
  },
  "Multithreading": {
    "path": "1. Thread Lifecycle: Creation, execution, termination → 2. Thread Safety: Race conditions, synchronization → 3. Locks: Mutex, read-write locks → 4. Thread Pools: Worker threads, task queues → 5. Atomic Operations: Thread-safe primitives → 6. Producer-Consumer: Coordination patterns → 7. Deadlock Prevention: Lock ordering, timeouts → 8. Performance: Context switching, cache locality → 9. Debugging: Race condition detection",
    "course": "Java Concurrency in Practice (Book) or Multithreading in Python (Real Python)"
  },
  "Windows OS": {
    "path": "1. Architecture: Kernel, user mode, system calls → 2. Process Management: Creation, scheduling → 3. Memory Management: Virtual memory, paging → 4. File System: NTFS, permissions, registry → 5. Device Drivers: Hardware abstraction → 6. Security: ACL, UAC, Windows Defender → 7. Networking: TCP/IP stack, WinSock → 8. Services: Windows services, SCM → 9. PowerShell: Automation, scripting",
    "course": "Windows Internals (Book) or Windows System Administration (Pluralsight)"
  },
  "Debugging Tools": {
    "path": "1. Debugger Basics: Breakpoints, stepping, inspection → 2. Memory Debugging: Heap analysis, leaks → 3. Performance Profiling: CPU, memory usage → 4. Log Analysis: Structured logging, correlation → 5. Network Debugging: Packet capture, analysis → 6. Distributed Tracing: Request correlation → 7. Static Analysis: Code quality, vulnerabilities → 8. Dynamic Analysis: Runtime behavior → 9. Post-mortem: Crash dumps, root cause analysis",
    "course": "Debugging Techniques (Pluralsight) or Effective Debugging (Book)"
  },
  "Design Patterns": {
    "path": "1. Creational Patterns: Singleton, factory, builder → 2. Structural Patterns: Adapter, decorator, facade → 3. Behavioral Patterns: Observer, strategy, command → 4. MVC Pattern: Model-view-controller → 5. Dependency Injection: IoC, service locator → 6. Repository Pattern: Data access abstraction → 7. Anti-patterns: Common mistakes to avoid → 8. Pattern Selection: When to use which pattern → 9. Modern Patterns: Microservices, event sourcing",
    "course": "Design Patterns (Coursera - University of Alberta) or Head First Design Patterns (Book)"
  },
  "Hack": {
    "path": "1. Language Basics: Syntax, types, functions → 2. Type System: Gradual typing, generics → 3. Async Programming: Async/await patterns → 4. Collections: Arrays, maps, sets → 5. Object-Oriented: Classes, interfaces, traits → 6. Functional Programming: Lambdas, immutability → 7. Error Handling: Exceptions, option types → 8. Testing: HackTest framework → 9. Performance: Optimization, profiling",
    "course": "Hack Language Tutorial (Facebook/Meta Documentation) or Learn Hack Programming (HackLang.org)"
  },
  "React": {
    "path": "1. JSX Syntax: Components, elements, expressions → 2. Components: Functional, class components → 3. State Management: useState, useReducer → 4. Props: Data flow, prop types → 5. Event Handling: Synthetic events, callbacks → 6. Lifecycle: useEffect, cleanup → 7. Context API: Global state management → 8. Routing: React Router, navigation → 9. Performance: Memoization, lazy loading → 10. Testing: Jest, React Testing Library",
    "course": "React - The Complete Guide (Udemy - Maximilian Schwarzmüller) or React Fundamentals (React Training)"
  },
  "JavaScript": {
    "path": "1. Language Fundamentals: Variables, types, functions → 2. DOM Manipulation: Selecting, modifying elements → 3. Event Handling: Listeners, event delegation → 4. Asynchronous Programming: Promises, async/await → 5. ES6+ Features: Arrow functions, destructuring → 6. Closures: Scope, lexical environment → 7. Prototypes: Inheritance, prototype chain → 8. Modules: Import/export, bundling → 9. Error Handling: Try/catch, error objects → 10. Testing: Jest, unit testing",
    "course": "The Complete JavaScript Course (Udemy - Jonas Schmedtmann) or JavaScript: Understanding the Weird Parts (Udemy)"
  },
  "API Development": {
    "path": "1. REST Principles: Resources, HTTP methods → 2. API Design: Endpoints, status codes → 3. Authentication: JWT, OAuth, API keys → 4. Documentation: OpenAPI, Swagger → 5. Versioning: URL, header, parameter → 6. Rate Limiting: Throttling, quotas → 7. Error Handling: Consistent error responses → 8. Testing: Postman, automated testing → 9. Security: Input validation, CORS → 10. Monitoring: Logging, metrics, alerting",
    "course": "REST API Design, Development & Management (Udemy) or API Design Patterns (Book)"
  },
  "GraphQL": {
    "path": "1. GraphQL Basics: Schema, types, queries → 2. Schema Definition: Types, fields, resolvers → 3. Queries: Fetching data, nested queries → 4. Mutations: Data modifications → 5. Subscriptions: Real-time updates → 6. Resolvers: Data fetching logic → 7. Apollo Server: GraphQL server setup → 8. Apollo Client: Frontend integration → 9. Caching: Query caching, normalization → 10. Security: Query depth limiting, authentication",
    "course": "Modern GraphQL Bootcamp (Udemy - Andrew Mead) or GraphQL Fundamentals (Apollo GraphQL)"
  },
  "Big Data Tools": {
    "path": "1. Hadoop Ecosystem: HDFS, MapReduce, YARN → 2. Apache Spark: RDDs, DataFrames, Spark SQL → 3. Data Storage: HBase, Cassandra, MongoDB → 4. Stream Processing: Kafka, Storm, Flink → 5. Data Warehousing: Hive, Impala, Presto → 6. Workflow Management: Airflow, Oozie → 7. Data Formats: Parquet, Avro, ORC → 8. Cluster Management: YARN, Mesos, Kubernetes → 9. Monitoring: Ganglia, Nagios, ELK stack",
    "course": "Big Data Specialization (Coursera - UC San Diego) or Complete Big Data & Hadoop Course (Udemy)"
  },
  "Communication Skills": {
    "path": "1. Active Listening: Understanding, empathy → 2. Clear Expression: Concise, structured communication → 3. Technical Writing: Documentation, proposals → 4. Presentation Skills: Structure, delivery, visuals → 5. Meeting Facilitation: Agenda, participation → 6. Conflict Resolution: Mediation, compromise → 7. Cross-cultural Communication: Awareness, adaptation → 8. Feedback: Giving/receiving constructive feedback → 9. Storytelling: Narrative, persuasion",
    "course": "Communication Skills (Coursera - University of California San Diego) or Technical Communication (edX - MIT)"
  },
  "Cross-Team Collaboration": {
    "path": "1. Stakeholder Management: Identification, engagement → 2. Project Coordination: Timeline, dependencies → 3. Knowledge Sharing: Documentation, presentations → 4. Agile Ceremonies: Standups, retrospectives → 5. Tools: Slack, JIRA, Confluence → 6. Negotiation: Win-win solutions → 7. Cultural Awareness: Team dynamics → 8. Remote Collaboration: Virtual meetings, async communication → 9. Mentoring: Knowledge transfer, guidance",
    "course": "Collaboration and Teamwork Skills (Coursera) or Leading Teams (Harvard Business School Online)"
  },
  "Scala": {
    "path": "1. Functional Programming: Immutability, higher-order functions → 2. Object-Oriented: Classes, traits, inheritance → 3. Pattern Matching: Case classes, extractors → 4. Collections: Lists, maps, sets, operations → 5. Concurrency: Actors, futures, parallel collections → 6. Type System: Generics, variance, implicits → 7. SBT: Build tool, dependencies → 8. Testing: ScalaTest, property-based testing → 9. Frameworks: Akka, Play Framework",
    "course": "Functional Programming in Scala (Coursera - EPFL) or Scala & Functional Programming Essentials (Udemy)"
  },
  "Kafka": {
    "path": "1. Messaging Concepts: Producers, consumers, topics → 2. Architecture: Brokers, partitions, replication → 3. Producer API: Sending messages, serialization → 4. Consumer API: Reading messages, offset management → 5. Kafka Streams: Stream processing, transformations → 6. Kafka Connect: Data integration, connectors → 7. Schema Registry: Schema evolution, compatibility → 8. Operations: Monitoring, tuning, maintenance → 9. Security: Authentication, authorization, encryption",
    "course": "Apache Kafka Series (Udemy - Stephane Maarek) or Kafka: The Definitive Guide (Book)"
  },
  "Finagle": {
    "path": "1. Service Framework: Client-server abstraction → 2. Protocol Support: HTTP, Thrift, MySQL → 3. Load Balancing: Strategies, health checking → 4. Circuit Breakers: Failure handling, recovery → 5. Filters: Request/response transformation → 6. Metrics: Monitoring, observability → 7. Service Discovery: Dynamic service location → 8. Timeouts: Request deadlines, retries → 9. Testing: Service mocking, integration tests",
    "course": "Finagle Documentation (Twitter) or Effective Scala (Twitter Engineering Blog)"
  },
  "MySQL": {
    "path": "1. Database Fundamentals: Tables, relationships, constraints → 2. SQL Queries: SELECT, JOIN, subqueries → 3. Data Manipulation: INSERT, UPDATE, DELETE → 4. Indexing: B-tree, performance optimization → 5. Transactions: ACID properties, isolation levels → 6. Stored Procedures: Functions, triggers → 7. Replication: Master-slave, high availability → 8. Performance Tuning: Query optimization, caching → 9. Backup/Recovery: Data protection strategies",
    "course": "MySQL Database Administration (Udemy) or MySQL for Developers (PlanetScale)"
  },
  "Thrift": {
    "path": "1. IDL Basics: Interface Definition Language → 2. Data Types: Primitives, collections, structs → 3. Service Definition: Methods, exceptions → 4. Code Generation: Multi-language support → 5. Transport Layer: Socket, HTTP, memory → 6. Protocol Layer: Binary, compact, JSON → 7. Server Types: Simple, threaded, non-blocking → 8. Client Usage: Synchronous, asynchronous → 9. Versioning: Schema evolution, compatibility",
    "course": "Apache Thrift Tutorial (Apache Documentation) or Thrift: The Missing Guide (Book)"
  },
  "Big Data Processing": {
    "path": "1. Batch Processing: MapReduce, Spark batch jobs → 2. Stream Processing: Real-time data processing → 3. Data Partitioning: Horizontal, vertical partitioning → 4. Distributed Computing: Parallelization strategies → 5. Data Formats: Parquet, Avro, ORC → 6. Data Compression: Algorithms, trade-offs → 7. Memory Management: Caching, spilling → 8. Fault Tolerance: Checkpointing, recovery → 9. Performance Optimization: Tuning, profiling",
    "course": "Big Data Analysis with Apache Spark (edX - UC Berkeley) or Spark: The Definitive Guide (Book)"
  },
  "Monitoring": {
    "path": "1. Metrics Collection: System, application metrics → 2. Alerting: Thresholds, escalation policies → 3. Dashboards: Visualization, KPIs → 4. Log Management: Centralized logging, analysis → 5. Distributed Tracing: Request flow tracking → 6. Health Checks: Endpoint monitoring → 7. SLA/SLI: Service level objectives → 8. Incident Response: On-call, post-mortems → 9. Tools: Prometheus, Grafana, ELK stack",
    "course": "Site Reliability Engineering (Google) or Monitoring and Observability (O'Reilly)"
  },
  "Observability": {
    "path": "1. Three Pillars: Metrics, logs, traces → 2. Telemetry: Data collection, instrumentation → 3. Distributed Tracing: Jaeger, Zipkin → 4. Structured Logging: JSON, correlation IDs → 5. APM Tools: Application performance monitoring → 6. Error Tracking: Exception monitoring → 7. Business Metrics: User behavior, KPIs → 8. Alerting: Intelligent alerting, noise reduction → 9. Debugging: Production troubleshooting",
    "course": "Observability Engineering (O'Reilly Book) or Distributed Systems Observability (Cindy Sridharan)"
  },
  "Hadoop": {
    "path": "1. HDFS: Distributed file system, replication → 2. MapReduce: Programming model, job execution → 3. YARN: Resource management, scheduling → 4. Hive: SQL-like queries, data warehousing → 5. Pig: Data flow language, transformations → 6. HBase: NoSQL database, column-family → 7. Sqoop: Data import/export, RDBMS integration → 8. Flume: Log data collection, streaming → 9. Cluster Management: Administration, monitoring",
    "course": "Hadoop Platform and Application Framework (Coursera - UC San Diego) or Hadoop: The Definitive Guide (Book)"
  },
  "Spark": {
    "path": "1. RDD Basics: Resilient Distributed Datasets → 2. DataFrames: Structured data processing → 3. Spark SQL: SQL queries, catalyst optimizer → 4. Streaming: DStreams, structured streaming → 5. MLlib: Machine learning library → 6. GraphX: Graph processing framework → 7. Cluster Management: Standalone, YARN, Kubernetes → 8. Performance Tuning: Caching, partitioning → 9. Deployment: Cluster modes, configuration",
    "course": "Apache Spark with Scala (Udemy - Frank Kane) or Learning Spark (O'Reilly Book)"
  },
  "Data Modeling": {
    "path": "1. Conceptual Modeling: Entities, relationships → 2. Logical Modeling: Normalization, keys → 3. Physical Modeling: Storage, indexing → 4. Dimensional Modeling: Star, snowflake schemas → 5. NoSQL Modeling: Document, key-value, graph → 6. Data Vault: Historical data modeling → 7. Schema Design: Performance considerations → 8. Data Governance: Quality, lineage → 9. Tools: ER/Studio, PowerDesigner, Lucidchart",
    "course": "Data Modeling Fundamentals (Pluralsight) or The Data Warehouse Toolkit (Book)"
  },
  "Data Warehousing": {
    "path": "1. Architecture: OLTP vs OLAP, ETL → 2. Dimensional Modeling: Facts, dimensions → 3. Schema Design: Star, snowflake, galaxy → 4. ETL Processes: Extract, transform, load → 5. Data Quality: Cleansing, validation → 6. Slowly Changing Dimensions: Type 1, 2, 3 → 7. Partitioning: Horizontal, vertical strategies → 8. Indexing: Bitmap, B-tree indexes → 9. Performance: Query optimization, aggregation",
    "course": "Data Warehousing for Business Intelligence (Coursera - University of Colorado) or Building a Data Warehouse (Inmon)"
  },
  "Redshift": {
    "path": "1. Architecture: Columnar storage, MPP → 2. Cluster Management: Nodes, parameter groups → 3. Data Loading: COPY command, bulk loading → 4. Query Optimization: Distribution, sort keys → 5. Compression: Encoding, storage efficiency → 6. Workload Management: Query queues, priorities → 7. Security: VPC, encryption, IAM → 8. Monitoring: CloudWatch, query performance → 9. Data Lake Integration: Spectrum, external tables",
    "course": "Amazon Redshift Deep Dive (A Cloud Guru) or Data Warehousing on AWS (AWS Training)"
  },
  "Snowflake": {
    "path": "1. Architecture: Separation of compute/storage → 2. Virtual Warehouses: Scaling, auto-suspend → 3. Data Loading: Snowpipe, bulk loading → 4. Time Travel: Historical data access → 5. Zero-copy Cloning: Instant database copies → 6. Secure Data Sharing: Cross-account sharing → 7. Streams: Change data capture → 8. Tasks: Scheduling, automation → 9. Performance: Clustering, caching",
    "course": "Snowflake Fundamentals (Snowflake University) or Snowflake Cookbook (O'Reilly)"
  },
  "S3": {
    "path": "1. Object Storage: Buckets, objects, keys → 2. Storage Classes: Standard, IA, Glacier → 3. Lifecycle Management: Automated transitions → 4. Versioning: Object versions, delete markers → 5. Security: IAM, bucket policies, ACLs → 6. Performance: Request patterns, transfer acceleration → 7. Event Notifications: Lambda triggers, SQS → 8. Cross-region Replication: Data durability → 9. Cost Optimization: Storage classes, monitoring",
    "course": "AWS S3 Masterclass (Udemy) or AWS Storage Services Overview (A Cloud Guru)"
  },
  "Glue": {
    "path": "1. Data Catalog: Metadata repository, crawlers → 2. ETL Jobs: Spark-based data transformation → 3. DataBrew: Visual data preparation → 4. Schema Registry: Schema evolution, compatibility → 5. Workflows: Job orchestration, scheduling → 6. Connections: Data source connectivity → 7. Development Endpoints: Interactive development → 8. Monitoring: Job metrics, CloudWatch → 9. Cost Optimization: Job bookmarking, partitioning",
    "course": "AWS Glue Tutorial (AWS Documentation) or Building Data Lakes on AWS (A Cloud Guru)"
  },
  "Athena": {
    "path": "1. Serverless Queries: Presto-based SQL engine → 2. Data Sources: S3, federated queries → 3. Table Creation: DDL, partition management → 4. Query Optimization: Columnar formats, compression → 5. Workgroups: Resource management, cost control → 6. Result Caching: Query performance improvement → 7. Security: IAM, encryption, access control → 8. JDBC/ODBC: Programmatic access → 9. Cost Management: Query optimization, data formats",
    "course": "Amazon Athena Deep Dive (AWS Documentation) or Serverless Analytics with Amazon Athena (A Cloud Guru)"
  },
  "Parquet": {
    "path": "1. Columnar Format: Storage benefits, compression → 2. Schema Evolution: Adding/removing columns → 3. Predicate Pushdown: Query optimization → 4. Nested Data: Complex types, arrays, maps → 5. Encoding: Dictionary, RLE, bit packing → 6. Metadata: Footer, column statistics → 7. Partitioning: Hive-style partitioning → 8. Tools: Spark, Pandas, Arrow integration → 9. Performance: Read/write optimization",
    "course": "Parquet Format Deep Dive (Apache Parquet Docs) or Efficient Data Formats for Analytics (Strata Conference)"
  },
  "Delta Lake": {
    "path": "1. ACID Transactions: Atomicity, consistency → 2. Time Travel: Historical data access → 3. Schema Evolution: Safe schema changes → 4. Upserts: Merge operations, CDC → 5. Data Versioning: Commit history, rollback → 6. Optimization: Z-order, vacuum operations → 7. Streaming: Real-time data ingestion → 8. Unity Catalog: Governance, lineage → 9. Performance: Caching, liquid clustering",
    "course": "Delta Lake Fundamentals (Databricks Academy) or Building Reliable Data Lakes with Delta Lake (Databricks)"
  },
  "Product Strategy": {
    "path": "1. Market Analysis: Customer needs, competitive landscape → 2. Vision & Mission: Product direction, goals → 3. Roadmapping: Feature prioritization, timeline → 4. Business Model: Revenue streams, value proposition → 5. Metrics: KPIs, success measurement → 6. Stakeholder Alignment: Communication, buy-in → 7. Go-to-Market: Launch strategy, positioning → 8. Portfolio Management: Product lifecycle → 9. Innovation: Emerging technologies, disruption",
    "course": "Product Management Fundamentals (Coursera - University of Virginia) or Product Strategy (Product School)"
  },
  "Roadmapping": {
    "path": "1. Strategic Alignment: Business goals, product vision → 2. Feature Prioritization: Value, effort, impact → 3. Timeline Planning: Milestones, dependencies → 4. Stakeholder Input: Customer feedback, business needs → 5. Resource Planning: Team capacity, constraints → 6. Communication: Visual roadmaps, storytelling → 7. Agile Integration: Sprint planning, iterations → 8. Risk Management: Contingency planning → 9. Continuous Updates: Feedback incorporation, pivots",
    "course": "Product Roadmapping (Product Plan) or Strategic Product Management (Udemy)"
  },
  "Agile Methodologies": {
    "path": "1. Agile Manifesto: Values, principles → 2. Scrum Framework: Roles, events, artifacts → 3. Kanban: Flow, WIP limits, continuous delivery → 4. User Stories: Acceptance criteria, definition of done → 5. Sprint Planning: Capacity, commitment → 6. Daily Standups: Progress, impediments → 7. Retrospectives: Continuous improvement → 8. Estimation: Story points, planning poker → 9. Scaling: SAFe, LeSS frameworks",
    "course": "Agile Development (University of Virginia - Coursera) or Certified Scrum Master (Scrum Alliance)"
  },
  "Scrum": {
    "path": "1. Scrum Roles: Product Owner, Scrum Master, Team → 2. Sprint Structure: Planning, execution, review → 3. Product Backlog: Prioritization, grooming → 4. Sprint Backlog: Task breakdown, commitment → 5. Daily Standups: Progress, impediments → 6. Sprint Review: Demo, stakeholder feedback → 7. Sprint Retrospective: Process improvement → 8. Definition of Done: Quality standards → 9. Metrics: Velocity, burndown charts",
    "course": "Professional Scrum Master (Scrum.org) or Scrum Fundamentals (Pluralsight)"
  },
  "Stakeholder Management": {
    "path": "1. Stakeholder Identification: Primary, secondary stakeholders → 2. Influence Mapping: Power, interest matrix → 3. Communication Planning: Frequency, channels → 4. Expectation Management: Setting, aligning expectations → 5. Conflict Resolution: Negotiation, compromise → 6. Feedback Collection: Surveys, interviews → 7. Decision Making: RACI, consensus building → 8. Relationship Building: Trust, rapport → 9. Change Management: Adoption, resistance",
    "course": "Stakeholder Engagement (Coursera) or Project Stakeholder Management (PMI)"
  },
  "User Stories": {
    "path": "1. Story Structure: As a, I want, so that → 2. Acceptance Criteria: Testable conditions → 3. Story Writing: INVEST criteria → 4. Epic Breakdown: Large features to stories → 5. User Personas: Target audience definition → 6. Story Mapping: User journey visualization → 7. Prioritization: MoSCoW, value-based → 8. Estimation: Story points, complexity → 9. Definition of Done: Completion criteria",
    "course": "User Story Mapping (Jeff Patton) or Writing Effective User Stories (Mountain Goat Software)"
  },
  "Market Research": {
    "path": "1. Research Design: Quantitative, qualitative methods → 2. Customer Segmentation: Demographics, psychographics → 3. Competitive Analysis: Direct, indirect competitors → 4. Survey Design: Questions, sampling methods → 5. Interview Techniques: User interviews, focus groups → 6. Data Collection: Primary, secondary sources → 7. Analysis"
        },
  
  "Product Strategy": {
    "path": "1. Market Analysis: Customer needs, competitor analysis → 2. Vision & Mission: Product positioning, value proposition → 3. Goal Setting: OKRs, KPIs, success metrics → 4. Prioritization: RICE, MoSCoW, value vs effort → 5. Roadmapping: Timeline planning, dependencies → 6. Stakeholder Alignment: Communication, buy-in → 7. Go-to-Market: Launch strategy, messaging → 8. Iteration: Feedback loops, pivoting strategies",
    "course": "Product Strategy (Coursera - University of Virginia) or Product Management Fundamentals (Udemy)"
  },
  "Roadmapping": {
    "path": "1. Strategic Planning: Long-term vision alignment → 2. Feature Prioritization: Impact vs effort matrices → 3. Timeline Estimation: Development cycles, dependencies → 4. Resource Planning: Team capacity, skill requirements → 5. Milestone Definition: Release planning, checkpoints → 6. Risk Assessment: Technical debt, market changes → 7. Communication: Stakeholder updates, transparency → 8. Agile Integration: Sprint planning, backlog management",
    "course": "Product Roadmaps (ProductPlan) or Strategic Product Management (edX - Boston University)"
  },
  "Agile Methodologies": {
    "path": "1. Agile Principles: Manifesto, values, mindset → 2. Scrum Framework: Roles, events, artifacts → 3. Sprint Planning: Story estimation, capacity planning → 4. Daily Standups: Communication, impediment removal → 5. Sprint Review: Demo, feedback collection → 6. Retrospectives: Continuous improvement → 7. Kanban: Visual workflow, WIP limits → 8. Scaling: SAFe, LeSS, Nexus frameworks",
    "course": "Agile Development (Coursera - University of Virginia) or Certified ScrumMaster (Scrum Alliance)"
  },
  "Scrum": {
    "path": "1. Scrum Theory: Empiricism, transparency, inspection → 2. Roles: Product Owner, Scrum Master, Development Team → 3. Events: Sprint, Planning, Daily Scrum, Review, Retrospective → 4. Artifacts: Product Backlog, Sprint Backlog, Increment → 5. User Stories: Writing, acceptance criteria → 6. Estimation: Planning poker, story points → 7. Velocity: Team performance metrics → 8. Scaling: Multiple teams, dependencies",
    "course": "Scrum Fundamentals (Scrum.org) or Professional Scrum Master (PSM I)"
  },
  "Stakeholder Management": {
    "path": "1. Stakeholder Identification: Primary, secondary, key influencers → 2. Power-Interest Grid: Prioritizing stakeholder engagement → 3. Communication Planning: Frequency, channels, formats → 4. Expectation Management: Setting realistic goals → 5. Conflict Resolution: Mediating competing interests → 6. Influence Strategies: Building coalitions, persuasion → 7. Feedback Collection: Regular check-ins, surveys → 8. Change Management: Managing resistance, adoption",
    "course": "Stakeholder Engagement (Coursera - Rice University) or Project Stakeholder Management (PMI)"
  },
  "User Stories": {
    "path": "1. Story Structure: As a/I want/So that format → 2. Acceptance Criteria: Definition of done → 3. Story Splitting: Vertical slicing, INVEST criteria → 4. Prioritization: Value, urgency, dependencies → 5. Estimation: Story points, relative sizing → 6. Epic Management: Large features, themes → 7. Backlog Refinement: Grooming, detailing → 8. Traceability: Requirements to features mapping",
    "course": "User Story Mapping (Mountain Goat Software) or Writing Great User Stories (Pluralsight)"
  },
  "Market Research": {
    "path": "1. Research Design: Primary vs secondary research → 2. Target Audience: Customer segmentation, personas → 3. Survey Design: Question types, bias avoidance → 4. Interview Techniques: User interviews, focus groups → 5. Competitive Analysis: Feature comparison, positioning → 6. Data Collection: Quantitative, qualitative methods → 7. Analysis: Statistical significance, insights extraction → 8. Reporting: Actionable recommendations, presentations",
    "course": "Market Research (Coursera - University of California Davis) or Consumer Research (edX - University of British Columbia)"
  },
  "A/B Testing": {
    "path": "1. Hypothesis Formation: Null/alternative hypotheses → 2. Test Design: Control vs treatment groups → 3. Sample Size: Statistical power, effect size → 4. Randomization: Traffic allocation, bias prevention → 5. Metrics Selection: Primary, secondary, guardrail metrics → 6. Statistical Analysis: Significance testing, confidence intervals → 7. Results Interpretation: Practical vs statistical significance → 8. Implementation: Feature flags, gradual rollouts",
    "course": "A/B Testing (Google Analytics Academy) or Experimentation for Improvement (Coursera - Duke University)"
  },
  "Customer Interviews": {
    "path": "1. Interview Planning: Objectives, target participants → 2. Question Design: Open-ended, unbiased questions → 3. Recruitment: Finding representative users → 4. Interview Techniques: Active listening, probing → 5. Data Recording: Notes, audio, video → 6. Analysis: Thematic analysis, pattern identification → 7. Synthesis: Insights, recommendations → 8. Follow-up: Validation, additional research",
    "course": "Customer Development (Steve Blank) or User Research Methods (Interaction Design Foundation)"
  },
  "Figma": {
    "path": "1. Interface Basics: Canvas, layers, properties panel → 2. Design Tools: Shapes, text, pen tool → 3. Components: Master components, instances, variants → 4. Auto Layout: Responsive design, constraints → 5. Prototyping: Interactions, transitions, overlays → 6. Design Systems: Styles, libraries, tokens → 7. Collaboration: Comments, sharing, handoff → 8. Plugins: Extending functionality, automation",
    "course": "Figma Masterclass (Udemy) or Figma UI/UX Design Essentials (YouTube - AJ&Smart)"
  },
  "JIRA": {
    "path": "1. Project Setup: Project types, configurations → 2. Issue Types: Stories, bugs, tasks, epics → 3. Workflows: Status transitions, approvals → 4. Boards: Scrum, Kanban, custom boards → 5. Reporting: Burndown, velocity, control charts → 6. JQL: Advanced search, filtering → 7. Customization: Fields, screens, permissions → 8. Integration: Confluence, Bitbucket, third-party tools",
    "course": "JIRA Fundamentals (Atlassian University) or Agile Project Management with JIRA (Udemy)"
  },
  "Confluence": {
    "path": "1. Page Creation: Templates, formatting, structure → 2. Collaboration: Comments, mentions, page sharing → 3. Macros: Dynamic content, integrations → 4. Space Management: Permissions, organization → 5. Templates: Standardizing documentation → 6. Integration: JIRA, development tools → 7. Search: Finding content, filters → 8. Administration: User management, app configuration",
    "course": "Confluence Fundamentals (Atlassian University) or Documentation Best Practices (Write the Docs)"
  },
  "Product Lifecycle Management": {
    "path": "1. Product Strategy: Vision, positioning, roadmap → 2. Market Research: Customer needs, competition → 3. Development: Requirements, design, build → 4. Launch: Go-to-market, marketing, sales enablement → 5. Growth: Feature expansion, market penetration → 6. Maturity: Optimization, cost management → 7. Decline: Sunset planning, migration → 8. Portfolio Management: Product mix, resource allocation",
    "course": "Product Lifecycle Management (Coursera - University of Virginia) or Product Management for AI & Data Science (Udemy)"
  },
  "KPI Definition": {
    "path": "1. Business Objectives: Aligning metrics to goals → 2. KPI Selection: Leading vs lagging indicators → 3. SMART Criteria: Specific, measurable, achievable → 4. Metric Hierarchy: North star, primary, secondary → 5. Baseline Establishment: Historical data, benchmarks → 6. Target Setting: Realistic, stretch goals → 7. Tracking: Dashboards, reporting cadence → 8. Action Planning: Performance improvement strategies",
    "course": "KPI and Performance Management (Coursera - UC Irvine) or Data-Driven Decision Making (edX - MIT)"
  },
  "Go-to-Market Strategy": {
    "path": "1. Market Segmentation: Target audience identification → 2. Value Proposition: Unique selling points → 3. Positioning: Competitive differentiation → 4. Pricing Strategy: Models, optimization → 5. Channel Strategy: Distribution, partnerships → 6. Marketing Mix: Product, price, place, promotion → 7. Sales Enablement: Training, materials, tools → 8. Launch Execution: Timeline, metrics, feedback loops",
    "course": "Go-to-Market Strategy (Coursera - University of Virginia) or Product Marketing Fundamentals (Product Marketing Alliance)"
  },
  "Competitive Analysis": {
    "path": "1. Competitor Identification: Direct, indirect, substitute → 2. Information Gathering: Public sources, tools → 3. Feature Comparison: Functionality, usability → 4. Pricing Analysis: Models, positioning → 5. Marketing Analysis: Messaging, channels → 6. SWOT Analysis: Strengths, weaknesses, opportunities, threats → 7. Positioning Map: Market landscape visualization → 8. Strategic Implications: Opportunities, threats, responses",
    "course": "Competitive Strategy (Coursera - Ludwig-Maximilians-Universität München) or Market Intelligence (edX - University of British Columbia)"
  },
  "Wireframing": {
    "path": "1. Low-Fidelity Sketching: Paper prototypes, basic layouts → 2. Digital Wireframing: Tools, templates → 3. Information Architecture: Content hierarchy, navigation → 4. User Flow: Task completion paths → 5. Responsive Design: Multi-device considerations → 6. Annotation: Specifications, interactions → 7. Iteration: Feedback incorporation, refinement → 8. Handoff: Development specifications, assets",
    "course": "UX Design Process (Google UX Design Certificate) or Wireframing Essentials (Interaction Design Foundation)"
  },
  "Game Physics": {
    "path": "1. Classical Mechanics: Force, mass, acceleration → 2. Collision Detection: AABB, sphere, polygon → 3. Collision Response: Impulse, momentum conservation → 4. Rigid Body Dynamics: Rotation, angular momentum → 5. Soft Body Physics: Springs, cloth simulation → 6. Fluid Dynamics: Particle systems, SPH → 7. Performance Optimization: Spatial partitioning, LOD → 8. Physics Engines: Integration, custom solutions",
    "course": "Game Physics (Coursera - University of Colorado) or Real-Time Physics Rendering (YouTube - Sebastian Lague)"
  },
  "AI for Games": {
    "path": "1. Pathfinding: A*, Dijkstra, navigation meshes → 2. State Machines: FSM, hierarchical state machines → 3. Behavior Trees: Composite nodes, decision making → 4. Steering Behaviors: Seek, flee, flocking → 5. Decision Making: Utility systems, planning → 6. Learning AI: Reinforcement learning, neural networks → 7. Procedural Generation: Content creation algorithms → 8. Performance: LOD, optimization techniques",
    "course": "AI for Games (Coursera - University of California Santa Cruz) or Game AI Programming (MIT OpenCourseWare)"
  },
  "Shader Programming": {
    "path": "1. Graphics Pipeline: Vertex, fragment processing → 2. GLSL/HLSL: Shader languages, syntax → 3. Vertex Shaders: Transformations, lighting → 4. Fragment Shaders: Texturing, effects → 5. Lighting Models: Phong, PBR, custom → 6. Texturing: UV mapping, sampling, filtering → 7. Advanced Effects: Post-processing, compute shaders → 8. Optimization: Performance, mobile considerations",
    "course": "Shader Development (Unity Learn) or Introduction to Computer Graphics (Coursera - UC San Diego)"
  },
  "OpenGL": {
    "path": "1. OpenGL Context: Window creation, initialization → 2. Rendering Pipeline: Buffers, shaders, drawing → 3. Transformations: Model, view, projection matrices → 4. Texturing: Loading, binding, sampling → 5. Lighting: Phong model, multiple lights → 6. Advanced Features: Framebuffers, instancing → 7. Modern OpenGL: VAOs, VBOs, core profile → 8. Performance: Batch drawing, GPU optimization",
    "course": "Learn OpenGL (learnopengl.com) or Computer Graphics with OpenGL (Coursera - University of London)"
  },
  "DirectX": {
    "path": "1. DirectX Basics: COM interfaces, device creation → 2. Direct3D: Rendering pipeline, resources → 3. Shaders: HLSL, vertex/pixel shaders → 4. Textures: Loading, sampling, render targets → 5. Input: DirectInput, handling user input → 6. Audio: DirectSound, 3D audio → 7. Advanced Features: Tessellation, compute shaders → 8. Performance: Profiling, optimization techniques",
    "course": "DirectX 11 Programming (YouTube - ChiliTomatoNoodle) or Real-Time 3D Graphics with DirectX (Microsoft Learn)"
  },
  "Animation Systems": {
    "path": "1. Keyframe Animation: Interpolation, curves → 2. Skeletal Animation: Bones, skinning, weights → 3. Blend Trees: State transitions, parameter blending → 4. Inverse Kinematics: IK chains, constraints → 5. Motion Capture: Data processing, retargeting → 6. Procedural Animation: Physics-based, algorithmic → 7. Animation Compression: Storage, streaming → 8. Runtime Optimization: LOD, culling, batching",
    "course": "Animation Programming (Unity Learn) or Computer Animation Algorithms (YouTube - ThinMatrix)"
  },
  "Multiplayer Networking": {
    "path": "1. Network Architecture: Client-server, P2P → 2. Protocol Selection: TCP vs UDP trade-offs → 3. State Synchronization: Authority, reconciliation → 4. Lag Compensation: Prediction, rollback → 5. Security: Anti-cheat, validation → 6. Scalability: Load balancing, instancing → 7. Optimization: Bandwidth, latency → 8. Testing: Network simulation, profiling",
    "course": "Multiplayer Game Programming (Coursera - University of California Santa Cruz) or Networked Games Architecture (YouTube - GDC)"
  },
  "Gameplay Programming": {
    "path": "1. Game Loop: Update cycles, timing → 2. Input Systems: Handling, mapping, responsiveness → 3. Game States: Menus, gameplay, transitions → 4. Entity Systems: Components, managers → 5. Scripting Integration: Lua, Python, visual scripting → 6. Event Systems: Messaging, decoupling → 7. Save Systems: Serialization, persistence → 8. Debugging: Tools, logging, profiling",
    "course": "Game Programming Patterns (gameprogrammingpatterns.com) or Gameplay Programming (Unity Learn)"
  },
  "Game Design": {
    "path": "1. Core Mechanics: Rules, systems, interactions → 2. Player Psychology: Motivation, engagement → 3. Balancing: Difficulty curves, progression → 4. Narrative Design: Storytelling, character development → 5. Level Design: Flow, pacing, challenges → 6. UI/UX: Interface design, usability → 7. Monetization: Business models, ethics → 8. Playtesting: Feedback, iteration",
    "course": "Game Design and Development (Coursera - Michigan State) or The Art of Game Design (Jesse Schell)"
  },
  "3D Modeling": {
    "path": "1. Modeling Fundamentals: Vertices, edges, faces → 2. Topology: Clean geometry, edge flow → 3. Modeling Techniques: Box modeling, sculpting → 4. UV Mapping: Texture coordinates, unwrapping → 5. Materials: PBR, texturing workflow → 6. Rigging: Bones, constraints, controls → 7. Animation: Keyframes, curves, timing → 8. Optimization: Poly count, LOD, mobile",
    "course": "Blender Fundamentals (Blender Guru) or 3D Modeling for Games (Coursera - University of California Santa Cruz)"
  },
  "Level Design": {
    "path": "1. Design Principles: Flow, pacing, balance → 2. Player Psychology: Navigation, exploration → 3. Spatial Design: Layout, landmarks, wayfinding → 4. Challenge Progression: Difficulty curves, learning → 5. Environmental Storytelling: Narrative through space → 6. Technical Constraints: Performance, memory → 7. Iteration: Playtesting, feedback, refinement → 8. Tools: Editors, scripting, automation",
    "course": "Level Design (Game Design Workshop) or Environment Art for Games (YouTube - Ryan Kingslien)"
  },
  "Visual Scripting": {
    "path": "1. Node-Based Logic: Flow control, variables → 2. Event Systems: Triggers, responses → 3. State Management: Conditions, transitions → 4. Data Flow: Input/output, type systems → 5. Debugging: Breakpoints, inspection → 6. Performance: Optimization, compilation → 7. Integration: Code interop, custom nodes → 8. Best Practices: Organization, reusability",
    "course": "Visual Scripting (Unity Learn) or Blueprint Visual Scripting (Unreal Engine Documentation)"
  },
  "Game Optimization": {
    "path": "1. Performance Profiling: CPU, GPU, memory usage → 2. Rendering Optimization: Draw calls, batching → 3. Asset Optimization: Textures, models, audio → 4. Code Optimization: Algorithms, data structures → 5. Memory Management: Allocation, pooling → 6. Platform-Specific: Mobile, console optimizations → 7. Tools: Profilers, analyzers → 8. Quality Settings: Scalable graphics, LOD",
    "course": "Game Optimization (Unity Learn) or Performance Optimization for Games (YouTube - Code Monkey)"
  },
  "Mobile Game Development": {
    "path": "1. Platform Differences: iOS vs Android constraints → 2. Touch Input: Gestures, responsiveness → 3. Performance: Battery, thermal management → 4. Screen Adaptation: Multiple resolutions, safe areas → 5. Platform Services: Analytics, ads, IAP → 6. Publishing: App stores, certification → 7. Monetization: F2P models, user retention → 8. Testing: Device testing, remote debugging",
    "course": "Mobile Game Development (Coursera - Michigan State) or Unity Mobile Game Development (Udemy)"
  },
  "VR/AR": {
    "path": "1. VR/AR Fundamentals: Immersion, presence, tracking → 2. Hardware: Headsets, controllers, sensors → 3. Spatial Computing: 6DOF, hand tracking → 4. User Interface: 3D UI, spatial interaction → 5. Performance: High framerate requirements → 6. Comfort: Motion sickness, ergonomics → 7. Development Tools: SDKs, engines → 8. Deployment: Platform stores, distribution",
    "course": "VR/AR Development (Coursera - University of California San Diego) or XR Development (Unity Learn)"
  },
  "Audio Integration": {
    "path": "1. Audio Basics: Formats, compression, streaming → 2. 3D Audio: Spatialization, attenuation → 3. Dynamic Music: Adaptive, interactive scoring → 4. Sound Effects: Triggers, layering, mixing → 5. Audio Middleware: Wwise, FMOD integration → 6. Performance: Memory, CPU optimization → 7. Platform Audio: Surround sound, headphones → 8. Testing: Audio QA, platform compliance",
    "course": "Game Audio Implementation (YouTube - Akash Thakkar) or Audio Programming for Games (Berklee Online)"
  },
  "Go": {
    "path": "1. Go Basics: Syntax, types, control structures → 2. Concurrency: Goroutines, channels, select → 3. Packages: Organization, imports, modules → 4. Error Handling: Error types, panic/recover → 5. Testing: Unit tests, benchmarks, table tests → 6. Web Development: HTTP servers, routing → 7. Database Integration: SQL drivers, ORMs → 8. Performance: Profiling, optimization, garbage collection",
    "course": "Go Programming Language (Coursera - University of California Irvine) or Learn Go Programming (Udemy)"
  },
  "Operating Systems": {
    "path": "1. OS Concepts: Processes, threads, scheduling → 2. Memory Management: Virtual memory, paging → 3. File Systems: Organization, permissions, I/O → 4. Inter-Process Communication: Pipes, sockets, shared memory → 5. Synchronization: Mutexes, semaphores, deadlocks → 6. Device Management: Drivers, interrupts → 7. Security: Authentication, access control → 8. Performance: System calls, optimization",
    "course": "Operating Systems (Coursera - Nanjing University) or Introduction to Operating Systems (Udacity)"
  },
  "Big-O Analysis": {
    "path": "1. Complexity Basics: Time vs space complexity → 2. Asymptotic Notation: Big-O, Omega, Theta → 3. Common Complexities: Constant, linear, logarithmic → 4. Recursive Analysis: Master theorem, recurrence relations → 5. Data Structure Analysis: Arrays, trees, graphs → 6. Algorithm Analysis: Sorting, searching, dynamic programming → 7. Amortized Analysis: Average case over sequences → 8. Practical Considerations: Constants, real-world performance",
    "course": "Algorithms Specialization (Coursera - Stanford) or Big-O Notation (Khan Academy)"
  },
  "Coding Interviews": {
    "path": "1. Problem-Solving Approach: Understanding, planning, coding → 2. Data Structures: Arrays, linked lists, trees, graphs → 3. Algorithms: Sorting, searching, dynamic programming → 4. System Design: Scalability, trade-offs → 5. Behavioral Questions: STAR method, examples → 6. Code Quality: Readability, edge cases, testing → 7. Communication: Thinking out loud, clarifying questions → 8. Practice: Mock interviews, coding challenges",
    "course": "Coding Interview Bootcamp (Udemy) or Cracking the Coding Interview (Book + YouTube)"
  },
  "Software Engineering Principles": {
    "path": "1. SOLID Principles: Single responsibility, open/closed → 2. Design Patterns: Creational, structural, behavioral → 3. Code Quality: Clean code, refactoring → 4. Testing: Unit, integration, TDD → 5. Version Control: Git workflows, branching strategies → 6. Documentation: API docs, code comments → 7. Code Review: Best practices, feedback → 8. Maintenance: Technical debt, legacy systems",
    "course": "Software Engineering (Coursera - University of Alberta) or Clean Code (Robert C. Martin)"
  },
  "Design Patterns": {
    "path": "1. Creational Patterns: Singleton, factory, builder → 2. Structural Patterns: Adapter, decorator, facade → 3. Behavioral Patterns: Observer, strategy, command → 4. Anti-Patterns: Code smells, common mistakes → 5. Language-Specific: Patterns in different languages → 6. Architectural Patterns: MVC, MVP, MVVM → 7. Concurrency Patterns: Producer-consumer, thread pool → 8. Application: When and how to use patterns",
    "course": "Design Patterns (Coursera - University of Alberta) or Head First Design Patterns (Book)"
  },
  "Cloud Computing": {
    "path": "1. Cloud Models: IaaS, PaaS, SaaS → 2. Deployment Models: Public, private, hybrid → 3. Virtualization: VMs, containers, serverless → 4. Storage: Object, block, file storage → 5. Networking: VPC, load balancers, CDN → 6. Security: IAM, encryption, compliance → 7. Monitoring: Logging, metrics, alerting → 8. Cost Optimization: Reserved instances, auto-scaling",
    "course": "Cloud Computing Concepts (Coursera - University of Illinois) or AWS Cloud Practitioner Essentials"
  },
  "Hack": {
    "path": "1. PHP Integration: Running alongside PHP → 2. Type System: Gradual typing, type annotations → 3. Generics: Type parameters, constraints → 4. Async Programming: Awaitable, coroutines → 5. Collections: Vector, Map, Set types → 6. XHP: XML-like syntax for UI → 7. Traits: Code reuse, mixins → 8. HHVM: Runtime environment, performance",
    "course": "Hack Language Tutorial (Facebook/Meta Documentation) or PHP to Hack Migration (Meta Developer Resources)"
  },
  "React": {
    "path": "1. JSX Syntax: Elements, expressions, components → 2. Components: Functional, class components → 3. Props: Data passing, prop types → 4. State: useState, state management → 5. Events: Handling, synthetic events → 6. Lifecycle: useEffect, component lifecycle → 7. Context: Global state, provider pattern → 8. Hooks: Custom hooks, rules of hooks → 9. Performance: Memoization, lazy loading",
    "course": "React - The Complete Guide (Udemy) or React Fundamentals (React Documentation)"
  },
  "GraphQL": {
    "path": "1. GraphQL Basics: Queries, mutations, subscriptions → 2. Schema Definition: Types, fields, resolvers → 3. Query Language: Syntax, variables, fragments → 4. Server Implementation: Apollo Server, resolvers → 5. Client Integration: Apollo Client, caching → 6. Authentication: JWT, context, permissions → 7. Performance: N+1 problem, DataLoader → 8. Real-time: Subscriptions, live queries",
    "course": "GraphQL with React (Udemy) or Full-Stack GraphQL (Apollo GraphQL)"
  },
  "Big Data Tools": {
    "path": "1. Hadoop Ecosystem: HDFS, MapReduce, YARN → 2. Apache Spark: RDDs, DataFrames, Spark SQL → 3. Apache Kafka: Message queues, streaming → 4. Data Processing: Batch vs stream processing → 5. NoSQL Databases: MongoDB, Cassandra, HBase → 6. Data Warehousing: Hive, Impala, Presto → 7. Workflow Management: Airflow, Oozie → 8. Cloud Platforms: AWS EMR, Google Dataproc",
    "course": "Big Data Specialization (Coursera - UC San Diego) or Apache Spark with Python (Udemy)"
  },
  "Communication Skills": {
    "path": "1. Verbal Communication: Clear speaking, active listening → 2. Written Communication: Emails, documentation, reports → 3. Presentation Skills: Structure, visual aids, delivery → 4. Non-verbal Communication: Body language, tone → 5. Interpersonal Skills: Empathy, rapport building → 6. Conflict Resolution: De-escalation, compromise → 7. Cross-cultural: Cultural awareness, adaptation → 8. Technical Communication: Explaining complex concepts",
    "course": "Communication Skills (Coursera - University of California San Diego) or Effective Communication (LinkedIn Learning)"
  },
  "Cross-Team Collaboration": {
    "path": "1. Team Dynamics: Roles, responsibilities, boundaries → 2. Communication Protocols: Meetings, updates, channels → 3. Project Coordination: Dependencies, timelines → 4. Conflict Management: Resolution strategies → 5. Knowledge Sharing: Documentation, cross-training → 6. Cultural Sensitivity: Diverse teams, inclusion → 7. Remote Collaboration: Tools, practices → 8. Feedback Culture: Giving, receiving constructive feedback",
    "course": "Teamwork Skills (Coursera - University of California Davis) or Cross-Functional Team Leadership (LinkedIn Learning)"
  },
  "Scala": {
    "path": "1. Scala Basics: Syntax, immutability, type inference → 2. Object-Oriented: Classes, traits, inheritance → 3. Functional Programming: Higher-order functions, immutability → 4. Collections: List, Map, Set, operations → 5. Pattern Matching: Case classes, extractors → 6. Concurrency: Actors, futures, parallel collections → 7. Type System: Generics, variance, implicits → 8. Build Tools: SBT, dependency management",
    "course": "Functional Programming in Scala (Coursera - EPFL) or Scala & Functional Programming Essentials (Udemy)"
  },
  "Video Streaming Architecture": {
    "path": "1. Video Encoding: Codecs, bitrate, quality → 2. Adaptive Streaming: HLS, DASH, quality switching → 3. CDN: Content distribution, edge caching → 4. Load Balancing: Traffic distribution, failover → 5. DRM: Content protection, licensing → 6. Analytics: Quality metrics, user behavior → 7. Scalability: Horizontal scaling, microservices → 8. Performance: Latency optimization, buffering",
    "course": "Video Streaming Systems (System Design Interview) or Building Video Streaming Applications (AWS Training)"
  },
  "Resilience Engineering": {
    "path": "1. Fault Tolerance: Graceful degradation, redundancy → 2. Circuit Breakers: Preventing cascade failures → 3. Retry Strategies: Exponential backoff, jitter → 4. Timeouts: Request timeouts, connection limits → 5. Bulkheads: Resource isolation, failure containment → 6. Monitoring: Health checks, alerting → 7. Chaos Engineering: Fault injection, testing → 8. Recovery: Disaster recovery, backup strategies",
    "course": "Building Resilient Systems (Pluralsight) or Site Reliability Engineering (Google SRE Book)"
  },
  "Cloud-Native Development": {
    "path": "1. Containerization: Docker, container orchestration → 2. Microservices: Service decomposition, communication → 3. Service Mesh: Istio, traffic management → 4. CI/CD: Pipeline automation, deployment strategies → 5. Configuration Management: ConfigMaps, secrets → 6. Observability: Logging, metrics, tracing → 7. Security: Container security, network policies → 8. Cloud Platforms: AWS, GCP, Azure services",
    "course": "Cloud Native Development (Linux Foundation) or Microservices with Spring Cloud (Udemy)"
  },
  "Cross-Browser Compatibility": {
    "path": "1. Browser Differences: Rendering engines, feature support → 2. CSS Compatibility: Vendor prefixes, fallbacks → 3. JavaScript Compatibility: Polyfills, transpilation → 4. Testing Tools: BrowserStack, cross-browser testing → 5. Progressive Enhancement: Base functionality, enhancements → 6. Feature Detection: Modernizr, capability queries → 7. Responsive Design: Viewport, media queries → 8. Performance: Loading, rendering optimization",
    "course": "Cross-Browser Web Development (Pluralsight) or Web Compatibility Testing (Mozilla Developer Network)"
  },
  "Swift": {
    "path": "1. Swift Basics: Syntax, optionals, type safety → 2. Object-Oriented: Classes, structs, protocols → 3. Functional Features: Closures, higher-order functions → 4. Memory Management: ARC, strong/weak references → 5. Error Handling: Try-catch, throwing functions → 6. Generics: Type parameters, constraints → 7. Concurrency: Grand Central Dispatch, async/await → 8. Interoperability: Objective-C bridging, C libraries",
    "course": "iOS Development with Swift (Coursera - University of Toronto) or Swift Programming (Apple Developer)"
  },
  "Objective-C": {
    "path": "1. C Foundation: Pointers, memory management → 2"
  },
  
  "Swift": {
    "path": "1. Swift Fundamentals: Syntax, variables, data types → 2. Control Flow: Conditionals, loops, switch statements → 3. Functions: Parameters, return types, closures → 4. Object-Oriented Programming: Classes, structs, inheritance → 5. Protocols: Interface definitions, delegation → 6. Optionals: Safe handling of nil values → 7. Error Handling: Try-catch mechanisms → 8. Memory Management: ARC, strong/weak references → 9. Generics: Type-safe generic programming → 10. Concurrency: async/await, actors",
    "course": "Swift Programming Fundamentals (Apple Developer) or iOS & Swift Bootcamp (Udemy - Angela Yu)"
  },
  "Objective-C": {
    "path": "1. C Foundations: Pointers, memory management → 2. Objective-C Syntax: Message passing, method calls → 3. Classes & Objects: Interface, implementation → 4. Memory Management: Manual retain/release, ARC → 5. Categories: Extending existing classes → 6. Protocols: Formal/informal protocols → 7. Foundation Framework: NSString, NSArray, NSDictionary → 8. Runtime: Dynamic typing, method swizzling → 9. Blocks: Anonymous functions, closures",
    "course": "Objective-C Programming (Big Nerd Ranch) or Objective-C Fundamentals (Pluralsight)"
  },
  "iOS Development": {
    "path": "1. Xcode Basics: Interface Builder, project structure → 2. UIKit Fundamentals: Views, view controllers → 3. Auto Layout: Constraints, responsive design → 4. Navigation: Tab bars, navigation controllers → 5. Data Persistence: Core Data, UserDefaults → 6. Networking: URLSession, JSON parsing → 7. App Lifecycle: States, background processing → 8. Testing: Unit tests, UI tests → 9. App Store: Provisioning, submission process",
    "course": "iOS App Development with Swift (Coursera - University of Toronto) or Complete iOS Development Bootcamp (Udemy)"
  },
  "macOS": {
    "path": "1. macOS Architecture: Darwin, kernel, frameworks → 2. AppKit Framework: Windows, menus, controls → 3. Document-Based Apps: NSDocument architecture → 4. Core Services: File system, preferences → 5. Cocoa Bindings: Model-view-controller → 6. Grand Central Dispatch: Concurrency patterns → 7. Sandboxing: Security, entitlements → 8. Distribution: Mac App Store, notarization → 9. System Integration: Services, extensions",
    "course": "macOS App Development (Ray Wenderlich) or Cocoa Programming for macOS (Big Nerd Ranch)"
  },
  "Xcode": {
    "path": "1. Interface Overview: Navigator, editor, utilities → 2. Project Setup: Targets, schemes, configurations → 3. Interface Builder: Storyboards, XIBs, constraints → 4. Code Editor: Syntax highlighting, autocomplete → 5. Debugging: Breakpoints, LLDB, view hierarchy → 6. Testing: Unit tests, UI tests, code coverage → 7. Instruments: Performance profiling, memory leaks → 8. Version Control: Git integration, source control → 9. Archive & Distribution: Build settings, provisioning",
    "course": "Xcode Essentials (Apple Developer Documentation) or Mastering Xcode (Ray Wenderlich)"
  },
  "UIKit": {
    "path": "1. View Hierarchy: UIView, subviews, superview → 2. View Controllers: Lifecycle, navigation → 3. Auto Layout: Constraints, stack views → 4. Table Views: Data source, delegate patterns → 5. Collection Views: Custom layouts, flow layout → 6. Gesture Recognizers: Touch handling, gestures → 7. Animations: Core Animation, view animations → 8. Custom Views: Drawing, CALayer → 9. Accessibility: VoiceOver, accessibility traits",
    "course": "UIKit Fundamentals (Ray Wenderlich) or iOS User Interface Design (Coursera)"
  },
  "Core Data": {
    "path": "1. Data Model: Entities, attributes, relationships → 2. NSManagedObject: Model classes, KVC → 3. NSManagedObjectContext: Object lifecycle, changes → 4. Persistent Store: SQLite, binary, in-memory → 5. Fetching: NSFetchRequest, predicates, sorting → 6. Relationships: One-to-one, one-to-many → 7. Migration: Lightweight, heavyweight migration → 8. Threading: Context concurrency, parent-child → 9. Performance: Batching, faulting, caching",
    "course": "Core Data Fundamentals (Ray Wenderlich) or Core Data by Tutorials (raywenderlich.com)"
  },
  "Security Best Practices": {
    "path": "1. Authentication: Multi-factor, biometric authentication → 2. Authorization: Role-based access control → 3. Encryption: Symmetric, asymmetric, hashing → 4. Secure Communication: TLS, certificate pinning → 5. Input Validation: Sanitization, injection prevention → 6. Session Management: Tokens, expiration → 7. Data Protection: Keychain, secure storage → 8. Code Security: Static analysis, dependency scanning → 9. Threat Modeling: Risk assessment, mitigation",
    "course": "Cybersecurity Fundamentals (Coursera - IBM) or Application Security (OWASP Learning)"
  },
  "UI/UX Design": {
    "path": "1. Design Principles: Typography, color, layout → 2. User Research: Personas, user journeys → 3. Information Architecture: Sitemaps, navigation → 4. Wireframing: Low-fidelity prototypes → 5. Visual Design: High-fidelity mockups → 6. Interaction Design: Micro-interactions, transitions → 7. Usability Testing: A/B testing, user feedback → 8. Accessibility: WCAG guidelines, inclusive design → 9. Design Systems: Components, style guides",
    "course": "Google UX Design Certificate (Coursera) or UI/UX Design Specialization (CalArts)"
  },
  "Go": {
    "path": "1. Language Basics: Syntax, variables, types → 2. Control Structures: If, for, switch statements → 3. Functions: Parameters, multiple returns → 4. Structs & Methods: Custom types, receivers → 5. Interfaces: Implicit implementation, polymorphism → 6. Goroutines: Concurrent programming → 7. Channels: Communication, synchronization → 8. Packages: Modules, imports, visibility → 9. Error Handling: Error interface, panic/recover → 10. Testing: Unit tests, benchmarks",
    "course": "Go Programming Language (Coursera - UC Irvine) or Learn Go Programming (Udemy)"
  },
  "MySQL": {
    "path": "1. Database Fundamentals: RDBMS concepts, normalization → 2. SQL Basics: SELECT, INSERT, UPDATE, DELETE → 3. Data Types: Numeric, string, date types → 4. Table Design: Primary keys, foreign keys → 5. Joins: Inner, outer, cross joins → 6. Indexing: B-tree, composite indexes → 7. Functions: String, date, aggregate functions → 8. Stored Procedures: Functions, triggers → 9. Performance Tuning: Query optimization, EXPLAIN → 10. Replication: Master-slave, clustering",
    "course": "MySQL Database Administration (Udemy) or MySQL for Developers (Pluralsight)"
  },
  "PostgreSQL": {
    "path": "1. PostgreSQL Basics: Installation, psql client → 2. Advanced Data Types: JSON, arrays, custom types → 3. Indexing: B-tree, GIN, GiST indexes → 4. Transactions: ACID properties, isolation levels → 5. Window Functions: Analytical queries → 6. Common Table Expressions: Recursive queries → 7. Extensions: PostGIS, full-text search → 8. Performance: Query planning, vacuum → 9. Replication: Streaming, logical replication → 10. Administration: Backup, recovery, monitoring",
    "course": "PostgreSQL Administration (Udemy) or PostgreSQL for Everybody (Coursera - University of Michigan)"
  },
  "Kafka": {
    "path": "1. Messaging Concepts: Pub-sub, event streaming → 2. Kafka Architecture: Brokers, topics, partitions → 3. Producers: Publishing messages, serialization → 4. Consumers: Consuming messages, consumer groups → 5. Partitioning: Key-based, round-robin → 6. Replication: Leader-follower, fault tolerance → 7. Kafka Connect: Data integration → 8. Kafka Streams: Stream processing → 9. Schema Registry: Avro, JSON schema → 10. Monitoring: JMX metrics, Kafka Manager",
    "course": "Apache Kafka Series (Udemy - Stephane Maarek) or Kafka Fundamentals (Confluent)"
  },
  "AWS": {
    "path": "1. Cloud Fundamentals: IaaS, PaaS, SaaS concepts → 2. Core Services: EC2, S3, VPC, IAM → 3. Compute: EC2, Lambda, ECS, EKS → 4. Storage: S3, EBS, EFS, Glacier → 5. Database: RDS, DynamoDB, Redshift → 6. Networking: VPC, CloudFront, Route 53 → 7. Security: IAM, KMS, Security Groups → 8. Monitoring: CloudWatch, CloudTrail → 9. DevOps: CodePipeline, CloudFormation → 10. Cost Management: Billing, cost optimization",
    "course": "AWS Cloud Practitioner (AWS Training) or AWS Solutions Architect Associate (A Cloud Guru)"
  },
  "GCP": {
    "path": "1. Google Cloud Basics: Projects, billing, IAM → 2. Compute: Compute Engine, Cloud Functions, GKE → 3. Storage: Cloud Storage, Persistent Disk → 4. Database: Cloud SQL, Firestore, BigQuery → 5. Networking: VPC, Load Balancing, CDN → 6. Machine Learning: AI Platform, AutoML → 7. Data Analytics: BigQuery, Dataflow → 8. DevOps: Cloud Build, Cloud Deploy → 9. Monitoring: Cloud Monitoring, Logging → 10. Security: Cloud IAM, Cloud KMS",
    "course": "Google Cloud Platform Fundamentals (Coursera - Google Cloud) or GCP Associate Cloud Engineer (A Cloud Guru)"
  },
  "Azure": {
    "path": "1. Azure Fundamentals: Subscriptions, resource groups → 2. Compute: Virtual Machines, App Service, Functions → 3. Storage: Blob Storage, File Storage, Disk Storage → 4. Database: SQL Database, Cosmos DB → 5. Networking: Virtual Network, Load Balancer → 6. Identity: Active Directory, RBAC → 7. DevOps: Azure DevOps, ARM templates → 8. Monitoring: Azure Monitor, Application Insights → 9. AI/ML: Cognitive Services, Machine Learning → 10. Security: Key Vault, Security Center",
    "course": "Microsoft Azure Fundamentals (Microsoft Learn) or Azure Administrator Associate (Pluralsight)"
  },
  "BigQuery": {
    "path": "1. Data Warehouse Concepts: OLAP, dimensional modeling → 2. BigQuery Architecture: Dremel, columnar storage → 3. SQL in BigQuery: Standard SQL, legacy SQL → 4. Data Loading: Batch, streaming, federated → 5. Table Management: Partitioning, clustering → 6. Query Optimization: Best practices, costs → 7. Access Control: IAM, dataset permissions → 8. Data Export: Formats, destinations → 9. Machine Learning: BQML, AutoML integration → 10. Monitoring: Query history, job monitoring",
    "course": "BigQuery for Data Analysts (Coursera - Google Cloud) or BigQuery Fundamentals (A Cloud Guru)"
  },
  "Data Modeling": {
    "path": "1. Conceptual Modeling: Entity-relationship diagrams → 2. Logical Modeling: Normalization, denormalization → 3. Physical Modeling: Indexes, partitioning → 4. Dimensional Modeling: Star schema, snowflake → 5. Data Vault: Hub, link, satellite tables → 6. NoSQL Modeling: Document, key-value, graph → 7. Time Series Modeling: Temporal data patterns → 8. Data Lineage: Tracking data flow → 9. Master Data Management: Reference data → 10. Data Quality: Validation rules, constraints",
    "course": "Data Modeling Fundamentals (Pluralsight) or The Data Warehouse Toolkit (Kimball Group)"
  },
  "Data Warehousing": {
    "path": "1. DW Concepts: OLTP vs OLAP, ETL vs ELT → 2. Architecture: Kimball vs Inmon approaches → 3. Dimensional Modeling: Facts, dimensions, hierarchies → 4. ETL Processes: Extract, transform, load → 5. Data Quality: Cleansing, validation, monitoring → 6. Performance: Indexing, partitioning, aggregation → 7. OLAP Cubes: Multidimensional analysis → 8. Slowly Changing Dimensions: SCD types → 9. Data Governance: Metadata, lineage → 10. Modern DW: Cloud, real-time, data lakes",
    "course": "Data Warehousing Fundamentals (IBM) or Building a Data Warehouse (Coursera - UC Davis)"
  },
  "Redshift": {
    "path": "1. Redshift Architecture: Clusters, nodes, slices → 2. Data Loading: COPY command, bulk loading → 3. Table Design: Distribution keys, sort keys → 4. Query Optimization: Explain plans, workload management → 5. Data Types: Compression, encoding → 6. Administration: Backup, restore, monitoring → 7. Security: Encryption, VPC, IAM → 8. Performance Tuning: Vacuum, analyze → 9. Spectrum: Querying S3 data → 10. Integration: BI tools, data pipelines",
    "course": "Amazon Redshift Deep Dive (AWS Training) or Data Warehousing on AWS (A Cloud Guru)"
  },
  "Snowflake": {
    "path": "1. Snowflake Architecture: Multi-cluster, separation of compute/storage → 2. Virtual Warehouses: Scaling, auto-suspend → 3. Data Loading: Bulk loading, continuous loading → 4. Tables: Micro-partitions, clustering → 5. Time Travel: Data retention, recovery → 6. Cloning: Zero-copy cloning → 7. Data Sharing: Secure data sharing → 8. Security: RBAC, network policies → 9. Performance: Query optimization, caching → 10. Integration: BI tools, data pipelines",
    "course": "Snowflake Fundamentals (Snowflake University) or Snowflake Data Warehouse (Udemy)"
  },
  "S3": {
    "path": "1. S3 Basics: Buckets, objects, keys → 2. Storage Classes: Standard, IA, Glacier → 3. Security: Bucket policies, ACLs, encryption → 4. Versioning: Object versions, lifecycle → 5. Performance: Multipart upload, transfer acceleration → 6. Event Notifications: Lambda triggers, SQS → 7. Cross-Region Replication: Disaster recovery → 8. Static Website Hosting: Web hosting → 9. Cost Optimization: Lifecycle policies → 10. Integration: CloudFront, data analytics",
    "course": "Amazon S3 Masterclass (Udemy) or AWS Storage Services (A Cloud Guru)"
  },
  "Glue": {
    "path": "1. ETL Concepts: Extract, transform, load processes → 2. Glue Architecture: Jobs, crawlers, catalog → 3. Data Catalog: Metadata management, schemas → 4. Crawlers: Auto-discovery, schema inference → 5. ETL Jobs: PySpark, Scala scripts → 6. Transformations: Built-in transforms, custom → 7. Job Scheduling: Triggers, workflows → 8. Monitoring: CloudWatch, job metrics → 9. Integration: S3, RDS, Redshift → 10. Security: IAM roles, encryption",
    "course": "AWS Glue ETL (Udemy) or Data Engineering on AWS (A Cloud Guru)"
  },
  "Athena": {
    "path": "1. Serverless Analytics: Query S3 data directly → 2. Presto Engine: SQL query engine → 3. Table Management: External tables, partitions → 4. Data Formats: Parquet, ORC, JSON → 5. Query Optimization: Partition pruning, compression → 6. Cost Optimization: Data formats, partitioning → 7. Integration: Glue catalog, QuickSight → 8. Security: IAM, bucket policies → 9. Performance: Query tuning, caching → 10. Use Cases: Ad-hoc analytics, log analysis",
    "course": "Amazon Athena Deep Dive (AWS Training) or Serverless Analytics with AWS (Pluralsight)"
  },
  "Parquet": {
    "path": "1. Columnar Storage: Benefits over row-based → 2. File Format: Structure, metadata → 3. Compression: Snappy, GZIP, LZO → 4. Schema Evolution: Adding/removing columns → 5. Encoding: Dictionary, bit-packing → 6. Partitioning: Directory-based partitioning → 7. Performance: Predicate pushdown, projection → 8. Tools: Spark, Pandas, Arrow integration → 9. Best Practices: Schema design, optimization → 10. Ecosystem: Hive, Impala, Drill support",
    "course": "Apache Parquet Tutorial (DataBricks) or Columnar Storage Deep Dive (Confluent)"
  },
  "Delta Lake": {
    "path": "1. Data Lake Challenges: ACID properties, schema → 2. Delta Architecture: Transaction log, versioning → 3. ACID Transactions: Consistency, isolation → 4. Time Travel: Historical data access → 5. Schema Evolution: Adding/modifying columns → 6. Data Quality: Constraints, expectations → 7. Streaming: Real-time data ingestion → 8. Performance: Z-ordering, bloom filters → 9. Integration: Spark, MLflow, Kafka → 10. Operations: Vacuum, optimize commands",
    "course": "Delta Lake Fundamentals (Databricks Academy) or Building Data Lakes with Delta Lake (Pluralsight)"
  },
  "Terraform": {
    "path": "1. Infrastructure as Code: Benefits, concepts → 2. Terraform Basics: Providers, resources → 3. Configuration Language: HCL syntax → 4. State Management: State files, remote backends → 5. Variables: Input variables, outputs → 6. Modules: Reusable configurations → 7. Provisioners: Custom scripts, actions → 8. Workspaces: Environment separation → 9. Best Practices: Code organization, security → 10. CI/CD Integration: Automated deployments",
    "course": "Terraform Associate Certification (HashiCorp Learn) or Terraform Deep Dive (Pluralsight)"
  },
  "Ansible": {
    "path": "1. Configuration Management: Automation concepts → 2. Ansible Architecture: Control node, managed nodes → 3. Inventory: Host groups, variables → 4. Playbooks: YAML, tasks, plays → 5. Modules: Built-in modules, custom modules → 6. Variables: Host vars, group vars, facts → 7. Templates: Jinja2 templating → 8. Roles: Code organization, reusability → 9. Handlers: Event-driven tasks → 10. Vault: Encryption, secrets management",
    "course": "Ansible for DevOps (Jeff Geerling) or Red Hat Ansible Automation (Red Hat Training)"
  },
  "Monitoring": {
    "path": "1. Monitoring Fundamentals: Metrics, logs, traces → 2. Infrastructure Monitoring: CPU, memory, disk → 3. Application Monitoring: APM, performance → 4. Log Management: Collection, parsing, analysis → 5. Alerting: Thresholds, notification channels → 6. Dashboards: Visualization, KPIs → 7. SLI/SLO: Service level objectives → 8. Synthetic Monitoring: Proactive testing → 9. Distributed Tracing: Request flows → 10. Incident Response: On-call, runbooks",
    "course": "Site Reliability Engineering (Google) or Monitoring and Observability (Udemy)"
  },
  "Grafana": {
    "path": "1. Grafana Basics: Installation, web interface → 2. Data Sources: Prometheus, InfluxDB, CloudWatch → 3. Dashboard Creation: Panels, queries → 4. Visualization: Graphs, tables, heatmaps → 5. Variables: Template variables, dynamic dashboards → 6. Alerting: Alert rules, notification channels → 7. User Management: Organizations, teams → 8. Plugins: Panel plugins, data source plugins → 9. Provisioning: Configuration as code → 10. Advanced Features: Annotations, links",
    "course": "Grafana Fundamentals (Grafana Labs) or Monitoring with Grafana (Pluralsight)"
  },
  "Prometheus": {
    "path": "1. Time Series Database: Metrics collection → 2. Architecture: Server, pushgateway, exporters → 3. Data Model: Metrics, labels, samples → 4. PromQL: Query language, functions → 5. Service Discovery: Auto-discovery mechanisms → 6. Exporters: Node exporter, custom exporters → 7. Alerting: Alertmanager, rules → 8. Storage: Local storage, remote storage → 9. Federation: Hierarchical setups → 10. Best Practices: Cardinality, naming conventions",
    "course": "Prometheus Monitoring (Udemy) or Monitoring with Prometheus (Linux Academy)"
  },
  "Cloud Infrastructure": {
    "path": "1. Cloud Computing Models: IaaS, PaaS, SaaS → 2. Multi-Cloud Strategy: Vendor selection, portability → 3. Network Architecture: VPCs, subnets, gateways → 4. Compute Services: VMs, containers, serverless → 5. Storage Solutions: Block, object, file storage → 6. Database Services: Managed databases, NoSQL → 7. Security: Identity, encryption, compliance → 8. Cost Management: Budgeting, optimization → 9. Disaster Recovery: Backup, replication → 10. Automation: Infrastructure as code",
    "course": "Cloud Computing Concepts (Coursera - University of Illinois) or Multi-Cloud Architecture (A Cloud Guru)"
  },
  "Networking": {
    "path": "1. OSI Model: Seven layers, protocols → 2. TCP/IP: Internet protocol suite → 3. Routing: Static, dynamic routing protocols → 4. Switching: VLANs, spanning tree → 5. DNS: Domain name resolution → 6. Load Balancing: Layer 4/7, algorithms → 7. Firewalls: Stateful, stateless filtering → 8. VPN: Site-to-site, remote access → 9. Network Security: Encryption, authentication → 10. Troubleshooting: Tools, methodologies",
    "course": "Computer Networking (Coursera - University of Washington) or CCNA Certification (Cisco)"
  },
  "Shell Scripting": {
    "path": "1. Shell Basics: Command line, file system → 2. Variables: Environment, local variables → 3. Control Structures: If, loops, case statements → 4. Functions: Definition, parameters, scope → 5. File Operations: Reading, writing, permissions → 6. Text Processing: Grep, sed, awk → 7. Process Management: Jobs, signals → 8. Error Handling: Exit codes, error checking → 9. Advanced Features: Arrays, regular expressions → 10. Debugging: Set options, logging",
    "course": "Shell Scripting Tutorial (LinuxCommand.org) or Bash Scripting (Udemy)"
  },
  "Load Balancing": {
    "path": "1. Load Balancing Concepts: Distribution algorithms → 2. Layer 4 vs Layer 7: Network vs application → 3. Algorithms: Round robin, least connections → 4. Health Checks: Active, passive monitoring → 5. Session Persistence: Sticky sessions → 6. SSL Termination: Certificate management → 7. High Availability: Failover, redundancy → 8. Global Load Balancing: DNS-based, anycast → 9. Performance: Connection pooling, caching → 10. Monitoring: Metrics, logging",
    "course": "Load Balancing Fundamentals (F5 Networks) or High Availability and Load Balancing (Pluralsight)"
  },
  "High Availability": {
    "path": "1. Availability Concepts: Uptime, SLA, RTO/RPO → 2. Redundancy: Active-active, active-passive → 3. Failover: Automatic, manual failover → 4. Clustering: Shared storage, quorum → 5. Database HA: Replication, clustering → 6. Network HA: Bonding, VRRP → 7. Monitoring: Health checks, alerting → 8. Disaster Recovery: Backup, restore procedures → 9. Testing: Chaos engineering, DR drills → 10. Documentation: Runbooks, procedures",
    "course": "High Availability Architecture (AWS) or Building Resilient Systems (O'Reilly)"
  },
  "Resilience Engineering": {
    "path": "1. Resilience Principles: Fault tolerance, graceful degradation → 2. Circuit Breakers: Fail-fast patterns → 3. Bulkhead Pattern: Isolation, compartmentalization → 4. Retry Logic: Exponential backoff, jitter → 5. Timeout Management: Request timeouts → 6. Chaos Engineering: Fault injection → 7. Observability: Monitoring, alerting → 8. Incident Response: Runbooks, post-mortems → 9. Capacity Planning: Scaling strategies → 10. Testing: Fault injection, load testing",
    "course": "Building Resilient Systems (O'Reilly) or Chaos Engineering (Netflix)"
  },
  "Observability": {
    "path": "1. Three Pillars: Metrics, logs, traces → 2. Metrics: Time series, aggregation → 3. Logging: Structured logging, centralization → 4. Distributed Tracing: Request correlation → 5. Instrumentation: Code, infrastructure → 6. Sampling: Trace sampling strategies → 7. Correlation: Linking metrics, logs, traces → 8. Alerting: Anomaly detection, thresholds → 9. Dashboards: Operational views → 10. SLI/SLO: Service level indicators",
    "course": "Observability Engineering (Honeycomb) or Distributed Systems Observability (O'Reilly)"
  },
  "Cloud-Native Development": {
    "path": "1. Cloud-Native Principles: 12-factor app methodology → 2. Microservices: Service decomposition → 3. Containerization: Docker, container best practices → 4. Orchestration: Kubernetes, container scheduling → 5. Service Mesh: Istio, traffic management → 6. CI/CD: Pipeline automation → 7. Configuration Management: External configuration → 8. Observability: Monitoring, logging, tracing → 9. Security: Container security, RBAC → 10. Patterns: Circuit breaker, bulkhead, retry",
    "course": "Cloud Native Computing Foundation (CNCF) courses or Cloud Native DevOps with Kubernetes (O'Reilly)"
  },
  "Video Streaming Architecture": {
    "path": "1. Video Fundamentals: Codecs, bitrates, formats → 2. Encoding: H.264, H.265, VP9, AV1 → 3. Adaptive Streaming: HLS, DASH protocols → 4. CDN: Content distribution, edge caching → 5. Transcoding: Multi-bitrate encoding → 6. Storage: Video file management → 7. DRM: Content protection, licensing → 8. Analytics: QoE metrics, player analytics → 9. Live Streaming: Real-time encoding, low latency → 10. Optimization: Bandwidth, quality trade-offs",
    "course": "Video Streaming Technology (Coursera) or Building Video Streaming Applications (Pluralsight)"
  },
  "Linux": {
    "path": "1. Linux Basics: File system, commands → 2. User Management: Users, groups, permissions → 3. Process Management: ps, kill, jobs → 4. Package Management: apt, yum, dnf → 5. System Services: systemd, init systems → 6. Network Configuration: interfaces, routing → 7. File Permissions: chmod, chown, ACLs → 8. Shell Scripting: Bash automation → 9. Log Management: rsyslog, journald → 10. Performance Monitoring: top, htop, sar",
    "course": "Linux Command Line Basics (Udacity) or Linux System Administration (Linux Academy)"
  },
  "Incident Response": {
    "path": "1. Incident Management: ITIL framework, processes → 2. On-Call Practices: Escalation, rotation → 3. Alerting: Alert fatigue, signal vs noise → 4. Troubleshooting: Systematic approach → 5. Communication: Status pages, stakeholder updates → 6. Post-Mortems: Blameless culture, learning → 7. Runbooks: Standard operating procedures → 8. Tools: PagerDuty, Opsgenie → 9. Metrics: MTTR, MTBF → 10. Continuous Improvement: Retrospectives",
    "course": "Site Reliability Engineering (Google) or Incident Response (PagerDuty University)"
  },
  "On-call Practices": {
    "path": "1. On-Call Fundamentals: Responsibilities, expectations → 2. Alert Design: Quality over quantity → 3. Escalation: Multi-tier support → 4. Handoff Procedures: Knowledge transfer → 5. Burnout Prevention: Rotation, workload balance → 6. Tools: Alert management, communication → 7. Documentation: Runbooks, troubleshooting guides → 8. Metrics: Response time, resolution time → 9. Training: Shadow shifts, knowledge sharing → 10. Post-Incident: Learning, improvement",
    "course": "On-Call Engineering (Google SRE) or Effective On-Call Practices (PagerDuty)"
  }

  }
        # Generate recommendations for each missing skill
        for skill in missing_skills:
            if skill in skill_recommendations:
                learning_path.append(skill_recommendations[skill])
            else:
                # Generic recommendation if specific one isn't available
                learning_path.append(f"Develop proficiency in {skill} through online courses and projects")
                
        # Add general advice if there are missing skills
        if missing_skills:
            learning_path.append({
    "path": "Create portfolio projects that showcase your skills in these areas",
    "course": "Not applicable"
})

            
        return learning_path
    
    def score_resume(self, student_id: str, goal: str, resume_text: str) -> Dict[str, Any]:
        """
        Score a resume against a goal and provide detailed insights.
        
        Args:
            student_id: Unique student identifier
            goal: Target position or domain
            resume_text: Full plain-text resume content
            
        Returns:
            Dictionary with score, matched_skills, missing_skills, and suggested_learning_path
        """
        if not isinstance(resume_text, str):
            resume_text = ""
        resume_text = resume_text.strip()
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", resume_text)
        if not cleaned.strip():
            return {
                "score": 0.0,
                    "matched_skills": [],
                    "missing_skills": [],
                    "suggested_learning_path": []
                }
        if not isinstance(resume_text, str):
            resume_text = ""

        resume_text = resume_text.strip()
        if not resume_text:
            goal = goal if goal in self.goals_data else self.default_goal
            return {
                "score": 0.0,
                "matched_skills": [],
                "missing_skills": self.goals_data.get(goal, []),
                "suggested_learning_path": self._generate_learning_path(self.goals_data.get(goal, []))
            }

        # Check if the goal is supported, otherwise use default
        if goal not in self.config["model_goals_supported"]:
            logger.warning(f"Goal '{goal}' not supported, using default: {self.config['default_goal_model']}")
            goal = self.config["default_goal_model"]
            
        # Extract skills from resume
        found_skills = self._extract_skills_from_resume(resume_text)
        
        # Get matched and missing skills
        matched_skills, missing_skills = self._get_matched_missing_skills(found_skills, goal)
        
        # Generate ML model score if model exists for this goal
        if goal in self.models and self.shared_vectorizer:
            X = self.shared_vectorizer.transform([resume_text])

            
            # Get probability score from the model (positive class probability)
            score = float(self.models[goal].predict_proba(X)[0][1])
        else:
            # Fallback scoring based on skill match percentage if model unavailable
            logger.warning(f"No model available for goal '{goal}', using skill-based scoring")
            required_skills = self.goals.get(goal, [])
            score = len(matched_skills) / max(len(required_skills), 1) if required_skills else 0.0
            
        # Generate learning path
        learning_path = self._generate_learning_path(missing_skills)
        
        # Prepare and return the result
        result = {
            "score": score,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "suggested_learning_path": learning_path
        }
        
        return result
    
    def evaluate_passing(self, score: float) -> bool:
        """
        Determine if a score meets the minimum passing threshold.
        
        Args:
            score: The resume match score
            
passes the thresholdpip         Returns:
            Boolean indicating if the score install joblib
        """
        return score >= self.config["minimum_score_to_pass"]