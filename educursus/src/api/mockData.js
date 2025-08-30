// src/api/mockData.js
export const mockCareers = [
    { id: 'data-scientist', title: 'AI & Data Scientist', description: 'Analyze complex data to find trends and make predictions using machine learning models.' },
    { id: 'ml-engineer', title: 'Machine Learning Engineer', description: 'Design and build production-level AI models and systems that learn from data.' },
    { id: 'cloud-architect', title: 'Cloud Solutions Architect', description: 'Design and manage scalable, secure, and robust cloud infrastructure for applications.' },
];

export const mockRoadmaps = {
    'data-scientist': [
        { name: 'Foundations', skills: ['Python Programming', 'Statistics & Probability', 'Linear Algebra'] },
        { name: 'Core Skills', skills: ['Data Wrangling with Pandas', 'Data Visualization (Matplotlib, Seaborn)', 'SQL Databases'] },
        { name: 'Machine Learning', skills: ['Scikit-Learn Fundamentals', 'Regression & Classification Models', 'Model Evaluation'] },
        { name: 'Advanced AI', skills: ['Deep Learning with TensorFlow/PyTorch', 'Natural Language Processing (NLP)', 'Big Data Technologies (Spark)'] },
        { name: 'Deployment', skills: ['Building REST APIs (Flask/FastAPI)', 'Containerization with Docker', 'Cloud AI Services (AWS/GCP)'] }
    ],
};