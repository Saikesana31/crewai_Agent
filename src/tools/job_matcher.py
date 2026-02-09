"""
Custom Job Matcher Tool
Efficiently matches CV against all job postings in CSV file
"""

import pandas as pd
import re
from typing import Dict, List, Tuple
from collections import Counter
import json


class JobMatcher:
    """Match CV skills and experience against job postings"""
    
    def __init__(self, cv_text: str):
        self.cv_text = cv_text.lower()
        self.skills = self._extract_skills()
        self.experience_years = self._extract_years_experience()
        
    def _extract_skills(self) -> set:
        """Extract technical skills from CV text"""
        # Common AI/ML skills to look for
        skill_keywords = {
            # Programming Languages
            'python', 'r', 'sql', 'java', 'javascript', 'scala',
            
            # ML Frameworks
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn', 
            'xgboost', 'lightgbm', 'catboost',
            
            # Deep Learning
            'deep learning', 'neural networks', 'cnn', 'rnn', 'lstm', 
            'transformer', 'bert', 'gpt', 'attention',
            
            # GenAI & LLM
            'langchain', 'langgraph', 'llm', 'large language model',
            'openai', 'gpt', 'rag', 'retrieval augmented generation',
            'prompt engineering', 'fine-tuning', 'embeddings',
            'vector database', 'agentic', 'multi-agent',
            
            # NLP
            'nlp', 'natural language processing', 'spacy', 'nltk', 
            'text classification', 'sentiment analysis', 'ner',
            'named entity recognition',
            
            # Computer Vision
            'computer vision', 'opencv', 'image processing', 
            'object detection', 'image segmentation', 'ocr',
            
            # Cloud & MLOps
            'aws', 'azure', 'gcp', 'sagemaker', 'lambda', 'bedrock',
            'docker', 'kubernetes', 'mlflow', 'ci/cd', 'mlops',
            
            # Databases
            'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch',
            'pinecone', 'chromadb', 'weaviate', 'faiss',
            
            # Big Data
            'spark', 'pyspark', 'hadoop', 'kafka', 'airflow', 'databricks',
            
            # Data Science
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
            'data science', 'machine learning', 'statistics',
            'a/b testing', 'experimentation',
            
            # Tools
            'git', 'github', 'jenkins', 'terraform', 'tableau'
        }
        
        found_skills = set()
        for skill in skill_keywords:
            if skill in self.cv_text:
                found_skills.add(skill)
                
        return found_skills
    
    def _extract_years_experience(self) -> int:
        """Extract years of experience from CV"""
        # Look for patterns like "4+ years", "5 years"
        patterns = [
            r'(\d+)\+?\s*years?',
            r'(\d+)\s*years? of',
        ]
        
        years = []
        for pattern in patterns:
            matches = re.findall(pattern, self.cv_text)
            years.extend([int(m) for m in matches])
        
        return max(years) if years else 4  # Default to 4 if not found
    
    def calculate_match_score(self, job_row: pd.Series) -> Tuple[int, List[str], List[str]]:
        """
        Calculate match score for a job posting
        Returns: (score, matching_points, gaps)
        """
        job_text = ""
        
        # Combine relevant job fields
        if pd.notna(job_row.get('title')):
            job_text += str(job_row['title']).lower() + " "
        if pd.notna(job_row.get('descriptionText')):
            job_text += str(job_row['descriptionText']).lower() + " "
        if pd.notna(job_row.get('companyDescription')):
            job_text += str(job_row['companyDescription']).lower() + " "
        
        score = 0
        matching_points = []
        gaps = []
        
        # 1. Skill matching (60% weight)
        job_skills = set()
        for skill in self.skills:
            if skill in job_text:
                job_skills.add(skill)
        
        if len(self.skills) > 0:
            skill_match_ratio = len(job_skills) / len(self.skills)
            score += int(skill_match_ratio * 60)
            
            # Top matching skills
            if job_skills:
                top_skills = list(job_skills)[:5]
                matching_points.append(f"Matching skills: {', '.join(top_skills)}")
        
        # 2. AI/ML specific keywords (20% weight)
        ai_keywords = ['ai', 'ml', 'machine learning', 'artificial intelligence', 
                       'deep learning', 'data science', 'llm', 'genai']
        ai_matches = sum(1 for kw in ai_keywords if kw in job_text)
        score += min(20, ai_matches * 3)
        
        if ai_matches > 0:
            matching_points.append("AI/ML focused role")
        
        # 3. Experience level (10% weight)
        seniority = str(job_row.get('seniorityLevel', '')).lower()
        if 'senior' in seniority or 'lead' in seniority or 'principal' in seniority:
            if self.experience_years >= 4:
                score += 10
                matching_points.append("Seniority level matches experience")
            else:
                gaps.append("May need more experience for senior role")
        elif 'entry' in seniority or 'junior' in seniority:
            if self.experience_years > 5:
                matching_points.append("Overqualified - could be good fit")
            score += 5
        else:
            score += 7  # Mid-level or not specified
        
        # 4. Cloud platform match (5% weight)
        cloud_platforms = ['aws', 'azure', 'gcp', 'google cloud']
        cloud_match = any(platform in job_text and platform in self.cv_text 
                         for platform in cloud_platforms)
        if cloud_match:
            score += 5
            matching_points.append("Cloud platform experience matches")
        
        # 5. Location preference (5% weight) - US locations preferred
        location = str(job_row.get('location', '')).lower()
        if any(state in location for state in ['tx', 'texas', 'dallas', 'ca', 'california', 
                                                 'ny', 'new york', 'wa', 'washington']):
            score += 5
        
        # Ensure score is within 0-100 range
        score = min(100, max(0, score))
        
        # Add gaps if low score
        if score < 40:
            gaps.append("Limited skill overlap with job requirements")
        
        return score, matching_points[:3], gaps[:3]
    
    def match_all_jobs(self, csv_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Match CV against all jobs in CSV
        Returns sorted DataFrame with match scores
        """
        print(f"Reading jobs from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"Found {len(df)} jobs to evaluate")
        
        # Calculate match scores for all jobs
        results = []
        for idx, row in df.iterrows():
            score, matching_points, gaps = self.calculate_match_score(row)
            
            result = {
                'rank': 0,  # Will be set after sorting
                'match_score': score,
                'job_title': row.get('title', 'N/A'),
                'company_name': row.get('companyName', 'N/A'),
                'location': row.get('location', 'N/A'),
                'seniority_level': row.get('seniorityLevel', 'N/A'),
                'employment_type': row.get('employmentType', 'N/A'),
                'posted_at': row.get('postedAt', 'N/A'),
                'apply_url': row.get('applyUrl', 'N/A'),
                'matching_points': ' | '.join(matching_points) if matching_points else 'General match',
                'gaps': ' | '.join(gaps) if gaps else 'None identified'
            }
            results.append(result)
            
            # Progress indicator
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(df)} jobs...")
        
        # Create DataFrame and sort by score
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('match_score', ascending=False)
        results_df['rank'] = range(1, len(results_df) + 1)
        
        # Reorder columns
        cols = ['rank', 'match_score', 'job_title', 'company_name', 'location', 
                'seniority_level', 'employment_type', 'matching_points', 'gaps', 
                'posted_at', 'apply_url']
        results_df = results_df[cols]
        
        print(f"\nCompleted! Processed all {len(results_df)} jobs")
        print(f"\nTop 10 matches:")
        print(results_df.head(10)[['rank', 'match_score', 'job_title', 'company_name', 'location']])
        
        # Save results if output path provided
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"\nResults saved to {output_path}")
        
        return results_df


def run_job_matcher(cv_path: str, jobs_csv_path: str, output_path: str = None) -> str:
    """
    Main function to run job matching
    """
    from tools.custom_tool import PDFSearchTool
    
    # Read CV
    print(f"Reading CV from {cv_path}...")
    pdf_tool = PDFSearchTool()
    cv_text = pdf_tool._run(cv_path)
    
    # Create matcher and process jobs
    matcher = JobMatcher(cv_text)
    results_df = matcher.match_all_jobs(jobs_csv_path, output_path)
    
    # Generate summary
    summary = f"""
Job Matching Analysis Complete!
================================

Total Jobs Analyzed: {len(results_df)}
CV Skills Identified: {len(matcher.skills)}
Estimated Experience: {matcher.experience_years}+ years

Top 15 Job Matches:
-------------------
"""
    
    for idx, row in results_df.head(15).iterrows():
        summary += f"\n{row['rank']}. {row['job_title']} at {row['company_name']}"
        summary += f"\n   Location: {row['location']}"
        summary += f"\n   Match Score: {row['match_score']}%"
        summary += f"\n   Key Points: {row['matching_points']}"
        if row['gaps'] != 'None identified':
            summary += f"\n   Gaps: {row['gaps']}"
        summary += "\n"
    
    summary += f"\nFull results available with all {len(results_df)} jobs ranked by match score."
    
    return summary


if __name__ == "__main__":
    # Test the matcher
    cv_path = "./src/data/Sai_AI_Ml_Resume.pdf"
    jobs_csv = "./src/data/jobs_list.csv"
    output_csv = "./src/data/job_matches_results.csv"
    
    summary = run_job_matcher(cv_path, jobs_csv, output_csv)
    print(summary)
