#!/usr/bin/env python3
"""
Standalone Job Matcher Script
Matches CV against all job postings and generates ranked results
"""

import sys
sys.path.insert(0, 'src')

from tools.job_matcher import run_job_matcher

if __name__ == "__main__":
    # Configuration
    cv_path = "./src/data/Sai_AI_Ml_Resume.pdf"
    jobs_csv = "./src/data/jobs_list.csv"
    output_csv = "./src/data/job_matches_results.csv"
    
    print("=" * 70)
    print("JOB MATCHER - Analyzing All Job Postings")
    print("=" * 70)
    print()
    
    try:
        # Run the matcher
        summary = run_job_matcher(cv_path, jobs_csv, output_csv)
        print("\n" + summary)
        
        print("\n" + "=" * 70)
        print("✅ SUCCESS! Check the output file for complete results:")
        print(f"   {output_csv}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
