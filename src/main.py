# used for custom inputs for your agent and tasks and  Intializing the crew

import sys
from crew import MatchtoJobsCrew


def run():

    inputs = {
        'path_to_cv': './src/data/Sai_AI_Ml_Resume.pdf',
        'path_to_jobs_csv': './src/data/jobs_list.csv',
    }

    result = MatchtoJobsCrew().crew().kickoff(inputs=inputs)
    print(result.raw)

if __name__ == '__main__':
    run()