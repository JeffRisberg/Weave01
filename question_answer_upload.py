import weave
import pandas as pd
from weave import Dataset
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize Weave
weave.init('question-answer')


def upload():

    # read the csv
    dataset_rows = pd.read_csv('datasets/hr-support-questions.csv', engine='python')

    # Create a dataset
    dataset = Dataset(name='hr-support-questions-data', rows=dataset_rows)

    # Publish the dataset
    weave.publish(dataset)

    # read the csv
    dataset_rows = pd.read_csv('datasets/knowledge-articles.csv', engine='python')

    # Create a dataset
    dataset = Dataset(name='knowledge-articles-data', rows=dataset_rows)

    # Publish the dataset
    weave.publish(dataset)

    log.info("done")

if __name__ == "__main__":
    upload()
