from .crew import Tel
from dotenv import load_dotenv
load_dotenv()


def process_billing_query(customer_email, query):

    crew = Tel().crew()

    res = crew.kickoff(
        inputs={
            'customer_email' : customer_email,
            'query' : query

        }
    )

    return res.raw