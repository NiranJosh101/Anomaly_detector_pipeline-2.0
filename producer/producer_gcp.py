import os
import json
import time
import pandas as pd
from kafka import KafkaProducer
from google.cloud import storage
from dotenv import load_dotenv

from src.logging_setup import logger
from src.exception_setup.exception import AnomalyDetectionException



load_dotenv()  

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
TOPIC_NAME = os.getenv("TOPIC_NAME", "raw_events")
GCS_BUCKET = os.getenv("GCS_BUCKET", None)
LOCAL_DATA_PATH = os.getenv("LOCAL_DATA_PATH", "data/sample.csv")


def fetch_gcs_file(bucket_name: str, prefix: str = ""):
    try:

        """Fetch CSV files from a GCS bucket and return as Pandas DataFrames."""
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)

        for blob in blobs:
            if blob.name.endswith(".csv"):
                logger.logging.info(f"Downloading {blob.name} from GCS")
                data = blob.download_as_text()
                df = pd.read_csv(pd.io.common.StringIO(data))
                yield df

    except Exception as e:
            raise AnomalyDetectionException(e, sys)


def create_producer():
    try:
        """Initialize and return a Kafka producer instance."""
        return KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=5,
        )

    except Exception as e:
            raise AnomalyDetectionException(e, sys)



def stream_data(producer: KafkaProducer, source="local"):
    """Continuously stream data to Kafka from local or GCS."""
    if source == "gcs" and GCS_BUCKET:
        data_stream = fetch_gcs_file(GCS_BUCKET)
    else:
        logger.logging.info(f"Reading local data from {LOCAL_DATA_PATH}")
        data_stream = [pd.read_csv(LOCAL_DATA_PATH)]

    for df in data_stream:
        logger.logging.info(f"Streaming {len(df)} records to Kafka topic '{TOPIC_NAME}'")
        for _, row in df.iterrows():
            record = row.to_dict()
            producer.send(TOPIC_NAME, value=record)
            print(f"Sent: {record}")
            time.sleep(1)  
        producer.flush()

    logger.logging.info("All data streamed successfully!")



if __name__ == "__main__":
    print("Connecting to Kafka...")
    producer = create_producer()
    print("Connected successfully!")

    source = "gcs" if GCS_BUCKET else "local"
    stream_data(producer, source)
