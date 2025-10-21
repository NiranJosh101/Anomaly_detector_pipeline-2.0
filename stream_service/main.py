import os
import sys
import asyncio
import json
from collections import deque
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from aiokafka import AIOKafkaConsumer

from src.exception_setup.exception import AnomalyDetectionException
from src.logging_setup import logger

app = FastAPI(title="Anomaly Stream API", version="1.0")


KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
RESULT_TOPIC = os.getenv("RESULT_TOPIC", "anomaly_predictions")


LATEST_RESULTS: deque = deque(maxlen=16)

# Kafka consumer 
consumer: AIOKafkaConsumer | None = None


async def consume_predictions():
    try:
        """Background task that consumes inference results from Kafka."""
        global consumer
        consumer = AIOKafkaConsumer(
            OUT_TOPIC,
            bootstrap_servers=KAFKA_BROKER,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
            group_id="stream_consumer_group"
        )

        await consumer.start()
        print(f"[INFO] Kafka consumer connected to {KAFKA_BROKER} — listening on topic: {RESULT_TOPIC}")

        try:
            async for msg in consumer:
                data = msg.value
                if isinstance(data, dict):
                    LATEST_RESULTS.append(data)
                else:
                    print(f"[WARN] Unexpected message format: {data}")
        except Exception as e:
            print(f"[ERROR] Kafka consumer crashed: {e}")
        finally:
            await consumer.stop()

    except Exception as e:
            raise AnomalyDetectionException(e, sys)

@app.on_event("startup")
async def startup_event():
    try:
        """Start background Kafka consumer on app startup."""
        asyncio.create_task(consume_predictions())

    except Exception as e:
            raise AnomalyDetectionException(e, sys)


@app.get("/latest", response_model=List[Dict[str, Any]])
async def get_latest_results(limit: int = 10):
    try:
        """Return the most recent predictions."""
        if not LATEST_RESULTS:
            raise HTTPException(status_code=404, detail="No results yet.")
        return list(LATEST_RESULTS)[-limit:]

    except Exception as e:
            raise AnomalyDetectionException(e, sys)


@app.get("/")
async def root():
    try:
        return {"status": "ok", "topic": OUT_TOPIC, "buffer_size": len(LATEST_RESULTS)}

    except Exception as e:
            raise AnomalyDetectionException(e, sys)
