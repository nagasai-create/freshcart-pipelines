
import argparse
import json
import random
import time
from datetime import datetime, timezone

import pika


EXCHANGE_NAME = "rapido-dispatch"

CITY_ZONES = {
    "bangalore": ["koramangala", "indiranagar", "whitefield"],
    "chennai": ["tnagar", "velachery", "adyar"],
}


def main():
    parser = argparse.ArgumentParser(
        description="Publish valid and invalid requests to RabbitMQ."
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--duration", type=int, default=120)
    args = parser.parse_args()

    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=args.host,
            port=5672,
            heartbeat=60,
            blocked_connection_timeout=30,
        )
    )

    channel = connection.channel()

    channel.exchange_declare(
        exchange=EXCHANGE_NAME,
        exchange_type="topic",
        durable=True,
    )

    print(f"Connected to {EXCHANGE_NAME}")
    print("Publishing every 2 seconds. Every 10th message is invalid.")

    request_count = 0
    start_time = time.monotonic()

    try:
        while time.monotonic() - start_time < args.duration:
            request_count += 1
            request_id = f"REQ-{request_count:05d}"

            if request_count % 10 == 0:
                # Deliberately invalid: zone field is missing.
                city = "bangalore"
                request_type = random.choice(["ride", "food"])

                routing_key = f"{city}.{request_type}"

                message = {
                    "id": request_id,
                    "city": city,
                    "type": request_type,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }

                label = "PUBLISHED BAD"

            else:
                city = random.choice(list(CITY_ZONES.keys()))
                zone = random.choice(CITY_ZONES[city])
                request_type = random.choice(["ride", "food"])

                routing_key = f"{city}.{zone}.{request_type}"

                message = {
                    "id": request_id,
                    "city": city,
                    "zone": zone,
                    "type": request_type,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }

                label = "PUBLISHED"

            channel.basic_publish(
                exchange=EXCHANGE_NAME,
                routing_key=routing_key,
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    content_type="application/json",
                    delivery_mode=2,
                ),
            )

            print(
                f"[{label}] {routing_key} | {json.dumps(message)}",
                flush=True,
            )

            time.sleep(2)

    except KeyboardInterrupt:
        print("\nDispatcher stopped by user.")

    finally:
        if connection.is_open:
            connection.close()

        print(f"Total published requests: {request_count}")


if __name__ == "__main__":
    main()
