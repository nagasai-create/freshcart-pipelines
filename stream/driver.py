
import argparse
import json
import pika


EXCHANGE_NAME = "rapido-dispatch"


def main():
    parser = argparse.ArgumentParser(
        description="Receive and validate RabbitMQ requests."
    )
    parser.add_argument("name", help="Driver name")
    parser.add_argument("pattern", help="RabbitMQ topic binding pattern")
    parser.add_argument("--host", default="localhost")
    args = parser.parse_args()

    print(f"Starting driver: {args.name}")
    print(f"Binding pattern: {args.pattern}")

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

    result = channel.queue_declare(
        queue="",
        exclusive=True,
        auto_delete=True,
    )
    queue_name = result.method.queue

    channel.queue_bind(
        exchange=EXCHANGE_NAME,
        queue=queue_name,
        routing_key=args.pattern,
    )

    channel.basic_qos(prefetch_count=1)

    print(f"Driver {args.name} is ready.")
    print(f"Private queue: {queue_name}")
    print(f"Listening for: {args.pattern}")
    print("-" * 55)

    def receive_request(ch, method, properties, body):
        try:
            message = json.loads(body.decode("utf-8"))

            # D3 validation: zone must be present and non-empty.
            if (
                not isinstance(message, dict)
                or not message.get("zone")
            ):
                print(
                    f"[NACK] Driver={args.name} | "
                    f"Missing zone | Routing key={method.routing_key} | "
                    f"Message={message}",
                    flush=True,
                )

                ch.basic_nack(
                    delivery_tag=method.delivery_tag,
                    requeue=False,
                )
                return

            print(
                f"[RECEIVED] Driver={args.name} | "
                f"Routing key={method.routing_key} | "
                f"Message={message}",
                flush=True,
            )

            ch.basic_ack(delivery_tag=method.delivery_tag)

            print(f"[ACK] Driver={args.name}", flush=True)

        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            print(
                f"[NACK] Driver={args.name} | Invalid message: {error}",
                flush=True,
            )

            ch.basic_nack(
                delivery_tag=method.delivery_tag,
                requeue=False,
            )

    channel.basic_consume(
        queue=queue_name,
        on_message_callback=receive_request,
        auto_ack=False,
    )

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print(f"\nDriver {args.name} stopped.")
    finally:
        if connection.is_open:
            connection.close()


if __name__ == "__main__":
    main()
