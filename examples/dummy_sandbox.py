import json
import sys


def main():
    # Simulate an isolated sandbox executing a tool and returning state diffs.
    # We output a JSON array of the states encountered.
    states = [
        "DB connection opened",
        "Queried user_id=123",
        "Returned 1 record",
        "Connection closed",
    ]
    print(json.dumps(states))
    sys.exit(0)


if __name__ == "__main__":
    main()
