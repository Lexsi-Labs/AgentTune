import json
from collections import Counter

INPUT_FILE = "logs/closed_loop/bulk_classified_failures.jsonl"


def analyze_results():
    root_causes = Counter()
    tools_called = Counter()
    failure_types = Counter()

    total = 0
    with open(INPUT_FILE) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            total += 1
            root_causes[data.get("root_cause", "unknown")] += 1
            failure_types[data.get("failure_type", "unknown")] += 1

            context = data.get("context", {})
            if "tool_called" in context:
                tools_called[context["tool_called"]] += 1

    print("=" * 50)
    print("📊 BULK EXPERIMENT ANALYSIS REPORT")
    print("=" * 50)
    print(f"Total Failed Trajectories Analyzed: {total}\\n")

    print("📈 ERROR TAXONOMY (Root Causes):")
    for rc, count in root_causes.most_common():
        print(f"  - {rc}: {count} ({count/total*100:.1f}%)")

    print("\\n🚨 DETECTION TYPES:")
    for ft, count in failure_types.most_common():
        print(f"  - {ft}: {count} ({count/total*100:.1f}%)")

    print("\\n🛠️ TOOLS THAT CRASHED/MISBEHAVED:")
    for tool, count in tools_called.most_common():
        print(f"  - {tool}: {count}")


if __name__ == "__main__":
    analyze_results()
