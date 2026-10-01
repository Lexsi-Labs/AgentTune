import json
import random
import uuid

OUTPUT_FILE = "tests/decide/fixtures/bulk_audit.jsonl"
NUM_EXPERIMENTS = 100


def generate_experiments():
    logs = []

    for _ in range(NUM_EXPERIMENTS):
        traj_id = f"traj_{uuid.uuid4().hex[:8]}"

        # Randomly pick a domain and an error scenario
        domain = random.choice(["text2sql", "math"])
        scenario = random.choice(
            [
                "tool_crash_hallucination",
                "tool_crash_wrong_tool",
                "low_judge_score_incomplete",
                "loop_collapse_routing",
                "loop_collapse_syntax",
            ]
        )

        if domain == "text2sql":
            if scenario == "tool_crash_hallucination":
                # Agent hallucinated a table
                logs.append(
                    {
                        "trajectory_id": traj_id,
                        "status": "error",
                        "stage_type": "tool_call",
                        "stage_name": "execute_sql",
                        "state_snapshot": {
                            "prompt": "Find all users who churned last month.",
                            "tool_called": "execute_sql",
                            "tool_input": "SELECT * FROM ChurnedUsers WHERE month = 'last';",
                        },
                        "error_details": "sqlite3.OperationalError: no such table: ChurnedUsers",
                    }
                )
            elif scenario == "tool_crash_wrong_tool":
                # Used math tool for SQL
                logs.append(
                    {
                        "trajectory_id": traj_id,
                        "status": "error",
                        "stage_type": "tool_call",
                        "stage_name": "calculator_tool",
                        "state_snapshot": {
                            "prompt": "Count the number of active subscriptions.",
                            "tool_called": "calculator",
                            "tool_input": "SELECT COUNT(*) FROM subscriptions WHERE active = True;",
                        },
                        "error_details": "SyntaxError: invalid syntax in math expression",
                    }
                )
            elif scenario == "low_judge_score_incomplete":
                # Missed a group by
                logs.append(
                    {
                        "trajectory_id": traj_id,
                        "status": "success",
                        "stage_type": "llm_judge",
                        "stage_name": "final_evaluation",
                        "state_snapshot": {
                            "prompt": "List the total revenue by department.",
                            "final_query": "SELECT department, revenue FROM sales;",
                            "issue": "Missing SUM and GROUP BY",
                        },
                        "result": {"score": 3.0},
                    }
                )
            elif scenario == "loop_collapse_routing":
                # Kept routing to math agent
                for _ in range(4):
                    logs.append(
                        {
                            "trajectory_id": traj_id,
                            "status": "success",
                            "stage_type": "agent_routing",
                            "stage_name": "route_to_math",
                            "state_snapshot": {
                                "prompt": "What is the average balance of users?",
                                "agent_called": "math_solver",
                            },
                        }
                    )
            elif scenario == "loop_collapse_syntax":
                # Kept trying bad syntax
                for _ in range(4):
                    logs.append(
                        {
                            "trajectory_id": traj_id,
                            "status": "success",
                            "stage_type": "tool_call",
                            "stage_name": "execute_sql",
                            "state_snapshot": {
                                "prompt": "Get top 5 users",
                                "tool_called": "execute_sql",
                                "tool_input": "SELECT * FROM users ORDER BY DESC LIMIT 5",
                            },
                        }
                    )
        else:  # MATH
            if scenario == "tool_crash_hallucination":
                # Hallucinated a python function
                logs.append(
                    {
                        "trajectory_id": traj_id,
                        "status": "error",
                        "stage_type": "tool_call",
                        "stage_name": "python_eval",
                        "state_snapshot": {
                            "prompt": "Calculate the roots of x^2 - 4 = 0",
                            "tool_called": "python_eval",
                            "tool_input": "find_roots(1, 0, -4)",
                        },
                        "error_details": "NameError: name 'find_roots' is not defined",
                    }
                )
            elif scenario == "tool_crash_wrong_tool":
                # Used SQL tool for math
                logs.append(
                    {
                        "trajectory_id": traj_id,
                        "status": "error",
                        "stage_type": "tool_call",
                        "stage_name": "execute_sql",
                        "state_snapshot": {
                            "prompt": "What is 15% of 1045?",
                            "tool_called": "execute_sql",
                            "tool_input": "15 / 100 * 1045",
                        },
                        "error_details": "sqlite3.OperationalError: near '15': syntax error",
                    }
                )
            elif scenario == "low_judge_score_incomplete":
                # Skipped reasoning steps
                logs.append(
                    {
                        "trajectory_id": traj_id,
                        "status": "success",
                        "stage_type": "llm_judge",
                        "stage_name": "final_evaluation",
                        "state_snapshot": {
                            "prompt": "Solve for x: 3x + 5 = 14",
                            "final_answer": "x = 3",
                            "issue": "Provided correct answer but no reasoning steps shown.",
                        },
                        "result": {"score": 4.5},
                    }
                )
            elif scenario == "loop_collapse_routing":
                # Kept routing to SQL agent
                for _ in range(4):
                    logs.append(
                        {
                            "trajectory_id": traj_id,
                            "status": "success",
                            "stage_type": "agent_routing",
                            "stage_name": "route_to_sql",
                            "state_snapshot": {
                                "prompt": "Compute the integral of 2x",
                                "agent_called": "sql_agent",
                            },
                        }
                    )
            elif scenario == "loop_collapse_syntax":
                for _ in range(4):
                    logs.append(
                        {
                            "trajectory_id": traj_id,
                            "status": "success",
                            "stage_type": "tool_call",
                            "stage_name": "calculator",
                            "state_snapshot": {
                                "prompt": "Evaluate 5 + 5",
                                "tool_called": "calculator",
                                "tool_input": "5 + 5 =",
                            },
                        }
                    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for log in logs:
            f.write(json.dumps(log) + "\n")

    print(
        f"Generated {len(logs)} audit events across {NUM_EXPERIMENTS} simulated trajectories in {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    generate_experiments()
