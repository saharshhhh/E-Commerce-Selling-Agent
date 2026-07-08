import json
from agent import SupportAgent

def run_tests():
    agent = SupportAgent()
    with open('test_cases.json', 'r') as f:
        test_cases = json.load(f)

    results = []
    print(f"Running {len(test_cases)} test cases...\n")

    for case in test_cases:
        question = case['question']
        expected = case['expected_mention']
        if isinstance(expected, str):
            expected = [expected]

        print(f"Question: {question}")
        actual = agent.ask(question)
        print(f"Actual: {actual[:100]}...")

        passed = any(phrase.lower() in actual.lower() for phrase in expected)
        print(f"Passed: {passed}")
        print("-" * 20)

        results.append({
            "question": question,
            "expected_mention": expected,
            "actual_answer": actual,
            "passed": passed
        })

    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    total_passed = sum(1 for r in results if r['passed'])
    print(f"Tests Complete: {total_passed}/{len(test_cases)} passed.")

if __name__ == "__main__":
    run_tests()