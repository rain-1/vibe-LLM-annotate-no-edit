#!/usr/bin/env python3
"""
Example annotation tasks showing the library's versatility.

The library is a general-purpose "prompted annotator" - you define
what you want to annotate via instructions, and it handles the
preservation guarantees.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_annotate import Document, Annotator
from llm_annotate.strategies import IndexLabelingStrategy, InlineDiffVerifyStrategy


# =============================================================================
# EXAMPLE ANNOTATION TASKS
# =============================================================================

TASKS = {
    "speaker_attribution": {
        "document": '''
"I can't believe you did that," she said, her voice trembling.

He shrugged. "What choice did I have?"

The room fell silent. Neither of them moved.

"You always have a choice," she whispered. "Always."
        '''.strip(),
        "instructions": "Identify the speaker for each piece of text. Use 'Narrator' for descriptive passages, character names or 'Character' for dialogue.",
        "annotation_types": ["Narrator", "Character", "She", "He"],
    },

    "sentiment_scoring": {
        "document": '''
The product arrived on time and was exactly as described.
However, the packaging was damaged and one item was broken.
Customer service was helpful but took three days to respond.
Overall, I would cautiously recommend this seller.
        '''.strip(),
        "instructions": "Score each sentence for sentiment: Positive, Negative, Neutral, or Mixed.",
        "annotation_types": ["Positive", "Negative", "Neutral", "Mixed"],
    },

    "toxicity_detection": {
        "document": '''
User1: Has anyone tried the new update?
User2: Yeah it's pretty good actually
User3: This is the worst update ever, the devs are idiots
User4: I disagree, I think they're doing their best
User5: User3 you're such a troll, go away
User1: Let's keep it civil please
        '''.strip(),
        "instructions": "Rate each message for toxicity: None, Mild, Moderate, or Severe.",
        "annotation_types": ["None", "Mild", "Moderate", "Severe"],
    },

    "code_review": {
        "document": '''
def process_user_input(data):
    query = "SELECT * FROM users WHERE id = " + data['id']
    result = db.execute(query)
    password = result['password']
    return eval(data.get('callback', 'None'))
        '''.strip(),
        "instructions": "Identify security issues in this code. Mark each line with: Safe, SQLInjection, InfoLeak, CodeExecution, or NeedsReview.",
        "annotation_types": ["Safe", "SQLInjection", "InfoLeak", "CodeExecution", "NeedsReview"],
    },

    "fact_checking": {
        "document": '''
The Great Wall of China is visible from space with the naked eye.
Water boils at 100°C at sea level under standard atmospheric pressure.
Humans only use 10% of their brain capacity.
The Earth orbits the Sun once every 365.25 days approximately.
        '''.strip(),
        "instructions": "Fact-check each statement. Label as: True, False, Misleading, or NeedsContext.",
        "annotation_types": ["True", "False", "Misleading", "NeedsContext"],
    },

    "reading_level": {
        "document": '''
The cat sat on the mat.
Photosynthesis is the process by which plants convert sunlight into energy.
The epistemological implications of quantum mechanics challenge our understanding of reality.
I like pizza.
        '''.strip(),
        "instructions": "Classify each sentence by reading level: Elementary, MiddleSchool, HighSchool, or College.",
        "annotation_types": ["Elementary", "MiddleSchool", "HighSchool", "College"],
    },

    "pii_detection": {
        "document": '''
Please contact John Smith at john.smith@example.com for more information.
His phone number is 555-123-4567 and he lives at 123 Main Street.
The meeting is scheduled for next Tuesday at 3pm.
Payment can be made to account number 1234-5678-9012-3456.
        '''.strip(),
        "instructions": "Identify PII (Personally Identifiable Information). Label each segment: NoPII, Name, Email, Phone, Address, Financial, or Other.",
        "annotation_types": ["NoPII", "Name", "Email", "Phone", "Address", "Financial"],
    },
}


def demo_task(task_name: str, strategy_name: str = "index_labeling"):
    """Run a demo annotation task."""
    if task_name not in TASKS:
        print(f"Unknown task: {task_name}")
        print(f"Available: {list(TASKS.keys())}")
        return

    task = TASKS[task_name]
    print(f"\n{'='*60}")
    print(f"TASK: {task_name}")
    print(f"{'='*60}")
    print(f"\nDocument:\n{task['document']}")
    print(f"\nInstructions: {task['instructions']}")
    print(f"Annotation types: {task['annotation_types']}")

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("\n[No API key - showing task setup only]")
        return

    # Run annotation
    annotator = Annotator(default_strategy=strategy_name)
    doc = Document(content=task['document'], name=f"{task_name}.txt")

    print(f"\nRunning with strategy: {strategy_name}")

    result = annotator.annotate(
        document=doc,
        instructions=task['instructions'],
        annotation_types=task['annotation_types'],
    )

    print(f"\nResults:")
    print(f"  Preserved: {result.document_preserved}")
    print(f"  Annotations: {len(result.annotations)}")

    for ann in result.annotations:
        preview = ann.anchor_text[:40] if ann.anchor_text else f"@{ann.position.offset}"
        print(f"  - [{ann.content}] {preview}...")


def list_tasks():
    """List all available annotation tasks."""
    print("\nAvailable Annotation Tasks:")
    print("-" * 40)
    for name, task in TASKS.items():
        print(f"\n{name}:")
        print(f"  Types: {', '.join(task['annotation_types'][:4])}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demo annotation tasks")
    parser.add_argument("task", nargs="?", help="Task to run")
    parser.add_argument("--strategy", "-s", default="index_labeling", help="Strategy to use")
    parser.add_argument("--list", "-l", action="store_true", help="List available tasks")
    args = parser.parse_args()

    if args.list or not args.task:
        list_tasks()
    else:
        demo_task(args.task, args.strategy)
