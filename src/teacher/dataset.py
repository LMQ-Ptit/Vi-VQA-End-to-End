"""
Teacher Dataset Module
Load và format data cho VLSP2023 VQA dataset
"""
import json
import os

def build_image_path(image_id, image_lookup, images_root="training-images"):
    """Build full path cho ảnh"""
    filename = image_lookup.get(str(image_id), f"{int(image_id):012d}.jpg")
    return f"{images_root}/{filename}"

def load_vqa_split(json_path, images_root="training-images"):
    """
    Load VQA split từ VLSP2023 JSON format

    Args:
        json_path: Path tới file JSON annotations
        images_root: Root directory chứa ảnh

    Returns:
        List of dict với keys: image_id, question, answer, image_path
    """
    with open(json_path, "r", encoding="utf-8") as f:
        train_json = json.load(f)

    image_lookup = train_json["images"]
    raw_data = []
    missing_images = 0

    for ann in train_json["annotations"].values():
        image_path = build_image_path(ann["image_id"], image_lookup, images_root)
        if os.path.exists(image_path):
            raw_data.append(
                {
                    "image_id": ann["image_id"],
                    "question": ann["question"],
                    "answer": ann["answer"],
                    "image_path": image_path,
                }
            )
        else:
            missing_images += 1

    print(f"Kept {len(raw_data)} samples, skipped {missing_images} missing images")
    return raw_data

def build_vqa_messages(question, answer):
    """Format messages cho SFTTrainer"""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        },
    ]

def format_vqa_example(example):
    """Format example cho Vision Language Modeling"""
    is_single = isinstance(example["question"], str)

    image_paths = [example["image_path"]] if is_single else example["image_path"]
    questions = [example["question"]] if is_single else example["question"]
    answers = [example["answer"]] if is_single else example["answer"]

    output_messages = []
    output_images = []

    for image_path, question, answer in zip(image_paths, questions, answers):
        output_messages.append(build_vqa_messages(question, answer))
        output_images.append([image_path])

    return {"messages": output_messages, "images": output_images}