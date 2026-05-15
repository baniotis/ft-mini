from openai import OpenAI

client = OpenAI(api_key="your_key")

print("Uploading training file...")
train_file = client.files.create(
    file=open("train5.jsonl", "rb"),
    purpose="fine-tune"
)

print("Uploading validation file...")
val_file = client.files.create(
    file=open("validation5.jsonl", "rb"),
    purpose="fine-tune"
)

print("Starting fine-tuning job...")
job = client.fine_tuning.jobs.create(
    training_file=train_file.id,
    validation_file=val_file.id,
    model="gpt-4o-mini-2024-07-18",
    hyperparameters={
        "n_epochs": 3
    }
)

print("JOB ID:", job.id)