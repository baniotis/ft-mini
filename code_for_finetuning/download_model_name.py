from openai import OpenAI

client = OpenAI(api_key="your_key")

job_id = "your_job_id"

job = client.fine_tuning.jobs.retrieve(job_id)

model_name = job.fine_tuned_model
print("Your fine-tuned model name is:")
print(model_name)