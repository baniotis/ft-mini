from openai import OpenAI

client = OpenAI(api_key="your_key")

job_id = "your_job_id"
status = client.fine_tuning.jobs.retrieve(job_id)
print("Fine-tune job status:")
print(status.status)