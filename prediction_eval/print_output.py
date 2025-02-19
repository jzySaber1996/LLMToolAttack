import json

with open("output_gpt_prediction.json", "r") as f_in:
    data = json.load(f_in)
    f_in.close()

for i in range(20):
    print(f"+ Example {i + 20}: Truth_support_knowledge: {data[i + 20]['support_knowledge']};\n "
          f"- Example {i + 20}: Predicted_support_knowledge: {data[i + 20]['pred_support_knowledge']}\n")